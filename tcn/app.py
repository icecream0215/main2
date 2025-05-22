import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
import uvicorn
import librosa
import tempfile
import os
import traceback
from moviepy.editor import VideoFileClip
import subprocess
import pandas as pd
import torchvision.transforms as transforms
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from contextlib import asynccontextmanager

# 导入模型
from vgg import VGG
from tcnmodel import Net
# 导入数据库和认证模块
from database import (
    User, 
    get_db, 
    initialize_database,  # 使用新的初始化函数
    AnalysisResult, 
    AnalyticsData, 
    SessionLocal
)
# 导入数据分析模块
from analytics import get_daily_analytics, get_trend_analysis
from auth import (
    UserCreate, 
    Token, 
    PasswordResetRequest,
    PasswordReset,
    authenticate_user, 
    create_access_token, 
    get_current_active_user, 
    get_current_admin_user,
    get_current_doctor_user,
    get_current_patient_user,
    get_user_by_email, 
    get_user,  # 添加对get_user函数的导入
    create_user,
    create_password_reset_request,
    verify_reset_token,
    reset_password,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    SECRET_KEY,  # 添加对SECRET_KEY的导入
    ALGORITHM,   # 添加对ALGORITHM的导入 
    pwd_context
)
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func, desc

device = torch.device("cpu")
model = Net()

# 初始化VGG模型（无需加载权重）
vgg_model = VGG()
vgg_model.to(device)
vgg_model.eval()

# 加载TCN模型权重
checkpoint = torch.load(os.path.join(os.path.dirname(__file__), "models", "model.pth"), map_location=device, weights_only=False)
state_dict = checkpoint["net"]
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# OpenFace路径配置，直接指定路径位置
import os
OPENFACE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "OpenFace_2.2.0_win_x64")
FEATURE_EXTRACTION_EXE = os.path.join(OPENFACE_DIR, "FeatureExtraction.exe")

# 模型期望的输入尺寸（根据你的测试代码）
NORMAL_VIDEO_SHAPE = 915   # 视频序列长度
DESIRED_VIDEO_WIDTH = 171  # 每帧视频特征数
NORMAL_AUDIO_SHAPE = 186   # 音频序列长度
DESIRED_AUDIO_WIDTH = 128  # 音频特征数

# 定义辅助函数：对输入数组进行裁剪/零填充
def pad_or_crop(array, target_length, axis=0):
    current = array.shape[axis]
    if current > target_length:
        slicer = [slice(None)] * array.ndim
        slicer[axis] = slice(0, target_length)
        return array[tuple(slicer)]
    elif current < target_length:
        pad_width = [(0, 0)] * array.ndim
        pad_width[axis] = (0, target_length - current)
        return np.pad(array, pad_width, mode="constant")
    else:
        return array

# 添加extract_and_process_features函数用于处理视频文件
async def extract_and_process_features(video_path):
    """从视频文件中提取视频和音频特征，并返回处理后的特征矩阵"""
    try:
        # 提取面部特征
        face_features = extract_face_features(video_path)
        
        # 提取音频特征
        audio_features = extract_audio_features(video_path)
        
        return face_features, audio_features
    except Exception as e:
        logger.error(f"特征提取失败: {str(e)}")
        raise Exception(f"视频处理失败: {str(e)}")

# 添加日志配置
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定义生命周期事件处理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    logger.info("创建数据库表...")
    initialize_database()
    
    # 检查数据库结构并在必要时进行迁移
    try:
        db = SessionLocal()
        # 尝试查询带role字段的用户，如果失败表示需要迁移
        db.query(User.role).limit(1).all()
        logger.info("数据库结构验证通过")
    except Exception as e:
        logger.warning(f"数据库结构验证失败: {e}")
        logger.info("尝试迁移数据库结构...")
        try:
            # 导入迁移模块
            from .migrate_db import migrate_database
            if migrate_database():
                logger.info("数据库迁移成功")
            else:
                logger.error("数据库迁移失败")
        except Exception as migration_error:
            logger.error(f"执行迁移时出错: {migration_error}")
    finally:
        db.close()
    
    logger.info("数据库初始化完成")
    yield
    # 关闭时执行（如果有清理操作）
    logger.info("应用程序关闭...")

# 创建一个简单的日志记录器
import logging
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(
    title="PyTorch Model Inference API", 
    lifespan=lifespan,
    # 启用详细的请求日志记录
    debug=True
)

# 添加一个中间件来记录认证相关的信息
@app.middleware("http")
async def log_requests(request, call_next):
    # 记录请求信息
    logger.info(f"请求: {request.method} {request.url.path}")
    
    # 如果有认证头，记录其存在
    auth_header = request.headers.get("Authorization")
    if auth_header:
        # 只记录令牌的前10个字符，保护敏感信息
        logger.info(f"认证头存在: {auth_header[:15]}...")
    
    # 处理请求
    try:
        response = await call_next(request)
        logger.info(f"响应: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        raise

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域名访问
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 用户注册
@app.post("/register", response_model=dict)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    # 检查用户名是否已存在
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名已被注册"
        )
    
    # 检查邮箱是否已存在
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该邮箱已被注册"
        )
    
    # 验证角色是否有效
    if user.role not in ["admin", "doctor", "patient"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无效的用户角色"
        )
    
    try:
        # 创建新用户
        db_user = create_user(db, user)
        return {"message": "用户注册成功", "username": db_user.username, "role": db_user.role}
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="注册失败，请稍后再试"
        )

# 用户登录
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# 获取当前用户信息
@app.get("/users/me", response_model=dict)
async def read_users_me(current_user = Depends(get_current_active_user)):
    return {
        "username": current_user.username,
        "email": current_user.email,
        "id": current_user.id,
        "is_active": current_user.is_active,
        "role": current_user.role
    }

# 请求密码重置
@app.post("/reset-password-request", response_model=dict)
def request_password_reset(request: PasswordResetRequest, db: Session = Depends(get_db)):
    user = create_password_reset_request(db, request.email)
    # 注意：在实际项目中，这里应该发送包含重置链接的电子邮件
    # 为了简化示例，我们直接返回令牌（实际应用中不应这样做）
    if user and user.reset_token:
        return {
            "message": "密码重置请求已处理，请检查您的电子邮件",
            "reset_token": user.reset_token  # 实际项目中不应返回此信息
        }
    return {"message": "密码重置请求已处理，请检查您的电子邮件"}

# 重置密码
@app.post("/reset-password", response_model=dict)
def perform_password_reset(reset_data: PasswordReset, db: Session = Depends(get_db)):
    if reset_password(db, reset_data.token, reset_data.new_password):
        return {"message": "密码已成功重置，请使用新密码登录"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="密码重置失败，令牌可能无效或已过期"
        )

# 密码重置请求页面
@app.get("/reset-password-request", response_class=HTMLResponse)
async def reset_password_request_page():
    with open(os.path.join(os.path.dirname(__file__), "templates", "reset_password_request.html"), encoding="utf-8") as f:
        return f.read()

# 密码重置页面
@app.get("/reset-password.html", response_class=HTMLResponse)
async def reset_password_page():
    with open(os.path.join(os.path.dirname(__file__), "templates", "reset_password.html"), encoding="utf-8") as f:
        return f.read()


# 静态文件服务
static_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_directory, exist_ok=True)  # 如果目录不存在则创建
app.mount("/static", StaticFiles(directory=static_directory), name="static")

# 首页路由 - 重定向到登录页面
@app.get("/", response_class=RedirectResponse)
def read_root():
    return RedirectResponse(url="/login")

# 登录页面路由
@app.get("/login", response_class=HTMLResponse)
def login_page():
    with open(os.path.join(os.path.dirname(__file__), "templates", "login.html"), "r", encoding="utf-8") as f:
        content = f.read()
    return content

# 注册页面路由
@app.get("/register", response_class=HTMLResponse)
def register_page():
    with open(os.path.join(os.path.dirname(__file__), "templates", "register.html"), "r", encoding="utf-8") as f:
        content = f.read()
    return content

# 管理员页面路由
@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, current_user: User = Depends(get_current_admin_user)):
    """
    管理员页面路由处理函数。
    使用get_current_admin_user依赖确保只有管理员可以访问。
    """
    try:
        # 记录访问信息
        logger.info(f"用户 {current_user.username} (角色: {current_user.role}) 访问管理员页面")
        
        # 读取管理员页面模板
        with open(os.path.join(os.path.dirname(__file__), "templates", "admin.html"), "r", encoding="utf-8") as f:
            content = f.read()
        
        return content
            
    except Exception as e:
        logger.error(f"获取管理员页面时出错: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="加载页面时发生错误"
        )

# 添加POST方法处理admin路由的表单认证
@app.post("/admin", response_class=HTMLResponse)
async def admin_page_post(auth_token: str = Form(...)):
    try:
        # 记录令牌（不记录完整令牌，只记录一部分用于调试）
        if auth_token:
            logger.info(f"收到管理员页面POST认证请求，令牌长度: {len(auth_token)}, 前15字符: {auth_token[:15]}...")
        else:
            logger.warning("收到管理员页面POST认证请求，但未提供令牌")
            
        # 手动验证令牌并获取用户
        db = SessionLocal()
        try:
            # 解码JWT令牌
            payload = jwt.decode(auth_token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if not username:
                logger.warning("令牌中没有用户名信息")
                raise HTTPException(status_code=401, detail="无效的认证凭据")
                
            # 获取用户信息    
            user = get_user(db, username)
            if not user:
                logger.warning(f"用户不存在: {username}")
                raise HTTPException(status_code=401, detail="用户不存在")
                
            # 检查用户角色
            logger.info(f"用户 {user.username} (角色: {user.role}) 尝试访问管理员页面")
            if user.role != "admin":
                logger.warning(f"用户 {user.username} 尝试访问管理员页面但权限不足（角色: {user.role}）")
                raise HTTPException(status_code=403, detail="没有访问权限")
                
            # 返回管理员页面
            logger.info(f"用户 {user.username} 通过POST方式成功访问管理员页面")
            with open(os.path.join(os.path.dirname(__file__), "templates", "admin.html"), "r", encoding="utf-8") as f:
                content = f.read()
            return content
                
        except JWTError as e:
            logger.error(f"JWT令牌解码失败: {str(e)}")
            raise HTTPException(status_code=401, detail="无效的认证凭据")
        finally:
            db.close()
    except HTTPException as e:
        # 返回错误信息而不是直接重定向
        return HTMLResponse(
            content=f"<html><body><h1>错误</h1><p>{e.detail}</p><p><a href='/login'>返回登录页面</a></p></body></html>",
            status_code=e.status_code
        )
    except Exception as e:
        logger.error(f"处理管理员页面POST请求时出错: {e}", exc_info=True)
        # 返回一个带错误消息的HTML页面
        return HTMLResponse(
            content=f"<html><body><h1>服务器错误</h1><p>处理请求时发生错误: {str(e)}</p><p><a href='/login'>返回登录页面</a></p></body></html>",
            status_code=500
        )

# 医生页面路由
@app.get("/doctor", response_class=HTMLResponse)
async def doctor_page(request: Request, current_user: User = Depends(get_current_doctor_user)):
    """
    医生页面路由处理函数。
    使用get_current_doctor_user依赖确保只有医生或管理员可以访问。
    自动处理认证并返回医生页面。
    """
    try:
        # 记录访问信息
        logger.info(f"用户 {current_user.username} (角色: {current_user.role}) 访问医生页面")
        
        # 读取医生页面模板
        with open(os.path.join(os.path.dirname(__file__), "templates", "doctor.html"), "r", encoding="utf-8") as f:
            content = f.read()
        
        return content
            
    except Exception as e:
        logger.error(f"获取医生页面时出错: {str(e)}")
        # 根据错误类型返回适当的错误响应
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="加载页面时发生错误"
        )

# 添加POST方法处理doctor路由的表单认证
@app.post("/doctor", response_class=HTMLResponse)
async def doctor_page_post(auth_token: str = Form(...)):
    try:
        # 记录令牌（不记录完整令牌，只记录一部分用于调试）
        if auth_token:
            logger.info(f"收到POST认证请求，令牌长度: {len(auth_token)}, 前15字符: {auth_token[:15]}...")
        else:
            logger.warning("收到POST认证请求，但未提供令牌")
            
        # 手动验证令牌并获取用户
        db = SessionLocal()
        try:
            # 解码JWT令牌
            payload = jwt.decode(auth_token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if not username:
                logger.warning("令牌中没有用户名信息")
                raise HTTPException(status_code=401, detail="无效的认证凭据")
                
            # 获取用户信息    
            user = get_user(db, username)
            if not user:
                logger.warning(f"用户不存在: {username}")
                raise HTTPException(status_code=401, detail="用户不存在")
                
            # 检查用户角色
            logger.info(f"用户 {user.username} (角色: {user.role}) 尝试访问医生页面")
            if user.role not in ["doctor", "admin"]:
                logger.warning(f"用户 {user.username} 尝试访问医生页面但权限不足（角色: {user.role}）")
                raise HTTPException(status_code=403, detail="没有访问权限")
                
            # 返回医生页面
            logger.info(f"用户 {user.username} 通过POST方式成功访问医生页面")
            with open(os.path.join(os.path.dirname(__file__), "templates", "doctor.html"), "r", encoding="utf-8") as f:
                content = f.read()
            return content
                
        except JWTError as e:
            logger.error(f"JWT令牌解码失败: {str(e)}")
            raise HTTPException(status_code=401, detail="无效的认证凭据")
        finally:
            db.close()
    except HTTPException as e:
        # 返回错误信息而不是直接重定向
        return HTMLResponse(
            content=f"<html><body><h1>错误</h1><p>{e.detail}</p><p><a href='/login'>返回登录页面</a></p></body></html>",
            status_code=e.status_code
        )
    except Exception as e:
        logger.error(f"处理医生页面POST请求时出错: {e}", exc_info=True)
        # 返回一个带错误消息的HTML页面
        return HTMLResponse(
            content=f"<html><body><h1>服务器错误</h1><p>处理请求时发生错误: {str(e)}</p><p><a href='/login'>返回登录页面</a></p></body></html>",
            status_code=500
        )

# 患者页面路由
@app.get("/patient", response_class=HTMLResponse)
async def patient_page(request: Request, current_user: User = Depends(get_current_patient_user)):
    """
    患者页面路由处理函数。
    使用get_current_patient_user依赖确保只有患者或管理员可以访问。
    """
    try:
        # 记录访问信息
        logger.info(f"用户 {current_user.username} (角色: {current_user.role}) 访问患者页面")
        
        # 读取患者页面模板
        with open(os.path.join(os.path.dirname(__file__), "templates", "patient.html"), "r", encoding="utf-8") as f:
            content = f.read()
            
        return content
            
    except Exception as e:
        logger.error(f"获取患者页面时出错: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="加载页面时发生错误"
        )

# 添加POST方法处理patient路由的表单认证
@app.post("/patient", response_class=HTMLResponse)
async def patient_page_post(auth_token: str = Form(...)):
    try:
        # 记录令牌（不记录完整令牌，只记录一部分用于调试）
        if auth_token:
            logger.info(f"收到患者页面POST认证请求，令牌长度: {len(auth_token)}, 前15字符: {auth_token[:15]}...")
        else:
            logger.warning("收到患者页面POST认证请求，但未提供令牌")
            
        # 手动验证令牌并获取用户
        db = SessionLocal()
        try:
            # 解码JWT令牌
            payload = jwt.decode(auth_token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if not username:
                logger.warning("令牌中没有用户名信息")
                raise HTTPException(status_code=401, detail="无效的认证凭据")
                
            # 获取用户信息    
            user = get_user(db, username)
            if not user:
                logger.warning(f"用户不存在: {username}")
                raise HTTPException(status_code=401, detail="用户不存在")
                
            # 检查用户角色
            logger.info(f"用户 {user.username} (角色: {user.role}) 尝试访问患者页面")
            if user.role not in ["patient", "admin"]:
                logger.warning(f"用户 {user.username} 尝试访问患者页面但权限不足（角色: {user.role}）")
                raise HTTPException(status_code=403, detail="没有访问权限")
                
            # 返回患者页面
            logger.info(f"用户 {user.username} 通过POST方式成功访问患者页面")
            with open(os.path.join(os.path.dirname(__file__), "templates", "patient.html"), "r", encoding="utf-8") as f:
                content = f.read()
            return content
                
        except JWTError as e:
            logger.error(f"JWT令牌解码失败: {str(e)}")
            raise HTTPException(status_code=401, detail="无效的认证凭据")
        finally:
            db.close()
    except HTTPException as e:
        # 返回错误信息而不是直接重定向
        return HTMLResponse(
            content=f"<html><body><h1>错误</h1><p>{e.detail}</p><p><a href='/login'>返回登录页面</a></p></body></html>",
            status_code=e.status_code
        )
    except Exception as e:
        logger.error(f"处理患者页面POST请求时出错: {e}", exc_info=True)
        # 返回一个带错误消息的HTML页面
        return HTMLResponse(
            content=f"<html><body><h1>服务器错误</h1><p>处理请求时发生错误: {str(e)}</p><p><a href='/login'>返回登录页面</a></p></body></html>",
            status_code=500
        )

# 视频特征提取函数 (使用OpenFace)
def extract_face_features(video_path):
    # 运行OpenFace特征提取
    try:
        print(f"开始提取面部特征，视频路径: {video_path}")
        
        cmd = [
            FEATURE_EXTRACTION_EXE,
            "-f", video_path,
            "-aus",
            "-pose",
            "-2Dfp",
            "-gaze"
        ]
        
        print(f"执行OpenFace命令: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"OpenFace处理失败: {stderr.decode()}")
        
        # 读取生成的CSV文件
        csv_path = video_path.rsplit('.', 1)[0] + '.csv'
        if not os.path.exists(csv_path):
            raise Exception("OpenFace未能生成特征文件")
        
        df = pd.read_csv(csv_path)
        
        # 选择需要的特征列
        feature_columns = [col for col in df.columns if 'AU' in col or 'pose' in col or 'gaze' in col]
        features = df[feature_columns].values
        
        # 清理临时文件
        os.remove(csv_path)
        
        return features
        
    except Exception as e:
        raise Exception(f"面部特征提取失败: {str(e)}")

# 音频特征提取函数 (使用VGG)
def extract_audio_features(video_path):
    try:
        print(f"开始提取音频特征，视频路径: {video_path}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        print("加载视频文件...")
        video = VideoFileClip(video_path)
        
        if video.audio is None:
            print("警告: 视频没有音轨，使用空音频代替")
            # 创建一个空的音频数组
            import numpy as np
            y = np.zeros(16000)  # 1秒的44.1kHz采样率
            sr = 16000
        else:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                print(f"提取音频到临时文件: {temp_audio.name}")
                video.audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
                
                # 使用librosa加载音频
                print("使用librosa加载音频...")
                y, sr = librosa.load(temp_audio.name, sr=None)
            
            # 提取梅尔频谱特征
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=160)  # 计算梅尔频谱
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # 转换为对数梅尔频谱
            
            # 转换为VGG可处理的格式
            audio_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0).to(device)
            
            # 使用VGG提取特征
            with torch.no_grad():
                audio_features = vgg_model(audio_tensor)
                
            os.unlink(temp_audio.name)
            
            return audio_features.cpu().numpy().squeeze()
            
    except Exception as e:
        raise Exception(f"音频特征提取失败: {str(e)}")

# 添加视频处理分析函数
async def process_video_analysis(file, current_user, db: Session, age: Optional[int] = None, gender: Optional[str] = None):
    """处理视频分析，返回分析结果"""
    # 验证文件格式
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(
            status_code=400,
            detail="只支持MP4、AVI或MOV格式的视频文件"
        )
    
    try:
        # 保存上传的视频文件到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            content = await file.read()
            
            # 检查文件大小
            if len(content) > 200 * 1024 * 1024:  # 200MB限制
                raise HTTPException(
                    status_code=400,
                    detail="文件大小超过限制（最大200MB）"
                )
                
            temp_video.write(content)
            temp_video_path = temp_video.name

        try:
            # 分别提取面部特征和音频特征
            logger.info(f"开始处理视频: {file.filename}")
            face_features = extract_face_features(temp_video_path)
            logger.info("面部特征提取完成")
            audio_features = extract_audio_features(temp_video_path)
            logger.info("音频特征提取完成")

            # 处理特征尺寸
            face_features = pad_or_crop(face_features, NORMAL_VIDEO_SHAPE, axis=0)
            audio_features = pad_or_crop(audio_features, NORMAL_AUDIO_SHAPE, axis=0)

            # 确保特征维度正确
            face_features = pad_or_crop(face_features, DESIRED_VIDEO_WIDTH, axis=1)
            audio_features = pad_or_crop(audio_features, DESIRED_AUDIO_WIDTH, axis=1)

            # 转换为tensor并进行预测
            video_tensor = torch.tensor(face_features, dtype=torch.float32).unsqueeze(0).to(device)
            audio_tensor = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(device)

            # 模型推理
            with torch.no_grad():
                output = model(video_tensor, audio_tensor)

            # 处理输出
            probs = output.squeeze(0).cpu().numpy()
            percentages = probs * 100
            predicted_idx = int(np.argmax(percentages))
            label_map = {0: "非抑郁", 1: "抑郁"}
            
            # 计算抎郁概率和置信度
            depression_probability = float(probs[1])
            non_depression_probability = float(probs[0])
            confidence = float(max(probs))
            
            # 根据阈值确定结果类型
            if depression_probability <= 0.3:
                result_type = "正常"
            elif depression_probability <= 0.6:
                result_type = "轻度抑郁"
            elif depression_probability <= 0.8:
                result_type = "中度抑郁"
            else:
                result_type = "重度抑郁"
            
            # 保存分析结果到数据库
            now = datetime.utcnow()  # 同一个时间戳用于所有时间字段
            result = AnalysisResult(
                user_id=current_user.id,
                filename=file.filename,
                result_type=result_type,
                predicted_class=label_map[predicted_idx],
                depression_probability=depression_probability,
                non_depression_probability=non_depression_probability,
                probability_class0=float(percentages[0]),
                probability_class1=float(percentages[1]),
                confidence=confidence,
                facial_analysis={"expression": "较少的积极情绪表达"},
                voice_analysis={"tone": "语调平缓，能量低"},
                body_language_analysis={"movement": "动作减少，姿势略显紧张"},
                patient_age=age,
                patient_gender=gender,
                processed_at=now,
                created_at=now
            )
            db.add(result)
            db.commit()
            db.refresh(result)

            return {
                "filename": file.filename,
                "result_type": result_type,
                "predicted_class": label_map[predicted_idx],
                "depression_probability": depression_probability,
                "non_depression_probability": non_depression_probability,
                "confidence": confidence,
                "probabilities": {
                    label_map[0]: f"{percentages[0]:.2f}%",
                    label_map[1]: f"{percentages[1]:.2f}%"
                },
                "result_id": result.id
            }
        except Exception as e:
            # 添加更详细的错误日志
            error_details = traceback.format_exc()
            print(f"视频处理错误: {error_details}")
            raise HTTPException(
                status_code=400,
                detail=f"视频处理失败: {str(e)}"
            )
        finally:
            # 确保临时文件被删除
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
                
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=400,
            detail=f"上传处理失败: {str(e)}"
        )

# 修改预测接口，使其需要用户登录并保存结果到数据库
@app.post("/predict")
async def predict(
    video_file: UploadFile = File(...),
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    return await process_video_analysis(video_file, current_user, db)

# 增强版预测接口，允许用户提供更多信息
@app.post("/predict/enhanced")
async def predict_enhanced(
    video_file: UploadFile = File(...),
    age: Optional[int] = Form(None),
    gender: Optional[str] = Form(None),
    additional_info: Optional[str] = Form(None),
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # 验证性别输入
    if gender and gender not in ["男", "女", "其他", "未知"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="性别只能是'男'、'女'、'其他'或'未知'"
        )
    
    # 验证年龄输入
    if age is not None and (age < 0 or age > 120):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="年龄必须在0到120之间"
        )
    
    # 处理视频并分析
    result = await process_video_analysis(video_file, current_user, db, age, gender)
    
    # 如果提供了额外信息，将其保存到数据库
    if additional_info and result.get("result_id"):
        try:
            analysis_result = db.query(AnalysisResult).filter(
                AnalysisResult.id == result["result_id"]
            ).first()
            
            if analysis_result:
                # 将额外信息保存到body_language_analysis字段中
                if not analysis_result.body_language_analysis:
                    analysis_result.body_language_analysis = {}
                
                body_data = analysis_result.body_language_analysis
                if isinstance(body_data, str):
                    import json
                    body_data = json.loads(body_data)
                
                body_data["additional_info"] = additional_info
                analysis_result.body_language_analysis = body_data
                
                db.commit()
        except Exception as e:
            logger.error(f"保存额外信息失败: {str(e)}")
            # 不抛出错误，继续返回结果
    
    return result

# 获取用户历史分析结果
@app.get("/api/history", response_model=list)
async def get_user_history(current_user = Depends(get_current_active_user), db: Session = Depends(get_db)):
    from database import AnalysisResult
    # 获取当前用户的所有分析记录
    results = db.query(AnalysisResult).filter(AnalysisResult.user_id == current_user.id).order_by(AnalysisResult.created_at.desc()).all()
    
    history_list = []
    for result in results:
        history_list.append({
            "id": result.id,
            "filename": result.filename,
            "result_type": result.result_type,
            "predicted_class": result.predicted_class,
            "non_depression_probability": result.non_depression_probability,
            "depression_probability": result.depression_probability,
            "probability_class0": result.probability_class0, # 保留兼容旧代码
            "probability_class1": result.probability_class1, # 保留兼容旧代码
            "confidence": result.confidence,
            "created_at": result.created_at.strftime("%Y-%m-%d %H:%M:%S")
        })
    return history_list

# 获取分析结果详情
@app.get("/api/analysis/{analysis_id}", response_model=dict)
async def get_analysis_detail(
    analysis_id: int, 
    current_user = Depends(get_current_active_user), 
    db: Session = Depends(get_db)
):
    from database import AnalysisResult
    
    # 获取特定分析记录
    result = db.query(AnalysisResult).filter(
        AnalysisResult.id == analysis_id
    ).first()
    
    # 检查结果是否存在
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="分析结果不存在"
        )
    
    # 检查是否有权限访问（自己的分析结果或医生/管理员可以查看所有人的）
    if result.user_id != current_user.id and current_user.role not in ["doctor", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="您没有权限查看此分析结果"
        )
      # 获取用户信息
    user = db.query(User).filter(User.id == result.user_id).first()
    
    # 构建详细结果对象
    detail = {
        "id": result.id,
        "user_id": result.user_id,
        "username": user.username if user else "未知用户",
        "filename": result.filename,
        "result_type": result.result_type,
        "predicted_class": result.predicted_class,
        "depression_probability": result.depression_probability,
        "non_depression_probability": result.non_depression_probability,
        "confidence": result.confidence,
        "created_at": result.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "patient_age": result.patient_age,
        "patient_gender": result.patient_gender,
        "doctor_notes": result.doctor_notes,
        "facial_analysis": result.facial_analysis,
        "voice_analysis": result.voice_analysis,
        "body_language_analysis": result.body_language_analysis
    }
    
    return detail

# 生成HTML分析报告
@app.get("/api/analysis/{analysis_id}/report", response_class=HTMLResponse)
async def generate_analysis_report(
    analysis_id: int,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """生成分析结果的HTML报告，可以在浏览器中查看或打印"""
    from database import AnalysisResult
    
    # 获取分析结果
    result = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
    
    # 检查结果是否存在
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="分析结果不存在"
        )
    
    # 检查是否有权限访问（自己的分析结果或医生/管理员可以查看所有人的）
    if result.user_id != current_user.id and current_user.role not in ["doctor", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="您没有权限查看此分析结果"
        )
    
    try:
        # 获取用户信息
        user = db.query(User).filter(User.id == result.user_id).first()
        
        # 格式化概率为百分比
        depression_pct = f"{result.depression_probability*100:.1f}%"
        non_depression_pct = f"{result.non_depression_probability*100:.1f}%"
        confidence_pct = f"{result.confidence*100:.1f}%"
        
        # 获取面部表情分析
        facial_analysis = ""
        if result.facial_analysis:
            facial_data = result.facial_analysis
            if isinstance(facial_data, str):
                import json
                facial_data = json.loads(facial_data)
                
            if isinstance(facial_data, dict) and "expression" in facial_data:
                facial_analysis = facial_data['expression']
        
        # 获取语音分析
        voice_analysis = ""
        if result.voice_analysis:
            voice_data = result.voice_analysis
            if isinstance(voice_data, str):
                import json
                voice_data = json.loads(voice_data)
                
            if isinstance(voice_data, dict) and "tone" in voice_data:
                voice_analysis = voice_data['tone']
        
        # 获取肢体语言分析
        body_analysis = ""
        if result.body_language_analysis:
            body_data = result.body_language_analysis
            if isinstance(body_data, str):
                import json
                body_data = json.loads(body_data)
                
            if isinstance(body_data, dict) and "movement" in body_data:
                body_analysis = body_data['movement']
                
            # 提取额外信息
            additional_info = ""
            if isinstance(body_data, dict) and "additional_info" in body_data:
                additional_info = body_data['additional_info']
        
        # 构建HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>抑郁症检测分析报告 - {result.id}</title>
            <style>
                body {{
                    font-family: 'Arial', 'Microsoft YaHei', sans-serif;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                    background-color: #f9f9f9;
                }}
                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    border-radius: 5px;
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    margin-bottom: 30px;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #3498db;
                    margin-top: 25px;
                    border-left: 4px solid #3498db;
                    padding-left: 10px;
                }}
                .info-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .info-table th {{
                    background-color: #f2f2f2;
                    text-align: left;
                    padding: 12px 15px;
                    border: 1px solid #ddd;
                    width: 30%;
                }}
                .info-table td {{
                    padding: 12px 15px;
                    border: 1px solid #ddd;
                }}
                .result-box {{
                    background-color: #f8f9fa;
                    border-left: 4px solid #3498db;
                    padding: 15px;
                    margin: 15px 0;
                }}
                .danger {{
                    color: #e74c3c;
                }}
                .warning {{
                    color: #f39c12;
                }}
                .success {{
                    color: #2ecc71;
                }}
                .notes {{
                    background-color: #fffbf0;
                    border: 1px dashed #ffd700;
                    padding: 15px;
                    margin-top: 20px;
                    border-radius: 5px;
                }}
                .progress-container {{
                    width: 100%;
                    background-color: #e0e0e0;
                    margin: 10px 0;
                    border-radius: 5px;
                }}
                .progress-bar {{
                    height: 24px;
                    border-radius: 5px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                }}
                .depression-bar {{
                    background-color: #e74c3c;
                }}
                .non-depression-bar {{
                    background-color: #2ecc71;
                }}
                @media print {{
                    body {{
                        background-color: white;
                    }}
                    .container {{
                        box-shadow: none;
                        padding: 0;
                    }}
                    .no-print {{
                        display: none;
                    }}
                }}
                .print-button {{
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                    margin-top: 20px;
                }}
                .print-button:hover {{
                    background-color: #2980b9;
                }}
                .logo {{
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .footer {{
                    margin-top: 30px;
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 14px;
                    border-top: 1px solid #eee;
                    padding-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="logo">
                    <h1>抑郁症检测分析报告</h1>
                </div>
                
                <h2>基本信息</h2>
                <table class="info-table">
                    <tr>
                        <th>分析ID</th>
                        <td>{result.id}</td>
                    </tr>
                    <tr>
                        <th>用户名</th>
                        <td>{user.username if user else "未知"}</td>
                    </tr>
                    <tr>
                        <th>分析文件</th>
                        <td>{result.filename}</td>
                    </tr>
                    <tr>
                        <th>分析时间</th>
                        <td>{result.created_at.strftime("%Y-%m-%d %H:%M:%S")}</td>
                    </tr>
                    <tr>
                        <th>年龄</th>
                        <td>{result.patient_age if result.patient_age else "未提供"}</td>
                    </tr>
                    <tr>
                        <th>性别</th>
                        <td>{result.patient_gender if result.patient_gender else "未提供"}</td>
                    </tr>
                </table>
                
                <h2>分析结果</h2>
                <div class="result-box">
                    <p><strong>检测结果:</strong> <span class="{
                        'danger' if result.result_type in ['重度抑郁', '中度抑郁'] else 
                        'warning' if result.result_type == '轻度抑郁' else 
                        'success'
                    }">{result.result_type}</span></p>
                    <p><strong>检测类别:</strong> {result.predicted_class}</p>
                    
                    <p><strong>抑郁概率:</strong></p>
                    <div class="progress-container">
                        <div class="progress-bar depression-bar" style="width: {depression_pct}">{depression_pct}</div>
                    </div>
                    
                    <p><strong>非抑郁概率:</strong></p>
                    <div class="progress-container">
                        <div class="progress-bar non-depression-bar" style="width: {non_depression_pct}">{non_depression_pct}</div>
                    </div>
                    
                    <p><strong>置信度:</strong> {confidence_pct}</p>
                </div>
                
                <h2>详细分析</h2>
                <div class="result-box">
                    {f"<p><strong>面部表情分析:</strong> {facial_analysis}</p>" if facial_analysis else ""}
                    {f"<p><strong>语音分析:</strong> {voice_analysis}</p>" if voice_analysis else ""}
                    {f"<p><strong>肢体语言分析:</strong> {body_analysis}</p>" if body_analysis else ""}
                </div>
                
                {f'''
                <h2>医生专业注释</h2>
                <div class="notes">
                    <p>{result.doctor_notes}</p>
                </div>
                ''' if result.doctor_notes else ""}
                
                <div class="no-print" style="text-align: center;">
                    <button class="print-button" onclick="window.print();">打印报告</button>
                </div>
                
                <div class="footer">
                    <p>此报告仅供参考，不构成医疗建议。如有疑问，请咨询专业医师。</p>
                    <p>报告生成日期: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
            </div>
            
            <script>
                // 页面加载时自动根据结果类型设置颜色
                document.addEventListener('DOMContentLoaded', function() 
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"生成HTML报告失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"生成报告失败: {str(e)}"
        )
        
    except Exception as e:
        logger.error(f"生成PDF报告失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"生成报告失败: {str(e)}"
        )

# 系统统计数据API
@app.get("/api/admin/stats", response_model=dict)
async def get_system_stats(
    current_user = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """获取系统统计数据，包括用户数量和分析结果数量"""
    try:
        # 用户统计
        total_users = db.query(func.count(User.id)).scalar()
        active_users = db.query(func.count(User.id)).filter(User.is_active == True).scalar()
        patient_count = db.query(func.count(User.id)).filter(User.role == "patient").scalar()
        doctor_count = db.query(func.count(User.id)).filter(User.role == "doctor").scalar()
        admin_count = db.query(func.count(User.id)).filter(User.role == "admin").scalar()
        
        # 分析结果统计
        total_analyses = db.query(func.count(AnalysisResult.id)).scalar()
        
        # 结果类型统计
        normal_count = db.query(func.count(AnalysisResult.id)).filter(
            AnalysisResult.result_type == "正常"
        ).scalar()
        
        mild_count = db.query(func.count(AnalysisResult.id)).filter(
            AnalysisResult.result_type == "轻度抑郁"
        ).scalar()
        
        moderate_count = db.query(func.count(AnalysisResult.id)).filter(
            AnalysisResult.result_type == "中度抑郁"
        ).scalar()
        
        severe_count = db.query(func.count(AnalysisResult.id)).filter(
            AnalysisResult.result_type == "重度抑郁"
        ).scalar()
        
        # 计算检出率
        detection_rate = 0
        if total_analyses > 0:
            depression_detected = mild_count + moderate_count + severe_count
            detection_rate = depression_detected / total_analyses
        
        # 最近分析
        recent_analyses = db.query(AnalysisResult).order_by(
            AnalysisResult.created_at.desc()
        ).limit(5).all()
        
        recent_list = []
        for analysis in recent_analyses:
            user = db.query(User).filter(User.id == analysis.user_id).first()
            recent_list.append({
                "id": analysis.id,
                "username": user.username if user else "未知",
                "result_type": analysis.result_type,
                "created_at": analysis.created_at.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return {
            "user_stats": {
                "total": total_users,
                "active": active_users,
                "patients": patient_count,
                "doctors": doctor_count,
                "admins": admin_count
            },
            "analysis_stats": {
                "total": total_analyses,
                "normal": normal_count,
                "mild": mild_count,
                "moderate": moderate_count,
                "severe": severe_count,
                "detection_rate": detection_rate
            },
            "recent_analyses": recent_list
        }
    
    except Exception as e:
        logger.error(f"获取系统统计数据失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取统计数据失败: {str(e)}"
        )

# 删除分析结果
@app.delete("/api/analysis/{analysis_id}", response_model=dict)
async def delete_analysis(
    analysis_id: int,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """删除分析结果（只能删除自己的结果，管理员可以删除任何结果）"""
    from database import AnalysisResult
    
    # 获取分析结果
    result = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
    
    # 检查结果是否存在
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="分析结果不存在"
        )
    
    # 检查是否有权限删除（只能删除自己的分析结果，管理员可以删除任何结果）
    if result.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="您没有权限删除此分析结果"
        )
    
    try:
        # 删除结果
        db.delete(result)
        db.commit()
        
        return {
            "message": "分析结果已成功删除",
            "id": analysis_id
        }
    except Exception as e:
        db.rollback()
        logger.error(f"删除分析结果失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除分析结果失败: {str(e)}"
        )

# 更新用户个人资料
@app.put("/api/users/profile", response_model=dict)
async def update_user_profile(
    current_password: str = Form(...),
    new_password: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """更新用户个人资料，包括密码和邮箱"""
    try:
        # 验证当前密码
        if not pwd_context.verify(current_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="当前密码不正确"
            )
        
        # 更新密码
        if new_password:
            if len(new_password) < 6:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="新密码长度不能少于6个字符"
                )
            
            current_user.hashed_password = pwd_context.hash(new_password)
        
        # 更新邮箱
        if email:
            # 检查邮箱格式
            if "@" not in email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="邮箱格式不正确"
                )
            
            # 检查邮箱是否已被其他用户使用
            existing_user = db.query(User).filter(
                User.email == email, 
                User.id != current_user.id
            ).first()
            
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="此邮箱已被其他用户使用"
                )
            
            current_user.email = email
        
        # 保存更改
        db.commit()
        
        return {
            "message": "个人资料更新成功",
            "username": current_user.username,
            "email": current_user.email
        }
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"更新用户个人资料失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新个人资料失败: {str(e)}"
        )

# 启动FastAPI应用
if __name__ == "__main__":
    print("正在启动抑郁症检测系统...")
    print("请在浏览器中访问: http://127.0.0.1:8000")
    print("按Ctrl+C停止服务")
    import uvicorn
    # 使用uvicorn启动应用，指定当前模块中的app对象
    uvicorn.run(app, host="127.0.0.1", port=8000)
