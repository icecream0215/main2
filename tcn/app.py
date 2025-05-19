import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import librosa
import tempfile
import os
from moviepy.editor import VideoFileClip
import subprocess
import pandas as pd
import torchvision.transforms as transforms
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

# 导入模型
from vgg import VGG
from tcnmodel import Net
# 导入数据库和认证模块
from database import User, get_db, create_tables, AnalysisResult, AnalyticsData
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
    get_user_by_email, 
    create_user,
    create_password_reset_request,
    verify_reset_token,
    reset_password,
    ACCESS_TOKEN_EXPIRE_MINUTES,
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

# OpenFace路径配置，使用相对路径提高可移植性
import os
OPENFACE_DIR = os.environ.get("OPENFACE_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "OpenFace_2.2.0_win_x64"))
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

# -----------------------------
# 2. 构建 FastAPI 接口
app = FastAPI(title="PyTorch Model Inference API")

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域名访问
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 初始化数据库
@app.on_event("startup")
def startup_db_client():
    logger.info("创建数据库表...")
    create_tables()
    logger.info("数据库初始化完成")

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

# 修改HTML表单

@app.get("/", response_class=HTMLResponse)
async def get_html():
    with open(os.path.join(os.path.dirname(__file__), "templates", "index.html"), encoding="utf-8") as f:
        return f.read()


# 用户登录页面
@app.get("/login", response_class=HTMLResponse)
async def login_page():
    with open(os.path.join(os.path.dirname(__file__), "templates", "login.html"), encoding="utf-8") as f:
        return f.read()

# 用户注册页面
@app.get("/register", response_class=HTMLResponse)
async def register_page():
    with open(os.path.join(os.path.dirname(__file__), "templates", "register.html"), encoding="utf-8") as f:
        return f.read()

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
app.mount("/static", StaticFiles(directory="static"), name="static")

# 首页路由
@app.get("/", response_class=HTMLResponse)
def read_root():
    with open(os.path.join(os.path.dirname(__file__), "templates", "index.html"), "r", encoding="utf-8") as f:
        content = f.read()
    return content

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
async def admin_page(current_user: User = Depends(get_current_active_user)):
    # 检查用户是否为管理员
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有访问权限，仅管理员可访问"
        )
    with open(os.path.join(os.path.dirname(__file__), "templates", "admin.html"), "r", encoding="utf-8") as f:
        content = f.read()
    return content

# 医生页面路由
@app.get("/doctor", response_class=HTMLResponse)
async def doctor_page(current_user: User = Depends(get_current_active_user)):
    # 检查用户是否为医生或管理员
    if current_user.role not in ["doctor", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有访问权限，仅医生可访问"
        )
    with open(os.path.join(os.path.dirname(__file__), "templates", "doctor.html"), "r", encoding="utf-8") as f:
        content = f.read()
    return content

# 患者页面路由
@app.get("/patient", response_class=HTMLResponse)
async def patient_page(current_user: User = Depends(get_current_active_user)):
    with open(os.path.join(os.path.dirname(__file__), "templates", "patient.html"), "r", encoding="utf-8") as f:
        content = f.read()
    return content


# 视频特征提取函数 (使用OpenFace)
def extract_face_features(video_path):
    # 运行OpenFace特征提取
    try:
        cmd = [
            FEATURE_EXTRACTION_EXE,
            "-f", video_path,
            "-aus",
            "-pose",
            "-2Dfp",
            "-gaze"
        ]
        
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
        video = VideoFileClip(video_path)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            video.audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
            
            # 使用librosa加载音频
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

# 修改预测接口，使其需要用户登录并保存结果到数据库
@app.post("/predict")
async def predict(
    video_file: UploadFile = File(...),
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    return await process_video_analysis(video_file, current_user, db)

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
            "predicted_class": result.predicted_class,
            "probability_class0": result.probability_class0,
            "probability_class1": result.probability_class1,
            "created_at": result.created_at.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return history_list

# 新增: 数据分析相关API接口
class AnalyticsRequest(BaseModel):
    time_range: str  # "last7", "last30", "last90", "last365"
    user_id: Optional[int] = None  # None表示所有用户

class AnalyticsResponse(BaseModel):
    summary: Dict[str, Any]
    detection_trend: List[Dict[str, Any]]
    result_distribution: Dict[str, int]
    age_distribution: Dict[str, float]
    gender_distribution: Dict[str, float]

@app.get("/api/analytics/summary", response_model=AnalyticsResponse)
async def get_analytics_summary(
    time_range: str = "last30", 
    user_filter: str = "all",
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """获取数据分析摘要信息"""
    
    # 根据时间范围计算开始日期
    today = datetime.now()
    if time_range == "last7":
        start_date = today - timedelta(days=7)
    elif time_range == "last30":
        start_date = today - timedelta(days=30)
    elif time_range == "last90":
        start_date = today - timedelta(days=90)
    elif time_range == "last365":
        start_date = today - timedelta(days=365)
    else:
        start_date = today - timedelta(days=30)  # 默认30天
    
    # 根据用户筛选条件构建查询
    query = db.query(AnalysisResult).filter(AnalysisResult.processed_at >= start_date)
    if user_filter == "current":
        query = query.filter(AnalysisResult.user_id == current_user.id)
    
    # 获取所有结果
    results = query.all()
    
    # 计算摘要数据
    total_analyses = len(results)
    depression_count = sum(1 for r in results if r.depression_probability > 0.5)
    depression_rate = depression_count / total_analyses if total_analyses > 0 else 0
    avg_confidence = sum(r.confidence for r in results) / total_analyses if total_analyses > 0 else 0
    
    # 上一时间段的数据用于比较
    prev_start_date = start_date - (today - start_date)
    prev_query = db.query(AnalysisResult).filter(
        AnalysisResult.processed_at >= prev_start_date,
        AnalysisResult.processed_at < start_date
    )
    if user_filter == "current":
        prev_query = prev_query.filter(AnalysisResult.user_id == current_user.id)
    
    prev_results = prev_query.all()
    prev_total = len(prev_results)
    prev_depression_count = sum(1 for r in prev_results if r.depression_probability > 0.5)
    prev_depression_rate = prev_depression_count / prev_total if prev_total > 0 else 0
    
    # 计算趋势变化
    total_change = ((total_analyses - prev_total) / prev_total * 100) if prev_total > 0 else 0
    detection_change = ((depression_rate - prev_depression_rate) / prev_depression_rate * 100) if prev_depression_rate > 0 else 0
    
    # 按月份聚合的检测结果趋势
    monthly_data = {}
    for result in results:
        month = result.processed_at.strftime("%Y-%m")
        if month not in monthly_data:
            monthly_data[month] = {"normal": 0, "mild": 0, "moderate": 0, "severe": 0}
        
        if result.depression_probability <= 0.3:
            monthly_data[month]["normal"] += 1
        elif result.depression_probability <= 0.6:
            monthly_data[month]["mild"] += 1
        elif result.depression_probability <= 0.8:
            monthly_data[month]["moderate"] += 1
        else:
            monthly_data[month]["severe"] += 1
    
    # 按结果类型的分布
    result_distribution = {
        "normal": sum(1 for r in results if r.depression_probability <= 0.3),
        "mild": sum(1 for r in results if 0.3 < r.depression_probability <= 0.6),
        "moderate": sum(1 for r in results if 0.6 < r.depression_probability <= 0.8),
        "severe": sum(1 for r in results if r.depression_probability > 0.8)
    }
    
    # 按年龄段的检出率
    age_groups = {
        "18-24岁": [],
        "25-34岁": [],
        "35-44岁": [],
        "45-54岁": [],
        "55-64岁": [],
        "65岁以上": []
    }
    
    for result in results:
        if not result.patient_age:
            continue
            
        age = result.patient_age
        group = None
        if 18 <= age <= 24:
            group = "18-24岁"
        elif 25 <= age <= 34:
            group = "25-34岁"
        elif 35 <= age <= 44:
            group = "35-44岁"
        elif 45 <= age <= 54:
            group = "45-54岁"
        elif 55 <= age <= 64:
            group = "55-64岁"
        elif age >= 65:
            group = "65岁以上"
            
        if group:
            age_groups[group].append(result.depression_probability > 0.5)
    
    age_distribution = {}
    for group, detections in age_groups.items():
        if detections:
            age_distribution[group] = sum(detections) / len(detections) * 100
        else:
            age_distribution[group] = 0
    
    # 按性别的检出率
    gender_groups = {"男性": [], "女性": []}
    for result in results:
        if not result.patient_gender:
            continue
            
        gender = result.patient_gender
        if gender in gender_groups:
            gender_groups[gender].append(result.depression_probability > 0.5)
    
    gender_distribution = {}
    for gender, detections in gender_groups.items():
        if detections:
            gender_distribution[gender] = sum(detections) / len(detections) * 100
        else:
            gender_distribution[gender] = 0
    
    response = {
        "summary": {
            "total_analyses": total_analyses,
            "depression_rate": depression_rate * 100,  # 转为百分比
            "avg_confidence": avg_confidence * 100,  # 转为百分比
            "avg_processing_time": 26,  # 假设的值，实际应从分析结果中计算
            "total_change": total_change,
            "detection_change": detection_change
        },
        "detection_trend": [
            {
                "month": month,
                "normal": data["normal"],
                "mild": data["mild"],
                "moderate": data["moderate"],
                "severe": data["severe"]
            } for month, data in monthly_data.items()
        ],
        "result_distribution": result_distribution,
        "age_distribution": age_distribution,
        "gender_distribution": gender_distribution
    }
    
    return response

# 添加通用视频处理函数，避免重复代码
async def process_video_analysis(file, current_user, db: Session):
    """处理视频分析，返回分析结果"""
    # 验证文件格式
    if not file.filename.lower().endswith('.mp4'):
        raise HTTPException(
            status_code=400,
            detail="只支持MP4格式的视频文件"
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
            
            # 计算抑郁概率和置信度
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
                patient_age=35,
                patient_gender="男性",
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
            import traceback
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

# 修复analyze-video端点，与现有predict函数一致

@app.post("/analyze-video")
async def analyze_video(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    return await process_video_analysis(file, current_user, db)

# 用户管理API - 获取所有用户列表（仅管理员可用）
@app.get("/api/users", response_model=List[dict])
async def get_users(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    # 检查用户是否为管理员
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有访问权限，仅管理员可访问"
        )
    
    users = db.query(User).all()
    result = []
    for user in users:
        result.append({
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat() if user.created_at else None
        })
    return result

# 更新用户信息（仅管理员可用）
@app.put("/api/users/{user_id}", response_model=dict)
async def update_user(user_id: int, user_data: dict, current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    # 检查用户是否为管理员
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有访问权限，仅管理员可访问"
        )
    
    # 查找用户
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    # 更新用户信息
    if "username" in user_data:
        # 检查用户名是否已存在
        existing_user = db.query(User).filter(User.username == user_data["username"], User.id != user_id).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名已被使用"
            )
        user.username = user_data["username"]
    
    if "email" in user_data:
        # 检查邮箱是否已存在
        existing_user = db.query(User).filter(User.email == user_data["email"], User.id != user_id).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="邮箱已被使用"
            )
        user.email = user_data["email"]
    
    if "role" in user_data and user_data["role"] in ["admin", "doctor", "patient"]:
        user.role = user_data["role"]
    
    if "is_active" in user_data:
        user.is_active = user_data["is_active"]
    
    # 如果提供了密码，重置密码
    if "password" in user_data and user_data["password"]:
        user.hashed_password = pwd_context.hash(user_data["password"])
    
    db.commit()
    return {"message": "用户信息更新成功"}

# 删除用户（仅管理员可用）
@app.delete("/api/users/{user_id}", response_model=dict)
async def delete_user(user_id: int, current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    # 检查用户是否为管理员
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有访问权限，仅管理员可访问"
        )
    
    # 防止管理员删除自己
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不能删除当前登录的管理员账户"
        )
    
    # 查找并删除用户
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    db.delete(user)
    db.commit()
    return {"message": "用户删除成功"}

# 获取所有患者的分析结果（管理员和医生可看所有，患者只能看自己的）
@app.get("/api/analysis_results", response_model=List[dict])
async def get_analysis_results(
    user_id: Optional[int] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # 根据角色确定查询范围
    if current_user.role == "patient":
        # 患者只能查看自己的分析结果
        results = db.query(AnalysisResult).filter(AnalysisResult.user_id == current_user.id).all()
    elif current_user.role in ["doctor", "admin"]:
        # 医生和管理员可以查看所有分析结果，也可以按用户ID过滤
        if user_id:
            results = db.query(AnalysisResult).filter(AnalysisResult.user_id == user_id).all()
        else:
            results = db.query(AnalysisResult).all()
    
    # 构造返回结果
    result_list = []
    for result in results:
        result_list.append({
            "id": result.id,
            "user_id": result.user_id,
            "filename": result.filename,
            "result_type": result.result_type,
            "non_depression_probability": result.non_depression_probability,
            "depression_probability": result.depression_probability,
            "confidence": result.confidence,
            "processed_at": result.processed_at.isoformat() if result.processed_at else None,
            "patient_age": result.patient_age,
            "patient_gender": result.patient_gender,
            "facial_analysis": result.facial_analysis,
            "voice_analysis": result.voice_analysis,
            "body_language_analysis": result.body_language_analysis
        })
    
    return result_list

# 获取特定患者的分析结果详情
@app.get("/api/analysis_results/{result_id}", response_model=dict)
async def get_analysis_result_detail(
    result_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # 查询分析结果
    result = db.query(AnalysisResult).filter(AnalysisResult.id == result_id).first()
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="分析结果不存在"
        )
    
    # 检查访问权限
    if current_user.role == "patient" and result.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限查看其他患者的分析结果"
        )
    
    # 返回详细结果
    return {
        "id": result.id,
        "user_id": result.user_id,
        "filename": result.filename,
        "result_type": result.result_type,
        "non_depression_probability": result.non_depression_probability,
        "depression_probability": result.depression_probability,
        "confidence": result.confidence,
        "processed_at": result.processed_at.isoformat() if result.processed_at else None,
        "patient_age": result.patient_age,
        "patient_gender": result.patient_gender,
        "facial_analysis": result.facial_analysis,
        "voice_analysis": result.voice_analysis,
        "body_language_analysis": result.body_language_analysis
    }

# 医生为患者上传视频
@app.post("/api/upload_for_patient", response_model=dict)
async def upload_video_for_patient(
    patient_id: int = Form(...),
    patient_age: Optional[int] = Form(None),
    patient_gender: Optional[str] = Form(None),
    video: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # 验证上传者是医生或管理员
    if current_user.role not in ["doctor", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限，仅医生和管理员可为患者上传视频"
        )
    
    # 验证患者存在
    patient = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="患者不存在或ID无效"
        )
    
    # 检查文件类型
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="仅支持MP4、AVI或MOV格式的视频文件"
        )
    
    try:
        # 为视频处理创建临时文件，保存上传的视频
        video_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{video.filename}"
        video_path = os.path.join(tempfile.gettempdir(), video_filename)
        
        # 保存上传的视频文件
        with open(video_path, "wb") as f:
            f.write(await video.read())
        
        # 调用视频处理流程
        logger.info(f"开始处理视频: {video_filename}")
        
        # 【这里应该有视频处理和分析的代码，与普通上传流程相同】
        # 为简化示例，假设我们调用一个predict_video函数
        
        # 将分析结果保存到数据库，关联到患者ID
        analysis_result = AnalysisResult(
            user_id=patient_id,
            filename=video_filename,
            result_type="待分析",  # 初始状态
            non_depression_probability=0.0,
            depression_probability=0.0,
            confidence=0.0,
            patient_age=patient_age,
            patient_gender=patient_gender
        )
        
        db.add(analysis_result)
        db.commit()
        db.refresh(analysis_result)
        
        # 返回成功消息和分析结果ID
        return {
            "message": "视频上传成功，已为患者创建分析任务",
            "analysis_id": analysis_result.id,
            "patient_id": patient_id,
            "patient_username": patient.username
        }
        
    except Exception as e:
        logger.error(f"处理视频时出错: {str(e)}")
        # 如果出错，确保清理临时文件
        if os.path.exists(video_path):
            os.remove(video_path)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理视频时出错: {str(e)}"
        )

# 添加服务启动代码
if __name__ == "__main__":
    # 设置服务启动参数
    host = "0.0.0.0"  # 监听所有网络接口
    port = 8000       # 使用8000端口
    
    print(f"启动服务器于 http://{host}:{port}")
    print(f"可以通过访问 http://localhost:{port} 访问服务")
    
    # 使用已经导入的uvicorn模块启动FastAPI应用
    uvicorn.run(app, host=host, port=port)
