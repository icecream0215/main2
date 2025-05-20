from datetime import datetime, timedelta
from typing import Optional
import secrets
import string

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import User, get_db

# 用于令牌生成的密钥和算法
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
RESET_TOKEN_EXPIRE_HOURS = 24

# 用户注册和令牌模型
class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: str = "patient"  # 默认为患者角色
    
class Token(BaseModel):
    access_token: str
    token_type: str
    
class TokenData(BaseModel):
    username: Optional[str] = None

# 密码重置请求模型
class PasswordResetRequest(BaseModel):
    email: str
    
class PasswordReset(BaseModel):
    token: str
    new_password: str

# 密码哈希工具
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2密码流处理
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 验证密码
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# 生成密码哈希
def get_password_hash(password):
    return pwd_context.hash(password)

# 通过用户名获取用户
def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

# 通过邮箱获取用户
def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

# 创建用户
def create_user(db: Session, user: UserCreate):
    # 确保角色是有效的
    if user.role not in ["admin", "doctor", "patient"]:
        user.role = "patient"  # 默认为患者
        
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=get_password_hash(user.password),
        role=user.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# 验证用户
def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

# 创建访问令牌
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# 获取当前用户
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

# 获取当前活跃用户
async def get_current_active_user(current_user = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="用户已被禁用")
    return current_user

# 检查用户是否有特定角色
async def check_user_role(required_roles: list, current_user = Depends(get_current_active_user)):
    if current_user.role not in required_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="权限不足，无法访问此资源"
        )
    return current_user

# 专门检查管理员权限的依赖
async def get_current_admin_user(current_user = Depends(get_current_active_user)):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限"
        )
    return current_user

# 专门检查医生权限的依赖
async def get_current_doctor_user(current_user = Depends(get_current_active_user)):
    if current_user.role not in ["doctor", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要医生或管理员权限"
        )
    return current_user

# 专门检查患者权限的依赖
async def get_current_patient_user(current_user = Depends(get_current_active_user)):
    if current_user.role not in ["patient", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要患者权限"
        )
    return current_user

# 创建密码重置令牌
def create_reset_token(length=32):
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

# 创建密码重置请求
def create_password_reset_request(db: Session, email: str):
    user = get_user_by_email(db, email)
    if not user:
        # 即使找不到用户也返回成功，防止邮箱枚举攻击
        return None
    
    reset_token = create_reset_token()
    expire_time = datetime.utcnow() + timedelta(hours=RESET_TOKEN_EXPIRE_HOURS)
    
    user.reset_token = reset_token
    user.reset_token_expires = expire_time
    db.commit()
    
    return user

# 验证密码重置令牌
def verify_reset_token(db: Session, token: str):
    user = db.query(User).filter(User.reset_token == token).first()
    
    if not user:
        return None
    
    # 检查令牌是否过期
    if user.reset_token_expires < datetime.utcnow():
        return None
        
    return user

# 重置密码
def reset_password(db: Session, token: str, new_password: str):
    user = verify_reset_token(db, token)
    
    if not user:
        return False
    
    # 更新密码并清除重置令牌
    user.hashed_password = get_password_hash(new_password)
    user.reset_token = None
    user.reset_token_expires = None
    db.commit()
    
    return True