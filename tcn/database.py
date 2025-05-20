import os
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Float, Text, JSON, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# 创建SQLite数据库引擎
SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# 创建数据库会话
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建Base类
Base = declarative_base()

# 用户模型
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    role = Column(String, default="patient")  # 用户角色: admin, doctor, patient
    reset_token = Column(String, nullable=True)
    reset_token_expires = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关系: 一个用户可以有多个分析结果
    analysis_results = relationship("AnalysisResult", back_populates="user")


# 分析结果模型
class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String, nullable=False)
    
    # 统一字段名
    result_type = Column(String, nullable=False)  # 例如："正常", "轻度抑郁", "中度抑郁", "重度抑郁"
    predicted_class = Column(String, nullable=True)  # 兼容旧代码
    
    # 概率字段
    non_depression_probability = Column(Float, nullable=False)
    depression_probability = Column(Float, nullable=False)
    probability_class0 = Column(Float, nullable=True)  # 兼容旧代码
    probability_class1 = Column(Float, nullable=True)  # 兼容旧代码
    
    confidence = Column(Float, nullable=False)
    processed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)  # 兼容旧代码
      # 详细结果存储为JSON
    facial_analysis = Column(JSON, nullable=True)
    voice_analysis = Column(JSON, nullable=True)
    body_language_analysis = Column(JSON, nullable=True)
    
    # 医生注释
    doctor_notes = Column(Text, nullable=True)
    
    # 用户数据
    patient_age = Column(Integer, nullable=True)
    patient_gender = Column(String, nullable=True)
    
    # 关联用户
    user = relationship("User", back_populates="analysis_results")


# 添加统计数据模型
class AnalyticsData(Base):
    __tablename__ = "analytics_data"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=datetime.utcnow)
    total_analyses = Column(Integer, default=0)
    detection_rate = Column(Float, default=0.0)
    avg_confidence = Column(Float, default=0.0)
    avg_processing_time = Column(Float, default=0.0)
    
    # 按严重程度划分的结果统计
    normal_count = Column(Integer, default=0)
    mild_count = Column(Integer, default=0)
    moderate_count = Column(Integer, default=0)
    severe_count = Column(Integer, default=0)
    
    # 其他统计数据
    male_detection_rate = Column(Float, nullable=True)
    female_detection_rate = Column(Float, nullable=True)

def initialize_database():
    """
    智能初始化数据库:
    1. 如果数据库文件不存在，创建所有表
    2. 如果数据库存在但缺少某些表，只创建缺少的表
    3. 如果所有表都存在，不做任何操作
    """
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    # 获取所有模型定义的表名
    metadata = Base.metadata
    required_tables = metadata.tables.keys()
    
    # 检查是否需要创建表
    missing_tables = set(required_tables) - set(existing_tables)
    
    if missing_tables:
        # 只创建缺少的表
        metadata.create_all(bind=engine, tables=[metadata.tables[table] for table in missing_tables])
        print(f"Created missing tables: {missing_tables}")
    else:
        print("Database is up-to-date, no tables need to be created.")

# 获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()