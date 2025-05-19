from database import AnalysisResult, AnalyticsData
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

def get_daily_analytics(db: Session, date: datetime, user_id: Optional[int] = None):
    """获取指定日期的分析数据"""
    start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + timedelta(days=1)
    
    # 构建基本查询
    query = db.query(AnalysisResult).filter(
        and_(
            AnalysisResult.processed_at >= start_date,
            AnalysisResult.processed_at < end_date
        )
    )
    
    # 如果指定了用户，筛选该用户的数据
    if user_id:
        query = query.filter(AnalysisResult.user_id == user_id)
    
    # 获取当日所有分析结果
    results = query.all()
    
    # 计算统计数据
    total = len(results)
    normal_count = sum(1 for r in results if r.depression_probability <= 0.3)
    mild_count = sum(1 for r in results if 0.3 < r.depression_probability <= 0.6)
    moderate_count = sum(1 for r in results if 0.6 < r.depression_probability <= 0.8)
    severe_count = sum(1 for r in results if r.depression_probability > 0.8)
    
    # 计算平均值
    avg_confidence = sum(r.confidence for r in results) / total if total > 0 else 0
    
    # 按性别划分的检出率
    male_results = [r for r in results if r.patient_gender == "男性"]
    female_results = [r for r in results if r.patient_gender == "女性"]
    
    male_detection_rate = sum(1 for r in male_results if r.depression_probability > 0.5) / len(male_results) if male_results else 0
    female_detection_rate = sum(1 for r in female_results if r.depression_probability > 0.5) / len(female_results) if female_results else 0
    
    # 创建或更新统计数据记录
    analytics_record = db.query(AnalyticsData).filter(
        func.date(AnalyticsData.date) == func.date(start_date)
    ).first()
    
    if not analytics_record:
        analytics_record = AnalyticsData(
            date=start_date,
            total_analyses=total,
            detection_rate=(total - normal_count) / total if total > 0 else 0,
            avg_confidence=avg_confidence,
            avg_processing_time=20.0,  # 假设的平均处理时间
            normal_count=normal_count,
            mild_count=mild_count,
            moderate_count=moderate_count,
            severe_count=severe_count,
            male_detection_rate=male_detection_rate,
            female_detection_rate=female_detection_rate
        )
        db.add(analytics_record)
    else:
        analytics_record.total_analyses = total
        analytics_record.detection_rate = (total - normal_count) / total if total > 0 else 0
        analytics_record.avg_confidence = avg_confidence
        analytics_record.normal_count = normal_count
        analytics_record.mild_count = mild_count
        analytics_record.moderate_count = moderate_count
        analytics_record.severe_count = severe_count
        analytics_record.male_detection_rate = male_detection_rate
        analytics_record.female_detection_rate = female_detection_rate
    
    db.commit()
    return analytics_record

def get_trend_analysis(
    db: Session, 
    start_date: datetime, 
    end_date: datetime, 
    user_id: Optional[int] = None,
    interval: str = "day"
) -> List[Dict[str, Any]]:
    """获取指定时间段内的趋势分析数据"""
    
    # 构建基本查询
    query = db.query(AnalysisResult).filter(
        and_(
            AnalysisResult.processed_at >= start_date,
            AnalysisResult.processed_at <= end_date
        )
    )
    
    # 如果指定了用户，筛选该用户的数据
    if user_id:
        query = query.filter(AnalysisResult.user_id == user_id)
    
    # 获取所有相关分析结果
    results = query.all()
    
    # 按指定间隔分组数据
    grouped_data = {}
    for result in results:
        if interval == "day":
            key = result.processed_at.strftime("%Y-%m-%d")
        elif interval == "week":
            # 计算该日期是一年中的第几周
            key = f"{result.processed_at.strftime('%Y')}-W{result.processed_at.isocalendar()[1]}"
        elif interval == "month":
            key = result.processed_at.strftime("%Y-%m")
        elif interval == "quarter":
            month = result.processed_at.month
            quarter = (month - 1) // 3 + 1
            key = f"{result.processed_at.year}-Q{quarter}"
        else:
            key = result.processed_at.strftime("%Y")
        
        if key not in grouped_data:
            grouped_data[key] = {
                "period": key,
                "total": 0,
                "normal": 0,
                "mild": 0,
                "moderate": 0,
                "severe": 0,
                "detection_rate": 0,
                "avg_confidence": 0
            }
        
        group = grouped_data[key]
        group["total"] += 1
        
        if result.depression_probability <= 0.3:
            group["normal"] += 1
        elif result.depression_probability <= 0.6:
            group["mild"] += 1
        elif result.depression_probability <= 0.8:
            group["moderate"] += 1
        else:
            group["severe"] += 1
        
        # 累加置信度，后面再计算平均值
        group["avg_confidence"] += result.confidence
    
    # 计算平均值和比率
    trend_data = []
    for key, group in grouped_data.items():
        total = group["total"]
        if total > 0:
            group["detection_rate"] = (group["mild"] + group["moderate"] + group["severe"]) / total * 100
            group["avg_confidence"] = group["avg_confidence"] / total * 100
        trend_data.append(group)
    
    # 按时间排序
    trend_data.sort(key=lambda x: x["period"])
    
    return trend_data