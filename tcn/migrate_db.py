import sqlite3
import os
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def migrate_database():
    """为users表添加缺失的role列"""
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "users.db")
    
    if not os.path.exists(db_path):
        logger.error(f"数据库文件不存在: {db_path}")
        return False
    
    # 连接数据库
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 检查users表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cursor.fetchone():
            logger.error("users表不存在")
            return False
        
        # 检查users表是否已有role列
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if "role" not in columns:
            logger.info("添加role列到users表")
            cursor.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'patient'")
            conn.commit()
            logger.info("成功添加role列")
            return True
        else:
            logger.info("role列已存在，无需迁移")
            return True
            
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"数据库迁移出错: {e}")
        return False
    finally:
        if conn:
            conn.close()
    
if __name__ == "__main__":
    logger.info("开始数据库迁移...")
    if migrate_database():
        logger.info("数据库迁移成功完成")
    else:
        logger.error("数据库迁移失败")
