# 认证路由测试脚本
import requests
import json
import sys
import os
from getpass import getpass

# 基本URL配置
BASE_URL = "http://127.0.0.1:8000"

def login(username, password, role):
    """尝试登录并返回令牌"""
    print(f"\n尝试以{role}身份登录 ({username})...")
    
    # 准备登录数据
    login_data = {
        "username": username,
        "password": password
    }
    
    # 发送登录请求
    response = requests.post(
        f"{BASE_URL}/token", 
        data=login_data
    )
    
    if response.status_code == 200:
        token_data = response.json()
        token = token_data.get("access_token")
        print(f"登录成功! 获取到令牌: {token[:15]}...")
        
        # 获取用户信息
        headers = {"Authorization": f"Bearer {token}"}
        user_response = requests.get(f"{BASE_URL}/users/me", headers=headers)
        
        if user_response.status_code == 200:
            user_info = user_response.json()
            print(f"用户信息: {json.dumps(user_info, ensure_ascii=False, indent=2)}")
            
            # 检查角色
            if user_info.get("role") != role:
                print(f"警告: 账户角色不匹配! 预期 '{role}', 实际 '{user_info.get('role')}'")
                return None
            
            return token
        else:
            print(f"获取用户信息失败: {user_response.status_code} - {user_response.text}")
            return None
    else:
        print(f"登录失败: {response.status_code} - {response.text}")
        return None

def test_post_route(url, token):
    """测试POST路由访问"""
    print(f"\n测试POST访问 {url}...")
    
    # 准备表单数据
    data = {"auth_token": token}
    
    # 发送POST请求
    response = requests.post(url, data=data)
    
    # 打印结果
    print(f"状态码: {response.status_code}")
    if 200 <= response.status_code < 300:
        print("访问成功!")
        return True
    else:
        print(f"访问失败: {response.text}")
        return False

def test_get_route_with_token(url, token):
    """测试带令牌的GET路由访问"""
    print(f"\n测试GET访问 {url} (带令牌)...")
    
    # 准备请求头
    headers = {"Authorization": f"Bearer {token}"}
    
    # 发送GET请求
    response = requests.get(url, headers=headers)
    
    # 打印结果
    print(f"状态码: {response.status_code}")
    if 200 <= response.status_code < 300:
        print("访问成功!")
        return True
    else:
        print(f"访问失败: {response.text}")
        return False

def main():
    """主函数 - 运行完整测试流程"""
    print("===== 抑郁症检测系统 - 认证路由测试 =====")
    
    # 提示输入用户名和密码
    print("\n请输入管理员账户信息:")
    admin_username = input("用户名: ")
    admin_password = getpass("密码: ")
    
    # 尝试管理员登录
    admin_token = login(admin_username, admin_password, "admin")
    
    if admin_token:
        # 测试直接GET访问管理员页面
        test_get_route_with_token(f"{BASE_URL}/admin", admin_token)
        
        # 测试POST访问管理员页面
        test_post_route(f"{BASE_URL}/admin", admin_token)
        
        # 测试访问其他角色页面
        test_get_route_with_token(f"{BASE_URL}/doctor", admin_token)
        test_get_route_with_token(f"{BASE_URL}/patient", admin_token)
    
    print("\n测试完成!")

if __name__ == "__main__":
    main()
