# 认证问题修复方案

## 问题描述

当管理员尝试登录系统时，遇到 "Method Not Allowed" 错误。这是因为系统中缺少处理 `/admin` 路由的 POST 方法处理程序。

## 分析

1. 现有的登录页面代码会根据用户角色，创建一个表单并通过 POST 方法提交令牌到相应的路由:
   - 管理员: `/admin` (POST)
   - 医生: `/doctor` (POST)
   - 患者: `/patient` (POST)

2. 系统中已经实现了处理 `/doctor` 和 `/patient` 路由的 POST 方法，但缺少处理 `/admin` 路由的 POST 方法。

## 解决方案

添加了处理 `/admin` 路由 POST 请求的函数，功能包括:
1. 接收并验证表单提交的认证令牌
2. 检查用户角色是否为 "admin"
3. 返回管理员页面内容或适当的错误消息

## 实现细节

在 `app.py` 中添加了 `admin_page_post` 函数，处理对 `/admin` 路由的 POST 请求:

```python
@app.post("/admin", response_class=HTMLResponse)
async def admin_page_post(auth_token: str = Form(...)):
    try:
        # 验证令牌
        db = SessionLocal()
        try:
            # 解码JWT令牌
            payload = jwt.decode(auth_token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            
            # 获取用户信息并验证角色
            user = get_user(db, username)
            if user.role != "admin":
                raise HTTPException(status_code=403, detail="没有访问权限")
                
            # 返回管理员页面
            with open(os.path.join(os.path.dirname(__file__), "templates", "admin.html"), "r", encoding="utf-8") as f:
                content = f.read()
            return content
        finally:
            db.close()
    except Exception as e:
        # 返回适当的错误信息
        return HTMLResponse(
            content=f"<html><body><h1>错误</h1><p>{str(e)}</p></body></html>",
            status_code=500
        )
```

## 测试

创建了 `test_auth_routes.py` 测试脚本，可以用来测试:
1. 管理员登录过程
2. 使用令牌通过 GET 方法访问 `/admin` 路由
3. 使用表单提交通过 POST 方法访问 `/admin` 路由

要运行测试，在命令行中执行:
```
python test_auth_routes.py
```

## 后续改进

1. 考虑统一三种用户角色的认证处理方式
2. 添加更详细的日志记录
3. 实现更友好的错误提示界面
