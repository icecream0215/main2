// DOM 加载完成后执行代码
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM 已加载，JavaScript 初始化中...');

    // 获取当前页面 URL 路径
    const currentPath = window.location.pathname;

    // 检查用户认证状态
    checkAuthStatus();

    // 设置全局点击事件来关闭下拉菜单
    document.addEventListener('click', function(event) {
        const dropdowns = document.querySelectorAll('.dropdown-menu.show');
        dropdowns.forEach(dropdown => {
            const parentDropdown = dropdown.closest('.dropdown');
            if (parentDropdown && !parentDropdown.contains(event.target)) {
                dropdown.classList.remove('show');
            }
        });
    });

    // 简单的表单验证
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            const requiredFields = form.querySelectorAll('[required]');
            let isValid = true;

            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    isValid = false;
                    field.classList.add('error');
                    
                    // 如果字段旁边没有错误提示，则添加一个
                    let nextElem = field.nextElementSibling;
                    if (!nextElem || !nextElem.classList.contains('error-message')) {
                        const errorMsg = document.createElement('div');
                        errorMsg.textContent = '此字段为必填项';
                        errorMsg.classList.add('error-message');
                        errorMsg.style.color = 'red';
                        errorMsg.style.fontSize = '0.8em';
                        errorMsg.style.marginTop = '5px';
                        field.parentNode.insertBefore(errorMsg, field.nextSibling);
                    }
                } else {
                    field.classList.remove('error');
                    let nextElem = field.nextElementSibling;
                    if (nextElem && nextElem.classList.contains('error-message')) {
                        nextElem.remove();
                    }
                }
            });

            if (!isValid) {
                event.preventDefault();
            }
        });
    });

    // 处理分析结果页面上的按钮事件
    setupAnalysisResultButtons();
    
    // 处理患者页面的功能
    setupPatientPageFunctions();
    
    // 处理医生页面的功能
    setupDoctorPageFunctions();
    
    // 处理管理员页面的功能
    setupAdminPageFunctions();
    
    // 处理导航栏的激活状态
    highlightActiveNavItem();
    
    // 设置视频上传预览
    setupVideoPreview();
});

// 检查用户认证状态
function checkAuthStatus() {
    const token = localStorage.getItem('access_token');
    const currentPath = window.location.pathname;
    const publicPages = ['/', '/login', '/register', '/reset-password-request', '/reset-password.html'];
    
    // 如果不在公共页面且没有token，重定向到登录页面
    if (!publicPages.includes(currentPath) && !token) {
        window.location.href = '/login';
        return;
    }
    
    // 如果有token，在每个页面上显示用户信息
    if (token) {
        fetch('/users/me', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('认证失败');
            }
            return response.json();
        })        .then(user => {
            // 保存用户数据到本地存储
            localStorage.setItem('user_data', JSON.stringify(user));
            localStorage.setItem('user_role', user.role);
            
            displayUserInfo(user);
            
            // 检查用户是否有权限访问当前页面
            checkPagePermission(user, currentPath);
        })
        .catch(error => {
            console.error('获取用户信息失败:', error);
            // 如果API调用失败，可能是token过期，清除并重定向到登录
            localStorage.removeItem('access_token');
            localStorage.removeItem('user_role');
            if (!publicPages.includes(currentPath)) {
                window.location.href = '/login';
            }
        });
    }
}

// 显示用户信息及下拉菜单
function displayUserInfo(user) {
    const userInfoContainer = document.getElementById('user-info');
    if (userInfoContainer) {
        userInfoContainer.innerHTML = `
            <div class="dropdown">
                <button class="dropdown-toggle">欢迎，${user.username} ▼</button>
                <div class="dropdown-menu">
                    <a href="#" class="user-profile-link">个人信息</a>
                    <a href="#" class="account-settings-link">账号设置</a>
                    <hr>
                    <a href="#" class="logout-link">退出登录</a>
                </div>
            </div>
        `;
        
        // 添加下拉菜单的点击交互
        const dropdownToggle = userInfoContainer.querySelector('.dropdown-toggle');
        const dropdownMenu = userInfoContainer.querySelector('.dropdown-menu');
        
        dropdownToggle.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation(); // 阻止事件冒泡
            dropdownMenu.classList.toggle('show');
        });
        
        // 添加菜单项的点击事件
        const profileLink = userInfoContainer.querySelector('.user-profile-link');
        const settingsLink = userInfoContainer.querySelector('.account-settings-link');
        const logoutLink = userInfoContainer.querySelector('.logout-link');
        
        if (profileLink) {
            profileLink.addEventListener('click', function(e) {
                e.preventDefault();
                showUserProfile();
            });
        }
        
        if (settingsLink) {
            settingsLink.addEventListener('click', function(e) {
                e.preventDefault();
                showAccountSettings();
            });
        }
        
        if (logoutLink) {
            logoutLink.addEventListener('click', function(e) {
                e.preventDefault();
                logout();
            });
        }
    }
}

// 显示用户个人信息对话框
function showUserProfile() {
    const userData = JSON.parse(localStorage.getItem('user_data') || '{}');
    
    // 先移除任何已存在的模态框
    const existingModals = document.querySelectorAll('.modal');
    existingModals.forEach(modal => {
        document.body.removeChild(modal);
    });
    
    // 创建模态对话框
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content">
            <span class="close">&times;</span>
            <h2>个人信息</h2>
            <div class="profile-info">
                <p><strong>用户名:</strong> ${userData.username || '未知'}</p>
                <p><strong>邮箱:</strong> ${userData.email || '未设置'}</p>
                <p><strong>角色:</strong> ${userData.role ? translateRole(userData.role) : '未知'}</p>
                <p><strong>账号状态:</strong> ${userData.is_active ? '活跃' : '已禁用'}</p>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // 关闭按钮功能
    const closeBtn = modal.querySelector('.close');
    closeBtn.addEventListener('click', function() {
        modal.style.display = 'none';
        setTimeout(() => {
            if (modal.parentNode) {
                document.body.removeChild(modal);
            }
        }, 300); // 给过渡动画一些时间
    });
    
    // 点击模态框外部关闭
    modal.addEventListener('click', function(event) {
        if (event.target === modal) {
            modal.style.display = 'none';
            setTimeout(() => {
                if (modal.parentNode) {
                    document.body.removeChild(modal);
                }
            }, 300);
        }
    });
    
    // 显示模态框
    setTimeout(() => {
        modal.style.display = 'block';
    }, 10);
}

// 显示账户设置对话框
function showAccountSettings() {
    // 获取当前用户数据
    const userData = JSON.parse(localStorage.getItem('user_data') || '{}');
    
    // 先移除任何已存在的模态框
    const existingModals = document.querySelectorAll('.modal');
    existingModals.forEach(modal => {
        if (modal.parentNode) {
            document.body.removeChild(modal);
        }
    });
    
    // 创建模态对话框
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content">
            <span class="close">&times;</span>
            <h2>账号设置</h2>
            <form id="account-settings-form">
                <div class="form-group">
                    <label for="current-password">当前密码</label>
                    <input type="password" id="current-password" name="current_password" required>
                </div>
                <div class="form-group">
                    <label for="new-password">新密码 (如需修改)</label>
                    <input type="password" id="new-password" name="new_password">
                </div>
                <div class="form-group">
                    <label for="confirm-password">确认新密码</label>
                    <input type="password" id="confirm-password" name="confirm_password">
                </div>
                <div class="form-group">
                    <label for="email">电子邮箱</label>
                    <input type="email" id="email" name="email" value="${userData.email || ''}" placeholder="输入新邮箱地址">
                </div>
                <div class="form-error" style="color: red; margin: 10px 0; display: none;"></div>
                <div class="form-actions">
                    <button type="submit" class="btn primary">保存更改</button>
                </div>
            </form>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // 关闭按钮功能
    const closeBtn = modal.querySelector('.close');
    closeBtn.addEventListener('click', function() {
        modal.style.display = 'none';
        setTimeout(() => {
            if (modal.parentNode) {
                document.body.removeChild(modal);
            }
        }, 300);
    });
    
    // 点击模态框外部关闭
    modal.addEventListener('click', function(event) {
        if (event.target === modal) {
            modal.style.display = 'none';
            setTimeout(() => {
                if (modal.parentNode) {
                    document.body.removeChild(modal);
                }
            }, 300);
        }
    });
    
    // 表单提交处理
    const form = modal.querySelector('#account-settings-form');
    const formError = modal.querySelector('.form-error');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const currentPassword = form.querySelector('#current-password').value;
        const newPassword = form.querySelector('#new-password').value;
        const confirmPassword = form.querySelector('#confirm-password').value;
        const email = form.querySelector('#email').value;
        
        // 简单验证
        if (!currentPassword) {
            formError.textContent = "请输入当前密码";
            formError.style.display = "block";
            return;
        }
        
        // 如果填写了新密码，验证两次输入是否一致
        if (newPassword && newPassword !== confirmPassword) {
            formError.textContent = "两次输入的新密码不一致";
            formError.style.display = "block";
            return;
        }
        
        formError.style.display = "none";
        const formData = new FormData();
        formData.append('current_password', currentPassword);
        
        if (newPassword) {
            formData.append('new_password', newPassword);
        }
        
        if (email && email !== userData.email) {
            formData.append('email', email);
        }
        
        const token = localStorage.getItem('access_token');
        
        // 显示加载状态
        const submitBtn = form.querySelector('button[type="submit"]');
        submitBtn.disabled = true;
        submitBtn.textContent = "保存中...";
        
        fetch('/api/users/profile', {
            method: 'PUT',
            headers: {
                'Authorization': `Bearer ${token}`
            },
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.detail || '更新失败');
                });
            }
            return response.json();
        })
        .then(data => {
            // 显示成功消息
            const successMsg = document.createElement('div');
            successMsg.className = 'success-message';
            successMsg.textContent = '个人资料已更新成功!';
            successMsg.style.color = 'green';
            successMsg.style.padding = '10px';
            successMsg.style.marginTop = '10px';
            successMsg.style.backgroundColor = '#f0f9f0';
            successMsg.style.border = '1px solid green';
            successMsg.style.borderRadius = '4px';
            
            form.appendChild(successMsg);
            
            // 更新本地存储的用户数据
            if (data.email) {
                userData.email = data.email;
                localStorage.setItem('user_data', JSON.stringify(userData));
            }
            
            // 重置表单
            form.reset();
            
            // 3秒后关闭模态框
            setTimeout(() => {
                modal.style.display = 'none';
                setTimeout(() => {
                    if (modal.parentNode) {
                        document.body.removeChild(modal);
                    }
                }, 300);
            }, 3000);
        })
        .catch(error => {
            formError.textContent = error.message || '更新个人资料时出错';
            formError.style.display = "block";
        })
        .finally(() => {
            // 恢复按钮状态
            submitBtn.disabled = false;
            submitBtn.textContent = "保存更改";
        });
    });
    
    // 显示模态框
    setTimeout(() => {
        modal.style.display = 'block';
    }, 10);
}

// 翻译角色名称
function translateRole(role) {
    const roleMap = {
        'admin': '管理员',
        'doctor': '医生',
        'patient': '患者'
    };
    return roleMap[role] || role;
}

// 检查页面权限
function checkPagePermission(user, currentPath) {
    if (currentPath === '/admin' && user.role !== 'admin') {
        window.location.href = '/login';
    } else if (currentPath === '/doctor' && !['doctor', 'admin'].includes(user.role)) {
        window.location.href = '/login';
    } else if (currentPath === '/patient' && user.role !== 'patient' && user.role !== 'admin') {
        window.location.href = '/login';
    }
}

// 处理登出
function logout() {
    // 显示登出确认对话框
    if (confirm('确定要退出登录吗？')) {
        // 清除所有用户相关的本地存储数据
        localStorage.removeItem('access_token');
        localStorage.removeItem('user_role');
        localStorage.removeItem('user_data');
        
        // 可以在这里添加API调用，告知服务器用户已登出
        console.log('用户已登出');
        
        // 显示短暂的登出消息
        const logoutMsg = document.createElement('div');
        logoutMsg.textContent = '正在退出登录...';
        logoutMsg.style.position = 'fixed';
        logoutMsg.style.top = '50%';
        logoutMsg.style.left = '50%';
        logoutMsg.style.transform = 'translate(-50%, -50%)';
        logoutMsg.style.padding = '20px';
        logoutMsg.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
        logoutMsg.style.color = 'white';
        logoutMsg.style.borderRadius = '5px';
        logoutMsg.style.zIndex = '10000';
        
        document.body.appendChild(logoutMsg);
        
        // 短暂延迟后重定向到登录页面
        setTimeout(() => {
            window.location.href = '/login';
        }, 1000);
    }
}

// 通用API请求处理函数
async function apiRequest(url, options = {}) {
    // 获取访问令牌
    const token = localStorage.getItem('access_token');
    
    // 合并默认选项
    const requestOptions = {
        headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
            ...options.headers
        },
        ...options
    };
    
    try {
        // 发起请求
        const response = await fetch(url, requestOptions);
        
        // 检查是否登录过期
        if (response.status === 401) {
            // 清除认证数据并重定向到登录页面
            localStorage.removeItem('access_token');
            localStorage.removeItem('user_role');
            localStorage.removeItem('user_data');
            window.location.href = '/login';
            throw new Error('认证已过期，请重新登录');
        }
        
        // 解析响应
        const data = await response.json();
        
        // 检查API响应是否成功
        if (!response.ok) {
            throw new Error(data.detail || '请求失败');
        }
        
        return data;
    } catch (error) {
        console.error('API请求错误:', error);
        throw error;
    }
}

// 辅助函数：根据结果类型返回颜色类名
function getResultColorClass(resultType) {
    switch(resultType) {
        case '正常':
            return 'text-success';
        case '轻度抑郁':
            return 'text-warning';
        case '中度抑郁':
            return 'text-warning-dark';
        case '重度抑郁':
            return 'text-danger';
        default:
            return '';
    }
}

// 辅助函数：加载历史数据
function loadHistoryData() {
    const historyContainer = document.getElementById('history-container');
    if (!historyContainer) return;
    
    historyContainer.innerHTML = '<p class="text-center">加载中...</p>';
    
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            if (data.length === 0) {
                historyContainer.innerHTML = '<p class="text-center">暂无历史记录</p>';
                return;
            }
            
            let tableHtml = `
            <div class="table-responsive">
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>日期</th>
                            <th>文件名</th>
                            <th>结果类型</th>
                            <th>抑郁概率</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            data.forEach((item, index) => {
                tableHtml += `
                    <tr>
                        <td>${index + 1}</td>
                        <td>${item.created_at}</td>
                        <td>${item.filename}</td>
                        <td class="${getResultColorClass(item.predicted_class)}">${item.predicted_class}</td>
                        <td>${(item.probability_class1 * 100).toFixed(2)}%</td>
                    </tr>
                `;
            });
            
            tableHtml += `
                    </tbody>
                </table>
            </div>
            `;
            
            historyContainer.innerHTML = tableHtml;
        })
        .catch(error => {
            historyContainer.innerHTML = '<p class="text-center">加载历史记录失败</p>';
            console.error('加载历史记录错误:', error);
        });
}

// 设置患者页面功能
function setupPatientPageFunctions() {
    console.log('设置患者页面功能...');
    const uploadForm = document.getElementById('upload-form');
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const videoFile = document.getElementById('video-upload').files[0];
            if (!videoFile) {
                alert('请选择视频文件');
                return;
            }
            
            // 显示上传进度条
            const progressBar = document.getElementById('upload-progress');
            progressBar.style.display = 'block';
            progressBar.value = 0;
            
            const formData = new FormData();
            formData.append('video_file', videoFile);
            
            const xhr = new XMLHttpRequest();
            
            // 设置进度跟踪
            xhr.upload.addEventListener('progress', function(e) {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    progressBar.value = percentComplete;
                }
            });
            
            // 处理请求完成
            xhr.addEventListener('load', function() {
                if (xhr.status >= 200 && xhr.status < 300) {
                    const response = JSON.parse(xhr.responseText);
                    displayAnalysisResult(response);
                    // 重新加载历史记录
                    loadHistoryData();
                } else {
                    alert('上传失败：' + xhr.statusText);
                }
                progressBar.style.display = 'none';
            });
            
            // 处理上传错误
            xhr.addEventListener('error', function() {
                alert('上传过程中发生错误');
                progressBar.style.display = 'none';
            });
            
            // 发送请求
            xhr.open('POST', '/api/analyze', true);
            xhr.setRequestHeader('Authorization', `Bearer ${localStorage.getItem('access_token')}`);
            xhr.send(formData);
        });
    }
    
    // 自动加载历史记录
    loadHistoryData();
}

// 设置视频预览功能
function setupVideoPreview() {
    const videoUpload = document.getElementById('video-upload');
    const previewContainer = document.getElementById('preview-container');
    
    if (videoUpload && previewContainer) {
        videoUpload.addEventListener('change', function(e) {
            const file = e.target.files[0];
            
            if (!file) {
                previewContainer.innerHTML = '';
                return;
            }
            
            // 检查文件类型
            const fileType = file.type;
            if (!fileType.startsWith('video/')) {
                previewContainer.innerHTML = '<p class="error">请选择有效的视频文件</p>';
                return;
            }
            
            // 创建视频预览
            const videoURL = URL.createObjectURL(file);
            previewContainer.innerHTML = `
                <div class="video-preview">
                    <video controls width="100%">
                        <source src="${videoURL}" type="${fileType}">
                        您的浏览器不支持视频标签。
                    </video>
                    <div class="file-info">
                        <p><strong>文件名:</strong> ${file.name}</p>
                        <p><strong>大小:</strong> ${formatFileSize(file.size)}</p>
                        <p><strong>类型:</strong> ${fileType}</p>
                    </div>
                </div>
            `;
        });
    }
}

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    
    return parseFloat((bytes / Math.pow(1024, i)).toFixed(2)) + ' ' + sizes[i];
}

// 显示分析结果
function displayAnalysisResult(result) {
    const resultContainer = document.getElementById('result-container');
    if (!resultContainer) return;
    
    let resultClass = '';
    switch(result.predicted_class) {
        case '正常':
            resultClass = 'result-normal';
            break;
        case '轻度抑郁':
            resultClass = 'result-mild';
            break;
        case '中度抑郁':
            resultClass = 'result-moderate';
            break;
        case '重度抑郁':
            resultClass = 'result-severe';
            break;
    }
    
    resultContainer.innerHTML = `
        <div class="result-card ${resultClass}">
            <h3>分析结果</h3>
            <div class="result-main">
                <p class="result-type">${result.predicted_class}</p>
                <p class="result-probability">抑郁概率: ${(result.probability_class1 * 100).toFixed(2)}%</p>
            </div>
            <div class="result-details">
                <p>分析ID: ${result.id}</p>
                <p>分析时间: ${result.created_at}</p>
                <p>文件名: ${result.filename}</p>
                <a href="/api/analysis/${result.id}" class="btn detail-btn">查看详情</a>
            </div>
        </div>
    `;
}
