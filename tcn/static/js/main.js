// DOM 加载完成后执行代码
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

    // 设置视频上传预览
    setupVideoPreview();
    
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
});

// 通用标签页切换函数
function showTab(tabId) {
    console.log(`切换到标签页: ${tabId}`);
    
    // 隐藏所有标签页内容
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => {
        content.classList.remove('active');
    });

    // 移除所有菜单项的激活状态
    const menuItems = document.querySelectorAll('.sidebar-menu .menu-item');
    menuItems.forEach(item => {
        item.classList.remove('active');
    });

    // 显示选定的标签页内容
    const selectedTab = document.getElementById(tabId);
    if (selectedTab) {
        selectedTab.classList.add('active');
        console.log(`${tabId} 标签页已显示`);
    } else {
        console.error(`未找到ID为 ${tabId} 的标签页`);
    }

    // 设置选定菜单项的激活状态
    // 获取标签页ID对应的菜单项（移除-tab后缀）
    const menuId = tabId.replace('-tab', '');
    const selectedMenuItem = document.querySelector(`.sidebar-menu a[href="#${menuId}"]`);
    if (selectedMenuItem) {
        selectedMenuItem.classList.add('active');
        console.log(`${menuId} 菜单项已激活`);
    } else {
        console.error(`未找到链接到 ${menuId} 的菜单项`);
    }
    
    // 更新URL哈希值以便页面刷新时保持相同标签页
    window.location.hash = menuId;
}

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
    
    historyContainer.innerHTML = `
        <div class="text-center p-4">
            <div class="loader"></div>
            <p class="text-muted mt-3">加载历史记录...</p>
        </div>
    `;
    
    // 使用实际API获取历史记录
    fetch('/api/history', {
        headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('获取历史记录失败');
            }
            return response.json();
        })
        .then(data => {
            if (!data || data.length === 0) {
                historyContainer.innerHTML = `
                    <div class="text-center p-4">
                        <i class="fas fa-history text-muted" style="font-size: 3rem; opacity: 0.3;"></i>
                        <p class="mt-3">暂无历史记录</p>
                        <p class="text-muted">您的检测记录将显示在此处</p>
                    </div>
                `;
                return;
            }
            
            let historyHtml = '';
            
            data.forEach((item, index) => {
                // 使用真实返回的字段
                const isDepressed = item.depression_probability > 0.5;
                const resultClass = isDepressed ? 'danger' : 'success';
                const resultText = item.result_type || (isDepressed ? '存在抑郁风险' : '情绪状态良好');
                const resultIcon = isDepressed ? 'fa-exclamation-triangle' : 'fa-check-circle';
                const depressionPercent = (item.depression_probability * 100).toFixed(1);
                
                // 日期处理
                const date = new Date(item.created_at);
                const formattedDate = date.toLocaleDateString('zh-CN');
                const formattedTime = date.toLocaleTimeString('zh-CN');
                
                historyHtml += `
                    <div class="history-item animate-fade-in" style="animation-delay: ${index * 0.1}s">
                        <div class="history-icon ${resultClass}">
                            <i class="fas ${resultIcon}"></i>
                        </div>
                        <div class="history-content">
                            <div class="history-title">
                                ${resultText}
                                <span class="badge badge-${resultClass} ml-2">
                                    ${depressionPercent}%
                                </span>
                            </div>
                            <div class="history-date">
                                <i class="fas fa-calendar-alt"></i> ${formattedDate}
                                <i class="fas fa-clock ml-1"></i> ${formattedTime}
                                <span class="ml-2"><i class="fas fa-file-video"></i> ${item.filename}</span>
                            </div>
                        </div>
                        <div class="history-actions">
                            <button class="btn btn-icon btn-sm btn-text" onclick="viewHistoryDetails(${item.id})">
                                <i class="fas fa-eye"></i>
                            </button>
                            <button class="btn btn-icon btn-sm btn-text" onclick="downloadHistoryReport(${item.id})">
                                <i class="fas fa-download"></i>
                            </button>
                        </div>
                    </div>
                `;
            });
            
            historyContainer.innerHTML = historyHtml;
        })
        .catch(error => {
            console.error('加载历史记录失败:', error);
            historyContainer.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-circle"></i>
                    <span>加载历史记录失败，请稍后重试</span>
                </div>
            `;
        });
}

// 辅助函数：为概览页面加载简化的历史数据
function loadOverviewHistoryData() {
    const historyContainer = document.getElementById('overview-history-container');
    if (!historyContainer) return;
    
    historyContainer.innerHTML = `
        <div class="text-center p-4">
            <div class="loader"></div>
            <p class="text-muted mt-3">加载历史记录...</p>
        </div>
    `;
    
    // 使用专门的API端点获取最近的历史记录
    fetch('/api/history/recent', {
        headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('获取历史记录失败');
            }
            return response.json();
        })
        .then(data => {
            if (!data || data.length === 0) {
                historyContainer.innerHTML = `
                    <div class="text-center p-4">
                        <i class="fas fa-history text-muted" style="font-size: 3rem; opacity: 0.3;"></i>
                        <p class="mt-3">暂无历史记录</p>
                        <p class="text-muted">您的检测记录将显示在此处</p>
                    </div>
                `;
                return;
            }
            
            let historyHtml = '';
            
            data.forEach((item, index) => {
                // 使用真实返回的字段
                const isDepressed = item.depression_probability > 0.5;
                const resultClass = isDepressed ? 'danger' : 'success';
                const resultText = item.result_type || (isDepressed ? '存在抑郁风险' : '情绪状态良好');
                const resultIcon = isDepressed ? 'fa-exclamation-triangle' : 'fa-check-circle';
                const depressionPercent = (item.depression_probability * 100).toFixed(1);
                
                // 日期处理
                const date = new Date(item.created_at);
                const formattedDate = date.toLocaleDateString('zh-CN');
                
                historyHtml += `
                    <div class="history-item animate-fade-in" style="animation-delay: ${index * 0.1}s">
                        <div class="history-icon ${resultClass}">
                            <i class="fas ${resultIcon}"></i>
                        </div>
                        <div class="history-content">
                            <div class="history-title">
                                ${resultText}
                                <span class="badge badge-${resultClass} ml-2">
                                    ${depressionPercent}%
                                </span>
                            </div>
                            <div class="history-date">
                                <i class="fas fa-calendar-alt"></i> ${formattedDate}
                                <span class="ml-2 text-truncate" style="max-width: 150px;"><i class="fas fa-file-video"></i> ${item.filename}</span>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            historyContainer.innerHTML = historyHtml;
            
            // 获取总数并添加查看更多按钮
            fetch('/api/history', {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('access_token')}`
                }
            })
                .then(response => response.json())
                .then(allData => {
                    if (allData && allData.length > 3) {
                        const viewMoreBtn = document.createElement('div');
                        viewMoreBtn.className = 'text-center p-3 mt-2';
                        viewMoreBtn.innerHTML = `
                            <button class="btn btn-text btn-sm" onclick="showTab('result-tab')">
                                <i class="fas fa-history"></i> 查看全部${allData.length}条记录
                            </button>
                        `;
                        historyContainer.appendChild(viewMoreBtn);
                    }
                })
                .catch(err => {
                    console.warn('获取总历史记录数失败:', err);
                });
        })
        .catch(error => {
            console.error('加载历史记录失败:', error);
            historyContainer.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-circle"></i>
                    <span>加载历史记录失败，请稍后重试</span>
                </div>
            `;
        });
}

// 新增：切换管理员控制台标签页
function switchTab(tabId) {
    console.log(`Switching to tab: ${tabId}`);
    // 隐藏所有标签页内容
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => {
        content.classList.remove('active');
        content.style.display = 'none'; // Ensure it's hidden
    });

    // 移除所有菜单项的激活状态
    const menuItems = document.querySelectorAll('.sidebar-menu .menu-item');
    menuItems.forEach(item => {
        item.classList.remove('active');
    });

    // 显示选定的标签页内容
    const selectedTab = document.getElementById(tabId);
    if (selectedTab) {
        selectedTab.classList.add('active');
        selectedTab.style.display = 'block'; // Ensure it's shown
        console.log(`${tabId} tab displayed.`);
    } else {
        console.error(`Tab with id ${tabId} not found.`);
        // Optionally display the first tab if the requested one is not found
        const firstTab = document.querySelector('.tab-content');
        if (firstTab) {
            firstTab.classList.add('active');
            firstTab.style.display = 'block';
        }
    }

    // 设置选定菜单项的激活状态
    const selectedMenuItem = document.querySelector(`.sidebar-menu a[href="#${tabId}"]`);
    if (selectedMenuItem) {
        selectedMenuItem.classList.add('active');
        console.log(`Menu item for ${tabId} activated.`);
    } else {
        console.error(`Menu item for tab ${tabId} not found.`);
    }
}


// 新增：初始化管理员仪表盘
function initAdminDashboard(user) {
    console.log('Initializing admin dashboard with user:', user);

    // Populate admin name in sidebar and welcome message
    const adminNameElements = document.querySelectorAll('#admin-name, #admin-welcome-name');
    adminNameElements.forEach(el => {
        if (el) el.textContent = user.username || '管理员';
    });
    
    // Populate user name in header dropdown
    const userNameHeader = document.getElementById('user-name');
    if (userNameHeader) {
        userNameHeader.textContent = user.username || '管理员';
    }

    // Populate current date
    const currentDateElement = document.getElementById('current-date');
    if (currentDateElement) {
        currentDateElement.textContent = new Date().toLocaleDateString('zh-CN', { year: 'numeric', month: 'long', day: 'numeric' });
    }

    // TODO: Fetch and display stats (total users, analyses, etc.)
    // fetchAdminStats(); 

    // Initialize charts
    initAdminCharts();

    // TODO: Fetch and display recent activity
    // fetchRecentActivity();

    // Setup event listeners for dropdowns in the header, if not already handled globally
    const headerDropdownToggle = document.querySelector('.main-content .dropdown-toggle');
    const headerDropdownMenu = document.querySelector('.main-content .dropdown-menu');

    if (headerDropdownToggle && headerDropdownMenu) {
        headerDropdownToggle.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            headerDropdownMenu.classList.toggle('show');
        });
    }
    
    // Ensure the default tab (dashboard) is shown if no specific tab is targeted
    // This is usually handled by the initial HTML structure or a redirect.
    // If a specific tab was bookmarked/linked, switchTab should handle it.
    // For now, we assume 'dashboard' is active by default in HTML.
    if (window.location.hash) {
        const targetTab = window.location.hash.substring(1);
        switchTab(targetTab);
    } else {
        // Ensure dashboard is active if no hash is present
        switchTab('dashboard');
    }
}

// 新增：初始化管理员图表
function initAdminCharts() {
    console.log('Initializing admin charts...');
    // Trend Chart
    const trendCtx = document.getElementById('trend-chart')?.getContext('2d');
    if (trendCtx) {
        new Chart(trendCtx, {
            type: 'line',
            data: {
                labels: ['一月', '二月', '三月', '四月', '五月', '六月', '七月'], // Sample labels
                datasets: [{
                    label: '检测数量',
                    data: [65, 59, 80, 81, 56, 55, 40], // Sample data
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        console.log('Trend chart initialized.');
    } else {
        console.warn('Trend chart canvas not found.');
    }

    // User Distribution Chart
    const userDistCtx = document.getElementById('user-distribution')?.getContext('2d');
    if (userDistCtx) {
        new Chart(userDistCtx, {
            type: 'doughnut',
            data: {
                labels: ['患者', '医生', '管理员'], // Sample labels
                datasets: [{
                    label: '用户类型',
                    data: [300, 50, 10], // Sample data
                    backgroundColor: [
                        'rgb(54, 162, 235)',
                        'rgb(75, 192, 192)',
                        'rgb(255, 205, 86)'
                    ],
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
            }
        });
        console.log('User distribution chart initialized.');
    } else {
        console.warn('User distribution chart canvas not found.');
    }
}

// Placeholder for fetching admin stats
async function fetchAdminStats() {
    console.log('Fetching admin stats...');
    try {
        // Replace with actual API endpoint
        // const stats = await apiRequest('/api/admin/stats'); 
        const stats = { // Mock data
            totalUsers: 124,
            totalAnalyses: 1842,
            depressionRate: 23.7,
            analysesToday: 24,
            userGrowthMonthly: 12,
            analysesGrowthMonthly: 8,
            depressionRateChangeMonthly: -2.4,
            analysesTodayChangeDaily: 15
        };

        document.getElementById('total-users').textContent = stats.totalUsers;
        document.getElementById('total-analyses').textContent = stats.totalAnalyses.toLocaleString();
        document.getElementById('depression-rate').textContent = `${stats.depressionRate}%`;
        document.getElementById('analyses-today').textContent = stats.analysesToday;

        // Update stat changes (example for total users)
        const totalUsersChangeEl = document.querySelector('#total-users + .stat-change');
        if (totalUsersChangeEl) {
            totalUsersChangeEl.innerHTML = `<i class="fas fa-arrow-${stats.userGrowthMonthly >= 0 ? 'up' : 'down'}"></i> ${Math.abs(stats.userGrowthMonthly)}% <span class="text-muted">本月</span>`;
            totalUsersChangeEl.className = `stat-change stat-${stats.userGrowthMonthly >= 0 ? 'up' : 'down'}`;
        }
        // ... update other stat changes similarly ...

        console.log('Admin stats populated:', stats);
    } catch (error) {
        console.error('Failed to fetch admin stats:', error);
        showAlert('加载管理员统计数据失败', 'danger');
    }
}

// Placeholder for fetching recent activity
async function fetchRecentActivity() {
    console.log('Fetching recent activity...');
    try {
        // Replace with actual API endpoint
        // const activities = await apiRequest('/api/admin/recent-activity?limit=5');
        const activities = [ // Mock data
            { icon: 'fa-user-plus', iconColor: 'blue', title: '新用户注册', subtitle: '张明 (patient) 加入了系统', time: '10分钟前' },
            { icon: 'fa-chart-line', iconColor: 'green', title: '完成视频分析', subtitle: '刘华 完成了视频分析，结果: 情绪良好', time: '32分钟前' },
            { icon: 'fa-exclamation-triangle', iconColor: 'orange', title: '风险提醒', subtitle: '王芳 的视频分析显示存在抑郁风险', time: '1小时前' },
            { icon: 'fa-cog', iconColor: 'purple', title: '系统更新', subtitle: '模型已更新至 v2.3.5 版本', time: '2小时前' },
        ];

        const activityListContainer = document.querySelector('.activity-list');
        if (activityListContainer) {
            activityListContainer.innerHTML = ''; // Clear existing mock data
            activities.forEach(activity => {
                const item = document.createElement('div');
                item.className = 'activity-item';
                item.innerHTML = `
                    <div class="activity-icon ${activity.iconColor}">
                        <i class="fas ${activity.icon}"></i>
                    </div>
                    <div class="activity-content">
                        <div class="activity-title">${activity.title}</div>
                        <div class="activity-subtitle">${activity.subtitle}</div>
                    </div>
                    <div class="activity-time">${activity.time}</div>
                `;
                activityListContainer.appendChild(item);
            });
            console.log('Recent activity populated.');
        } else {
            console.warn('Activity list container not found.');
        }
    } catch (error) {
        console.error('Failed to fetch recent activity:', error);
        showAlert('加载最近活动失败', 'danger');
    }
}

function setupAdminPageFunctions() {
    console.log('Setting up admin page functions...');
    // This function is called on DOMContentLoaded.
    // initAdminDashboard will be called by checkAuth in admin.html once user is authenticated.
    // However, if user data is already in localStorage, we might want to initialize sooner.
    const userData = JSON.parse(localStorage.getItem('user_data'));
    const userRole = localStorage.getItem('user_role');

    if (userData && userRole === 'admin' && window.location.pathname.startsWith('/admin')) {
        // If on admin page and admin data is available, initialize.
        // This handles cases like page refresh.
        // initAdminDashboard(userData); // checkAuth in admin.html will also call this. Avoid double call for now.
                                     // Let's rely on checkAuth in admin.html to be the single source of truth for calling initAdminDashboard.
    }
    
    // Event listeners for elements that are always present, regardless of dynamic content
    // For example, the main save settings button if it's always in the DOM.
    const saveSettingsButton = document.getElementById('save-settings-btn');
    if (saveSettingsButton) {
        saveSettingsButton.addEventListener('click', () => {
            // Placeholder for actual save function
            console.log('Save settings button clicked');
            // Example: collectAndSaveSystemSettings(); 
            showAlert('设置已保存 (模拟)', 'success');
        });
    }

    // Model threshold slider update
    const thresholdSlider = document.getElementById('model-threshold');
    const thresholdValueDisplay = document.getElementById('threshold-value');
    if (thresholdSlider && thresholdValueDisplay) {
        thresholdSlider.addEventListener('input', function() {
            thresholdValueDisplay.textContent = this.value;
        });
    }
    
    // Add other general admin event listeners here if needed
    // e.g., for static buttons in System Settings or Logs if their actions don't depend on dynamic data loading.
    // For functions tied to specific tabs, they can be called within initAdminDashboard or when a tab is switched.

    // Example: Add event listeners for filter buttons in User Management (if they are static)
    const userSearchBtn = document.getElementById('search-btn'); // In User Management
    if (userSearchBtn) {
        userSearchBtn.addEventListener('click', searchUsers);
    }
    // ... and so on for other static buttons or elements.
}

// Placeholder for searchUsers function
function searchUsers() {
    console.log('searchUsers called');
    const searchTerm = document.getElementById('user-search')?.value;
    const roleFilter = document.getElementById('role-filter')?.value;
    const statusFilter = document.getElementById('status-filter')?.value;
    console.log('Search criteria:', { searchTerm, roleFilter, statusFilter });
    showAlert(`正在搜索用户 (模拟): ${searchTerm || '所有'}`, 'info');
    // TODO: Implement actual user search and table update
    // fetchAndDisplayUsers({ searchTerm, roleFilter, statusFilter });
}

// Placeholder for resetFilters function
function resetFilters() {
    console.log('resetFilters called');
    const userSearchInput = document.getElementById('user-search');
    const roleFilterSelect = document.getElementById('role-filter');
    const statusFilterSelect = document.getElementById('status-filter');

    if(userSearchInput) userSearchInput.value = '';
    if(roleFilterSelect) roleFilterSelect.value = '';
    if(statusFilterSelect) statusFilterSelect.value = '';
    
    showAlert('筛选条件已重置 (模拟)', 'info');
    // TODO: Implement actual filter reset and table update
    // fetchAndDisplayUsers(); // Fetch all users
}

// Placeholder for updateChart function (Dashboard trend chart)
function updateChart(period) {
    console.log(`updateChart called with period: ${period}`);
    showAlert(`更新图表数据: ${period} (模拟)`, 'info');
    // TODO: Fetch data for the selected period and update the trend chart
    // For example:
    // const trendChart = Chart.getChart('trend-chart');
    // if (trendChart) {
    //     // Fetch new data based on period
    //     // trendChart.data.labels = newLabels;
    //     // trendChart.data.datasets[0].data = newData;
    //     // trendChart.update();
    // }
}

// 更新趋势图表数据
function updateTrendChart(days, event) {
    console.log(`更新趋势图表，显示${days}天的数据`);
    
    // 更新按钮激活状态
    const buttons = document.querySelectorAll('.chart-actions .btn-group .btn');
    buttons.forEach(btn => {
        btn.classList.remove('active');
    });
    
    // 如果有点击事件，则设置点击的按钮为激活状态
    if (event && event.target) {
        const clickedBtn = event.target;
        clickedBtn.classList.add('active');
    }
    
    // 显示加载状态
    const chartCanvas = document.getElementById('mood-chart');
    if (chartCanvas) {
        const chart = Chart.getChart(chartCanvas);
        if (chart) {
            chart.data.datasets[0].data = [];
            chart.data.datasets[1].data = [];
            chart.update();
        }
    }
    
    // 从API获取数据
    fetch(`/api/trend?days=${days}`, {
        headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('获取趋势数据失败');
        }
        return response.json();
    })
    .then(data => {
        if (!data || !data.dates || data.dates.length === 0) {
            showAlert('没有找到符合条件的数据', 'warning');
            return;
        }
        
        // 更新图表
        if (chartCanvas) {
            const chart = Chart.getChart(chartCanvas);
            if (chart) {
                chart.data.labels = data.dates;
                chart.data.datasets[0].data = data.mood_scores;
                chart.data.datasets[1].data = data.depression_risks;
                chart.update();
                
                const period = typeof days === 'string' && days === 'all' ? '全部' : `${days}天`;
                showAlert(`已更新图表，显示${period}的数据`, 'info');
            }
        }
    })
    .catch(error => {
        console.error('加载趋势数据失败:', error);
        showAlert('加载趋势数据失败，请稍后重试', 'danger');
    });
}

// 初始化概览页面的情绪趋势小图表
function initOverviewMoodChart() {
    const chartCanvas = document.getElementById('overview-mood-chart');
    if (!chartCanvas) return;
    
    console.log('初始化概览页面情绪趋势小图表...');
    
    // 初始化空图表，风格简化版
    const chart = new Chart(chartCanvas.getContext('2d'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: '情绪指数',
                    data: [],
                    borderColor: 'rgba(96, 165, 250, 1)',
                    backgroundColor: 'rgba(96, 165, 250, 0.2)',
                    borderWidth: 2,
                    pointRadius: 2,
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    display: false
                },
                x: {
                    display: false
                }
            }
        }
    });
    
    // 从API获取最近7天数据
    fetch('/api/trend?days=7', {
        headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('获取趋势数据失败');
        }
        return response.json();
    })
    .then(data => {
        if (!data || !data.dates || data.dates.length === 0) {
            console.log('无趋势数据，显示默认图表');
            return;
        }
        
        // 更新图表数据
        chart.data.labels = data.dates;
        chart.data.datasets[0].data = data.mood_scores;
        chart.update();
    })
    .catch(error => {
        console.error('加载概览趋势数据失败:', error);
    });
}

// 处理医生页面的功能
function setupDoctorPageFunctions() {
    console.log('设置医生页面功能...');
    
    // 检查当前页面是否为医生页面
    if (window.location.pathname.indexOf('doctor') === -1) {
        return;
    }
    
    console.log('检测到医生页面，初始化功能...');
    
    // 初始化页面数据
    const userData = JSON.parse(localStorage.getItem('user_data') || '{}');
    
    // 设置问候语
    const userGreeting = document.getElementById('user-greeting');
    if (userGreeting) {
        userGreeting.textContent = `欢迎回来，${userData.username || '医生'}！`;
    }
    
    // 初始化默认标签页
    if (typeof showTab === 'function') {
        const hash = window.location.hash;
        if (hash) {
            // 如果URL中有标签页信息，显示对应标签页
            const tabId = hash.replace('#', '');
            showTab(`${tabId}-tab`);
        } else {
            // 否则默认显示概览标签页
            showTab('dashboard-tab');
        }
    } else {
        console.error('showTab函数未定义');
    }
    
    // 初始化图表
    initDoctorCharts();
    
    // 加载患者数据
    loadPatientList();
    
    // 加载最近分析记录
    loadRecentAnalyses();
}

// 初始化医生图表
function initDoctorCharts() {
    console.log('初始化医生图表...');
    
    // 患者抑郁程度分布图
    const distributionChart = document.getElementById('patientDistributionChart');
    if (distributionChart) {
        new Chart(distributionChart.getContext('2d'), {
            type: 'pie',
            data: {
                labels: ['无抑郁', '轻度抑郁', '中度抑郁', '重度抑郁'],
                datasets: [{
                    data: [65, 20, 10, 5],
                    backgroundColor: [
                        'rgba(34, 197, 94, 0.7)',
                        'rgba(234, 179, 8, 0.7)',
                        'rgba(249, 115, 22, 0.7)',
                        'rgba(239, 68, 68, 0.7)'
                    ],
                    borderColor: [
                        'rgba(34, 197, 94, 1)',
                        'rgba(234, 179, 8, 1)',
                        'rgba(249, 115, 22, 1)',
                        'rgba(239, 68, 68, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right'
                    }
                }
            }
        });
    }
    
    // 干预效果趋势图
    const trendChart = document.getElementById('interventionTrendChart');
    if (trendChart) {
        const months = ['一月', '二月', '三月', '四月', '五月'];
        
        new Chart(trendChart.getContext('2d'), {
            type: 'line',
            data: {
                labels: months,
                datasets: [{
                    label: '干预前',
                    data: [68, 72, 70, 65, 75],
                    borderColor: 'rgba(249, 115, 22, 1)',
                    backgroundColor: 'rgba(249, 115, 22, 0.1)',
                    fill: true
                }, {
                    label: '干预后',
                    data: [50, 48, 40, 35, 30],
                    borderColor: 'rgba(34, 197, 94, 1)',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: '抑郁指数 (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: '月份'
                        }
                    }
                }
            }
        });
    }
}

// 加载患者列表
function loadPatientList() {
    console.log('加载患者列表...');
    const patientTableBody = document.getElementById('patientTableBody');
    
    if (!patientTableBody) return;
    
    // 显示加载状态
    patientTableBody.innerHTML = `
        <tr>
            <td colspan="4" class="text-center p-4">
                <div class="loader"></div>
                <p class="mt-3 text-muted">加载患者数据...</p>
            </td>
        </tr>
    `;
    
    // 从API获取患者数据
    fetch('/api/doctor/patients', {
        headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('获取患者数据失败');
        }
        return response.json();
    })
    .then(patients => {
        // 清空现有内容
        patientTableBody.innerHTML = '';
        
        if (!patients || patients.length === 0) {
            patientTableBody.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center p-4">
                        <i class="fas fa-users text-muted" style="font-size: 2rem; opacity: 0.3;"></i>
                        <p class="mt-3">暂无患者数据</p>
                    </td>
                </tr>
            `;
            return;
        }
        
        // 添加患者行
        patients.forEach((patient, index) => {
            // 确定风险级别和徽章样式
            let badgeClass, riskText;
            const status = patient.status?.toLowerCase() || 'unknown';
            
            switch (status) {
                case '良好':
                case '稳定':
                    badgeClass = 'badge-success';
                    riskText = patient.status;
                    break;
                case '需关注':
                    badgeClass = 'badge-warning';
                    riskText = patient.status;
                    break;
                case '高风险':
                    badgeClass = 'badge-danger';
                    riskText = patient.status;
                    break;
                default:
                    badgeClass = 'badge-info';
                    riskText = '未评估';
            }
            
            // 创建患者行
            const row = document.createElement('tr');
            row.className = 'animate-fade-in';
            row.style.animationDelay = `${index * 0.05}s`;
            
            // 格式化最后分析日期
            let formattedLastDate = patient.last_analysis_date || '未记录';
            if (formattedLastDate !== '未记录') {
                try {
                    const date = new Date(formattedLastDate);
                    formattedLastDate = date.toLocaleDateString('zh-CN');
                } catch (e) {
                    console.warn('日期格式化失败:', e);
                }
            }
            
            row.innerHTML = `
                <td>
                    <div class="d-flex align-center gap-sm">
                        <div class="patient-avatar-sm">${patient.name?.charAt(0) || '?'}</div>
                        <div>
                            <div class="font-medium">${patient.name || '未知患者'}</div>
                            <div class="text-sm text-muted">${patient.age || '--'}岁 | ${patient.gender || '--'}</div>
                        </div>
                    </div>
                </td>
                <td>${formattedLastDate}</td>
                <td><span class="badge ${badgeClass}">${riskText}</span></td>
                <td>
                    <div class="d-flex gap-sm">
                        <button class="btn btn-icon-only btn-sm" title="查看详情" onclick="viewPatientDetails(${patient.id})">
                            <i class="fas fa-eye"></i>
                        </button>
                        <button class="btn btn-icon-only btn-sm" title="编辑患者" onclick="editPatient(${patient.id})">
                            <i class="fas fa-edit"></i>
                        </button>
                        <button class="btn btn-icon-only btn-sm" title="发送消息" onclick="messagePatient(${patient.id})">
                            <i class="fas fa-comment"></i>
                        </button>
                    </div>
                </td>
            `;
            
            patientTableBody.appendChild(row);
        });
    })
    .catch(error => {
        console.error('加载患者列表失败:', error);
        patientTableBody.innerHTML = `
            <tr>
                <td colspan="4">
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle"></i>
                        <span>加载患者数据失败，请稍后重试</span>
                    </div>
                </td>
            </tr>
        `;
    });
}

// 加载最近分析记录
function loadRecentAnalyses() {
    console.log('加载最近分析记录...');
    const analysisTable = document.getElementById('recentAnalysisTable');
    
    if (!analysisTable) return;
    
    // 显示加载状态
    analysisTable.innerHTML = `
        <tr>
            <td colspan="7" class="text-center p-4">
                <div class="loader"></div>
                <p class="mt-3 text-muted">加载分析记录...</p>
            </td>
        </tr>
    `;
    
    // 从API获取分析记录
    fetch('/api/doctor/recent_analyses', {
        headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('获取分析记录失败');
        }
        return response.json();
    })
    .then(analyses => {
        // 清空现有内容
        analysisTable.innerHTML = '';
        
        if (!analyses || analyses.length === 0) {
            analysisTable.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center p-4">
                        <i class="fas fa-chart-line text-muted" style="font-size: 2rem; opacity: 0.3;"></i>
                        <p class="mt-3">暂无分析记录</p>
                    </td>
                </tr>
            `;
            return;
        }
        
        // 添加分析记录行
        analyses.forEach((analysis, index) => {
            // 确定结果样式
            let badgeClass;
            const result = analysis.result.toLowerCase();
            
            if (result.includes('正常')) {
                badgeClass = 'badge-success';
            } else if (result.includes('轻度')) {
                badgeClass = 'badge-info';
            } else if (result.includes('中度')) {
                badgeClass = 'badge-warning';
            } else if (result.includes('重度')) {
                badgeClass = 'badge-danger';
            } else {
                badgeClass = 'badge-secondary';
            }
            
            // 计算置信度百分比
            const confidencePercent = (analysis.confidence * 100).toFixed(1);
            
            // 格式化日期
            let formattedDate = analysis.date;
            try {
                const date = new Date(analysis.date);
                formattedDate = date.toLocaleDateString('zh-CN');
            } catch (e) {
                console.warn('日期格式化失败:', e);
            }
            
            // 创建分析记录行
            const row = document.createElement('tr');
            row.className = 'animate-fade-in';
            row.style.animationDelay = `${index * 0.05}s`;
            
            // 生成随机ID（实际应使用后端提供的ID）
            const analysisId = analysis.id || Math.floor(Math.random() * 10000);
            
            row.innerHTML = `
                <td>#${analysisId}</td>
                <td>
                    <div class="d-flex align-center gap-sm">
                        <div class="patient-avatar-sm">${analysis.patient_name?.charAt(0) || '?'}</div>
                        <div>${analysis.patient_name || '未知患者'}</div>
                    </div>
                </td>
                <td>视频分析</td>
                <td><span class="badge ${badgeClass}">${analysis.result}</span></td>
                <td>
                    <div class="progress-wrapper">
                        <div class="progress-bar" style="width: ${confidencePercent}%"></div>
                        <span>${confidencePercent}%</span>
                    </div>
                </td>
                <td>${formattedDate}</td>
                <td>
                    <div class="d-flex gap-sm">
                        <button class="btn btn-icon-only btn-sm" title="查看详情" onclick="viewAnalysisDetails(${analysisId})">
                            <i class="fas fa-eye"></i>
                        </button>
                        <button class="btn btn-icon-only btn-sm" title="导出报告" onclick="exportAnalysisReport(${analysisId})">
                            <i class="fas fa-download"></i>
                        </button>
                    </div>
                </td>
            `;
            
            analysisTable.appendChild(row);
        });
    })
    .catch(error => {
        console.error('加载分析记录失败:', error);
        analysisTable.innerHTML = `
            <tr>
                <td colspan="7">
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle"></i>
                        <span>加载分析记录失败，请稍后重试</span>
                    </div>
                </td>
            </tr>
        `;
    });
}

// 查看患者详情
function viewPatientDetails(patientId) {
    console.log(`查看患者详情，ID: ${patientId}`);
    // 实际实现中，这里应该打开一个患者详情的模态框或页面
    showAlert(`查看患者ID：${patientId}的详细信息`, 'info');
    
    // 模拟后端API调用
    setTimeout(() => {
        // 显示加载中状态
        document.getElementById('patients-tab').innerHTML += `
            <div id="patient-details-modal" class="modal" style="display: block;">
                <div class="modal-content">
                    <div class="modal-header">
                        <h3>患者详细信息</h3>
                        <button class="close-btn" onclick="closePatientDetails()">&times;</button>
                    </div>
                    <div class="modal-body">
                        <div class="loader-container text-center p-5">
                            <div class="loader"></div>
                            <p class="mt-3">正在加载患者信息...</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // 模拟加载完成，显示患者详情
        setTimeout(() => {
            document.querySelector('#patient-details-modal .modal-body').innerHTML = `
                <div class="patient-info">
                    <div class="patient-header d-flex gap-lg mb-4">
                        <div class="patient-avatar-lg">张</div>
                        <div>
                            <h3 class="m-0">张明</h3>
                            <p class="text-muted">28岁 | 男 | 患者ID: ${patientId}</p>
                            <div class="badge badge-warning">中等抑郁风险</div>
                        </div>
                    </div>
                    
                    <div class="tabs">
                        <div class="tab-item active">基本信息</div>
                        <div class="tab-item">检测历史</div>
                        <div class="tab-item">干预记录</div>
                    </div>
                    
                    <div class="tab-content active">
                        <div class="form-group">
                            <label>联系电话</label>
                            <p>135-1234-5678</p>
                        </div>
                        <div class="form-group">
                            <label>电子邮箱</label>
                            <p>zhangming@example.com</p>
                        </div>
                        <div class="form-group">
                            <label>首次就诊</label>
                            <p>2025年3月15日</p>
                        </div>
                        <div class="form-group">
                            <label>主诊医生</label>
                            <p>李医生</p>
                        </div>
                        <div class="form-group">
                            <label>备注</label>
                            <p>患者报告近期工作压力增大，睡眠质量下降。</p>
                        </div>
                    </div>
                </div>
                <div class="modal-footer mt-4">
                    <button class="btn btn-outline" onclick="closePatientDetails()">关闭</button>
                    <button class="btn btn-primary">编辑信息</button>
                </div>
            `;
        }, 1500);
    }, 300);
}

// 关闭患者详情
function closePatientDetails() {
    const modal = document.getElementById('patient-details-modal');
    if (modal) {
        modal.remove();
    }
}

// 编辑患者信息
function editPatient(patientId) {
    console.log(`编辑患者信息，ID: ${patientId}`);
    showAlert(`编辑患者ID：${patientId}的信息`, 'info');
    // 实际实现中，这里应该打开一个编辑患者信息的表单
}

// 给患者发送消息
function messagePatient(patientId) {
    console.log(`给患者发送消息，ID: ${patientId}`);
    showAlert(`给患者ID：${patientId}发送消息`, 'info');
    // 实际实现中，这里应该打开一个发送消息的对话框
}

// 查看分析详情
function viewAnalysisDetails(analysisId) {
    console.log(`查看分析详情，ID: ${analysisId}`);
    showAlert('正在加载分析详情...', 'info');
    
    // 切换到分析结果标签页
    showTab('analysis-tab');
    
    // 从API获取分析详情
    fetch(`/api/analysis/${analysisId}`, {
        headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('获取分析详情失败');
        }
        return response.json();
    })
    .then(data => {
        // 创建模态框
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.id = 'analysis-details-modal';
        
        // 计算抑郁概率百分比
        const depressionPercent = (data.depression_probability * 100).toFixed(1);
        const nonDepressionPercent = (data.non_depression_probability * 100).toFixed(1);
        
        // 格式化日期
        let formattedDate;
        try {
            const date = new Date(data.created_at);
            formattedDate = `${date.toLocaleDateString('zh-CN')} ${date.toLocaleTimeString('zh-CN')}`;
        } catch (e) {
            formattedDate = data.created_at || '未知时间';
        }
        
        // 确定结果徽章样式
        let badgeClass;
        if (data.result_type === '正常') {
            badgeClass = 'badge-success';
        } else if (data.result_type === '轻度抑郁') {
            badgeClass = 'badge-info';
        } else if (data.result_type === '中度抑郁') {
            badgeClass = 'badge-warning';
        } else if (data.result_type === '重度抑郁') {
            badgeClass = 'badge-danger';
        } else {
            badgeClass = 'badge-secondary';
        }
        
        // 构建模态框内容
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>分析结果详情 #${data.id}</h3>
                    <button class="close-btn" onclick="closeAnalysisDetails()">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="analysis-meta mb-4">
                        <div class="d-flex justify-between mb-3">
                            <div>
                                <strong>患者:</strong> ${data.username || '未知'}
                            </div>
                            <div>
                                <strong>分析时间:</strong> ${formattedDate}
                            </div>
                        </div>
                        <div class="d-flex justify-between mb-3">
                            <div>
                                <strong>文件名:</strong> ${data.filename || '未知'}
                            </div>
                            <div>
                                <strong>结果:</strong> <span class="badge ${badgeClass}">${data.result_type}</span>
                            </div>
                        </div>
                        
                        <div class="card bg-light p-3 mb-3">
                            <h4 class="mb-2">抑郁概率分析</h4>
                            <div class="mb-2">
                                <label>抑郁概率:</label>
                                <div class="progress-wrapper">
                                    <div class="progress-bar bg-danger" style="width: ${depressionPercent}%"></div>
                                    <span>${depressionPercent}%</span>
                                </div>
                            </div>
                            <div class="mb-2">
                                <label>非抑郁概率:</label>
                                <div class="progress-wrapper">
                                    <div class="progress-bar bg-success" style="width: ${nonDepressionPercent}%"></div>
                                    <span>${nonDepressionPercent}%</span>
                                </div>
                            </div>
                        </div>
                        
                        <h4 class="mb-2">详细分析</h4>
                        <div class="analysis-details-grid">
                            ${data.facial_analysis ? `
                            <div class="analysis-detail-card">
                                <h5>面部表情分析</h5>
                                <p>${data.facial_analysis.expression || '未提供面部表情分析数据'}</p>
                            </div>
                            ` : ''}
                            
                            ${data.voice_analysis ? `
                            <div class="analysis-detail-card">
                                <h5>语音分析</h5>
                                <p>${data.voice_analysis.tone || '未提供语音分析数据'}</p>
                            </div>
                            ` : ''}
                            
                            ${data.body_language_analysis ? `
                            <div class="analysis-detail-card">
                                <h5>肢体语言分析</h5>
                                <p>${data.body_language_analysis.movement || '未提供肢体语言分析数据'}</p>
                            </div>
                            ` : ''}
                        </div>
                        
                        <div class="mt-4">
                            <h4>医生注释</h4>
                            <div class="notes-editor-container">
                                <textarea id="doctor-notes" class="form-control" rows="3" placeholder="在此处添加对该分析结果的备注...">${data.doctor_notes || ''}</textarea>
                                <button id="save-notes-btn" class="btn btn-primary mt-2" onclick="saveAnalysisNotes(${data.id})">保存备注</button>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-outline" onclick="closeAnalysisDetails()">关闭</button>
                    <button class="btn btn-primary" onclick="exportAnalysisReport(${data.id})">导出报告</button>
                </div>
            </div>
        `;
        
        // 添加到页面
        document.body.appendChild(modal);
        
        // 显示模态框
        setTimeout(() => {
            modal.style.display = 'block';
        }, 10);
    })
    .catch(error => {
        console.error('获取分析详情失败:', error);
        showAlert('获取分析详情失败，请稍后重试', 'danger');
    });
}

// 关闭分析详情模态框
function closeAnalysisDetails() {
    const modal = document.getElementById('analysis-details-modal');
    if (modal) {
        modal.style.display = 'none';
        setTimeout(() => {
            modal.remove();
        }, 300);
    }
}

// 保存医生注释
function saveAnalysisNotes(analysisId) {
    const notesTextarea = document.getElementById('doctor-notes');
    const notes = notesTextarea ? notesTextarea.value : '';
    
    showAlert('正在保存备注...', 'info');
    
    // 这里应该调用API保存备注
    // 模拟API调用
    setTimeout(() => {
        showAlert('备注已保存', 'success');
    }, 1000);
}

// 导出分析报告
function exportAnalysisReport(analysisId) {
    console.log(`导出分析报告，ID: ${analysisId}`);
    showAlert('正在生成报告，请稍候...', 'info');
    
    // 直接打开报告在新窗口
    window.open(`/api/analysis/${analysisId}/report`, '_blank');
}

// 设置患者页面功能
function setupPatientPageFunctions() {
    console.log('设置患者页面功能...');
    const uploadForm = document.getElementById('upload-form');
    const resultContainer = document.getElementById('result-container');
    
    // 检查当前页面是否为患者页面
    if (window.location.pathname.indexOf('patient') === -1) {
        return;
    }
    
    console.log('检测到患者页面，初始化功能...');
    
    // 初始化默认标签页 (概览)
    if (typeof showTab === 'function') {
        const hash = window.location.hash;
        if (hash) {
            // 如果URL中有标签页信息，显示对应标签页
            const tabId = hash.replace('#', '');
            showTab(`${tabId}-tab`);
        } else {
            // 否则默认显示概览标签页
            showTab('overview-tab');
        }
    } else {
        console.error('showTab函数未定义');
    }
    
    // 初始化情绪趋势图表
    initMoodChart();
    initOverviewMoodChart();
    
    // 处理视频上传表单
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const videoFile = document.getElementById('video-upload').files[0];
            if (!videoFile) {
                showAlert('请选择视频文件', 'warning');
                return;
            }
            
            // 显示上传进度
            const progressContainer = document.getElementById('upload-progress-container');
            const progressBar = document.getElementById('upload-progress');
            const progressText = document.getElementById('progress-percentage');
            
            if (progressContainer) {
                progressContainer.style.display = 'block';
            }
            
            if (progressBar) {
                progressBar.style.width = '0%';
            }
            
            // 显示加载提示
            if (resultContainer) {
                resultContainer.innerHTML = `
                    <div class="card animate-fade-in">
                        <div class="card-body text-center">
                            <div class="loader"></div>
                            <p class="mt-3">正在处理视频，请稍候...</p>
                            <p class="text-muted">视频分析可能需要几分钟时间</p>
                        </div>
                    </div>
                `;
            }
            
            const formData = new FormData();
            formData.append('video_file', videoFile);
            
            const xhr = new XMLHttpRequest();
            
            // 设置进度跟踪
            xhr.upload.addEventListener('progress', function(e) {
                if (e.lengthComputable) {
                    const percentComplete = Math.floor((e.loaded / e.total) * 100);
                    if (progressBar) {
                        progressBar.style.width = percentComplete + '%';
                    }
                    if (progressText) {
                        progressText.textContent = percentComplete + '%';
                    }
                }
            });
            
            // 处理请求完成
            xhr.addEventListener('load', function() {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        const data = JSON.parse(xhr.responseText);
                        displayAnalysisResult(data);
                        
                        // 自动切换到结果标签页
                        showTab('result-tab');
                        
                        // 刷新历史记录
                        loadHistoryData();
                        loadOverviewHistoryData();
                    } catch (error) {
                        console.error('解析响应失败:', error);
                        showAlert('处理结果失败', 'danger');
                    }
                } else {
                    console.error('请求失败:', xhr.status);
                    showAlert('视频分析请求失败', 'danger');
                }
            });
            
            // 处理上传错误
            xhr.addEventListener('error', function() {
                console.error('上传失败');
                showAlert('上传失败，请检查网络连接', 'danger');
            });
            
            // 发送请求
            xhr.open('POST', '/predict', true);
            xhr.setRequestHeader('Authorization', `Bearer ${localStorage.getItem('access_token')}`);
            xhr.send(formData);
        });
    }
    
    // 加载历史记录
    loadHistoryData();
    loadOverviewHistoryData();
    
    // 初始化个人资料表单
    initProfileForm();
}

// 设置视频上传预览
function setupVideoPreview() {
    const videoInput = document.getElementById('video-upload');
    const previewContainer = document.getElementById('video-preview-container');
    
    if (!videoInput || !previewContainer) return;
    
    console.log('设置视频上传预览功能...');
    
    videoInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (!file) {
            previewContainer.innerHTML = '';
                       return;
        }
        
        // 检查是否为视频文件
        if (!file.type.startsWith('video/')) {
            showAlert('请上传有效的视频文件', 'warning');
            previewContainer.innerHTML = '';
            return;
        }
        
        // 创建视频预览
        const videoPreview = document.createElement('video');
        videoPreview.classList.add('video-preview');
        videoPreview.controls = true;
        videoPreview.src = URL.createObjectURL(file);
        
        // 清空预览容器并添加视频预览
        previewContainer.innerHTML = '';
        previewContainer.appendChild(videoPreview);
        
        // 显示文件信息
        const fileInfo = document.createElement('div');
        fileInfo.classList.add('file-info', 'mt-2');
        
        // 转换文件大小为可读格式
        const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
        
        fileInfo.innerHTML = `
            <p><strong>文件名:</strong> ${file.name}</p>
            <p><strong>类型:</strong> ${file.type}</p>
            <p><strong>大小:</strong> ${fileSizeMB} MB</p>
        `;
        
        previewContainer.appendChild(fileInfo);
    });
}

// 初始化情绪趋势图表
function initMoodChart() {
    const chartCanvas = document.getElementById('mood-chart');
    if (!chartCanvas) return;
    
    console.log('初始化情绪趋势图表...');
    
    // 初始化空图表
    const chart = new Chart(chartCanvas.getContext('2d'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: '情绪指数',
                    data: [],
                    borderColor: 'rgba(96, 165, 250, 1)',
                    backgroundColor: 'rgba(96, 165, 250, 0.2)',
                    borderWidth: 2,
                    pointRadius: 3,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: '抑郁风险',
                    data: [],
                    borderColor: 'rgba(252, 165, 165, 1)',
                    backgroundColor: 'rgba(252, 165, 165, 0.2)',
                    borderWidth: 2,
                    pointRadius: 3,
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
    
    // 从API获取数据
    fetch('/api/trend?days=30', {
        headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('获取趋势数据失败');
        }
        return response.json();
    })
    .then(data => {
        if (!data || !data.dates || data.dates.length === 0) {
            console.log('无趋势数据，显示默认图表');
            // 没有数据时显示空图表即可
            return;
        }
        
        // 更新图表数据
        chart.data.labels = data.dates;
        chart.data.datasets[0].data = data.mood_scores;
        chart.data.datasets[1].data = data.depression_risks;
        chart.update();
    })
    .catch(error => {
        console.error('加载趋势数据失败:', error);
    });
}

// 辅助函数：显示警告消息
function showAlert(message, type = 'info') {
    console.log(`显示提醒: ${message} (类型: ${type})`);
    
    // 首先查找页面上已有的alert容器
    let alertContainer = document.getElementById('alert-container');
    
    // 如果容器不存在，则创建一个
    if (!alertContainer) {
        alertContainer = document.createElement('div');
        alertContainer.id = 'alert-container';
        alertContainer.className = 'alert-container';
        alertContainer.style.position = 'fixed';
        alertContainer.style.top = '20px';
        alertContainer.style.right = '20px';
        alertContainer.style.maxWidth = '350px';
        alertContainer.style.zIndex = '9999';
        document.body.appendChild(alertContainer);
    }
    
    // 创建提示框
    const alertBox = document.createElement('div');
    alertBox.className = `alert alert-${type}`;
    alertBox.style.padding = '10px 15px';
    alertBox.style.marginBottom = '10px';
    alertBox.style.borderRadius = '4px';
    alertBox.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.2)';
    alertBox.style.opacity = '0';
    alertBox.style.transition = 'opacity 0.3s ease-in-out';
    
    // 设置不同类型提示的颜色
    switch (type) {
        case 'success':
            alertBox.style.backgroundColor = '#dff0d8';
            alertBox.style.borderColor = '#d6e9c6';
            alertBox.style.color = '#3c763d';
            break;
        case 'info':
            alertBox.style.backgroundColor = '#d9edf7';
            alertBox.style.borderColor = '#bce8f1';
            alertBox.style.color = '#31708f';
            break;
        case 'warning':
            alertBox.style.backgroundColor = '#fcf8e3';
            alertBox.style.borderColor = '#faebcc';
            alertBox.style.color = '#8a6d3b';
            break;
        case 'danger':
            alertBox.style.backgroundColor = '#f2dede';
            alertBox.style.borderColor = '#ebccd1';
            alertBox.style.color = '#a94442';
            break;
        default:
            alertBox.style.backgroundColor = '#f8f9fa';
            alertBox.style.borderColor = '#ddd';
            alertBox.style.color = '#333';
    }
    
    // 添加关闭按钮
    const closeButton = document.createElement('button');
    closeButton.innerHTML = '&times;';
    closeButton.className = 'close-btn';
    closeButton.style.marginLeft = '10px';
    closeButton.style.border = 'none';
    closeButton.style.background = 'transparent';
    closeButton.style.float = 'right';
    closeButton.style.fontSize = '18px';
    closeButton.style.fontWeight = 'bold';
    closeButton.style.lineHeight = '1';
    closeButton.style.cursor = 'pointer';
    closeButton.style.color = 'inherit';
    closeButton.style.opacity = '0.5';
    
    closeButton.addEventListener('click', () => {
        fadeOut(alertBox);
    });
    
    // 添加消息内容
    const messageSpan = document.createElement('span');
    messageSpan.innerHTML = message;
    
    // 组合提示框
    alertBox.appendChild(closeButton);
    alertBox.appendChild(messageSpan);
    
    // 添加到容器
    alertContainer.appendChild(alertBox);
    
    // 淡入显示
    setTimeout(() => {
        alertBox.style.opacity = '1';
    }, 10);
    
    // 3秒后自动消失
    setTimeout(() => {
        fadeOut(alertBox);
    }, 3000);
    
    // 淡出函数
    function fadeOut(element) {
        element.style.opacity = '0';
        setTimeout(() => {
            if (element.parentNode) {
                element.parentNode.removeChild(element);
            }
        }, 300);
    }
}

// 辅助函数：下载历史记录报告
function downloadHistoryReport(historyId) {
    console.log(`下载历史记录报告，ID: ${historyId}`);
    // 实际实现中，这里应该调用API生成并下载报告
    showAlert(`历史记录报告（ID：${historyId}）下载功能尚未实现`, 'info');
}

// 辅助函数：查看历史记录详情
function viewHistoryDetails(historyId) {
    console.log(`查看历史记录详情，ID: ${historyId}`);
    // 实际实现中，这里应该打开一个历史记录详情的模态框或页面
    showAlert(`查看历史记录ID：${historyId}的详细信息`, 'info');
}

// 辅助函数：初始化用户管理表格
function initUserManagementTable() {
    console.log('初始化用户管理表格...');
    // TODO: 实现用户管理表格的初始化逻辑
}

// 辅助函数：实现用户搜索功能
function searchUsers() {
    console.log('实现用户搜索功能...');
    // TODO: 实现用户搜索的具体逻辑
}

// 辅助函数：重置筛选条件
function resetFilters() {
    console.log('重置筛选条件...');
    // TODO: 实现重置筛选条件的具体逻辑
}

// 辅助函数：更新图表数据
function updateChart(period) {
    console.log(`更新图表数据，周期: ${period}...`);
    // TODO: 实现更新图表数据的具体逻辑
}

// 注意: viewAnalysisDetails 和 exportAnalysisReport 函数已在文件的前面部分定义

// 辅助函数：初始化个人资料表单
function initProfileForm() {
    const profileForm = document.getElementById('profile-form');
    if (!profileForm) return;
    
    console.log('初始化个人资料表单...');
    
    // 获取用户数据并填充表单
    const userData = JSON.parse(localStorage.getItem('user_data') || '{}');
    
    const usernameInput = document.getElementById('username');
    const emailInput = document.getElementById('email');
    
    if (usernameInput) usernameInput.value = userData.username || '';
    if (emailInput) emailInput.value = userData.email || '';
    
    // 处理表单提交
    profileForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const currentPassword = document.getElementById('current-password').value;
        const newPassword = document.getElementById('new-password').value;
        const confirmPassword = document.getElementById('confirm-password').value;
        const email = emailInput ? emailInput.value : '';
        
        if (!currentPassword) {
            showAlert('请输入当前密码', 'warning');
            return;
        }
        
        if (newPassword && newPassword !== confirmPassword) {
            showAlert('两次输入的新密码不一致', 'warning');
            return;
        }
        
        // 构建更新数据
        const formData = new FormData();
        formData.append('current_password', currentPassword);
        
        if (newPassword) {
            formData.append('new_password', newPassword);
        }
        
        if (email && email !== userData.email) {
            formData.append('email', email);
        }
        
        // 发送请求更新个人资料
        const token = localStorage.getItem('access_token');
        
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
            showAlert('个人资料更新成功!', 'success');
            
            // 更新本地存储的用户数据
            if (data.email) {
                userData.email = data.email;
                localStorage.setItem('user_data', JSON.stringify(userData));
            }
            
            // 清空密码字段
            document.getElementById('current-password').value = '';
            document.getElementById('new-password').value = '';
            document.getElementById('confirm-password').value = '';
        })
        .catch(error => {
            showAlert(error.message || '更新个人资料时出错', 'danger');
        });
    });
    
    console.log('个人资料表单初始化完成');
}

// 处理分析结果页面上的按钮事件
function setupAnalysisResultButtons() {
    console.log('设置分析结果页面按钮事件...');
    
    // 绑定查看详情按钮事件
    document.addEventListener('click', function(e) {
        if (e.target && e.target.classList.contains('view-details-btn')) {
            const resultId = e.target.dataset.resultId;
            if (resultId) {
                viewAnalysisDetails(resultId);
            }
        }
    });
    
    // 绑定导出报告按钮事件
    document.addEventListener('click', function(e) {
        if (e.target && e.target.classList.contains('export-report-btn')) {
            const resultId = e.target.dataset.resultId;
            if (resultId) {
                exportAnalysisReport(resultId);
            }
        }
    });
    
    // 绑定删除结果按钮事件
    document.addEventListener('click', function(e) {
        if (e.target && e.target.classList.contains('delete-result-btn')) {
            const resultId = e.target.dataset.resultId;
            if (resultId && confirm('确定要删除这条记录吗？此操作不可恢复。')) {
                deleteAnalysisResult(resultId);
            }
        }
    });
}

// 删除分析结果的函数
function deleteAnalysisResult(resultId) {
    console.log(`删除分析结果，ID: ${resultId}`);
    showAlert('正在删除记录...', 'info');
    
    fetch(`/api/analysis/${resultId}`, {
        method: 'DELETE',
        headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('删除记录失败');
        }
        showAlert('记录已成功删除', 'success');
        
        // 刷新历史记录列表
        loadHistoryData();
        
        // 如果概览页面也有历史记录，也刷新它
        if (document.getElementById('overview-history-container')) {
            loadOverviewHistoryData();
        }
    })
    .catch(error => {
        console.error('删除记录失败:', error);
        showAlert('删除记录失败，请稍后重试', 'danger');
    });
}

// 处理导航栏的激活状态
function highlightActiveNavItem() {
    console.log('高亮当前导航菜单项...');
    
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-item a, .sidebar-menu a');
    
    navLinks.forEach(link => {
        // 移除所有激活状态
        link.classList.remove('active');
        
        // 获取链接的href属性
        const href = link.getAttribute('href');
        if (!href) return;
        
        // 检查当前路径是否匹配链接路径
        if (href === '/' && currentPath === '/') {
            // 首页
            link.classList.add('active');
        } else if (href !== '/' && currentPath.startsWith(href)) {
            // 子页面，如 /patient, /doctor, /admin 等
            link.classList.add('active');
        }
    });
    
    // 处理URL中的哈希值对应的标签页
    const hash = window.location.hash;
    if (hash && typeof showTab === 'function') {
        const tabId = hash.replace('#', '');
        if (document.getElementById(`${tabId}-tab`)) {
            showTab(`${tabId}-tab`);
        }
    }
}