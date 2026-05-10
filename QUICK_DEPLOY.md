# ⚡ 快速部署指南 (3步完成)

## 📝 你现在需要做的:

### 步骤1: 在GitHub创建仓库 (2分钟)

1. 访问: https://github.com/new
2. 填写:
   - 仓库名: `rag-smart-customer-service`
   - 描述: `智能客服RAG系统 - 已优化`
   - 公开仓库(Public)
   - **不要**勾选"Initialize with README"
3. 点击 "Create repository"

---

### 步骤2: 推送代码到GitHub (1分钟)

**方法A: 使用脚本(推荐)**

双击运行项目目录下的: `push_to_github.bat`

输入你的GitHub用户名,自动完成推送

**方法B: 手动执行命令**

```bash
# 替换YOUR_USERNAME为你的GitHub用户名
git remote add origin https://github.com/YOUR_USERNAME/rag-smart-customer-service.git
git branch -M main
git push -u origin main
```

---

### 步骤3: 部署到Streamlit Cloud (3分钟)

1. 访问: https://streamlit.io/cloud
2. 点击 "Sign in with GitHub" (用GitHub账号登录)
3. 授权后点击 "New app"
4. 选择:
   - Repository: 你的仓库
   - Branch: `main`
   - Main file path: `app_qa.py`
5. **重要**: 在 "Advanced settings" → "Secrets" 中添加:
   ```toml
   DASHSCOPE_API_KEY = "sk-你的API密钥"
   ```
6. 点击 "Deploy!"

等待2-5分钟,部署完成后你会获得一个URL:
`https://xxx.streamlit.app`

---

## ✅ 完成!

将这个URL添加到简历中,面试官可以直接访问体验!

---

## 🔑 如何获取API Key?

1. 访问: https://dashscope.console.aliyun.com/
2. 注册/登录阿里云
3. 进入 "API-KEY管理"
4. 创建新Key或复制现有Key

---

## 💡 提示

- Streamlit Cloud免费套餐: 每月750小时
- 20分钟无活动会自动休眠
- 下次访问会自动唤醒(约30秒)
- 代码更新后push到GitHub,Streamlit Cloud会自动重新部署
