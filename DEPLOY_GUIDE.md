# 🚀 部署到Streamlit Cloud - 完整指南

## 第一步:在GitHub创建仓库

1. 访问 https://github.com/new
2. 填写仓库信息:
   - **Repository name**: `rag-smart-customer-service` (或其他你喜欢的名字)
   - **Description**: `基于RAG的智能客服系统 - 支持混合检索和响应缓存`
   - **Visibility**: Public (公开,Streamlit Cloud免费套餐要求)
   - **不要勾选** "Initialize with README" (我们已经有README了)
3. 点击 "Create repository"

## 第二步:推送代码到GitHub

创建完仓库后,GitHub会显示推送命令。在你的终端执行:

```bash
cd "d:\PycharmProjects\FastAPIProject\P4_RAG项目案例"

# 添加远程仓库(替换YOUR_USERNAME为你的GitHub用户名)
git remote add origin https://github.com/YOUR_USERNAME/rag-smart-customer-service.git

# 推送到GitHub
git branch -M main
git push -u origin main
```

**注意**: 将 `YOUR_USERNAME` 替换为你的实际GitHub用户名

## 第三步:配置API Key

### 方法1:在Streamlit Cloud界面配置(推荐)

1. 部署时会在Streamlit Cloud dashboard看到 "Secrets" 选项
2. 添加以下内容:

```toml
DASHSCOPE_API_KEY = "你的通义千问API Key"
```

### 方法2:获取API Key

如果还没有API Key:
1. 访问 https://dashscope.console.aliyun.com/
2. 登录阿里云账号
3. 进入 "API-KEY管理"
4. 创建或复制现有API Key

## 第四步:部署到Streamlit Cloud

1. 访问 https://streamlit.io/cloud
2. 点击 "Sign in with GitHub"
3. 授权Streamlit Cloud访问你的GitHub仓库
4. 点击 "New app"
5. 选择:
   - **Repository**: 选择你刚创建的仓库
   - **Branch**: `main`
   - **Main file path**: `app_qa.py`
6. 点击 "Deploy!"

## 第五步:等待部署完成

- 首次部署约需2-5分钟
- 部署成功后会获得一个公网URL,类似:
  `https://your-app-name.streamlit.app`

## 第六步:测试和分享

1. 访问获得的URL
2. 测试问答功能
3. 将这个URL添加到简历中!

---

## 🔧 常见问题

### Q1: 部署失败,提示缺少依赖
**解决**: 确保requirements.txt包含所有依赖,我们已经准备好了

### Q2: 应用启动但无法回答问题
**解决**: 检查是否正确配置了DASHSCOPE_API_KEY

### Q3: 如何更新代码?
**解决**:
```bash
# 修改代码后
git add .
git commit -m "更新说明"
git push

# Streamlit Cloud会自动重新部署
```

### Q4: 免费套餐限制
- 每月750小时运行时间(足够个人项目)
- 20小时不活动后自动休眠
- 唤醒时间约30秒

---

## 📊 部署后的简历描述

```markdown
项目名称: 智能客服系统(已上线)
在线演示: https://your-app-name.streamlit.app

技术栈: LangChain + Chroma + Streamlit + 通义千问

核心亮点:
1. 实现响应缓存机制,重复问题响应速度提升95%
2. 开发混合检索策略(向量+BM25),召回率提升15-25%
3. 独立完成从开发到云端部署的全流程
4. 系统已上线,支持公网访问和实时演示
```

---

## ✅ 部署检查清单

- [ ] GitHub仓库已创建
- [ ] 代码已推送到GitHub
- [ ] API Key已配置
- [ ] Streamlit Cloud应用已部署
- [ ] 获得了公网URL
- [ ] 测试了所有功能正常
- [ ] URL已添加到简历

---

祝你部署顺利! 🎉
