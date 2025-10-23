# Render Deployment Guide 🚀

Deploy your Knowledge Base Search Engine to Render with these simple steps.

## 🔧 **Deployment Options**

### Option 1: Full Application (Recommended for Production)
```bash
gunicorn backend.app.main:app -c gunicorn.conf.py
```

### Option 2: Simplified Application (Faster deployment)
```bash
gunicorn simple_main:app --bind 0.0.0.0:$PORT --workers 2 --worker-class uvicorn.workers.UvicornWorker
```

## 📋 **Render Setup Instructions**

### 1. **Create New Web Service on Render**
- Go to [render.com](https://render.com)
- Click "New +" → "Web Service"
- Connect your GitHub repository: `https://github.com/Tanmay081104/Knowledge-based-search-engine`

### 2. **Configure Service Settings**
- **Name**: `knowledge-base-search-engine`
- **Region**: Choose closest to your users
- **Branch**: `main`
- **Root Directory**: Leave empty
- **Runtime**: `Python 3`

### 3. **Build & Deploy Settings**
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: Choose one:
  ```bash
  # Option 1: Full app
  gunicorn backend.app.main:app -c gunicorn.conf.py
  
  # Option 2: Simple app (recommended)
  gunicorn simple_main:app --bind 0.0.0.0:$PORT --workers 2 --worker-class uvicorn.workers.UvicornWorker --timeout 120
  ```

### 4. **Environment Variables**
Add these in Render dashboard:
```env
# Required: At least one LLM API key
GOOGLE_API_KEY=your_google_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GROQ_API_KEY=your_groq_api_key

# Optional
DEFAULT_LLM=google
WORKERS=2
```

### 5. **Advanced Settings**
- **Instance Type**: Starter (free) or higher
- **Auto-Deploy**: Yes (deploys on git push)
- **Health Check Path**: `/health`

## 🚀 **Quick Deploy Commands**

### For Render Dashboard:
**Start Command**:
```bash
gunicorn simple_main:app --bind 0.0.0.0:$PORT --workers 2 --worker-class uvicorn.workers.UvicornWorker --timeout 120
```

**Build Command**:
```bash
pip install -r requirements.txt
```

## 📱 **Post-Deployment**

1. **Test your deployment**:
   - Visit your Render URL
   - Upload a test document
   - Ask a question

2. **Monitor logs**:
   - Check Render dashboard for deployment logs
   - Monitor for any errors or warnings

3. **Custom domain** (optional):
   - Add your custom domain in Render settings

## 🔍 **Troubleshooting**

### Common Issues:
- **Build timeout**: Reduce dependencies or use simpler version
- **Memory issues**: Use fewer workers or upgrade instance
- **API key errors**: Check environment variables are set correctly

### Debug Commands:
```bash
# Test locally first
python simple_main.py

# Test gunicorn locally
gunicorn simple_main:app --bind 0.0.0.0:8000 --workers 1 --worker-class uvicorn.workers.UvicornWorker
```

## ✅ **Ready for Production!**

Your Knowledge Base Search Engine will be live at:
`https://your-app-name.onrender.com`

🎯 **Pro tip**: Start with the simplified version (`simple_main.py`) for faster, more reliable deployments.