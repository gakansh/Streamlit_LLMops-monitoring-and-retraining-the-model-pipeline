# 🧠 LLMOps Dashboard – AI Summarization & Monitoring

🚀 **A Streamlit-based AI application** that provides **text summarization, LLM monitoring, and retraining functionalities**. The system integrates with Hugging Face models and supports **automatic retraining** based on user feedback.

---

## 📌 Features
✅ **AI-powered text summarization** using `facebook/bart-large-cnn`  
✅ **Real-time monitoring** (ROUGE score, user ratings, data drift detection)  
✅ **Automatic retraining pipeline** for low-rated summaries  
✅ **Database logging** (SQLite-based interaction tracking)  
✅ **AWS Deployment-ready** with full setup instructions  

---

## ⚡️ Getting Started

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/gakansh/Streamlit_LLMops-monitoring-and-retraining-the-model-pipeline
cd Streamlit_LLMops-monitoring-and-retraining-the-model-pipeline

pip install -r requirements.txt

streamlit run ap.py
✅ The app will be available at http://localhost:8501