# 🏭 AI-Powered Predictive Maintenance & Digital Twin System

An end-to-end multi-modal industrial predictive maintenance system for smart factories, featuring Explainable AI (XAI), RAG grounding, Multi-Agent orchestration, and Digital Twin What-If simulations.

## 📌 System Architecture & Pipeline
1. **Multi-Modal Data Engineering (Stage I):** Tabular telemetry, temporal rolling stats, synthetic component images, operator notes, and technical PDF SOPs.
2. **Predictive ML & Deep Learning (Stage II):** XGBoost Baseline Classifier & PyTorch LSTM Sequence Forecaster logged in MLflow.
3. **Computer Vision & NLP Subsystems (Stage III):** Defect classification and operator log intent extraction.
4. **RAG Grounding Engine (Stage IV):** Vector search over machinery safety SOPs using FAISS and MiniLM embeddings.
5. **Multi-Agent Orchestration (Stage V):** Vision, Predictive Maintenance, Knowledge, and Planning agents collaborating under Pydantic schemas.
6. **Explainable AI (Stage VI):** Native feature importance and vision severity scoring.
7. **Digital Twin Engine (Stage VII):** 12-hour forward horizon What-If state-space scenario simulations.
8. **Human-in-the-Loop Governance & Web App (Stage VIII & IX):** Interactive Streamlit web app with decision audit logging (`hitl_decision_audit_log.json`) and automated ReportLab PDF export.

## 🚀 How to Launch on Streamlit Community Cloud
1. Connect your GitHub repository `uzreshayesha/hackathon` to [share.streamlit.io](https://share.streamlit.io).
2. Set the main file path to `app.py`.
3. Click **Deploy!**
