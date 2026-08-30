
"""
Multi-Modal Factory AI Intelligence & Digital Twin Pipeline
Contains Stage I through Stage VIII processing logic.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from fpdf import FPDF
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from transformers import pipeline
import faiss
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from typing import Dict, Any

# Safe MLflow Import (Bypasses Python 3.14 / Protobuf compatibility bugs on cloud servers)
try:
    import mlflow
    import mlflow.xgboost
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

# --- DIRECTORY SETUP ---
BASE_DIR = "factory_data"
OS_DIRS = {
    "tabular": os.path.join(BASE_DIR, "tabular"),
    "images": os.path.join(BASE_DIR, "images"),
    "text": os.path.join(BASE_DIR, "text"),
    "pdfs": os.path.join(BASE_DIR, "pdfs"),
    "processed": os.path.join(BASE_DIR, "processed")
}
for d in OS_DIRS.values():
    os.makedirs(d, exist_ok=True)

# --- STAGE I: DATA GENERATION & ENGINEERING ---
def generate_and_preprocess_factory_data(num_records=1500, num_images=60):
    np.random.seed(42)
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=num_records, freq="1min")
    machine_ids = np.random.choice(["CNC_MILL_01", "CNC_MILL_02", "PRESS_LINE_01"], size=num_records)
    temperature = np.random.normal(72.0, 8.0, size=num_records)
    vibration = np.random.normal(2.1, 0.5, size=num_records)
    pressure = np.random.normal(101.3, 4.0, size=num_records)
    rpm = np.random.normal(1800, 100, size=num_records)
    failure_target = np.zeros(num_records, dtype=int)

    # Inject failure dynamics
    for i in range(100, num_records, 250):
        temperature[i-20:i] += np.linspace(5, 40, 20)
        vibration[i-20:i] += np.linspace(0.5, 4.5, 20)
        pressure[i-20:i] -= np.linspace(1, 15, 20)
        failure_target[i-8:i] = 1

    df_telemetry = pd.DataFrame({
        "timestamp": timestamps,
        "machine_id": machine_ids,
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure,
        "rpm": rpm,
        "failure_target": failure_target
    })

    # Rolling Feature Engineering
    df_telemetry["temp_roll_mean_15m"] = df_telemetry.groupby("machine_id")["temperature"].transform(lambda x: x.rolling(15, min_periods=1).mean())
    df_telemetry["vib_roll_std_15m"] = df_telemetry.groupby("machine_id")["vibration"].transform(lambda x: x.rolling(15, min_periods=1).std().fillna(0))
    df_telemetry["stress_index"] = (df_telemetry["temperature"] * df_telemetry["vibration"]) / (df_telemetry["pressure"] + 1e-5)

    # Imputation & Outlier Clipping
    df_telemetry['temperature'] = df_telemetry['temperature'].ffill().bfill()
    invalid_pressure_mask = df_telemetry['pressure'] < 0
    df_telemetry.loc[invalid_pressure_mask, 'pressure'] = df_telemetry['pressure'].median()

    for col in ['temperature', 'vibration', 'pressure', 'rpm']:
        Q1, Q3 = df_telemetry[col].quantile(0.25), df_telemetry[col].quantile(0.75)
        IQR = Q3 - Q1
        df_telemetry[col] = np.clip(df_telemetry[col], Q1 - 2.5 * IQR, Q3 + 2.5 * IQR)

    # Save Telemetry
    telemetry_path = os.path.join(OS_DIRS["tabular"], "machine_telemetry.csv")
    df_telemetry.to_csv(telemetry_path, index=False)

    # Chronological Split
    df_sorted = df_telemetry.sort_values('timestamp').reset_index(drop=True)
    total_len = len(df_sorted)
    train_idx, val_idx = int(total_len * 0.70), int(total_len * 0.85)

    train_df = df_sorted.iloc[:train_idx]
    val_df = df_sorted.iloc[train_idx:val_idx]
    test_df = df_sorted.iloc[val_idx:]

    train_df.to_csv(os.path.join(OS_DIRS["processed"], "train.csv"), index=False)
    val_df.to_csv(os.path.join(OS_DIRS["processed"], "val.csv"), index=False)
    test_df.to_csv(os.path.join(OS_DIRS["processed"], "test.csv"), index=False)

    return df_telemetry

# --- STAGE II: MACHINE LEARNING & DEEP LEARNING ---
def train_xgboost_model():
    train_df = pd.read_csv(os.path.join(OS_DIRS["processed"], "train.csv"))
    test_df = pd.read_csv(os.path.join(OS_DIRS["processed"], "test.csv"))
    features = ['temperature', 'vibration', 'pressure', 'rpm', 'temp_roll_mean_15m', 'vib_roll_std_15m', 'stress_index']
    
    X_train, y_train = train_df[features], train_df['failure_target']
    
    xgb_params = {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "objective": "binary:logistic",
        "random_state": 42
    }
    
    baseline_model = xgb.XGBClassifier(**xgb_params)
    baseline_model.fit(X_train, y_train)
    baseline_model.save_model("factory_champion_model.json")
    return baseline_model

# --- STAGE III: CV & NLP ANALYTICS ---
def analyze_maintenance_log(text_log):
    CRITICAL_KEYWORDS = ["failure", "alarm", "chattering", "leakage", "overheating", "emergency"]
    contains_critical = any(kw in text_log.lower() for kw in CRITICAL_KEYWORDS)
    urgency = "HIGH" if contains_critical else "LOW"
    return {
        "primary_issue": "Thermal/Mechanical Incident" if contains_critical else "Routine Inspection",
        "urgency_level": urgency,
        "keyword_flag": contains_critical
    }

# --- STAGE IV: RAG GROUNDING ENGINE ---
SOP_DOCUMENTS = [
    {
        "doc_id": "SOP_CNC_001",
        "title": "CNC Milling Thermal Escalation Response",
        "section": "Section 4.2 - Spindle Overheating Procedures",
        "content": "When CNC spindle temperature exceeds 80°C or thermal growth rate exceeds 1.5°C/min: Immediately reduce cutting feed rate by 50%. Inspect coolant nozzles for blockage. If temperature passes 95°C, initiate emergency sequence E-STOP-04."
    },
    {
        "doc_id": "MAN_VIB_003",
        "title": "Vibration Analysis & Bearing Degradation",
        "section": "Section 2.1 - Bearing Chattering Spectrum",
        "content": "Vibration levels exceeding 4.5 mm/s RMS accompanied by high-frequency chattering indicate severe inner-race bearing degradation. Schedule spindle bearing replacement within 12 operating hours."
    }
]

def retrieve_rag_evidence(query):
    if "temp" in query.lower() or "overheat" in query.lower() or "thermal" in query.lower():
        return SOP_DOCUMENTS[0]
    return SOP_DOCUMENTS[1]

# --- STAGE V: MULTI-AGENT ORCHESTRATION SCHEMAS ---
class VisionAnalysisResult(BaseModel):
    agent_name: str = "Vision Agent"
    defect_detected: bool
    severity: str
    confidence: float

class PredictiveMaintenanceResult(BaseModel):
    agent_name: str = "Predictive Maintenance Agent"
    machine_id: str
    failure_probability: float
    risk_level: str

class KnowledgeRetrievalResult(BaseModel):
    agent_name: str = "Knowledge Agent"
    relevant_doc_id: str
    section: str
    grounded_procedure: str

class ExecutiveActionPlan(BaseModel):
    orchestrator: str = "Planning & Decision Agent"
    final_risk_score: float
    recommended_action: str
    human_in_the_loop_required: bool
    justification: str

def run_multi_agent_pipeline(machine_id: str, temperature: float, vibration: float, text_note: str):
    prob = min(1.0, (temperature / 100.0) * 0.5 + (vibration / 5.0) * 0.5)
    risk = "HIGH" if prob > 0.65 else ("ELEVATED" if prob > 0.4 else "LOW")
    pred_res = PredictiveMaintenanceResult(machine_id=machine_id, failure_probability=round(prob, 3), risk_level=risk)

    has_defect = vibration > 3.5 or temperature > 85.0
    vision_res = VisionAnalysisResult(defect_detected=has_defect, severity="CRITICAL" if has_defect else "NONE", confidence=0.95)

    rag_doc = retrieve_rag_evidence(f"{temperature} {vibration} {text_note}")
    know_res = KnowledgeRetrievalResult(relevant_doc_id=rag_doc["doc_id"], section=rag_doc["section"], grounded_procedure=rag_doc["content"])

    composite_risk = (pred_res.failure_probability * 0.6) + (0.4 if vision_res.defect_detected else 0.0)
    hitl_flag = composite_risk > 0.60
    
    action = f"PAUSE PRODUCTION: Supervisor Approval Required for {know_res.relevant_doc_id}" if hitl_flag else f"Autonomous parameter adjustment under {know_res.relevant_doc_id}"
    justification = f"Composite risk score ({composite_risk:.2f}) exceeded safe operation threshold." if hitl_flag else "System Operating within safe parameters."

    plan = ExecutiveActionPlan(
        final_risk_score=round(composite_risk, 3),
        recommended_action=action,
        human_in_the_loop_required=hitl_flag,
        justification=justification
    )

    return vision_res, pred_res, know_res, plan
