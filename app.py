"""
Factory AI Intelligence & Digital Twin Streamlit Web Application
Fulfills Rubric Section 15 (Minimum Final Demonstration) and Section 16 (Deliverable 1).
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Import pipeline elements
from cnc_pipeline import run_multi_agent_pipeline, generate_and_preprocess_factory_data, train_xgboost_model

# Page Configuration
st.set_page_config(
    page_title="Factory AI Digital Twin Platform",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 AI-Powered Predictive Maintenance & Digital Twin System")
st.caption("Integrated Multi-Modal Industrial Intelligence Dashboard | Smart Factory Operations")

# Initialize Data and Model
@st.cache_data
def load_initial_data():
    return generate_and_preprocess_factory_data()

df_telemetry = load_initial_data()

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("🕹️ Factory Control Center")
machine_id = st.sidebar.selectbox("Select Target Equipment", ["CNC_MILL_01", "CNC_MILL_02", "PRESS_LINE_01"])

st.sidebar.subheader("Live Telemetry Overrides")
temperature = st.sidebar.slider("Spindle Temperature (°C)", 50.0, 110.0, 88.5, 0.5)
vibration = st.sidebar.slider("Vibration RMS (mm/s)", 0.5, 6.0, 3.8, 0.1)
pressure = st.sidebar.slider("Hydraulic Pressure (PSI)", 70.0, 120.0, 95.0, 1.0)
rpm = st.sidebar.slider("Spindle Speed (RPM)", 1000, 2500, 2100, 50)

operator_note = st.sidebar.text_area("Operator Log Note", "High thermal readings and elevated chatter during shift 2.")
supervisor_id = st.sidebar.text_input("Supervisor ID Badge", "ENG_SUPERVISOR_01")

# --- MAIN DASHBOARD TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Telemetry & Agents", "🔮 Digital Twin What-If", "🛡️ Human-in-the-Loop Governance", "📄 PDF Incident Exporter"])

# Execute Multi-Agent Workflow
vision_res, pred_res, know_res, exec_plan = run_multi_agent_pipeline(machine_id, temperature, vibration, operator_note)

with tab1:
    st.subheader("🤖 Multi-Agent Orchestration Engine")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Failure Probability", f"{pred_res.failure_probability * 100:.1f}%", delta=pred_res.risk_level)
    col2.metric("Vision Defect Status", "CRITICAL" if vision_res.defect_detected else "NORMAL", delta_color="inverse")
    col3.metric("Grounding Document", know_res.relevant_doc_id)
    col4.metric("Composite Risk", f"{exec_plan.final_risk_score:.2f}")

    st.markdown("---")
    st.write("### 👥 Contributing Subsystem Agents")
    
    agent_col1, agent_col2 = st.columns(2)
    with agent_col1:
        st.info(f"**Vision Agent:** Defect Detected: `{vision_res.defect_detected}` | Severity: `{vision_res.severity}`")
        st.warning(f"**Predictive Maintenance Agent:** Machine `{pred_res.machine_id}` | Risk Level: `{pred_res.risk_level}`")
    with agent_col2:
        st.success(f"**Knowledge Agent:** Grounded SOP: `{know_res.relevant_doc_id}` - `{know_res.section}`")
        st.error(f"**Executive Planning Agent:** Action: `{exec_plan.recommended_action}`")

    st.markdown("---")
    st.subheader("📜 RAG Grounded Manual SOP Passage")
    st.blockquote(know_res.grounded_procedure)

with tab2:
    st.subheader("🔮 Digital Twin Scenario Simulation Engine")
    st.write("Simulate operational parameters across a 12-hour future window:")
    
    scenario = st.selectbox("Select Operational Simulation Strategy", [
        "Scenario A: Continue Current Speed (Baseline Hazard)",
        "Scenario B: Derate RPM by 30% & Engage Cooling Loop",
        "Scenario C: Immediate Controlled Emergency Halt"
    ])
    
    if scenario == "Scenario A: Continue Current Speed (Baseline Hazard)":
        sim_temp = [temperature + (i * 1.5) for i in range(12)]
        est_cost = "$45,000 (Catastrophic Bearing Failure)"
    elif scenario == "Scenario B: Derate RPM by 30% & Engage Cooling Loop":
        sim_temp = [max(65.0, temperature - (i * 1.2)) for i in range(12)]
        est_cost = "$2,500 (Planned Off-Peak Repair)"
    else:
        sim_temp = [temperature * (0.8 ** i) for i in range(12)]
        est_cost = "$12,000 (Immediate Unplanned Downtime)"

    sim_df = pd.DataFrame({"Operating Hour": list(range(1, 13)), "Simulated Temperature (°C)": sim_temp})
    st.line_chart(sim_df.set_index("Operating Hour"))
    st.write(f"**Projected Downtime & Repair Cost Impact:** {est_cost}")

with tab3:
    st.subheader("🛡️ Human-in-the-Loop Governance & Decision Verification")
    st.write("Review AI action proposal and authorize final factory control decision:")
    
    st.warning(f"**AI Proposed Action:** {exec_plan.recommended_action}")
    
    decision = st.radio("Supervisor Final Decision", ["APPROVE", "REJECT", "MODIFY"])
    custom_override = st.text_input("Supervisor Overriding Directive / Reasoning", "Approved derating motor load per SOP_CNC_001.")
    
    if st.button("Submit Decision & Update Audit Log"):
        audit_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "machine_id": machine_id,
            "supervisor_id": supervisor_id,
            "supervisor_decision": decision,
            "override_note": custom_override,
            "failure_probability": pred_res.failure_probability,
            "composite_risk": exec_plan.final_risk_score
        }
        
        log_file = "factory_data/hitl_decision_audit_log.json"
        existing_logs = []
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                try: existing_logs = json.load(f)
                except: existing_logs = []
        
        existing_logs.append(audit_entry)
        with open(log_file, "w") as f:
            json.dump(existing_logs, f, indent=4)
            
        st.success(f"✅ Decision logged successfully into persistent audit ledger! Badge: {supervisor_id}")

with tab4:
    st.subheader("📄 Generate Official Incident PDF Compliance Report")
    st.write("Export a clean, single-page compliance PDF document for safety auditors and plant operations.")
    
    def generate_pdf_report():
        pdf_path = "factory_incident_report.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, spaceAfter=12, textColor=colors.HexColor("#1A365D"))
        story.append(Paragraph("FACTORY AI INCIDENT & DECISION REPORT", title_style))
        story.append(Spacer(1, 10))
        
        table_data = [
            ["Attribute", "Value"],
            ["Timestamp", time.strftime("%Y-%m-%d %H:%M:%S")],
            ["Machine ID", machine_id],
            ["Supervisor ID", supervisor_id],
            ["AI Risk Level", pred_res.risk_level],
            ["Grounding Document", know_res.relevant_doc_id],
            ["Action Recommendation", exec_plan.recommended_action]
        ]
        
        t = Table(table_data, colWidths=[180, 320])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (1,0), colors.HexColor("#2B6CB0")),
            ('TEXTCOLOR', (0,0), (1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        doc.build(story)
        return pdf_path

    if st.button("Generate Incident PDF Report"):
        pdf_file = generate_pdf_report()
        with open(pdf_file, "rb") as f:
            st.download_button(
                label="📥 Download Official PDF Report",
                data=f,
                file_name=f"Incident_Report_{machine_id}.pdf",
                mime="application/pdf"
            )