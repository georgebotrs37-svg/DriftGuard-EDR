import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="DriftGuard EDR - SOC Triage Advisor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark SOC Cyber Theme)
st.markdown("""
<style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stMetric { background-color: #1e293b; border-radius: 8px; padding: 12px; border: 1px solid #334155; }
    .badge-critical { background-color: #ef4444; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .badge-suspicious { background-color: #f59e0b; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .badge-benign { background-color: #10b981; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_edr_bundle():
    bundle = joblib.load('models/edr_pipeline.joblib')
    return bundle

bundle = load_edr_bundle()
preprocessor = bundle['preprocessor']
model = bundle['model']
threshold = bundle['threshold']

# Sidebar Header
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("DriftGuard EDR")
st.sidebar.caption("Machine Learning Endpoint Detection & Triage")
st.sidebar.markdown(f"**Operational Decision Threshold:** `{threshold:.2f}` *(SOC Alert-Budget Optimized)*")

# Preset Scenarios for Quick Analyst Triage
st.sidebar.markdown("### ⚡ Quick Presets")
scenario = st.sidebar.selectbox(
    "Load Simulated EDR Event:",
    ["Custom Input", "Ransomware Dropper (Phishing)", "Legitimate Admin PowerShell", "Stealthy Living-off-the-Land (LOLBin)", "Normal Office Worker"]
)

# Default values based on scenario
if scenario == "Ransomware Dropper (Phishing)":
    def_proc, def_parent = "powershell.exe", "winword.exe"
    def_cmd, def_depth, def_sys = 320, 5, 0
    def_cpu, def_mem = 45.0, 480.0
    def_net, def_ips = 6, 4
    def_fw, def_fd, def_rw = 85, 20, 18
    def_h, def_elev, def_th = 650, 1, 22
elif scenario == "Legitimate Admin PowerShell":
    def_proc, def_parent = "powershell.exe", "cmd.exe"
    def_cmd, def_depth, def_sys = 145, 3, 1
    def_cpu, def_mem = 15.0, 220.0
    def_net, def_ips = 1, 1
    def_fw, def_fd, def_rw = 12, 1, 4
    def_h, def_elev, def_th = 280, 1, 8
elif scenario == "Stealthy Living-off-the-Land (LOLBin)":
    def_proc, def_parent = "svchost.exe", "explorer.exe"
    def_cmd, def_depth, def_sys = 65, 3, 1
    def_cpu, def_mem = 12.0, 180.0
    def_net, def_ips = 2, 1
    def_fw, def_fd, def_rw = 3, 0, 2
    def_h, def_elev, def_th = 210, 1, 6
elif scenario == "Normal Office Worker":
    def_proc, def_parent = "chrome.exe", "explorer.exe"
    def_cmd, def_depth, def_sys = 40, 3, 1
    def_cpu, def_mem = 5.0, 160.0
    def_net, def_ips = 2, 1
    def_fw, def_fd, def_rw = 1, 0, 1
    def_h, def_elev, def_th = 110, 0, 10
else:
    def_proc, def_parent = "powershell.exe", "explorer.exe"
    def_cmd, def_depth, def_sys = 80, 3, 1
    def_cpu, def_mem = 10.0, 200.0
    def_net, def_ips = 2, 1
    def_fw, def_fd, def_rw = 5, 1, 2
    def_h, def_elev, def_th = 180, 0, 8

# Main Header
st.title("🛡️ EDR Malicious Process Triage Console")
st.markdown("Analyze process telemetry, evaluate attack probability, and dispatch containment directives.")

# Telemetry Input Form
col1, col2, col3 = st.columns(3)

proc_options = ['powershell.exe', 'cmd.exe', 'svchost.exe', 'explorer.exe', 'chrome.exe', 'rundll32.exe', 'certutil.exe', 'teams.exe', 'code.exe']
parent_options = ['explorer.exe', 'services.exe', 'cmd.exe', 'powershell.exe', 'winword.exe', 'excel.exe', 'system', 'svchost.exe']

with col1:
    st.subheader("Process Metadata")
    proc_name = st.selectbox("Process Name:", proc_options, index=proc_options.index(def_proc) if def_proc in proc_options else 0)
    parent_proc = st.selectbox("Parent Process:", parent_options, index=parent_options.index(def_parent) if def_parent in parent_options else 0)
    cmd_len = st.slider("Command Line Length (Chars):", 10, 1500, def_cmd)
    path_depth = st.slider("Directory Path Depth:", 1, 10, def_depth)
    is_sys = st.radio("Standard OS Path (System32):", [1, 0], index=0 if def_sys == 1 else 1, format_func=lambda x: "Yes (System32 / ProgramFiles)" if x == 1 else "No (Temp / AppData / UserPath)")

with col2:
    st.subheader("System & Disk Activity")
    cpu_pct = st.slider("CPU Usage (%):", 0.0, 100.0, float(def_cpu))
    mem_mb = st.slider("Memory Resident (MB):", 10.0, 4000.0, float(def_mem))
    file_writes = st.slider("File Write Operations:", 0, 300, def_fw)
    file_deletes = st.slider("File Delete Operations:", 0, 100, def_fd)
    reg_writes = st.slider("Registry Modifications:", 0, 100, def_rw)

with col3:
    st.subheader("Network & Execution Context")
    net_conns = st.slider("Active Network Connections:", 0, 30, def_net)
    remote_ips = st.slider("Distinct Remote IPs:", 0, 15, def_ips)
    handles = st.slider("Resource Handles Opened:", 10, 2000, def_h)
    is_elevated = st.radio("Elevated / Admin Privilege:", [1, 0], index=0 if def_elev == 1 else 1, format_func=lambda x: "Yes (Elevated / SYSTEM)" if x == 1 else "No (Standard User)")
    threads = st.slider("Active Thread Count:", 1, 64, def_th)

# Inference Preparation
input_df = pd.DataFrame([{
    'process_name': proc_name,
    'parent_process_name': parent_proc,
    'command_line_len': cmd_len,
    'process_path_depth': path_depth,
    'is_system_path': is_sys,
    'cpu_pct': cpu_pct,
    'memory_mb': mem_mb,
    'network_conns': net_conns,
    'remote_ips': remote_ips,
    'file_writes': file_writes,
    'file_deletes': file_deletes,
    'registry_writes': reg_writes,
    'handles_count': handles,
    'is_elevated': is_elevated,
    'thread_count': threads
}])

# Execute Model Prediction
X_proc = preprocessor.transform(input_df)
malicious_prob = float(model.predict_proba(X_proc)[0, 1])

st.markdown("---")
st.subheader("📊 SOC Triage Verdict & Containment Directives")

res_col1, res_col2 = st.columns([1, 2])

with res_col1:
    st.metric(
        label="Malicious Attack Probability",
        value=f"{malicious_prob:.1%}",
        delta=f"{(malicious_prob - threshold):+.1%} vs Threshold ({threshold:.2f})",
        delta_color="inverse"
    )
    
    if malicious_prob >= 0.70:
        st.markdown("<div class='badge-critical' style='text-align:center; padding:10px;'>🚨 CRITICAL SEVERITY (P1)</div>", unsafe_allow_html=True)
    elif malicious_prob >= threshold:
        st.markdown("<div class='badge-suspicious' style='text-align:center; padding:10px;'>⚠️ ELEVATED SUSPICION (P2)</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='badge-benign' style='text-align:center; padding:10px;'>✅ BENIGN / PERMITTED (P3)</div>", unsafe_allow_html=True)

with res_col2:
    if malicious_prob >= 0.70:
        st.error("### 🛑 Automated SOC Recommendation: ISOLATE ENDPOINT & KILL PROCESS")
        st.markdown("""
        * **1. Immediate Host Isolation:** Sever endpoint network access to prevent lateral movement.
        * **2. Terminate Process Tree:** Kill target PID and all spawned children.
        * **3. Memory Dump Collection:** Dump volatile memory for offline forensic analysis.
        * **4. Incident Escalation:** Page on-call Incident Response (IR) team immediately.
        """)
    elif malicious_prob >= threshold:
        st.warning("### ⚠️ Automated SOC Recommendation: RESTRICT & ASSIGN ANALYST REVIEW")
        st.markdown("""
        * **1. Process Sandboxing:** Freeze thread execution or throttle outbound network bandwidth.
        * **2. Triage Assignment:** Route event to Tier-1 SOC queue with 30-minute SLA.
        * **3. Context Verification:** Verify if host user is an active IT admin running authorized scripts.
        """)
    else:
        st.success("### ✅ Automated SOC Recommendation: ALLOW EXECUTION & LOG TELEMETRY")
        st.markdown("""
        * **1. Permit Execution:** Process telemetry conforms to benign baseline parameters.
        * **2. Continuous Monitoring:** Telemetry archived in central SIEM for 90-day retention.
        """)

# Risk Factors Attribution
st.markdown("---")
st.subheader("🔍 Top Risk Factors Contributing to This Score")
risk_factors = []
if file_writes > 30:
    risk_factors.append(f"High file write rate ({file_writes} writes) indicates potential ransomware or file dropper.")
if parent_proc in ['winword.exe', 'excel.exe']:
    risk_factors.append(f"Abnormal parent process lineage: `{proc_name}` was spawned by `{parent_proc}` (Office Macro Dropper).")
if is_sys == 0:
    risk_factors.append("Process executable is running outside official OS directories (Temp/AppData execution).")
if cmd_len > 150:
    risk_factors.append(f"Unusually long command line ({cmd_len} characters) commonly seen in Base64 obfuscation.")
if remote_ips >= 3:
    risk_factors.append(f"Multiple outbound IP connections ({remote_ips} IPs) indicates C2 communication or data exfiltration.")
if is_elevated == 1 and proc_name in ['powershell.exe', 'cmd.exe']:
    risk_factors.append("Interactive administrative shell running with elevated SYSTEM permissions.")

if risk_factors:
    for rf in risk_factors:
        st.markdown(f"- 🔴 **Risk Indicator:** {rf}")
else:
    st.markdown("- 🟢 **Safe:** No abnormal telemetry thresholds exceeded.")
