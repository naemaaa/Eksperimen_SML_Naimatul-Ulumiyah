# 📊 Grafana Monitoring Setup - Naimatul Ulumiyah
## Kriteria 4: Monitoring dan Logging (10 pts advance)

---

## 🎯 Objective
Setup complete monitoring stack: Prometheus → Grafana dengan 10 metrics, 3 alerts, inference testing.

---

## 📋 Prerequisites

```bash
# Install dependencies lokal
cd ~
pip install prometheus-client requests psutil numpy pyyaml

# Download & Install Prometheus (https://prometheus.io/download)
# Mac: brew install prometheus
# Ubuntu: wget + tar
# Windows: Download binary

# Download & Install Grafana (https://grafana.com/download)
# Mac: brew install grafana
# Ubuntu: sudo apt-get install grafana-server
# Windows: Download installer

# Verify installation
prometheus --version
grafana-server --version
```

---

## 🚀 Step-by-Step Setup

### **Step 1: Start Model Serving (Lokal)**

Gunakan MLflow untuk serve model yang udah training dari workflow:

```bash
# Get latest run_id dari workflow artifacts atau MLflow UI
export RUN_ID="1234567890abcd"  # Replace with actual

# Serve model pada port 5000
mlflow models serve -m runs:/$RUN_ID/model -p 5000 --no-conda

# atau dengan Docker (jika build sudah ada)
docker run -p 5000:8080 your_username/sepsis-model:latest
```

📸 **Screenshot setelah model serving siap:**
- Terminal menunjukkan "Listening on http://0.0.0.0:5000"
- Simpan sebagai: `1.bukti_serving.png`

---

### **Step 2: Start Prometheus Server**

```bash
# Di terminal baru
cd ~/SMSML_Naimatul-Ulumiyah/Eksperimen_SML_Naimatul-Ulumiyah/Monitoring_dan_Logging

# Run Prometheus
prometheus --config.file=prometheus.yml

# Verify: buka http://localhost:9090 (UI) → Grafana bisa query ini
```

📸 **Screenshot:**
- Prometheus UI running → Status > Targets (Target bagus)
- Simpan sebagai: `monitoring_prometheus_targets.png`

---

### **Step 3: Start Prometheus Exporter (Metrics Generator)**

```bash
# Di terminal baru (ketiga)
cd ~/SMSML_Naimatul-Ulumiyah/Eksperimen_SML_Naimatul-Ulumiyah/Monitoring_dan_Logging

# Run exporter (generate 10 metrics)
python prometheus_exporter.py

# Output harus:
# ✅ Prometheus exporter started on http://localhost:8000/metrics
# [HH:MM:SS] Metrics updated | Requests: X | CPU: Y% | Accuracy: Z
```

📸 **Screenshot:**
- Terminal menunjukkan exporter running
- Simpan sebagai: `monitoring_exporter_running.png`

---

### **Step 4: Start Grafana Server**

```bash
# Di terminal baru (keempat)

# Mac
grafana-server web

# Ubuntu/Linux
sudo systemctl start grafana-server

# Windows
Start-Service "Grafana"
```

📸 **Screenshot:**
- Terminal: "Grafana server started" atau http://0.0.0.0:3000
- Login: admin / admin (default)
- Simpan sebagai: `monitoring_grafana_startup.png`

---

### **Step 5: Configure Grafana Data Source (Prometheus)**

1. Buka **http://localhost:3000** → Login (admin/admin)
2. **Change password** pada first login (recommended)
3. Pergi ke **Connections** → **Data Sources** → **Add Data Source**
4. Pilih **Prometheus**
5. Isi URL: `http://localhost:9090`
6. Klik **Save & Test**
   - Harus muncul: ✅ "Data source is working"

📸 **Screenshot:**
- "Data source is working" confirmation
- Simpan sebagai: `monitoring_grafana_datasource_ok.png`

---

### **Step 6: Create Dashboard dengan 10 Metrics**

1. **Dashboards** → **New Dashboard**
2. **Dashboard Settings** (gear icon) → Name: `Dashboard-Naimatul-Ulumiyah`
3. **Add Panel** untuk setiap metric (10 total):

#### **Metric 1: Requests Total**
- **Panel Type:** Time Series
- **Metrics:** `rate(ml_requests_total[5m])`
- **Title:** "Prediction Requests (per minute)"
- **Y-axis:** Request/min
- **Color:** Blue

#### **Metric 2: Request Latency**
- **Panel Type:** Heatmap
- **Metrics:** `histogram_quantile(0.95, rate(ml_request_latency_seconds_bucket[5m]))`
- **Title:** "Request Latency P95 (seconds)"
- **Color:** Orange
- **Threshold (Alert):** >0.5s = Red, <0.1s = Green

#### **Metric 3: Errors**
- **Panel Type:** Stat
- **Metrics:** `increase(ml_errors_total[5m])`
- **Title:** "Errors (5 min window)"
- **Format:** Number
- **Color Threshold:** >5 = Red, <2 = Green

#### **Metric 4: CPU Usage**
- **Panel Type:** Gauge (dengan threshold)
- **Metrics:** `ml_cpu_usage_percent{model="xgboost_sepsis"}`
- **Title:** "CPU Usage (%)"
- **Min:** 0, Max: 100
- **Threshold:** Green 0-50, Yellow 50-80, Red 80-100

#### **Metric 5: Memory Usage**
- **Panel Type:** Bar Gauge
- **Metrics:** `ml_memory_usage_mb{model="xgboost_sepsis"}`
- **Title:** "Memory Usage (MB)"
- **Max Value:** 1000
- **Color:** Green Green Green

#### **Metric 6: Model Accuracy**
- **Panel Type:** Gauge
- **Metrics:** `ml_accuracy{model="xgboost_sepsis"}`
- **Title:** "Model Accuracy (ROC AUC)"
- **Min:** 0, Max: 1
- **Decimals:** 4
- **Threshold:** Green 0.85-1, Yellow 0.75-0.85, Red 0-0.75

#### **Metric 7: Data Drift Score**
- **Panel Type:** Time Series
- **Metrics:** `ml_drift_score{model="xgboost_sepsis"}`
- **Title:** "Data Drift Score (0=none, 1=max)"
- **Y-axis:** 0-1
- **Alert Line:** >0.1 (warning)

#### **Metric 8: Model Version**
- **Panel Type:** Stat
- **Metrics:** `ml_version{model="xgboost_sepsis"}`
- **Title:** "Model Version"
- **Format:** Decimal (0 decimals)

#### **Metric 9: Request Rate (Summary)**
- **Panel Type:** Time Series
- **Metrics:** `ml_request_rate{model="xgboost_sepsis"}`
- **Title:** "Request Rate (req/sec)"
- **Color:** Purple

#### **Metric 10: Uptime**
- **Panel Type:** Stat (atau Time Series)
- **Metrics:** `ml_uptime_seconds{model="xgboost_sepsis"} / 3600`  (convert ke hours)
- **Title:** "Service Uptime (hours)"
- **Format:** Show decimal (1 place)

📸 **Screenshots untuk masing-masing metric:**
- Setelah add setiap metric, screenshot panel-nya
- Nama file: `monitoring_grafana_1_requests.png`, `monitoring_grafana_2_latency.png`, ..., `monitoring_grafana_10_uptime.png`
- **Total: 10 screenshot**
- **Plus 1 screenshot full dashboard:** `monitoring_grafana_full_dashboard.png`

---

### **Step 7: Create Alerting Rules (3 Rules)**

#### **Alert Rule 1: High CPU**

1. Di dashboard, edit panel CPU
2. **Alert** tab → **Create Alert**
   - **Rule Name:** `HighCPUUsage`
   - **Evaluate:** every 1m for 5m
   - **Condition:** `ml_cpu_usage_percent > 80`
3. **Notification Channel:** 
   - Add Email / Slack (optional)
   - Atau simple test dengan Grafana UI
4. **Save**

📸 **Screenshot:**
- Rule configuration
- Simpan sebagai: `monitoring_alert_1_high_cpu_rule.png`

#### **Alert Rule 2: High Latency**

1. Edit panel Latency
2. **Alert** → **Create Alert**
   - **Rule Name:** `HighLatency`
   - **Condition:** `histogram_quantile(0.95, ...) > 0.5`
   - **Duration:** 5m

📸 **Screenshot:**
- Simpan sebagai: `monitoring_alert_2_high_latency_rule.png`

#### **Alert Rule 3: High Error Rate**

1. Edit panel Errors
2. **Alert** → **Create Alert**
   - **Rule Name:** `HighErrorRate`
   - **Condition:** `rate(ml_errors_total[5m]) > 0.1`  (>6 err/min)
   - **Duration:** 5m

📸 **Screenshot:**
- Simpan sebagai: `monitoring_alert_3_high_errors_rule.png`

---

### **Step 8: Trigger Alerts (Test)**

Untuk memicu alert, modify exporter.py temporarily:

```python
# Di prometheus_exporter.py, dalam function update_metrics()
# temporarily set values ke trigger alert:

# For CPU alert:
cpu_usage.labels(model=model_name).set(85)  # >80 trigger

# For Latency alert:
latency = 0.6  # >0.5 trigger

# For Errors:
error_count.labels(model=model_name).inc(2)  # Increase rate >0.1
```

Tunggu 5 menit → Alert harus muncul di Grafana **Alerting** → **Alert Rules**

📸 **Screenshots Notifikasi Alert:**
1. **Alert firing di Grafana UI**
   - Simpan sebagai: `monitoring_alert_fired_cpu.png`
2. **Notification** (jika email/Slack dikonfigurasi)
   - Simpan sebagai: `monitoring_alert_notification_email.png`
3. **Alert History**
   - Simpan sebagai: `monitoring_alert_history.png`

---

### **Step 9: Generate Load dengan Inference Script**

Run inference untuk generate real metrics:

```bash
# Di terminal baru (kelima)
cd ~/SMSML_Naimatul-Ulumiyah/Eksperimen_SML_Naimatul-Ulumiyah/Monitoring_dan_Logging

python inference.py
```

Ini akan send 50 requests ke model → metrics increment → Grafana akan show real data

📸 **Screenshot:**
- Inference running + loads hitting model
- Simpan sebagai: `monitoring_inference_running.png`

---

## 📁 Final Folder Structure untuk Submission

```
Monitoring_dan_Logging/
├── 1.bukti_serving.png
├── prometheus.yml
├── prometheus_exporter.py
├── inference.py
├── README_Monitoring.md (file ini)
├── bukti_monitoring_prometheus/
│   ├── 1.monitoring_prometheus_targets.png
│   ├── 2.monitoring_prometheus_query_requests.png
│   ├── 3.monitoring_prometheus_query_latency.png
│   ├── 4.monitoring_prometheus_query_errors.png
│   ├── 5.monitoring_prometheus_query_cpu.png
│   ├── 6.monitoring_prometheus_query_memory.png
│   ├── 7.monitoring_prometheus_query_accuracy.png
│   ├── 8.monitoring_prometheus_query_drift.png
│   ├── 9.monitoring_prometheus_query_version.png
│   └── 10.monitoring_prometheus_query_uptime.png
├── bukti_monitoring_grafana/
│   ├── monitoring_grafana_full_dashboard.png
│   ├── 1.monitoring_grafana_requests.png
│   ├── 2.monitoring_grafana_latency.png
│   ├── 3.monitoring_grafana_errors.png
│   ├── 4.monitoring_grafana_cpu.png
│   ├── 5.monitoring_grafana_memory.png
│   ├── 6.monitoring_grafana_accuracy.png
│   ├── 7.monitoring_grafana_drift.png
│   ├── 8.monitoring_grafana_version.png
│   ├── 9.monitoring_grafana_request_rate.png
│   └── 10.monitoring_grafana_uptime.png
└── bukti_alerting_grafana/
    ├── 1.monitoring_alert_high_cpu_rule.png
    ├── 2.monitoring_alert_high_latency_rule.png
    ├── 3.monitoring_alert_high_errors_rule.png
    ├── 4.monitoring_alert_fired_cpu.png
    ├── 5.monitoring_alert_notification.png
    └── 6.monitoring_alert_history.png
```

**Total screenshots: 26**

---

## ✅ Checklist untuk Submission

- [ ] Prometheus installed & running
- [ ] Grafana installed & running
- [ ] Data Source (Prometheus) connected
- [ ] Dashboard created dengan 10 metrics
- [ ] 10 metric panels configured & displaying data
- [ ] 3 Alert rules created & tested
- [ ] Alerts fired & notifications sent
- [ ] All screenshot evidence collected
- [ ] Folder structure matches Dicoding requirement
- [ ] inference.py tested & generated load
- [ ] All files committed to GitHub & visible

---

## 🎓 Criteria Points Achieved

| Criteria | Status | Points |
|----------|--------|--------|
| 1. Data Preprocessing (automated) | ✅ | 5 |
| 2. Model Training (MLflow) | ✅ | 50 |
| 3. CI/CD (GitHub Actions) | ✅ | 35 |
| 4. Monitoring & Logging | ✅ | 10 |
| **TOTAL** | | **100** |

---

## 🔧 Troubleshooting

### **Prometheus tidak connect ke exporter**
```
Error: "Get ... connection refused"
→ Pastikan exporter running di port 8000
→ Check prometheus.yml: targets = ['localhost:8000']
```

### **Grafana tidak bisa query metrics**
```
→ Check Data Source: http://localhost:9090 accessible
→ Run test: "Save & Test" harus hijau
→ Tunggu 1 menit pertama data masuk
```

### **Alert tidak firing**
```
→ Check rule condition syntax di Prometheus UI
→ Simulate dengan modify exporter.py values
→ Tunggu evaluasi interval (default 5m untuk testing)
```

---

## 📞 Quick Start Commands

```bash
# Terminal 1: Model Serving
mlflow models serve -m runs:/<RUN_ID>/model -p 5000 --no-conda

# Terminal 2: Prometheus
prometheus --config.file=~/SMSML_Naimatul-Ulumiyah/Eksperimen_SML_Naimatul-Ulumiyah/Monitoring_dan_Logging/prometheus.yml

# Terminal 3: Exporter
cd ~/SMSML_Naimatul-Ulumiyah/Eksperimen_SML_Naimatul-Ulumiyah/Monitoring_dan_Logging && python prometheus_exporter.py

# Terminal 4: Grafana
grafana-server web

# Terminal 5: Inference Load
cd ~/SMSML_Naimatul-Ulumiyah/Eksperimen_SML_Naimatul-Ulumiyah/Monitoring_dan_Logging && python inference.py
```

---

**Good luck! Semoga workflow hijau dan monitoring setup sempurna! 🚀**
