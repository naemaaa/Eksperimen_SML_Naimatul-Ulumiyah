# 📊 Grafana Dashboard Quick Reference
## Minimal Setup untuk Kriteria 4 Dicoding

---

## ⚡ 5-Minute Quick Setup

### **1. Start Required Services (4 terminals)**

**Terminal 1 - Model Serving:**
```bash
# Dapatkan run_id dari: GitHub Actions artifacts atau MLflow UI
export RUN_ID="your_run_id_here"
mlflow models serve -m runs:/$RUN_ID/model -p 5000 --no-conda
```

**Terminal 2 - Prometheus:**
```bash
cd ~/SMSML_Naimatul-Ulumiyah/Eksperimen_SML_Naimatul-Ulumiyah/Monitoring_dan_Logging
prometheus --config.file=prometheus.yml
# Akses: http://localhost:9090
```

**Terminal 3 - Exporter (Metrics Generator):**
```bash
cd ~/SMSML_Naimatul-Ulumiyah/Eksperimen_SML_Naimatul-Ulumiyah/Monitoring_dan_Logging
python prometheus_exporter.py
```

**Terminal 4 - Grafana:**
```bash
# Mac
grafana-server web

# Linux
sudo systemctl start grafana-server

# Windows
net start Grafana
```

### **2. Connect Prometheus to Grafana (2 min)**

1. Open `http://localhost:3000`
2. Login: `admin` / `admin` (default password)
3. Change password (required on first login)
4. **Connections** → **Data Sources** → **Add Data Source**
5. Select **Prometheus**
6. URL: `http://localhost:9090`
7. Click **Save & Test** (should show ✅ "Data source is working")

### **3. Create Dashboard (5 min)**

1. **Dashboards** → **New Dashboard**
2. **Dashboard Settings** (gear icon):
   - Name: `Dashboard-Naimatul-Ulumiyah`
   - Save
3. **Add Panel** and copy these metric queries:

---

## 📈 10 Metrics - Copy/Paste Queries

**Metric 1: Requests (Rate)**
```
rate(ml_requests_total[5m])
```
Panel Type: Time Series

---

**Metric 2: Latency (P95)**
```
histogram_quantile(0.95, sum(rate(ml_request_latency_seconds_bucket[5m])) by (le))
```
Panel Type: Heatmap / Graph

---

**Metric 3: Errors**
```
increase(ml_errors_total[5m])
```
Panel Type: Stat (number)

---

**Metric 4: CPU**
```
ml_cpu_usage_percent{model="xgboost_sepsis"}
```
Panel Type: Gauge
- Threshold: 0 (green) - 50 (yellow) - 80 (red) - 100

---

**Metric 5: Memory**
```
ml_memory_usage_mb{model="xgboost_sepsis"}
```
Panel Type: Bar Gauge
- Max: 1000

---

**Metric 6: Accuracy**
```
ml_accuracy{model="xgboost_sepsis"}
```
Panel Type: Gauge
- Min: 0, Max: 1

---

**Metric 7: Drift Score**
```
ml_drift_score{model="xgboost_sepsis"}
```
Panel Type: Time Series
- Y-axis: 0-1

---

**Metric 8: Version**
```
ml_version{model="xgboost_sepsis"}
```
Panel Type: Stat (number)
- Decimals: 0

---

**Metric 9: Request Rate**
```
ml_request_rate{model="xgboost_sepsis"}
```
Panel Type: Time Series

---

**Metric 10: Uptime (Hours)**
```
ml_uptime_seconds{model="xgboost_sepsis"} / 3600
```
Panel Type: Stat
- Unit: short
- Decimals: 1

---

## 🚨 3 Alert Rules - Setup in Grafana UI

### **Alert 1: High CPU (>80%)**
1. Edit CPU panel
2. **Alert** tab → **Create Alert**
   - Name: `HighCPUUsage`
   - Condition: `ml_cpu_usage_percent > 80`
   - Evaluate: every 1m for 5m
3. Notification: Email atau Slack (optional, atau test tanpa)
4. Save

### **Alert 2: High Latency (>0.5s)**
1. Edit Latency panel
2. **Alert** → **Create Alert**
   - Name: `HighLatency`
   - Condition: `histogram_quantile(0.95, ...) > 0.5`
   - Duration: 5m
3. Save

### **Alert 3: High Errors (>10/min)**
1. Edit Errors panel
2. **Alert** → **Create Alert**
   - Name: `HighErrorRate`
   - Condition: `rate(ml_errors_total[5m]) > 0.1666`
   - Duration: 5m
3. Save

---

## 📸 Screenshots untuk Submission

**10 Metric Panels:**
```
1.monitoring_grafana_requests.png
2.monitoring_grafana_latency.png
3.monitoring_grafana_errors.png
4.monitoring_grafana_cpu.png
5.monitoring_grafana_memory.png
6.monitoring_grafana_accuracy.png
7.monitoring_grafana_drift.png
8.monitoring_grafana_version.png
9.monitoring_grafana_request_rate.png
10.monitoring_grafana_uptime.png
```

**Plus Full Dashboard:**
```
monitoring_grafana_full_dashboard.png
```

**3 Alert Rules:**
```
monitoring_alert_1_high_cpu_rule.png
monitoring_alert_2_high_latency_rule.png
monitoring_alert_3_high_errors_rule.png
```

**Alert Firing (Optional):**
```
monitoring_alert_fired_cpu.png
monitoring_alert_notification.png
```

---

## ⚡ Minimal Version (If Short on Time)

**Minimum for Points:**
- ✅ Dashboard dengan 10 metric panels (screenshots)
- ✅ 3 Alert rules configured (screenshots)
- ✅ Prometheus data source connected
- ✅ Evidence folder structure

You can skip:
- Detailed alert notifications
- Inference load testing
- Multiple alert firings

---

## 🔍 Troubleshooting

**"No data in panel"**
→ Check Data Source: Save & Test harus hijau
→ Tunggu 1-2 menit untuk data masuk ke Prometheus/Grafana

**"Can't connect to Prometheus"**
→ Verify URL di Data Source: `http://localhost:9090`
→ Cek Prometheus running: `http://localhost:9090/graph`

**"Metrics not showing"**
→ Check exporter running: `http://localhost:8000/metrics`
→ Di Prometheus: Status → Targets, harus hijau "UP"

**"Alert not firing"**
→ Edit alert, cek kondisi/threshold
→ Modify exporter.py temporarily untuk trigger (set CPU=85)
→ Tunggu evaluation interval (5m)

---

## ✅ Submission Checklist

```
[ ] Dashboard created: "Dashboard-Naimatul-Ulumiyah"
[ ] 10 metrics visible & showing data
[ ] 3 alert rules created (High CPU, High Latency, High Errors)
[ ] Screenshots collected (11 + 3 = 14 min)
[ ] Optional: Alert fired & notification sent
[ ] Folder structure: bukti_monitoring_prometheus/, bukti_monitoring_grafana/, bukti_alerting_grafana/
[ ] SETUP_GUIDE.md filled
[ ] STATUS_SUMMARY.md prepared
[ ] All pushed to GitHub repo
```

---

## 🎓 Expected Proof Points

- ✅ Prometheus configured & scraping metrics
- ✅ Grafana connected to Prometheus
- ✅ 10 distinct metrics displayed on dashboard
- ✅ Metrics showing real data (not static)
- ✅ 3 Alerting rules defined
- ✅ Evidence in proper folder structure

---

**Expected Time: 15-20 min untuk quick setup**  
**Expected Points: 8-10 / 10**

Good luck! 🚀
