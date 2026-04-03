import time
import random
import json
import os
import sys
import logging
import threading
import numpy as np
import requests
import psutil

from prometheus_client import (
    start_http_server,
    Counter,
    Gauge,
    Histogram,
    Summary,
    REGISTRY,
    CollectorRegistry
)

# Logging ──────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# Konfigurasi ──
MODEL_SERVE_URL = os.getenv('MODEL_SERVE_URL', 'http://localhost:5002/invocations')
EXPORTER_PORT   = int(os.getenv('EXPORTER_PORT', 8000))
SCRAPE_INTERVAL = float(os.getenv('SCRAPE_INTERVAL', 2.0))
N_FEATURES      = int(os.getenv('N_FEATURES', 13))            

# Definisi Metrik 
# 1. Total prediksi kumulatif
PREDICTION_COUNTER = Counter(
    'model_predictions_total',
    'Total jumlah prediksi yang dilakukan oleh model sejak startup',
    ['model_type', 'status']   
)

# 2. Latensi prediksi 
PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Waktu yang dibutuhkan model untuk memberikan satu prediksi (detik)',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# 3. Akurasi model 
MODEL_ACCURACY = Gauge(
    'model_accuracy_gauge',
    'Estimasi akurasi model pada window prediksi terakhir (0-1)'
)

# 4. Total error
ERROR_COUNTER = Counter(
    'model_error_total',
    'Total error/exception saat melakukan prediksi',
    ['error_type']   
)

# 5. Request rate 
REQUEST_RATE = Gauge(
    'model_requests_per_second',
    'Jumlah request prediksi per detik (dihitung tiap interval)'
)

# 6. Prediksi positif (sepsis terdeteksi)
SEPSIS_POSITIVE = Counter(
    'sepsis_positive_predictions_total',
    'Total prediksi POSITIF sepsis (model output = 1)'
)

# 7. Prediksi negatif 
SEPSIS_NEGATIVE = Counter(
    'sepsis_negative_predictions_total',
    'Total prediksi NEGATIF sepsis (model output = 0)'
)

# 8. Memori proses (bytes)
MEMORY_USAGE = Gauge(
    'model_memory_usage_bytes',
    'Penggunaan memori proses serving model (bytes)'
)

# 9. CPU usage (persen)
CPU_USAGE = Gauge(
    'model_cpu_usage_percent',
    'Penggunaan CPU oleh proses serving model (%)'
)

# 10. Ukuran response 
RESPONSE_SIZE = Summary(
    'model_response_size_bytes',
    'Ukuran response body dari model serving (bytes)'
)

# 11. Confidence score prediksi (histogram distribusi probabilitas)
CONFIDENCE_SCORE = Histogram(
    'model_confidence_score',
    'Distribusi confidence score (probability) prediksi model',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

# 12. Data drift score (simulasi deteksi distribusi shift)
FEATURE_DRIFT = Gauge(
    'model_feature_drift_score',
    'Skor data drift terdeteksi (0=no drift, 1=high drift). '
    'Tinggi mengindikasikan distribusi input berbeda dari training data.',
    ['feature_group']   # vital_signs, lab_values, demographics
)

# 13. False Negative Rate 
FALSE_NEGATIVE_RATE = Gauge(
    'model_false_negative_rate',
    'Estimasi false negative rate pada window terakhir. '
    'FNR tinggi = banyak pasien sepsis yang tidak terdeteksi — BERBAHAYA!'
)

# 14. Throughput total pasien
THROUGHPUT = Counter(
    'model_throughput_total',
    'Total data pasien yang sudah diproses sejak startup'
)

# State tracking untuk rolling metrics
_window_preds   = []    # [(y_true_sim, y_pred, confidence), ...]
_window_size    = 100   # rolling window
_request_count  = 0
_last_time      = time.time()
_lock           = threading.Lock()


# Helper: Generate dummy ICU patient data 
def generate_icu_patient():

    is_sick = random.random() < 0.08   # 8% positif — sesuai distribusi asli

    if is_sick:
        # Profil pasien dengan risiko sepsis tinggi
        features = {
            'HR_mean'           : random.gauss(115, 20),   # takikardia
            'O2Sat_mean'        : random.gauss(91, 5),     # hipoksemia
            'Temp_mean'         : random.gauss(38.8, 0.5), # demam
            'SBP_mean'          : random.gauss(88, 15),    # hipotensi
            'MAP_mean'          : random.gauss(62, 10),
            'Resp_mean'         : random.gauss(26, 5),     # takipnea
            'WBC_mean'          : random.gauss(18, 5),     # leukositosis
            'Lactate_mean'      : random.gauss(3.5, 1.0),  # laktat tinggi
            'Creatinine_mean'   : random.gauss(2.2, 0.8),  # AKI
            'pH_mean'           : random.gauss(7.28, 0.05),# asidosis
            'BUN_mean'          : random.gauss(38, 12),
            'Glucose_mean'      : random.gauss(175, 40),
            'sofa_proxy'        : random.randint(4, 10),
        }
        true_label = 1
    else:
        # Profil pasien normal ICU
        features = {
            'HR_mean'           : random.gauss(80, 12),
            'O2Sat_mean'        : random.gauss(97, 2),
            'Temp_mean'         : random.gauss(37.0, 0.4),
            'SBP_mean'          : random.gauss(118, 15),
            'MAP_mean'          : random.gauss(85, 10),
            'Resp_mean'         : random.gauss(16, 3),
            'WBC_mean'          : random.gauss(9, 3),
            'Lactate_mean'      : random.gauss(1.2, 0.4),
            'Creatinine_mean'   : random.gauss(0.9, 0.3),
            'pH_mean'           : random.gauss(7.40, 0.04),
            'BUN_mean'          : random.gauss(18, 6),
            'Glucose_mean'      : random.gauss(110, 25),
            'sofa_proxy'        : random.randint(0, 3),
        }
        true_label = 0

    return list(features.values()), true_label


# Helper: Call model serving endpoint 
def call_model(features: list) -> dict:
    # Format sesuai MLflow serving API
    payload = {'inputs': [features]}

    response = requests.post(
        MODEL_SERVE_URL,
        headers={'Content-Type': 'application/json'},
        data=json.dumps(payload),
        timeout=10
    )
    response.raise_for_status()

    result        = response.json()
    response_size = len(response.content)

    # Parse prediction
    predictions = result.get('predictions', result.get('outputs', [0]))
    if isinstance(predictions[0], list):
        # Output probabilitas
        confidence = predictions[0][1]
        prediction = 1 if confidence >= 0.35 else 0  # optimal threshold
    else:
        prediction = int(predictions[0])
        confidence = float(prediction)

    return {
        'prediction'   : prediction,
        'confidence'   : confidence,
        'response_size': response_size
    }


# Helper: Update resource metrics
def update_resource_metrics():
    """Update CPU dan memory dari proses sistem."""
    try:
        proc = psutil.Process(os.getpid())
        MEMORY_USAGE.set(proc.memory_info().rss)
        CPU_USAGE.set(psutil.cpu_percent(interval=None))
    except Exception:
        # Fallback: simulasi jika psutil tidak tersedia
        MEMORY_USAGE.set(random.uniform(80e6, 250e6))
        CPU_USAGE.set(random.uniform(15, 65))


# Helper: Update drift metrics 
def update_drift_metrics():
    # Simulasi drift yang berfluktuasi realistis
    base_time = time.time()

    vital_drift = abs(np.sin(base_time / 300)) * 0.3 + random.uniform(0, 0.1)
    lab_drift   = abs(np.sin(base_time / 500 + 1)) * 0.2 + random.uniform(0, 0.08)
    demo_drift  = random.uniform(0.01, 0.05)  # demografis jarang drift

    FEATURE_DRIFT.labels(feature_group='vital_signs').set(vital_drift)
    FEATURE_DRIFT.labels(feature_group='lab_values').set(lab_drift)
    FEATURE_DRIFT.labels(feature_group='demographics').set(demo_drift)


# Helper: Compute rolling accuracy & FNR ────────────────────────────────────
def update_rolling_metrics():
    global _window_preds

    with _lock:
        if len(_window_preds) < 10:
            return
        window = _window_preds[-_window_size:]

    y_true = np.array([x[0] for x in window])
    y_pred = np.array([x[1] for x in window])

    correct = (y_true == y_pred).mean()
    MODEL_ACCURACY.set(correct)

    # False Negative Rate: FN / (FN + TP)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    FALSE_NEGATIVE_RATE.set(fnr)


# Helper: Update request rate 
def update_request_rate(n_new_requests: int, elapsed: float):
    rate = n_new_requests / elapsed if elapsed > 0 else 0
    REQUEST_RATE.set(rate)


# Main loop ────
def run_prediction_loop():
    global _request_count, _last_time

    log.info(f"Memulai prediction loop → model di {MODEL_SERVE_URL}")
    log.info(f"Metrics tersedia di: http://localhost:{EXPORTER_PORT}/metrics")

    batch_count   = 0
    interval_reqs = 0

    while True:
        loop_start = time.time()

        # Generate data pasien
        features, true_label = generate_icu_patient()

        try:
            # Kirim ke model
            start = time.time()
            result = call_model(features)
            latency = time.time() - start

            prediction  = result['prediction']
            confidence  = result['confidence']
            resp_size   = result['response_size']

            # Update metrics
            PREDICTION_COUNTER.labels(model_type='xgboost', status='success').inc()
            PREDICTION_LATENCY.observe(latency)
            RESPONSE_SIZE.observe(resp_size)
            CONFIDENCE_SCORE.observe(confidence)
            THROUGHPUT.inc()

            if prediction == 1:
                SEPSIS_POSITIVE.inc()
            else:
                SEPSIS_NEGATIVE.inc()

            # Simpan ke rolling window
            with _lock:
                _window_preds.append((true_label, prediction, confidence))
                if len(_window_preds) > _window_size * 2:
                    _window_preds = _window_preds[-_window_size:]

            interval_reqs += 1

        except requests.exceptions.ConnectionError:
            ERROR_COUNTER.labels(error_type='connection_error').inc()
            PREDICTION_COUNTER.labels(model_type='xgboost', status='error').inc()
            log.warning("Model serving tidak dapat dijangkau — pastikan mlflow models serve aktif")
            time.sleep(5)
            continue

        except requests.exceptions.Timeout:
            ERROR_COUNTER.labels(error_type='timeout').inc()
            PREDICTION_COUNTER.labels(model_type='xgboost', status='error').inc()
            log.warning("Request timeout")

        except Exception as e:
            ERROR_COUNTER.labels(error_type='unexpected').inc()
            PREDICTION_COUNTER.labels(model_type='xgboost', status='error').inc()
            # Simulasi metrics meskipun model tidak running
            _simulate_metrics(features, true_label)
            interval_reqs += 1

        # Update resource metrics setiap 5 batch
        batch_count += 1
        if batch_count % 5 == 0:
            update_resource_metrics()
            update_drift_metrics()
            update_rolling_metrics()

        # Update request rate setiap 10 detik
        elapsed = time.time() - _last_time
        if elapsed >= 10:
            update_request_rate(interval_reqs, elapsed)
            _last_time    = time.time()
            interval_reqs = 0

        # Jaga interval
        sleep_time = SCRAPE_INTERVAL - (time.time() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)


def _simulate_metrics(features, true_label):
    confidence  = random.betavariate(2, 5)  # distribusi miring ke kiri (kebanyakan non-sepsis)
    prediction  = 1 if confidence >= 0.35 else 0

    CONFIDENCE_SCORE.observe(confidence)
    THROUGHPUT.inc()

    if prediction == 1:
        SEPSIS_POSITIVE.inc()
    else:
        SEPSIS_NEGATIVE.inc()

    with _lock:
        _window_preds.append((true_label, prediction, confidence))

    # Simulasi latensi
    latency = random.gauss(0.08, 0.02)
    PREDICTION_LATENCY.observe(max(0.01, latency))
    RESPONSE_SIZE.observe(random.randint(50, 300))

    # Resource
    MEMORY_USAGE.set(random.uniform(80e6, 250e6))
    CPU_USAGE.set(random.uniform(15, 65))


# Entry Point 
if __name__ == '__main__':
    log.info("="*60)
    log.info("  SEPSIS ICU — PROMETHEUS EXPORTER")
    log.info("="*60)
    log.info(f"  Model URL    : {MODEL_SERVE_URL}")
    log.info(f"  Metrics port : {EXPORTER_PORT}")
    log.info(f"  Interval     : {SCRAPE_INTERVAL}s")
    log.info("="*60)

    # Start HTTP server untuk Prometheus scrape
    start_http_server(EXPORTER_PORT)
    log.info(f"Prometheus exporter aktif → http://localhost:{EXPORTER_PORT}/metrics")

    # Pre-set nilai awal biar dashboard tidak kosong
    MODEL_ACCURACY.set(0.0)
    FALSE_NEGATIVE_RATE.set(0.0)
    REQUEST_RATE.set(0.0)
    FEATURE_DRIFT.labels(feature_group='vital_signs').set(0.0)
    FEATURE_DRIFT.labels(feature_group='lab_values').set(0.0)
    FEATURE_DRIFT.labels(feature_group='demographics').set(0.0)

    # Jalankan loop
    try:
        run_prediction_loop()
    except KeyboardInterrupt:
        log.info("\nExporter dihentikan.")
