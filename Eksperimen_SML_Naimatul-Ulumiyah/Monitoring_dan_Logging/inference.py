import json
import time
import random
import argparse
import statistics
import numpy as np
import requests
from datetime import datetime

MODEL_URL       = 'http://localhost:5002/invocations'
OPTIMAL_THRESHOLD = 0.35    

# Warna terminal 
RED    = '\033[91m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
BLUE   = '\033[94m'
BOLD   = '\033[1m'
RESET  = '\033[0m'


# Generate patient data 
def generate_patient(profile: str = 'random') -> dict:
    """
    Generate data pasien ICU sintetis.
    profile: 'random' | 'healthy' | 'sepsis' | 'borderline'
    """
    if profile == 'sepsis':
        raw = {
            'HR_mean'          : random.gauss(118, 15),
            'O2Sat_mean'       : random.gauss(90, 4),
            'Temp_mean'        : random.gauss(39.1, 0.6),
            'SBP_mean'         : random.gauss(85, 12),
            'MAP_mean'         : random.gauss(60, 8),
            'Resp_mean'        : random.gauss(28, 6),
            'WBC_mean'         : random.gauss(20, 6),
            'Lactate_mean'     : random.gauss(4.0, 1.2),
            'Creatinine_mean'  : random.gauss(2.5, 0.9),
            'pH_mean'          : random.gauss(7.26, 0.06),
            'BUN_mean'         : random.gauss(42, 14),
            'Glucose_mean'     : random.gauss(185, 45),
            'sofa_proxy'       : random.randint(5, 11),
        }
    elif profile == 'healthy':
        raw = {
            'HR_mean'          : random.gauss(76, 10),
            'O2Sat_mean'       : random.gauss(98, 1.5),
            'Temp_mean'        : random.gauss(36.9, 0.3),
            'SBP_mean'         : random.gauss(122, 12),
            'MAP_mean'         : random.gauss(88, 8),
            'Resp_mean'        : random.gauss(15, 2),
            'WBC_mean'         : random.gauss(8, 2),
            'Lactate_mean'     : random.gauss(1.1, 0.3),
            'Creatinine_mean'  : random.gauss(0.85, 0.2),
            'pH_mean'          : random.gauss(7.41, 0.03),
            'BUN_mean'         : random.gauss(15, 4),
            'Glucose_mean'     : random.gauss(105, 20),
            'sofa_proxy'       : random.randint(0, 2),
        }
    elif profile == 'borderline':
        raw = {
            'HR_mean'          : random.gauss(100, 12),
            'O2Sat_mean'       : random.gauss(94, 3),
            'Temp_mean'        : random.gauss(38.2, 0.4),
            'SBP_mean'         : random.gauss(100, 14),
            'MAP_mean'         : random.gauss(72, 10),
            'Resp_mean'        : random.gauss(20, 4),
            'WBC_mean'         : random.gauss(13, 4),
            'Lactate_mean'     : random.gauss(2.2, 0.6),
            'Creatinine_mean'  : random.gauss(1.4, 0.5),
            'pH_mean'          : random.gauss(7.35, 0.04),
            'BUN_mean'         : random.gauss(26, 8),
            'Glucose_mean'     : random.gauss(145, 30),
            'sofa_proxy'       : random.randint(2, 5),
        }
    else:  # random
        p = random.choices(
            ['healthy', 'sepsis', 'borderline'],
            weights=[0.75, 0.08, 0.17]
        )[0]
        return generate_patient(p)

    return raw


def patient_to_features(patient: dict) -> list:
    return list(patient.values())


# Call model ───
def predict(features: list, threshold: float = OPTIMAL_THRESHOLD) -> dict:
    payload = {'inputs': [features]}

    start    = time.time()
    response = requests.post(
        MODEL_URL,
        headers={'Content-Type': 'application/json'},
        data=json.dumps(payload),
        timeout=10
    )
    latency  = time.time() - start
    response.raise_for_status()

    result      = response.json()
    predictions = result.get('predictions', result.get('outputs', [[0, 1]]))

    # Handle berbagai format output MLflow
    if isinstance(predictions[0], list):
        probs      = predictions[0]
        confidence = probs[1] if len(probs) > 1 else probs[0]
    elif isinstance(predictions[0], (int, float)):
        confidence = float(predictions[0])
    else:
        confidence = 0.5

    prediction = 1 if confidence >= threshold else 0

    return {
        'prediction' : prediction,
        'confidence' : confidence,
        'latency_ms' : latency * 1000,
        'label'      : 'SEPSIS' if prediction == 1 else 'Non-Sepsis',
        'risk_level' : _risk_level(confidence),
    }


def _risk_level(confidence: float) -> str:
    if confidence >= 0.7:  return f"{RED}CRITICAL{RESET}"
    if confidence >= 0.5:  return f"{YELLOW}HIGH{RESET}"
    if confidence >= 0.35: return f"{YELLOW}MODERATE{RESET}"
    if confidence >= 0.2:  return f"{BLUE}LOW{RESET}"
    return f"{GREEN}MINIMAL{RESET}"


# Clinical interpretation
def clinical_interpretation(patient: dict, result: dict) -> str:
    flags = []

    if patient.get('HR_mean', 0) > 100:
        flags.append("Takikardia (HR > 100)")
    if patient.get('O2Sat_mean', 100) < 94:
        flags.append("Hipoksemia (SpO2 < 94%)")
    if patient.get('Temp_mean', 37) > 38.3:
        flags.append("Demam (Temp > 38.3°C)")
    if patient.get('Lactate_mean', 0) > 2.0:
        flags.append(f"Laktat tinggi ({patient['Lactate_mean']:.1f} mmol/L)")
    if patient.get('SBP_mean', 120) < 90:
        flags.append("Hipotensi (SBP < 90 mmHg)")
    if patient.get('WBC_mean', 10) > 12 or patient.get('WBC_mean', 10) < 4:
        flags.append(f"WBC abnormal ({patient['WBC_mean']:.1f})")
    if patient.get('sofa_proxy', 0) >= 4:
        flags.append(f"SOFA score tinggi ({patient['sofa_proxy']})")

    if not flags:
        return "  Tidak ada tanda klinis sepsis yang signifikan."

    lines = [f"  • {f}" for f in flags]
    return "\n".join(lines)


# Display single prediction 
def display_prediction(i: int, patient: dict, result: dict, verbose: bool = True):
    label_color = RED if result['prediction'] == 1 else GREEN

    print(f"\n{'─'*55}")
    print(f"  Pasien #{i+1:03d}  |  {label_color}{BOLD}{result['label']}{RESET}"
          f"  |  Risk: {result['risk_level']}")
    print(f"  Confidence : {result['confidence']:.3f}"
          f"  |  Latency: {result['latency_ms']:.1f}ms")

    if verbose:
        print(f"\n  {BLUE}Tanda klinis:{RESET}")
        print(clinical_interpretation(patient, result))

        print(f"\n  {BLUE}Key values:{RESET}")
        print(f"  HR={patient.get('HR_mean',0):.0f}  "
              f"O2={patient.get('O2Sat_mean',0):.1f}%  "
              f"Temp={patient.get('Temp_mean',0):.1f}°C  "
              f"Lactate={patient.get('Lactate_mean',0):.1f}  "
              f"SOFA={patient.get('sofa_proxy',0)}")


# Summary stats 
def print_summary(results: list, latencies: list, elapsed: float):
    total    = len(results)
    positive = sum(1 for r in results if r['prediction'] == 1)
    negative = total - positive

    print(f"\n{'='*55}")
    print(f"  {BOLD}INFERENCE SUMMARY{RESET}")
    print(f"{'='*55}")
    print(f"  Total prediksi   : {total}")
    print(f"  Sepsis (+)       : {RED}{positive}{RESET} ({positive/total*100:.1f}%)")
    print(f"  Non-Sepsis (-)   : {GREEN}{negative}{RESET} ({negative/total*100:.1f}%)")
    print(f"  Total waktu      : {elapsed:.2f}s")
    print(f"  Throughput       : {total/elapsed:.1f} pred/s")
    print()
    print(f"  Latensi (ms):")
    print(f"    Mean   : {statistics.mean(latencies):.1f}")
    print(f"    Median : {statistics.median(latencies):.1f}")
    print(f"    P95    : {sorted(latencies)[int(len(latencies)*0.95)]:.1f}")
    print(f"    Max    : {max(latencies):.1f}")
    print(f"{'='*55}")


# Main
def main():
    parser = argparse.ArgumentParser(description='Sepsis ICU Model Inference')
    parser.add_argument('--n',        type=int,  default=20,    help='Jumlah prediksi')
    parser.add_argument('--quiet',    action='store_true',      help='Hanya tampilkan ringkasan')
    parser.add_argument('--stress',   action='store_true',      help='Stress test (tanpa delay)')
    parser.add_argument('--single',   action='store_true',      help='Single prediction dengan interpretasi penuh')
    parser.add_argument('--profile',  type=str,  default='random',
                        choices=['random', 'healthy', 'sepsis', 'borderline'],
                        help='Profil pasien yang di-generate')
    args = parser.parse_args()

    print(f"\n{BOLD}SEPSIS ICU — Model Inference{RESET}")
    print(f"   URL      : {MODEL_URL}")
    print(f"   Threshold: {OPTIMAL_THRESHOLD}")
    print(f"   Profil   : {args.profile}")
    print(f"   Waktu    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Test koneksi dulu
    try:
        test_patient  = generate_patient('healthy')
        test_features = patient_to_features(test_patient)
        test_result   = predict(test_features)
        print(f"\n{GREEN}Koneksi ke model serving berhasil{RESET}\n")
    except Exception as e:
        print(f"\n{RED}Tidak dapat terhubung ke model serving: {e}{RESET}")
        print(f"   Pastikan sudah menjalankan:")
        print(f"   mlflow models serve -m RUN_ID/artifacts/xgboost_model -p 5001 --no-conda")
        return

    # Single mode
    if args.single:
        profile = 'random' if args.profile == 'random' else args.profile
        patient  = generate_patient(profile)
        features = patient_to_features(patient)
        result   = predict(features)
        display_prediction(0, patient, result, verbose=True)
        return

    # Batch inference
    results   = []
    latencies = []
    start_all = time.time()

    for i in range(args.n):
        patient  = generate_patient(args.profile)
        features = patient_to_features(patient)

        try:
            result = predict(features)
            results.append(result)
            latencies.append(result['latency_ms'])

            if not args.quiet:
                display_prediction(i, patient, result, verbose=not args.stress)

        except Exception as e:
            print(f"{RED}  Error pada prediksi #{i+1}: {e}{RESET}")

        if not args.stress:
            time.sleep(0.3)

    elapsed = time.time() - start_all

    if results:
        print_summary(results, latencies, elapsed)


if __name__ == '__main__':
    main()
