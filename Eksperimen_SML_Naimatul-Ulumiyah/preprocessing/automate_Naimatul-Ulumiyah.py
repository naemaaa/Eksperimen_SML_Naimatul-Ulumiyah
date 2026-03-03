import pandas as pd
import numpy as np
import os
import glob
import argparse
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

VITAL_COLS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
LAB_COLS   = [
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets'
]
DEMO_COLS      = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']
CLINICAL_COLS  = VITAL_COLS + LAB_COLS
SEED           = 42



# STEP 1: LOAD DATA
def load_all_patients(data_dir: str) -> pd.DataFrame:
    # normalize path
    data_dir = os.path.expanduser(data_dir)
    data_dir = os.path.abspath(data_dir)

    # first, look for direct .psv files
    all_files = sorted(glob.glob(os.path.join(data_dir, '*.psv')))
    # if nothing found, try recursive search inside provided directory
    if not all_files and os.path.isdir(data_dir):
        log.info(f"Tidak ada .psv langsung di {data_dir}, mencoba rekursif...")
        all_files = sorted(glob.glob(os.path.join(data_dir, '**', '*.psv'), recursive=True))
    # if still none, search entire workspace (cwd)
    if not all_files:
        log.info("Mencari .psv di seluruh working directory sebagai upaya terakhir...")
        all_files = sorted(glob.glob(os.path.join(os.getcwd(), '**', '*.psv'), recursive=True))
    if not all_files:
        raise FileNotFoundError(
            f"Tidak ada file .psv ditemukan. Coba jalankan dengan --data_dir mengarah ke folder berisi .psv.\n"
            f"Dicari: {data_dir} dan subfolders; juga dari working dir {os.getcwd()}"
        )
    log.info(f"Ditemukan {len(all_files):,} file pasien di '{data_dir}'")

    dfs = []
    for i, filepath in enumerate(all_files):
        patient_id = os.path.basename(filepath).replace('.psv', '')
        df = pd.read_csv(filepath, sep='|')
        df.insert(0, 'patient_id', patient_id)
        dfs.append(df)
        if (i + 1) % 5000 == 0:
            log.info(f"  Loaded {i+1:,}/{len(all_files):,} ...")

    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"Merge selesai → {combined.shape[0]:,} baris, {combined.shape[1]} kolom")
    log.info(f"Pasien unik: {combined['patient_id'].nunique():,}")
    return combined


# STEP 2: FEATURE ENGINEERING

def _sofa_proxy(row: dict) -> int:
    score = 0
    cr = row.get('Creatinine_mean', np.nan)
    if not np.isnan(cr):
        if cr >= 3.5:  score += 3
        elif cr >= 2.0: score += 2
        elif cr >= 1.2: score += 1

    bili = row.get('Bilirubin_total_mean', np.nan)
    if not np.isnan(bili):
        if bili >= 12:  score += 3
        elif bili >= 6: score += 2
        elif bili >= 2: score += 1

    plat = row.get('Platelets_mean', np.nan)
    if not np.isnan(plat):
        if plat < 20:    score += 4
        elif plat < 50:  score += 3
        elif plat < 100: score += 2
        elif plat < 150: score += 1
    return score


def aggregate_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Feature engineering: agregasi per pasien...")
    rows   = []
    groups = list(df.groupby('patient_id'))

    for i, (pid, group) in enumerate(groups):
        row = {'patient_id': pid}
        row['SepsisLabel'] = int(group['SepsisLabel'].max())

        for col in DEMO_COLS:
            row[col] = group[col].iloc[0]

        row['ICULOS_max']      = group['ICULOS'].max()
        row['n_observations']  = len(group)

        for col in CLINICAL_COLS:
            series = group[col].dropna()
            if len(series) > 0:
                row[f'{col}_mean'] = series.mean()
                row[f'{col}_std']  = series.std() if len(series) > 1 else 0.0
                row[f'{col}_min']  = series.min()
                row[f'{col}_max']  = series.max()
                row[f'{col}_last'] = series.iloc[-1]
            else:
                for s in ['mean', 'std', 'min', 'max', 'last']:
                    row[f'{col}_{s}'] = np.nan
            row[f'{col}_missing_rate'] = group[col].isnull().mean()

        for col in VITAL_COLS:
            series = group[col].dropna()
            row[f'{col}_trend'] = (
                (series.iloc[-1] - series.iloc[0]) / len(series)
                if len(series) >= 3 else 0.0
            )

        row['sofa_proxy'] = _sofa_proxy(row)
        rows.append(row)

        if (i + 1) % 5000 == 0:
            log.info(f"  Progress: {i+1:,}/{len(groups):,}")

    result = pd.DataFrame(rows)
    log.info(f"Agregasi selesai → shape: {result.shape}")
    return result


# STEP 3: CLEANING

def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    log.info(f"Missing sebelum imputasi: {X.isnull().sum().sum():,}")
    X_imputed = X.fillna(X.median())
    log.info(f"Missing setelah imputasi: {X_imputed.isnull().sum().sum()}")
    return X_imputed


def remove_low_quality_features(X: pd.DataFrame,
                                 corr_threshold: float = 0.98) -> pd.DataFrame:
    # Zero variance
    zero_var = X.columns[X.std() == 0].tolist()
    if zero_var:
        log.info(f"Hapus {len(zero_var)} fitur zero-variance")
        X = X.drop(columns=zero_var)

    # High correlation
    corr_mat  = X.corr().abs()
    upper_tri = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    high_corr = [c for c in upper_tri.columns if any(upper_tri[c] > corr_threshold)]
    if high_corr:
        log.info(f"Hapus {len(high_corr)} fitur highly correlated (>{corr_threshold})")
        X = X.drop(columns=high_corr)

    log.info(f"Shape setelah cleaning: {X.shape}")
    return X


 # STEP 4: SPLIT + SMOTE + SCALE
 
def split_and_balance(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    log.info(f"Split → train: {X_train.shape}, test: {X_test.shape}")

    log.info("Menerapkan SMOTE pada training set...")
    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    log.info(f"  Post-SMOTE → train: {X_train_res.shape} | "
             f"Sepsis: {y_train_res.sum():,} | Non-Sepsis: {(y_train_res==0).sum():,}")

    log.info("StandardScaler normalisasi...")
    scaler        = StandardScaler()
    X_train_sc    = pd.DataFrame(scaler.fit_transform(X_train_res), columns=X.columns)
    X_test_sc     = pd.DataFrame(scaler.transform(X_test),          columns=X.columns)

    return X_train_sc, X_test_sc, y_train_res, y_test


# STEP 5: SIMPAN OUTPUT

def save_outputs(X_train, X_test, y_train, y_test,
                 X_full, y_full, output_dir: str = 'preprocessing/sepsis_dataset'):
    # ensure output directory is inside preprocessing
    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy(); train_df['SepsisLabel'] = y_train.values
    test_df  = X_test.copy();  test_df['SepsisLabel']  = y_test.values
    full_df  = X_full.copy();  full_df['SepsisLabel']  = y_full.values

    train_path = os.path.join(output_dir, 'sepsis_preprocessing_train.csv')
    test_path  = os.path.join(output_dir, 'sepsis_preprocessing_test.csv')
    full_path  = os.path.join(output_dir, 'sepsis_preprocessing.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)
    full_df.to_csv(full_path,   index=False)

    log.info(f"Tersimpan:")
    log.info(f"{train_path} → {train_df.shape}")
    log.info(f"{test_path}  → {test_df.shape}")
    log.info(f"{full_path}  → {full_df.shape}")


# MAIN PIPELINE

def preprocess(data_dir: str, output_dir: str = '.'):
    log.info("  SEPSIS ICU PREPROCESSING PIPELINE")

    # 1. Load
    df_raw = load_all_patients(data_dir)

    # 2. Feature engineering
    df_patient = aggregate_patient_features(df_raw)

    # 3. Pisahkan X dan y
    exclude_cols = ['patient_id', 'SepsisLabel']
    feature_cols = [c for c in df_patient.columns if c not in exclude_cols]
    X = df_patient[feature_cols].copy()
    y = df_patient['SepsisLabel'].copy()

    # 4. Clean
    X = handle_missing_values(X)
    X = remove_low_quality_features(X)

    # 5. Split + SMOTE + Scale
    X_train, X_test, y_train, y_test = split_and_balance(X, y)

    # 6. Simpan
    save_outputs(X_train, X_test, y_train, y_test, X, y, output_dir)

    log.info("  PIPELINE SELESAI — Data siap untuk training model!")

    return X_train, X_test, y_train, y_test


# CLI ENTRY POINT
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sepsis ICU Automated Preprocessing Pipeline'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='sepsis_preprocessing/training_setA/training',
        help='Path ke folder berisi file .psv pasien (default: sepsis_preprocessing/training_setA/training)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='Eksperimen_SML_Naimatul-Ulumiyah/preprocessing/sepsis_dataset',
        help='Folder output untuk menyimpan CSV (default: preprocessing/sepsis_dataset)'
    )
    args = parser.parse_args()

    preprocess(data_dir=args.data_dir, output_dir=args.output_dir)
