import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

# Load data from local preprocessed folder (robust to working dir)
print("Loading data...")
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(script_dir, 'sepsis_preprocessing')
train_df = pd.read_csv(os.path.join(data_dir, 'sepsis_preprocessing_train.csv'))
test_df  = pd.read_csv(os.path.join(data_dir, 'sepsis_preprocessing_test.csv'))

X_train = train_df.drop('SepsisLabel', axis=1)
y_train = train_df['SepsisLabel']
X_test  = test_df.drop('SepsisLabel', axis=1)
y_test  = test_df['SepsisLabel']

print(f"Train shape : {X_train.shape}")
print(f"Test shape  : {X_test.shape}")

# Setup MLflow
mlflow.set_experiment('Sepsis_ICU_Prediction')

with mlflow.start_run(run_name='RandomForest_Autolog'):

    mlflow.sklearn.autolog()

    print("\nTraining RandomForest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("HASIL EVALUASI")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC   : {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Non-Sepsis', 'Sepsis']))

print("\n Training selesai! Jalankan: mlflow ui")
print("   Buka browser: http://localhost:5000")
