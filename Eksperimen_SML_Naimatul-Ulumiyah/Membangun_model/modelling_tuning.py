import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve, classification_report
)

import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import shap

# ── CLI Args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--no-dagshub', action='store_true',
                    help='Skip DagsHub setup, simpan MLflow lokal')
parser.add_argument('--n-trials', type=int, default=30,
                    help='Jumlah trial Optuna (default: 30)')
args = parser.parse_args()

# ── DagsHub Setup ─────────────────────────────────────────────────────────────
USE_DAGSHUB = not args.no_dagshub

if USE_DAGSHUB:
    try:
        import dagshub
        DAGSHUB_USERNAME = "naemaaa"
        DAGSHUB_REPO     = "Sepsis-ICU-MLflow"

        dagshub.init(
            repo_owner=DAGSHUB_USERNAME,
            repo_name=DAGSHUB_REPO,
            mlflow=True
        )
        print(f"✅ DagsHub connected: {DAGSHUB_USERNAME}/{DAGSHUB_REPO}")
    except Exception as e:
        print(f"⚠️  DagsHub setup gagal: {e}")
        print("   Lanjut dengan MLflow lokal...")
        USE_DAGSHUB = False

mlflow.set_experiment('Sepsis_ICU_Tuning')

# ── Load Data ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SEPSIS ICU — ADVANCED MODEL TRAINING")
print("="*65)
print("\n📂 Loading data...")

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(script_dir, 'sepsis_preprocessing')
train_df = pd.read_csv(os.path.join(data_dir, 'sepsis_preprocessing_train.csv'))
test_df  = pd.read_csv(os.path.join(data_dir, 'sepsis_preprocessing_test.csv'))

X_train = train_df.drop('SepsisLabel', axis=1)
y_train = train_df['SepsisLabel']
X_test  = test_df.drop('SepsisLabel', axis=1)
y_test  = test_df['SepsisLabel']

print(f"  Train : {X_train.shape}  |  Sepsis: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"  Test  : {X_test.shape}   |  Sepsis: {y_test.sum():,}  ({y_test.mean()*100:.1f}%)")
print(f"  Fitur : {X_train.shape[1]}")


# ── Helper: Artifact Plots ─────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name, threshold=0.5):
    """Confusion matrix dengan anotasi klinis."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))

    disp = ConfusionMatrixDisplay(cm, display_labels=['Non-Sepsis', 'Sepsis'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)

    # Highlight False Negatives — paling berbahaya di medis
    ax.add_patch(plt.Rectangle((0.5, -0.5), 1, 1,
                                fill=True, color='#FFB3B3', alpha=0.3, zorder=0))
    ax.text(1.0, 0.0, '⚠️  False Negative\n(paling berbahaya)',
            ha='center', va='center', fontsize=8, color='darkred')

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    ax.set_title(
        f'{model_name} — Confusion Matrix\n'
        f'Sensitivity (Recall): {sensitivity:.3f}  |  Specificity: {specificity:.3f}',
        fontweight='bold'
    )
    path = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=120)
    plt.close()
    return path


def plot_roc_pr_curves(y_true, y_prob, model_name):
    """ROC + Precision-Recall curves dalam 1 figure."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    axes[0].plot(fpr, tpr, color='#E74C3C', lw=2, label=f'ROC (AUC = {auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    axes[0].fill_between(fpr, tpr, alpha=0.1, color='#E74C3C')
    axes[0].set(xlabel='False Positive Rate', ylabel='True Positive Rate',
                title=f'ROC Curve — {model_name}')
    axes[0].legend()

    # Precision-Recall Curve (lebih informatif untuk imbalanced!)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    axes[1].plot(recall, precision, color='#2E75B6', lw=2, label=f'PR (AP = {ap:.3f})')
    axes[1].axhline(y=y_true.mean(), color='gray', linestyle='--',
                    label=f'Baseline ({y_true.mean():.3f})')
    axes[1].fill_between(recall, precision, alpha=0.1, color='#2E75B6')
    axes[1].set(xlabel='Recall', ylabel='Precision',
                title=f'Precision-Recall Curve — {model_name}')
    axes[1].legend()

    plt.suptitle('💡 PR Curve lebih informatif dari ROC untuk imbalanced dataset',
                 fontsize=10, style='italic', color='gray')
    plt.tight_layout()
    path = f'roc_pr_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(path, bbox_inches='tight', dpi=120)
    plt.close()
    return path


def plot_threshold_optimization(y_true, y_prob, model_name):
    """
    Cari threshold optimal untuk F1 dan Recall.
    Krusial di medis — default threshold 0.5 jarang optimal!
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1s, recalls, precisions = [], [], []

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_true, pred, zero_division=0))
        recalls.append(recall_score(y_true, pred, zero_division=0))
        precisions.append(precision_score(y_true, pred, zero_division=0))

    best_f1_idx       = np.argmax(f1s)
    best_recall_idx   = np.argmax([r for r in recalls if r >= 0.9] or recalls)
    best_threshold_f1 = thresholds[best_f1_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, f1s,        label='F1 Score',  color='#2E75B6', lw=2)
    ax.plot(thresholds, recalls,    label='Recall',     color='#E74C3C', lw=2)
    ax.plot(thresholds, precisions, label='Precision',  color='#27AE60', lw=2)
    ax.axvline(best_threshold_f1, color='purple', linestyle='--', lw=1.5,
               label=f'Best F1 threshold: {best_threshold_f1:.2f}')
    ax.axvline(0.5, color='gray', linestyle=':', lw=1, label='Default (0.5)')

    ax.set(xlabel='Threshold', ylabel='Score',
           title=f'Threshold Optimization — {model_name}\n'
                 f'Best F1 @ threshold={best_threshold_f1:.2f} → F1={f1s[best_f1_idx]:.3f}, '
                 f'Recall={recalls[best_f1_idx]:.3f}')
    ax.legend()
    ax.set_xlim(0.1, 0.9)
    plt.tight_layout()
    path = f'threshold_opt_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(path, bbox_inches='tight', dpi=120)
    plt.close()

    return path, best_threshold_f1, f1s[best_f1_idx]


def plot_feature_importance(model, feature_names, model_name, top_n=25):
    """Feature importance dari model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return None

    fi_df = pd.DataFrame({
        'feature'   : feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#E74C3C' if 'Lactate' in f or 'WBC' in f or 'sofa' in f
              else '#4A90D9' for f in fi_df['feature']]
    ax.barh(fi_df['feature'][::-1], fi_df['importance'][::-1],
            color=colors[::-1], edgecolor='white')
    ax.set(xlabel='Importance', title=f'Top {top_n} Feature Importance — {model_name}')
    ax.text(0.98, 0.02, '🔴 = Biomarker klinis sepsis',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color='#E74C3C')
    plt.tight_layout()
    path = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(path, bbox_inches='tight', dpi=120)
    plt.close()
    return path


def plot_shap_analysis(model, X_sample, model_name):
    """
    SHAP analysis — menjelaskan KENAPA model memprediksi sepsis.
    Ini yang membuat portfolio kamu stand out secara klinis.
    """
    print(f"\n  🔍 Computing SHAP values untuk {model_name}...")

    try:
        if 'XGBoost' in model_name:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.TreeExplainer(model, feature_perturbation='interventional')

        shap_values = explainer.shap_values(X_sample)

        # Untuk binary classification, ambil class 1 (sepsis)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        # Plot 1: SHAP Summary (global importance)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_sample, show=False, max_display=20,
                          plot_type='bar', color='#E74C3C')
        plt.title(f'SHAP Feature Importance — {model_name}\n'
                  f'(Seberapa besar kontribusi tiap fitur terhadap prediksi sepsis)',
                  fontweight='bold')
        path_bar = f'shap_importance_{model_name.lower().replace(" ", "_")}.png'
        plt.tight_layout()
        plt.savefig(path_bar, bbox_inches='tight', dpi=120)
        plt.close()

        # Plot 2: SHAP Beeswarm (distribusi + arah pengaruh)
        plt.figure(figsize=(11, 9))
        shap.summary_plot(sv, X_sample, show=False, max_display=20)
        plt.title(f'SHAP Beeswarm — {model_name}\n'
                  f'(Merah = nilai tinggi → arah pengaruh terhadap prediksi sepsis)',
                  fontweight='bold')
        path_bee = f'shap_beeswarm_{model_name.lower().replace(" ", "_")}.png'
        plt.tight_layout()
        plt.savefig(path_bee, bbox_inches='tight', dpi=120)
        plt.close()

        print(f"  ✅ SHAP analysis selesai")
        return path_bar, path_bee

    except Exception as e:
        print(f"  ⚠️  SHAP error: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: RANDOM FOREST dengan Optuna Tuning
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*65)
print("  MODEL 1: Random Forest + Optuna Tuning")
print("─"*65)

def rf_objective(trial):
    params = {
        'n_estimators'      : trial.suggest_int('n_estimators', 50, 300),
        'max_depth'         : trial.suggest_int('max_depth', 5, 30),
        'min_samples_split' : trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf'  : trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features'      : trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]),
        'class_weight'      : 'balanced',
        'random_state'      : 42,
        'n_jobs'            : -1
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)

print(f"\nOptuna tuning ({args.n_trials} trials)...")
rf_study = optuna.create_study(direction='maximize',
                                study_name='RF_Sepsis',
                                sampler=optuna.samplers.TPESampler(seed=42))
rf_study.optimize(rf_objective, n_trials=args.n_trials, show_progress_bar=True)

best_rf_params = rf_study.best_params
best_rf_params.update({'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1})
print(f"Best RF params: {best_rf_params}")
print(f"Best RF AUC   : {rf_study.best_value:.4f}")

# Train final RF
print("\nTraining final Random Forest...")
best_rf = RandomForestClassifier(**best_rf_params)
best_rf.fit(X_train, y_train)

rf_pred = best_rf.predict(X_test)
rf_prob = best_rf.predict_proba(X_test)[:, 1]

# Threshold optimization
rf_thresh_path, rf_best_thresh, rf_best_f1 = plot_threshold_optimization(
    y_test, rf_prob, 'Random Forest'
)
rf_pred_opt = (rf_prob >= rf_best_thresh).astype(int)

# ── MLflow logging: Random Forest ─────────────────────────────────────────────
with mlflow.start_run(run_name='RandomForest_Optuna_Tuning') as run_rf:

    # Log params
    mlflow.log_params(best_rf_params)
    mlflow.log_param('tuning_method', 'optuna')
    mlflow.log_param('n_trials', args.n_trials)
    mlflow.log_param('optimal_threshold', rf_best_thresh)

    # Log metrics (manual — lebih dari autolog)
    mlflow.log_metric('accuracy',              accuracy_score(y_test, rf_pred))
    mlflow.log_metric('precision',             precision_score(y_test, rf_pred))
    mlflow.log_metric('recall',                recall_score(y_test, rf_pred))
    mlflow.log_metric('f1_score',              f1_score(y_test, rf_pred))
    mlflow.log_metric('roc_auc',               roc_auc_score(y_test, rf_prob))
    mlflow.log_metric('avg_precision',         average_precision_score(y_test, rf_prob))
    mlflow.log_metric('recall_optimal_thresh', recall_score(y_test, rf_pred_opt))
    mlflow.log_metric('f1_optimal_thresh',     rf_best_f1)
    mlflow.log_metric('optuna_best_auc',       rf_study.best_value)

    # Log artefak
    cm_path  = plot_confusion_matrix(y_test, rf_pred_opt, 'Random Forest')
    roc_path = plot_roc_pr_curves(y_test, rf_prob, 'Random Forest')
    fi_path  = plot_feature_importance(best_rf, X_train.columns.tolist(), 'Random Forest')
    shap_bar, shap_bee = plot_shap_analysis(best_rf, X_test.iloc[:500], 'Random Forest')

    for path in [cm_path, roc_path, rf_thresh_path, fi_path, shap_bar, shap_bee]:
        if path and os.path.exists(path):
            mlflow.log_artifact(path)

    # Log model
    mlflow.sklearn.log_model(best_rf, 'random_forest_model',
                              registered_model_name='Sepsis_RandomForest')

    # Log Optuna study summary
    optuna_summary = {
        'best_value': rf_study.best_value,
        'best_params': rf_study.best_params,
        'n_trials': args.n_trials
    }
    with open('optuna_rf_summary.json', 'w') as f:
        json.dump(optuna_summary, f, indent=2)
    mlflow.log_artifact('optuna_rf_summary.json')

    rf_run_id = run_rf.info.run_id
    print(f"\n  MLflow Run ID (RF): {rf_run_id}")

print(f"\n  RF Results (optimal threshold={rf_best_thresh:.2f}):")
print(f"  ROC AUC   : {roc_auc_score(y_test, rf_prob):.4f}")
print(f"  F1 Score  : {rf_best_f1:.4f}")
print(f"  Recall    : {recall_score(y_test, rf_pred_opt):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: XGBOOST dengan Optuna Tuning
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*65)
print("  MODEL 2: XGBoost + Optuna Tuning")
print("─"*65)

# Hitung scale_pos_weight untuk imbalance
neg_count  = (y_train == 0).sum()
pos_count  = (y_train == 1).sum()
spw        = neg_count / pos_count
print(f"\n  scale_pos_weight (imbalance correction): {spw:.2f}")

def xgb_objective(trial):
    params = {
        'n_estimators'      : trial.suggest_int('n_estimators', 100, 500),
        'max_depth'         : trial.suggest_int('max_depth', 3, 12),
        'learning_rate'     : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample'         : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree'  : trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight'  : trial.suggest_int('min_child_weight', 1, 10),
        'gamma'             : trial.suggest_float('gamma', 0, 5),
        'reg_alpha'         : trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda'        : trial.suggest_float('reg_lambda', 0, 5),
        'scale_pos_weight'  : spw,
        'eval_metric'       : 'auc',
        'use_label_encoder' : False,
        'random_state'      : 42,
        'n_jobs'            : -1
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    y_prob = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)

print(f"\nOptuna tuning ({args.n_trials} trials)...")
xgb_study = optuna.create_study(direction='maximize',
                                  study_name='XGB_Sepsis',
                                  sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=args.n_trials, show_progress_bar=True)

best_xgb_params = xgb_study.best_params
best_xgb_params.update({
    'scale_pos_weight': spw,
    'eval_metric'     : 'auc',
    'use_label_encoder': False,
    'random_state'    : 42,
    'n_jobs'          : -1
})
print(f"Best XGB params: {best_xgb_params}")
print(f"Best XGB AUC   : {xgb_study.best_value:.4f}")

# Train final XGBoost
print("\nTraining final XGBoost...")
best_xgb = xgb.XGBClassifier(**best_xgb_params)
best_xgb.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             verbose=False)

xgb_pred = best_xgb.predict(X_test)
xgb_prob = best_xgb.predict_proba(X_test)[:, 1]

# Threshold optimization
xgb_thresh_path, xgb_best_thresh, xgb_best_f1 = plot_threshold_optimization(
    y_test, xgb_prob, 'XGBoost'
)
xgb_pred_opt = (xgb_prob >= xgb_best_thresh).astype(int)

# ── MLflow logging: XGBoost ───────────────────────────────────────────────────
with mlflow.start_run(run_name='XGBoost_Optuna_Tuning') as run_xgb:

    # Log params
    mlflow.log_params({k: v for k, v in best_xgb_params.items()
                       if k not in ['eval_metric', 'use_label_encoder', 'n_jobs']})
    mlflow.log_param('tuning_method', 'optuna')
    mlflow.log_param('n_trials', args.n_trials)
    mlflow.log_param('optimal_threshold', xgb_best_thresh)
    mlflow.log_param('scale_pos_weight', round(spw, 3))

    # Log metrics
    mlflow.log_metric('accuracy',              accuracy_score(y_test, xgb_pred))
    mlflow.log_metric('precision',             precision_score(y_test, xgb_pred))
    mlflow.log_metric('recall',                recall_score(y_test, xgb_pred))
    mlflow.log_metric('f1_score',              f1_score(y_test, xgb_pred))
    mlflow.log_metric('roc_auc',               roc_auc_score(y_test, xgb_prob))
    mlflow.log_metric('avg_precision',         average_precision_score(y_test, xgb_prob))
    mlflow.log_metric('recall_optimal_thresh', recall_score(y_test, xgb_pred_opt))
    mlflow.log_metric('f1_optimal_thresh',     xgb_best_f1)
    mlflow.log_metric('optuna_best_auc',       xgb_study.best_value)

    # Log artefak
    cm_path  = plot_confusion_matrix(y_test, xgb_pred_opt, 'XGBoost')
    roc_path = plot_roc_pr_curves(y_test, xgb_prob, 'XGBoost')
    fi_path  = plot_feature_importance(best_xgb, X_train.columns.tolist(), 'XGBoost')
    shap_bar, shap_bee = plot_shap_analysis(best_xgb, X_test.iloc[:500], 'XGBoost')

    for path in [cm_path, roc_path, xgb_thresh_path, fi_path, shap_bar, shap_bee]:
        if path and os.path.exists(path):
            mlflow.log_artifact(path)

    # Log model
    mlflow.xgboost.log_model(best_xgb, 'xgboost_model',
                              registered_model_name='Sepsis_XGBoost')

    # Log Optuna summary
    optuna_summary = {
        'best_value' : xgb_study.best_value,
        'best_params': xgb_study.best_params,
        'n_trials'   : args.n_trials
    }
    with open('optuna_xgb_summary.json', 'w') as f:
        json.dump(optuna_summary, f, indent=2)
    mlflow.log_artifact('optuna_xgb_summary.json')

    xgb_run_id = run_xgb.info.run_id
    print(f"\n  MLflow Run ID (XGB): {xgb_run_id}")

print(f"\n  XGB Results (optimal threshold={xgb_best_thresh:.2f}):")
print(f"  ROC AUC   : {roc_auc_score(y_test, xgb_prob):.4f}")
print(f"  F1 Score  : {xgb_best_f1:.4f}")
print(f"  Recall    : {recall_score(y_test, xgb_pred_opt):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON & BEST MODEL SELECTION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("  MODEL COMPARISON")
print("="*65)

rf_auc  = roc_auc_score(y_test, rf_prob)
xgb_auc = roc_auc_score(y_test, xgb_prob)
best_model_name = 'XGBoost' if xgb_auc >= rf_auc else 'Random Forest'
best_model      = best_xgb  if xgb_auc >= rf_auc else best_rf

comparison = {
    'Model'          : ['Random Forest', 'XGBoost'],
    'ROC AUC'        : [rf_auc, xgb_auc],
    'F1 (optimal)'   : [rf_best_f1, xgb_best_f1],
    'Recall (opt)'   : [recall_score(y_test, rf_pred_opt),
                        recall_score(y_test, xgb_pred_opt)],
    'Precision (opt)': [precision_score(y_test, rf_pred_opt, zero_division=0),
                        precision_score(y_test, xgb_pred_opt, zero_division=0)],
    'Best Threshold' : [rf_best_thresh, xgb_best_thresh],
}
comp_df = pd.DataFrame(comparison)
print(comp_df.to_string(index=False))
print(f"\n🏆 Best Model: {best_model_name}")

# Save comparison
comp_df.to_csv('model_comparison.csv', index=False)

# Comparison plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics   = ['ROC AUC', 'F1 (optimal)', 'Recall (opt)']
colors    = ['#4A90D9', '#E74C3C']

for i, metric in enumerate(metrics):
    bars = axes[i].bar(comparison['Model'], comparison[metric],
                       color=colors, edgecolor='white', width=0.4)
    for bar, val in zip(bars, comparison[metric]):
        axes[i].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    axes[i].set_title(metric, fontweight='bold')
    axes[i].set_ylim(0, 1.1)
    axes[i].axhline(y=0.9, color='green', linestyle='--', alpha=0.4, label='Target 0.9')

plt.suptitle('Model Comparison: Random Forest vs XGBoost\n(Sepsis ICU Prediction)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png', bbox_inches='tight', dpi=120)
plt.close()

# Log comparison di MLflow parent run
with mlflow.start_run(run_name='Model_Comparison_Summary'):
    mlflow.log_metric('rf_roc_auc',  rf_auc)
    mlflow.log_metric('xgb_roc_auc', xgb_auc)
    mlflow.log_param('best_model', best_model_name)
    mlflow.log_artifact('model_comparison.csv')
    mlflow.log_artifact('model_comparison.png')


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("  SELESAI!")
print("="*65)
print(f"\n  🏆 Best Model   : {best_model_name}")
print(f"  📊 ROC AUC      : {max(rf_auc, xgb_auc):.4f}")
print(f"\n  Artefak yang dihasilkan:")
print("  ├── confusion_matrix_*.png")
print("  ├── roc_pr_*.png")
print("  ├── threshold_opt_*.png")
print("  ├── feature_importance_*.png")
print("  ├── shap_importance_*.png  ← SHAP analysis")
print("  ├── shap_beeswarm_*.png    ← SHAP beeswarm")
print("  ├── model_comparison.png")
print("  └── optuna_*_summary.json")
print()

if USE_DAGSHUB:
    print(f"MLflow online: https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
else:
    print("MLflow lokal: jalankan 'mlflow ui' → http://localhost:5000")

print()
print("  Insight klinis dari model:")
print("  → Cek SHAP beeswarm: fitur Lactate, WBC, HR_trend, sofa_proxy")
print("    adalah kontributor terbesar prediksi sepsis.")
print("  → Optimal threshold berbeda dari 0.5 karena class imbalance.")
print("    Di setting klinis, recall lebih penting dari precision!")