import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef,
    balanced_accuracy_score, average_precision_score
)


def evaluate_imbalanced(y_true, y_pred, y_scores=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    recall_safe = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_vuln = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_safe = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_vuln = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    g_mean = np.sqrt(recall_safe * recall_vuln) if (recall_safe * recall_vuln) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    auc_roc = None
    pr_auc = None
    if y_scores is not None:
        try:
            auc_roc = roc_auc_score(y_true, y_scores)
            pr_auc = average_precision_score(y_true, y_scores)
        except Exception:
            pass

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'balanced_accuracy': balanced_acc,
        'g_mean': g_mean,
        'specificity': specificity,
        'recall_safe': recall_safe,
        'recall_vuln': recall_vuln,
        'f1_safe': f1_safe,
        'f1_vuln': f1_vuln,
        'auc_roc': auc_roc,
        'pr_auc': pr_auc,
        'confusion_matrix': (tn, fp, fn, tp),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }


def print_imbalanced_report(metrics, dataset_name="BE-ALL"):
    print("=" * 80)
    print(f"Evaluation Report - {dataset_name}")
    print("=" * 80)
    print("\nMain Metrics (for imbalanced data):")
    print(f"    MCC: {metrics['mcc']:.4f}")
    print(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"    G-Mean: {metrics['g_mean']:.4f}")
    if metrics['pr_auc'] is not None:
        print(f"    PR-AUC: {metrics['pr_auc']:.4f}")
    if metrics['auc_roc'] is not None:
        print(f"    ROC-AUC: {metrics['auc_roc']:.4f}")

    print("\nPer-class Metrics:")
    print(f"   Recall (Safe): {metrics['recall_safe']:.4f}")
    print(f"   Recall (Vuln): {metrics['recall_vuln']:.4f}")
    print(f"   F1 (Safe): {metrics['f1_safe']:.4f}")
    print(f"   F1 (Vuln): {metrics['f1_vuln']:.4f}")

    print("\nTraditional Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1']:.4f}")

    print("\nConfusion Matrix:")
    print(f"   TN: {metrics['tn']:5d}   FP: {metrics['fp']:5d}")
    print(f"   FN: {metrics['fn']:5d}   TP: {metrics['tp']:5d}")
    print("=" * 80)