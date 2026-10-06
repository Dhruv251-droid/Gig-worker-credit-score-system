import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
import pandas as pd

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def calculate_fairness_metrics(y_true, y_pred, sensitive_attr):
    # sensitive_attr is binary (0 or 1)
    # y_pred is binary
    group_0 = (sensitive_attr == 0)
    group_1 = (sensitive_attr == 1)
    
    # Demographic Parity
    prob_1_given_0 = y_pred[group_0].mean() if group_0.sum() > 0 else 0
    prob_1_given_1 = y_pred[group_1].mean() if group_1.sum() > 0 else 0
    dp_diff = abs(prob_1_given_0 - prob_1_given_1)
    
    # Equal Opportunity (True Positive Rate difference)
    pos_group_0 = group_0 & (y_true == 1)
    pos_group_1 = group_1 & (y_true == 1)
    
    tpr_0 = y_pred[pos_group_0].mean() if pos_group_0.sum() > 0 else 0
    tpr_1 = y_pred[pos_group_1].mean() if pos_group_1.sum() > 0 else 0
    eopp_diff = abs(tpr_0 - tpr_1)
    
    # Equalized Odds (Max of TPR diff and FPR diff)
    neg_group_0 = group_0 & (y_true == 0)
    neg_group_1 = group_1 & (y_true == 0)
    
    fpr_0 = y_pred[neg_group_0].mean() if neg_group_0.sum() > 0 else 0
    fpr_1 = y_pred[neg_group_1].mean() if neg_group_1.sum() > 0 else 0
    
    eodds_diff = max(abs(tpr_0 - tpr_1), abs(fpr_0 - fpr_1))
    
    return {
        'Demographic Parity Diff': dp_diff,
        'Equal Opportunity Diff': eopp_diff,
        'Equalized Odds Diff': eodds_diff
    }

def evaluate_model(y_true, y_prob, sensitive_attr=None, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5
        
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)
    
    results = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC': auc,
        'Brier Score': brier,
        'ECE': ece
    }
    
    if sensitive_attr is not None:
        fairness = calculate_fairness_metrics(y_true, y_pred, sensitive_attr)
        results.update(fairness)
        
    return results
