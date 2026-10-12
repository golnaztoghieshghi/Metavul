# Experimental Results

This directory contains the experimental results of MetaVul on six benchmark datasets.

## Contents

The `tables/` directory contains the following CSV files:

| File | Dataset | Description |
|------|---------|-------------|
| `beall_results.csv` | BE-ALL | Results on the BE-ALL dataset |
| `rmall_results.csv` | RM-ALL | Results on the RM-ALL dataset |
| `hyall_results.csv` | HY-ALL | Results on the HY-ALL dataset |
| `reveal_results.csv` | ReVeal | Results on the ReVeal dataset |
| `primevul_results.csv` | PrimeVul | Results on the PrimeVul dataset |
| `bigvulsub_results.csv` | Big-Vulsub | Results on the Big-Vulsub dataset |

## Metrics

Each CSV file contains the following metrics:

| Column | Description |
|--------|-------------|
| `dataset` | Name of the dataset |
| `method` | Method used (MetaVul) |
| `best_threshold` | Optimal decision threshold |
| `accuracy` | Overall accuracy |
| `precision` | Precision (positive predictive value) |
| `recall` | Recall (sensitivity, true positive rate) |
| `f1` | F1-score |
| `fpr` | False Positive Rate |
| `fnr` | False Negative Rate |
| `mcc` | Matthews Correlation Coefficient |
| `balanced_accuracy` | Balanced accuracy |
| `g_mean` | Geometric mean of sensitivity and specificity |
| `roc_auc` | Area under the ROC curve |
| `pr_auc` | Area under the Precision-Recall curve |
| `tn` | True Negatives |
| `fp` | False Positives |
| `fn` | False Negatives |
| `tp` | True Positives |

## Reproducing the Results

To reproduce these results, run:

```
src/main.py
```
The script will generate the CSV files in this directory.

## Notes

All metrics are reported as decimal values (e.g., 0.9890 = 98.90%).

The best threshold was selected by optimizing a weighted combination of F1-score and recall.

Class weighting (power = 0.7) and combined Focal + Dice loss were used to handle class imbalance.
