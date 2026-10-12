# MetaVul: A Novel Meta-Learning Ensemble Framework with CodeBERT for Source Code Vulnerability Detection 

This repository contains the source code and experimental data for the paper **"MetaVul: A Novel Meta-Learning Ensemble Framework with CodeBERT for Source Code Vulnerability Detection "**.

## Overview

MetaVul is a novel meta-learning ensemble framework for source code vulnerability detection. It dynamically combines three specialized Graph Neural Network (GNN) detectors (GCN, GAT, and APPNP) with CodeBERT semantic features using a neural meta-learner, instead of simple voting mechanisms. The framework also applies class weighting and adaptive decision threshold selection to address class imbalance in vulnerability datasets.

## Repository Structure

```
MetaVul/
├── src/                      # Source code
│   ├── graph_extractor.py    # Graph extraction (AST, CFG, PDG)
│   ├── models.py             # GNN models, CodeBERT fusion, meta-learner
│   ├── dataset.py            # Dataset loading and preprocessing
│   ├── losses.py             # Combined Focal + Dice loss
│   ├── metrics.py            # Evaluation metrics (MCC, G-Mean, PR-AUC, ...)
│   └── main.py               # Main training and evaluation script
├── results/
│   └── tables/               # Experimental results (CSV files)
├── docs/
│   └── dataset_links.md      # Links to publicly available datasets
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Clone the repository and install the required dependencies:

```
git clone https://github.com/golnaztoghieshghi/MetaVul.git
cd MetaVul
pip install -r requirements.txt
```

## Usage

Run the main training and evaluation script:

```
python src/main.py
```

The script performs the following steps:

1. Loads the VulDeePecker dataset from Hugging Face
2. Filters samples by CWE-119 for the BE-ALL setting
3. Extracts AST, CFG, and PDG graph representations
4. Trains MetaVul with combined Focal + Dice loss
5. Evaluates the model and saves results to `results/tables/`

## Datasets

The following publicly available datasets were used in this study:

- **VulDeePecker:** https://huggingface.co/datasets/claudios/VulDeePecker
- **ReVeal:** https://huggingface.co/datasets/claudios/ReVeal
- **PrimeVul:** https://huggingface.co/datasets/starsofchance/PrimeVul
- **Big-Vulsub:** https://huggingface.co/datasets/mahdin70/balanced_merged_bigvul_primevul

The derived datasets **BE-ALL**, **RM-ALL**, and **HY-ALL** were constructed by combining and preprocessing the above publicly available sources.

## Results

Experimental results, including accuracy, precision, recall, F1-score, MCC, Balanced Accuracy, G-Mean, PR-AUC, and confusion matrices, are available in the `results/tables/` directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the corresponding author.
