# MetaVul: A Meta-Learning Ensemble Framework for Source Code Vulnerability Detection

This repository contains the source code for the paper "MetaVul: A Meta-Learning Ensemble Framework for Source Code Vulnerability Detection".

## Overview
MetaVul combines three GNN detectors (GCN, GAT, APPNP) with CodeBERT semantic features using a neural meta-learner for accurate vulnerability detection.

## Datasets
The following publicly available datasets were used:
- VulDeePecker: https://huggingface.co/datasets/claudios/VulDeePecker
- ReVeal: https://huggingface.co/datasets/claudios/ReVeal
- PrimeVul: https://huggingface.co/datasets/starsofchance/PrimeVul
- Big-Vulsub: https://huggingface.co/datasets/mahdin70/balanced_merged_bigvul_primevul

## Installation
```bash
pip install -r requirements.txt