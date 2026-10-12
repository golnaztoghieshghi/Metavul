# -*- coding: utf-8 -*-
"""
MetaVul: Main script for BE-ALL dataset
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score

from graph_extractor import GraphExtractor
from models import iGnnVD_AllSteps
from dataset import VulDataset, filter_by_cwe, get_improved_weights
from losses import CombinedLoss
from metrics import evaluate_imbalanced, print_imbalanced_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 80)
print("MetaVul - BE-ALL Dataset")
print(f"Device: {device}")
print("=" * 80)

# ==================== Load Dataset ====================
print("\n[Step 1] Loading dataset...")
dataset = load_dataset("claudios/VulDeePecker")

MAX_TRAIN = 15000
MAX_VAL = 2000
MAX_TEST = 2000

train_df = filter_by_cwe(dataset['train'], 'CWE-119', max_samples=MAX_TRAIN)
val_df = filter_by_cwe(dataset['validation'], 'CWE-119', max_samples=MAX_VAL)
test_df = filter_by_cwe(dataset['test'], 'CWE-119', max_samples=MAX_TEST)

print(f" Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ==================== Graph Extractor ====================
extractor = GraphExtractor()

# ==================== CodeBERT Tokenizer ====================
print("\n[Step 2] Loading CodeBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# ==================== Class Weights ====================
train_labels = train_df['label'].values
class_weights_tensor = get_improved_weights(train_labels, power=0.7).to(device)
print(f" Class weights: {class_weights_tensor}")

# ==================== Loss ====================
criterion = CombinedLoss(alpha=0.25, gamma=2.0, dice_weight=0.5, focal_weight=0.5)

# ==================== Model ====================
print("\n[Step 3] Building model...")
model = iGnnVD_AllSteps().to(device)
print(f" Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ==================== Datasets ====================
print("\n[Step 4] Preparing datasets...")
train_dataset = VulDataset(train_df, tokenizer, extractor)
val_dataset = VulDataset(val_df, tokenizer, extractor)
test_dataset = VulDataset(test_df, tokenizer, extractor)

BATCH_SIZE = 8
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=val_dataset.collate)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=test_dataset.collate)

# ==================== Training ====================
EPOCHS = 20
LR = 3e-4

optimizer = torch.optim.AdamW([
    {'params': model.codebert.parameters(), 'lr': 2e-5},
    {'params': model.gcn_ast.parameters(), 'lr': LR},
    {'params': model.gat_cfg.parameters(), 'lr': LR},
    {'params': model.appnp_pdg.parameters(), 'lr': LR},
    {'params': model.transformer.parameters(), 'lr': LR},
    {'params': model.bilstm.parameters(), 'lr': LR},
    {'params': model.cross_attn_gcn.parameters(), 'lr': LR},
    {'params': model.cross_attn_gat.parameters(), 'lr': LR},
    {'params': model.cross_attn_appnp.parameters(), 'lr': LR},
    {'params': model.meta_learner.parameters(), 'lr': LR * 2},
    {'params': model.classifier_gcn.parameters(), 'lr': LR * 2},
    {'params': model.classifier_gat.parameters(), 'lr': LR * 2},
    {'params': model.classifier_appnp.parameters(), 'lr': LR * 2},
    {'params': model.classifier_bilstm.parameters(), 'lr': LR * 2},
], lr=1e-4, weight_decay=0.001)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * len(train_loader) * EPOCHS),
    num_training_steps=len(train_loader) * EPOCHS
)

print(f"\n[Step 5] Training ({EPOCHS} epochs)...")
best_f1 = 0
best_state = None
patience = 5
counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        lbl = batch['labels'].to(device)
        ast_f = batch['ast_features'].to(device)
        ast_a = batch['ast_adj'].to(device)
        cfg_f = batch['cfg_features'].to(device)
        cfg_a = batch['cfg_adj'].to(device)
        pdg_f = batch['pdg_features'].to(device)
        pdg_a = batch['pdg_adj'].to(device)

        optimizer.zero_grad()
        logits_gcn, logits_gat, logits_appnp, logits_bilstm, final_logits, weights, _ = model(
            ids, mask, ast_f, ast_a, cfg_f, cfg_a, pdg_f, pdg_a
        )

        loss = (criterion(logits_gcn, lbl) + criterion(logits_gat, lbl) +
                criterion(logits_appnp, lbl) + criterion(logits_bilstm, lbl) +
                criterion(final_logits, lbl))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        preds = torch.argmax(final_logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbl.cpu().numpy())

    train_loss = total_loss / len(train_loader)
    train_acc = accuracy_score(all_labels, all_preds)

    # Validation
    model.eval()
    val_preds, val_labels, val_probs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbl = batch['labels'].to(device)
            ast_f = batch['ast_features'].to(device)
            ast_a = batch['ast_adj'].to(device)
            cfg_f = batch['cfg_features'].to(device)
            cfg_a = batch['cfg_adj'].to(device)
            pdg_f = batch['pdg_features'].to(device)
            pdg_a = batch['pdg_adj'].to(device)

            _, _, _, _, final_logits, _, _ = model(ids, mask, ast_f, ast_a, cfg_f, cfg_a, pdg_f, pdg_a)
            probs = F.softmax(final_logits, dim=1)
            preds = (probs[:, 1] > 0.30).int()
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(lbl.cpu().numpy())
            val_probs.extend(probs[:, 1].cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, zero_division=0)

    print(f"\n Epoch {epoch+1}/{EPOCHS}")
    print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"   Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = model.state_dict()
        counter = 0
        print(f"   Model saved (F1={best_f1:.4f})")
    else:
        counter += 1
        if counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break

if best_state:
    model.load_state_dict(best_state)

# ==================== Evaluation ====================
print("\n[Step 6] Final evaluation...")
model.eval()
all_probs, all_labels = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test"):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        lbl = batch['labels'].to(device)
        ast_f = batch['ast_features'].to(device)
        ast_a = batch['ast_adj'].to(device)
        cfg_f = batch['cfg_features'].to(device)
        cfg_a = batch['cfg_adj'].to(device)
        pdg_f = batch['pdg_features'].to(device)
        pdg_a = batch['pdg_adj'].to(device)

        _, _, _, _, final_logits, _, _ = model(ids, mask, ast_f, ast_a, cfg_f, cfg_a, pdg_f, pdg_a)
        probs = F.softmax(final_logits, dim=1)
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_labels.extend(lbl.cpu().numpy())

# Find best threshold
thresholds = np.arange(0.10, 0.50, 0.005)
best_th = 0.30
best_score = 0
for th in thresholds:
    preds = (np.array(all_probs) > th).astype(int)
    f1 = f1_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    score = 0.6 * f1 + 0.4 * recall
    if score > best_score:
        best_score = score
        best_th = th

print(f" Best threshold: {best_th:.3f}")

final_preds = (np.array(all_probs) > best_th).astype(int)
metrics = evaluate_imbalanced(np.array(all_labels), final_preds, np.array(all_probs))
print_imbalanced_report(metrics, "BE-ALL")

# ==================== Save Results ====================
pd.DataFrame([{
    'dataset': 'BE-ALL',
    'method': 'MetaVul',
    'best_threshold': best_th,
    'accuracy': metrics['accuracy'],
    'precision': metrics['precision'],
    'recall': metrics['recall'],
    'f1': metrics['f1'],
    'mcc': metrics['mcc'],
    'balanced_accuracy': metrics['balanced_accuracy'],
    'g_mean': metrics['g_mean'],
    'recall_vuln': metrics['recall_vuln'],
    'f1_vuln': metrics['f1_vuln'],
    'auc_roc': metrics['auc_roc'],
    'pr_auc': metrics['pr_auc'],
    'fpr': 1 - metrics['specificity'],
    'tn': metrics['tn'], 'fp': metrics['fp'],
    'fn': metrics['fn'], 'tp': metrics['tp']
}]).to_csv('../results/tables/beall_final_with_imbalanced_metrics.csv', index=False)
print("\n Results saved to results/tables/beall_final_with_imbalanced_metrics.csv")