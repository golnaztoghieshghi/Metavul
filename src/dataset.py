import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd


def filter_by_cwe(dataset_split, target_cwe, max_samples=None):
    data = []
    for item in dataset_split:
        code = item.get('functionSource', item.get('code', ''))
        label = item.get('label', 0)
        cwe = item.get('cwe', '')
        if code and len(str(code)) > 20:
            if target_cwe in str(cwe):
                data.append({'code': str(code), 'label': int(label)})
        if max_samples and len(data) >= max_samples:
            break
    return pd.DataFrame(data)


def get_improved_weights(y_train, power=0.7):
    majority = np.sum(y_train == 0)
    minority = np.sum(y_train == 1)
    if minority > 0:
        weight_positive = (majority / minority) ** power
    else:
        weight_positive = 1.0
    return torch.tensor([1.0, weight_positive], dtype=torch.float)


class VulDataset(Dataset):
    def __init__(self, df, tokenizer, extractor):
        self.samples = [(row['code'], row['label']) for _, row in df.iterrows()]
        self.tokenizer = tokenizer
        self.extractor = extractor
        print(f"   {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code, label = self.samples[idx]
        tokens = self.extractor.tokenize(code, max_tokens=80)
        ast_f, ast_a = self.extractor.extract_ast_graph(tokens)
        cfg_f, cfg_a = self.extractor.extract_cfg_graph(tokens)
        pdg_f, pdg_a = self.extractor.extract_pdg_graph(tokens)
        enc = self.tokenizer(code, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'ast_features': ast_f, 'ast_adj': ast_a,
            'cfg_features': cfg_f, 'cfg_adj': cfg_a,
            'pdg_features': pdg_f, 'pdg_adj': pdg_a,
            'label': torch.tensor(label, dtype=torch.long)
        }

    def collate(self, batch):
        max_nodes = max(b['ast_features'].shape[0] for b in batch)
        input_ids = torch.stack([b['input_ids'] for b in batch])
        mask = torch.stack([b['attention_mask'] for b in batch])
        labels = torch.stack([b['label'] for b in batch])
        ast_f, ast_a, cfg_f, cfg_a, pdg_f, pdg_a = [], [], [], [], [], []
        for b in batch:
            n = b['ast_features'].shape[0]
            ast_f.append(F.pad(b['ast_features'], (0, 0, 0, max_nodes - n), value=0))
            ast_a.append(F.pad(b['ast_adj'], (0, max_nodes - n, 0, max_nodes - n), value=0))
            cfg_f.append(F.pad(b['cfg_features'], (0, 0, 0, max_nodes - n), value=0))
            cfg_a.append(F.pad(b['cfg_adj'], (0, max_nodes - n, 0, max_nodes - n), value=0))
            pdg_f.append(F.pad(b['pdg_features'], (0, 0, 0, max_nodes - n), value=0))
            pdg_a.append(F.pad(b['pdg_adj'], (0, max_nodes - n, 0, max_nodes - n), value=0))
        return {
            'input_ids': input_ids, 'attention_mask': mask, 'labels': labels,
            'ast_features': torch.stack(ast_f), 'ast_adj': torch.stack(ast_a),
            'cfg_features': torch.stack(cfg_f), 'cfg_adj': torch.stack(cfg_a),
            'pdg_features': torch.stack(pdg_f), 'pdg_adj': torch.stack(pdg_a)
        }