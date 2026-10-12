import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        return F.relu(self.linear(torch.bmm(adj, x)))


class GCN(nn.Module):
    def __init__(self, in_dim=7, hidden_dim=64, out_dim=64):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.att = nn.Linear(hidden_dim, 1)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, adj):
        x = self.dropout(F.relu(self.gcn1(x, adj)))
        x = self.gcn2(x, adj)
        att_w = torch.softmax(self.att(x), dim=1)
        return self.out((x * att_w).sum(dim=1))


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=2):
        super().__init__()
        self.heads = heads
        self.linear = nn.Linear(in_dim, out_dim * heads)

    def forward(self, x, adj):
        batch, n, in_dim = x.shape
        x = self.linear(x).view(batch, n, self.heads, -1)
        x = x.transpose(1, 2)
        return x.mean(dim=1)


class GAT(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=64, out_dim=64, heads=2):
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden_dim, heads)
        self.gat2 = GATLayer(hidden_dim, hidden_dim, heads)
        self.att = nn.Linear(hidden_dim, 1)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, adj):
        x = self.dropout(self.gat1(x, adj))
        x = self.gat2(x, adj)
        att_w = torch.softmax(self.att(x), dim=1)
        return self.out((x * att_w).sum(dim=1))


class APPNP(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=64, out_dim=64, alpha=0.1, K=10):
        super().__init__()
        self.alpha = alpha
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.att = nn.Linear(hidden_dim, 1)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, adj):
        x = self.mlp(x)
        x0 = x
        for _ in range(self.K):
            x = (1 - self.alpha) * torch.bmm(adj, x) + self.alpha * x0
        att_w = torch.softmax(self.att(x), dim=1)
        return self.out((x * att_w).sum(dim=1))


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=64, num_heads=2, num_layers=1, ff_dim=128, dropout=0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size=768, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True,
                            batch_first=True, dropout=0.2)
        self.out = nn.Linear(hidden_size * 2, 64)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.out(lstm_out.mean(dim=1))


class CrossAttentionFusion(nn.Module):
    def __init__(self, gnn_dim=64, codebert_dim=768, output_dim=64):
        super().__init__()
        self.codebert_proj = nn.Linear(codebert_dim, gnn_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=gnn_dim, num_heads=4, batch_first=True)
        self.fc = nn.Linear(gnn_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, gnn_feat, codebert_feat):
        codebert_proj = self.codebert_proj(codebert_feat)
        gnn_expanded = gnn_feat.unsqueeze(1)
        codebert_expanded = codebert_proj.unsqueeze(1)
        attended, _ = self.cross_attn(gnn_expanded, codebert_expanded, codebert_expanded)
        attended = attended.squeeze(1)
        combined = torch.cat([attended, gnn_feat], dim=1)
        return self.dropout(F.relu(self.fc(combined)))


class AttentionMetaLearner(nn.Module):
    def __init__(self, num_classifiers=4, feature_dim=2, hidden_dim=32):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=2, batch_first=True)
        self.fc1 = nn.Linear(num_classifiers * feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classifiers)
        self.dropout = nn.Dropout(0.2)

    def forward(self, probs):
        attn_output, attn_weights = self.attention(probs, probs, probs)
        combined = probs + 0.3 * attn_output
        combined = combined.view(combined.size(0), -1)
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return F.softmax(self.fc3(x), dim=1), attn_weights


class iGnnVD_AllSteps(nn.Module):
    def __init__(self):
        super().__init__()
        print("Loading CodeBERT...")
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.codebert_dropout = nn.Dropout(0.2)

        self.gcn_ast = GCN(in_dim=7, hidden_dim=64, out_dim=64)
        self.gat_cfg = GAT(in_dim=5, hidden_dim=64, out_dim=64)
        self.appnp_pdg = APPNP(in_dim=6, hidden_dim=64, out_dim=64)

        self.transformer = TransformerEncoder(input_dim=64, num_heads=2, num_layers=1)
        self.bilstm = BiLSTMEncoder(input_size=768, hidden_size=64, num_layers=1)

        self.cross_attn_gcn = CrossAttentionFusion(gnn_dim=64, codebert_dim=768, output_dim=64)
        self.cross_attn_gat = CrossAttentionFusion(gnn_dim=64, codebert_dim=768, output_dim=64)
        self.cross_attn_appnp = CrossAttentionFusion(gnn_dim=64, codebert_dim=768, output_dim=64)

        self.classifier_gcn = nn.Linear(768 + 64, 2)
        self.classifier_gat = nn.Linear(768 + 64, 2)
        self.classifier_appnp = nn.Linear(768 + 64, 2)
        self.classifier_bilstm = nn.Linear(768 + 64, 2)

        self.meta_learner = AttentionMetaLearner(num_classifiers=4, feature_dim=2, hidden_dim=32)

    def forward(self, input_ids, attention_mask,
                ast_features, ast_adj,
                cfg_features, cfg_adj,
                pdg_features, pdg_adj):
        code_out = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        code_emb = self.codebert_dropout(code_out.last_hidden_state)
        code_emb_pooled = code_emb.mean(dim=1)

        gcn_raw = self.gcn_ast(ast_features, ast_adj)
        gat_raw = self.gat_cfg(cfg_features, cfg_adj)
        appnp_raw = self.appnp_pdg(pdg_features, pdg_adj)

        gcn_enhanced = self.cross_attn_gcn(gcn_raw, code_emb_pooled)
        gat_enhanced = self.cross_attn_gat(gat_raw, code_emb_pooled)
        appnp_enhanced = self.cross_attn_appnp(appnp_raw, code_emb_pooled)

        gnn_stack = torch.stack([gcn_enhanced, gat_enhanced, appnp_enhanced], dim=1)
        gnn_enhanced = self.transformer(gnn_stack)

        gcn_final = gnn_enhanced[:, 0, :]
        gat_final = gnn_enhanced[:, 1, :]
        appnp_final = gnn_enhanced[:, 2, :]

        bilstm_out = self.bilstm(code_emb)

        logits_gcn = self.classifier_gcn(torch.cat([code_emb_pooled, gcn_final], dim=1))
        logits_gat = self.classifier_gat(torch.cat([code_emb_pooled, gat_final], dim=1))
        logits_appnp = self.classifier_appnp(torch.cat([code_emb_pooled, appnp_final], dim=1))
        logits_bilstm = self.classifier_bilstm(torch.cat([code_emb_pooled, bilstm_out], dim=1))

        probs_gcn = F.softmax(logits_gcn, dim=1)
        probs_gat = F.softmax(logits_gat, dim=1)
        probs_appnp = F.softmax(logits_appnp, dim=1)
        probs_bilstm = F.softmax(logits_bilstm, dim=1)

        all_probs = torch.stack([probs_gcn, probs_gat, probs_appnp, probs_bilstm], dim=1)
        weights, attn_weights = self.meta_learner(all_probs)

        final_logits = (weights[:, 0:1] * logits_gcn +
                        weights[:, 1:2] * logits_gat +
                        weights[:, 2:3] * logits_appnp +
                        weights[:, 3:4] * logits_bilstm)

        return logits_gcn, logits_gat, logits_appnp, logits_bilstm, final_logits, weights, attn_weights