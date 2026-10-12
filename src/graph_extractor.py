import re
import numpy as np
import torch


class GraphExtractor:
    def __init__(self):
        self.dangerous = {'strcpy', 'strcat', 'sprintf', 'gets', 'scanf', 'eval',
                          'malloc', 'calloc', 'realloc', 'free', 'memcpy', 'memmove'}
        self.control = {'if', 'else', 'for', 'while', 'do', 'switch', 'case'}
        self.operators = {'+', '-', '*', '/', '%', '=', '==', '!=', '<', '>'}

    def tokenize(self, code, max_tokens=80):
        code = re.sub(r'//.*?$|/\*.*?\*/', '', code, flags=re.MULTILINE | re.DOTALL)
        tokens = re.findall(r'\b\w+\b', code)[:max_tokens]
        return tokens if tokens else ['empty']

    def extract_ast_graph(self, tokens):
        n = len(tokens)
        features = []
        for i, t in enumerate(tokens):
            feat = [1.0 if t.isalpha() else 0.0,
                    1.0 if t.isdigit() else 0.0,
                    1.0 if t in self.control else 0.0,
                    1.0 if t in self.operators else 0.0,
                    1.0 if t in self.dangerous else 0.0,
                    min(len(t) / 20, 1.0),
                    1.0 if t in self.dangerous else 0.0]
            features.append(feat)
        adj = np.zeros((n, n))
        for i in range(n - 1):
            adj[i][i + 1] = adj[i + 1][i] = 1
        for i in range(n):
            for j in range(i + 1, min(i + 3, n)):
                adj[i][j] = adj[j][i] = 1
        row_sum = adj.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1, row_sum)
        return torch.tensor(features, dtype=torch.float), torch.tensor(adj / row_sum, dtype=torch.float)

    def extract_cfg_graph(self, tokens):
        n = len(tokens)
        features = []
        for i, t in enumerate(tokens):
            feat = [1.0 if t in self.control else 0.0,
                    1.0 if t in {'if', 'else'} else 0.0,
                    1.0 if t.isalpha() else 0.0,
                    i / max(n, 1),
                    1.0 if t in self.dangerous else 0.0]
            features.append(feat)
        adj = np.zeros((n, n))
        for i in range(n - 1):
            adj[i][i + 1] = adj[i + 1][i] = 1
        row_sum = adj.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1, row_sum)
        return torch.tensor(features, dtype=torch.float), torch.tensor(adj / row_sum, dtype=torch.float)

    def extract_pdg_graph(self, tokens):
        n = len(tokens)
        features = []
        for i, t in enumerate(tokens):
            feat = [1.0 if t in self.dangerous else 0.0,
                    1.0 if t.isalpha() else 0.0,
                    1.0 if t.isdigit() else 0.0,
                    min(len(t) / 20, 1.0),
                    i / max(n, 1),
                    1.0 if t in self.operators else 0.0]
            features.append(feat)
        adj = np.zeros((n, n))
        for i in range(n - 1):
            adj[i][i + 1] = adj[i + 1][i] = 1
        for i in range(n):
            for j in range(i + 1, min(i + 5, n)):
                if tokens[i] == tokens[j]:
                    adj[i][j] = adj[j][i] = 1
        row_sum = adj.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1, row_sum)
        return torch.tensor(features, dtype=torch.float), torch.tensor(adj / row_sum, dtype=torch.float)