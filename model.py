from typing import Dict

import torch
from torch.nn import Embedding
import torch.nn as nn

    
class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, m_type = 'lstm', dropout = 0.5, bidirectional = False, nhead=16):
        super(BasicBlock, self).__init__()
        self.m_type = m_type
        self.num_layers = hidden_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        if self.m_type == 'lstm':
            self.block = nn.LSTM(input_dim, output_dim, self.num_layers,dropout = self.dropout, bidirectional = self.bidirectional, batch_first=True)
        elif self.m_type == 'rnn': 
            self.block = nn.RNN(input_dim, output_dim, self.num_layers,dropout = self.dropout, bidirectional = self.bidirectional, batch_first=True)
        elif self.m_type == 'gru': 
            self.block = nn.GRU(input_dim, output_dim, self.num_layers,dropout = self.dropout, bidirectional = self.bidirectional, batch_first=True)
        elif self.m_type == 'transformer':
            self.block = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                *[nn.TransformerEncoderLayer(d_model=output_dim, dim_feedforward=2048, nhead=nhead, dropout = self.dropout, batch_first=True) for i in range(self.num_layers)]
            )
    def forward(self, x):
        if self.m_type == 'lstm':
            out, (hidden_state,_) = self.block(x)
        elif self.m_type == 'rnn':
            out, hidden_state = self.block(x)
        elif self.m_type == 'gru':
            out, hidden_state = self.block(x)
        elif self.m_type == 'transformer':
            out = self.block(x)
        else:
             raise ValueError(f"Unknown model: {self.m_type}")
        return out
    
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256, dropout=0.5, nhead=16, m_type = 'lstm'):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim, hidden_layers, dropout=dropout, m_type = m_type, nhead=nhead),
            # nn.Linear(hidden_dim, output_dim)
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )
    def forward(self, x):
        x = self.fc(x)
        # print(x.shape)
        x = self.linear(x[:,-1])
        x = torch.reshape(x,(-1,))
        return x

