import torch
import torch.nn as nn

class LSTM2Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.5, device='cpu'):
        super(LSTM2Model, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 2-layer LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Move model to the specified device
        self.to(device)
    
    def forward(self, x, hidden=None):
        # x shape: [batch_size, seq_length]
        x = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        output, hidden = self.lstm(x, hidden)  # output: [batch_size, seq_length, hidden_dim]
        output = self.dropout(output)
        output = self.fc(output)  # [batch_size, seq_length, vocab_size]
        return output, hidden
    
    def init_hidden(self, batch_size):
        # 2 layers of hidden states directly on the correct device
        h0 = torch.zeros(2, batch_size, self.lstm.hidden_size, device=self.device)
        c0 = torch.zeros(2, batch_size, self.lstm.hidden_size, device=self.device)
        return (h0, c0)