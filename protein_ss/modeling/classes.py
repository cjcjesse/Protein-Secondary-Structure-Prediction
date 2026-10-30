import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -torch.log(torch.tensor(10000.0)) / d_model)

        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)

        return x + self.pe[:, :seq_len, :]

class ProteinMLM(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_len, dropout, mlm_vocab, padding_idx, n_vocab=20):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(mlm_vocab, d_model, padding_idx=padding_idx) # includes masked and padding token idx since inputs contain them, (batch_size, max_len, d_model)
        self.pos_enc = PositionalEncoding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers, enable_nested_tensor=False)
        self.layer_norm = nn.LayerNorm(d_model) # one final normalisation
        self.head = nn.Linear(d_model, n_vocab) # logit score preds across actual aa residue excluding masked and padding token idx

    def encode(self, X, pad_mask=None):
        '''
        a separate encode instance method since objective is MLM self-supervised training of ProteinMLM to serve as encoder-body
        makes it easy to convert ProteinMLM into an generalizable encoder that generates encodings for any input protein sequence
        '''
        x = self.emb(X)
        x = self.pos_enc(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x, src_key_padding_mask=pad_mask) # pad_mask prevents attention from being attended to padding tokens
        x = self.layer_norm(x)
        
        return x

    def forward(self, X, pad_mask=None):
        x = self.encode(X, pad_mask)
        output = self.head(x)
        
        return output

class FrozenMLMEncoder(nn.Module):
    def __init__(self, mlm):
        super().__init__()
        mlm.eval()
        for param in mlm.parameters():
            param.requires_grad = False
        
        self.model = [mlm] # store in regular Python list so parameters in frozen mlm are not registered by nn.Module
        self.out_dim = mlm.d_model

    def forward(self, X, pad_mask=None):
        '''
        intention is to: 1) NEVER call .train() on frozen encoder, 2) when iterating across X, Y, lengths, is_padding in train/val_loader,
        AFTER pushing .to(DEVICE), FrozenMLMEncoder will take in X and is_padding for src_key_padding_mask
        '''    
        with torch.no_grad():
            x = self.model[0].encode(X, pad_mask) # set postions with real residue to False
            
        return x # encodings prior to MLM prediction, (batch_size, max_len, d_model)

class CNNBiLSTMTagger(nn.Module):
    def __init__(self, encoder, kernel_size=3, conv_layers=2, hidden_size=64, num_classes=3, dropout=0.1, pack=True):
        super().__init__()
        self.encoder = encoder # (batch_size, max_len, d_model)
        pad = kernel_size // 2 # cancels out shrinkage from input - kernel size; floor((input - kernel + 2 * pad) / stride=1) + 1

        self.cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(encoder.out_dim if i == 0 else hidden_size, hidden_size, kernel_size, padding=pad),
                nn.ReLU()
            ) for i in range(conv_layers)
        ])
        self.norm = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(conv_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
        self.pack = pack
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            batch_first=True, 
            bidirectional=True
        )
        self.head = nn.Linear(2 * hidden_size, num_classes) # (batch_size, max_len, hidden_size * n_dirs) -> (batch_size, max_len, num_classes)

    def forward(self, X, lengths, pad_mask):
        valid = (~pad_mask).unsqueeze(1) # (batch_size, 1, max_len)

        x = self.encoder(X, pad_mask) # (batch_size, max_len, d_model)
        x = x.transpose(1, 2) # (batch_size, d_model, max_len) for Conv1D layer
        
        for idx, layer in enumerate(self.cnn):
            x = x * valid # zeros out values at padding token positions, (batch_size, d_model/hidden_size, max_len)
            x = layer(x).transpose(1, 2) # (batch_size, hidden_size, max_len) -> (batch_size, max_len, hidden_size)
            x = self.norm[idx](x)
            x = self.dropout(x).transpose(1, 2) # (batch_size, max_len, hidden_size) -> (batch_size, hidden_size, max_len)
        x = x.transpose(1, 2) # (batch_size, max_len, hidden_size)

        max_len = x.size(1)
        if self.pack:
            p = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            output, _ = self.lstm(p)
            output, _ = pad_packed_sequence(output, batch_first=True, total_length=max_len)
        else:
            output, _ = self.lstm(x) # (batch_size, max_len, hidden_size * n_dirs)

        output = self.head(output)
        
        return output # (batch_size, max_len, num_classes)