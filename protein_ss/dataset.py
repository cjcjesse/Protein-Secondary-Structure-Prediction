import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
from protein_ss.config import RAW_DATA_DIR

def download(name):
    path = hf_hub_download(repo_id='proteinea/secondary_structure_prediction', filename=name, repo_type='dataset', local_dir=RAW_DATA_DIR)

    return pd.read_csv(path)

def clean(df, max_len, seed, n=None):
    df = df[['input', 'dssp3']].dropna() # fancy indexing to isolate only input and dssp3 cols, drop examples with at least 1 NaN
    df = df.loc[df['input'].str.len() == df['dssp3'].str.len(), :] # filter for rows where len of input primary seq is == len of target seq
    df = df.loc[df['input'].str.len() <= max_len, :] # filter for rows where input primary seq len is <= MAX_LEN for computational efficiency
    
    if n is not None and len(df) > n: # limit the total no. of training samples to a reasonable no.
        df = df.sample(n, random_state=seed)

    return df.reset_index(drop=True) # reset row int indices and return cleaned df

def encode_seq(input, target, input_tokenizer, target_tokenizer, ignore_idx, x_idx):
    #input: input str of AA characters to be encoded into int indices from a single example
    #target: input str of state labels to be encoded into int indices from a single target
    
    x = np.array([input_tokenizer.get(aa, x_idx) for aa in input], dtype=np.int64)
    y = np.array([target_tokenizer.get(label, ignore_idx) for label in target],  dtype=np.int64)

    return x, y

def collate(batch, pad_idx, ignore_idx):
    lengths = torch.tensor([len(input) for input, _ in batch])
    max_len = int(lengths.max())

    X = torch.full((len(batch), max_len), pad_idx,  dtype=torch.long) # tensor of (batch_size, max_len) where elements are all padding token idx
    Y = torch.full((len(batch), max_len), ignore_idx, dtype=torch.long) # tensor of (batch_size, max_len) where elements are all ignore idx
    is_padding = torch.ones((len(batch), max_len), dtype=torch.bool) # tensor of (batch_size, max_len) where elements are True/all padding

    for idx, (input, target) in enumerate(batch):
        X[idx, :len(input)] = input
        Y[idx, :len(target)] = target
        is_padding[idx, :len(input)] = False # set postions with real residue to False

    return X, Y, lengths, is_padding

def mlm_collate(batch, pad_idx, ignore_idx, mask_idx, x_idx, generator=None, p_mask=0.15, n_vocab=20):
    batch_size = len(batch)
    lengths = torch.tensor([len(input) for input in batch]) # collects a list of int length per example in mini-batch (batch), casting it into a 1d-tensor
    max_len = lengths.max().item() # returns max seq len

    X = torch.full((batch_size, max_len), pad_idx, dtype=torch.long) # create tensor of matching dims as input batch with all padding token idx
    is_real = torch.zeros((batch_size, max_len), dtype=torch.bool) # start as all False, target to form a boolean mask of X where 1 for true AA residue, 0 otherwise
    
    for idx, input in enumerate(batch): # iterating per EXAMPLES in a given mini-batch (batch)
        X[idx, :len(input)] = input
        is_real[idx, :len(input)] = True 

    dice = torch.rand(batch_size, max_len, generator=generator) # a random number per position, torch.rand samples each element from [0, 1)
    chosen = (dice < p_mask) & is_real & (X != x_idx) # bool mask where 1 indicates positions chosen to be masked/corrupted/retained; 0 for unmasked, unknown residue X or padding tokens

    Y = torch.full((batch_size, max_len), ignore_idx, dtype=torch.long) # create tensor of matching dims as input batch (which should match targets) with all ignore idx (eventually to mark unmasked or padding token positions)
    Y[chosen] = X[chosen] # replace positions that were chosen to be masked/corrupted/retained with actual AA residue idx as labels

    dice2 = torch.rand(batch_size, max_len, generator=generator)
    to_mask = chosen & (dice2 < 0.8) # bool mask where 1 indicates chosen positions that are to be masked
    to_corrupt = chosen & (dice2 >= 0.8) & (dice2 < 0.9)  # bool mask where 1 indicates chosen positions that are to be corrupted
    total_corrupt = int(to_corrupt.sum().item()) # total no. of chosen positions picked to be corrupted

    X[to_mask] = mask_idx
    X[to_corrupt] = torch.randint(0, n_vocab, size=(total_corrupt, ), generator=generator) # create a 1d tensor of sampled ints between 0 to 20, element-wise allocation into chosen positions picked for corrupting
    
    is_padding = ~is_real # to be passed as input into src_key_padding_mask

    return X, Y, is_padding # (batch_size, max_len)


class MLMDataset(Dataset): # same generic custom dataset class definition for padded sequence data
    def __init__(self, df, input_tokenizer, x_idx):
        self.items = [np.array([input_tokenizer.get(aa, x_idx) for aa in input], dtype=np.int64) for input in df['input']] # list of 1D-arrays where elements are idx encoded aa per input seq str
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.items[idx]) # 1D-tensor where elements are idx encoded aa per input seq str

class SecondaryStructureDataset(Dataset):
    def __init__(self, df, input_tokenizer, target_tokenizer, ignore_idx, x_idx):
        self.items = [encode_seq(input, target, input_tokenizer, target_tokenizer, ignore_idx, x_idx) for input, target in zip(df['input'], df['dssp3'])] # 1D-numpy arrays where elements are idx encoded AA or state labels
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        x, y = self.items[idx]

        return torch.from_numpy(x), torch.from_numpy(y) # tuple of 1D-tensors where elements are idx encoded AA or state labels