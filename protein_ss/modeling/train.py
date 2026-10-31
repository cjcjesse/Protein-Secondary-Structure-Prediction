import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import protein_ss.modeling.classes as ss_classes
import optuna

from tqdm.auto import tqdm
from torchmetrics.classification import Accuracy, MatthewsCorrCoef
from protein_ss.config import MODELS_DIR

def objective(trial, train_loader, val_loader, max_len, mlm_vocab, epochs, padding_idx, ignore_idx, seed, device, generator, weight_decay=1e-3, n_vocab=20):
    try:
        torch.manual_seed(seed) # reset seed order per trial to remove training noise
        
        d_model = trial.suggest_categorical('d_model', [128, 256, 512])
        nhead = trial.suggest_categorical('nhead', [n for n in range(2, 5) if d_model % n == 0]) # between 2 to 4 nheads, return only those that d_model can be cleanly divisible by
        num_layers = trial.suggest_int('num_layers', 2, 4)
        lr = trial.suggest_float('lr', 1e-4, 2e-3, log=True)
        dropout = trial.suggest_float('dropout', 0.0, 0.3)
        dim_feedforward = d_model * 2
        
        model = ss_classes.ProteinMLM(d_model, nhead, num_layers, dim_feedforward, max_len, dropout, mlm_vocab, padding_idx, n_vocab).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

        best_eval_metric = float('-inf')
        acc_score = Accuracy(task='multiclass', num_classes=n_vocab, average='micro').to(device)

        #for epoch in tqdm(range(epochs), desc='Epochs', leave=False):
        for epoch in range(epochs):
            model.train()
            for X, Y, pad in train_loader:
                masked = Y != ignore_idx # bool mask where 1 if not padding or unmasked
                if masked.sum() == 0:
                    continue # check if mini-batch by chance has no chosen position for masking/corrupting/retention, continue to next mini-batch if so

                X, Y, pad = X.to(device), Y.to(device), pad.to(device) # (batch_size, max_len, d_model), (batch_size, max_len), (batch_size, max_len)
                optimizer.zero_grad()
                outputs = model(X, pad) # (batch_size, max_len, n_vocab)
                loss = criterion(outputs.reshape(-1, n_vocab), Y.reshape(-1)) # reshape to: (batch_size * max_len, n_vocab), (batch_size * max_len, )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            acc_score.reset()
            with torch.no_grad():
                generator.manual_seed(seed)
                for X, Y, pad in val_loader:
                    masked = Y != ignore_idx # bool mask where 1 if not padding or unmasked
                    if masked.sum() == 0:
                        continue # check if mini-batch by chance has no chosen position for masking/corrupting/retention, continue to next mini-batch if so
                    
                    X, Y, pad = X.to(device), Y.to(device), pad.to(device) # (batch_size, max_len, d_model), (batch_size, max_len), (batch_size, max_len)
                    outputs = model(X, pad) # (batch_size, max_len, n_vocab)
                    preds = outputs.argmax(-1)
                    acc_score.update(preds[masked], Y[masked])

            epoch_eval_metric = acc_score.compute().item()

            if epoch_eval_metric > best_eval_metric:
                best_eval_metric = epoch_eval_metric

            scheduler.step(epoch_eval_metric)

            trial.report(epoch_eval_metric, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_eval_metric
    finally:
        if 'model' in locals():
            del model
        if 'optimizer' in locals():
            del optimizer
        torch.cuda.empty_cache()

def train_val_loop(train_loader, val_loader, d_model, nhead, num_layers, lr, dropout, max_len, mlm_vocab, epochs, padding_idx, ignore_idx, seed, device, generator, weight_decay=1e-3, n_vocab=20):
    state_dict_path = MODELS_DIR / './ss_proj_mlm.pt'
    dim_feedforward = d_model * 2
    model = ss_classes.ProteinMLM(d_model, nhead, num_layers, dim_feedforward, max_len, dropout, mlm_vocab, padding_idx, n_vocab).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    best_eval_metric = float('-inf')
    train_epoch_losses = []
    val_epoch_losses = []
    acc_score = Accuracy(task='multiclass', num_classes=n_vocab, average='micro').to(device)

    with tqdm(range(epochs), desc='Epochs', leave=False) as pbar:
        for epoch in pbar:
            model.train()
            train_loss = 0.0
            val_loss = 0.0
            train_total = 0
            val_total = 0

            for X, Y, pad in train_loader:
                masked = Y != ignore_idx # bool mask where 1 if not padding or unmasked
                if masked.sum() == 0:
                    continue # check if mini-batch by chance has no chosen position for masking/corrupting/retention, continue to next mini-batch if so

                X, Y, pad = X.to(device), Y.to(device), pad.to(device) # (batch_size, max_len, d_model), (batch_size, max_len), (batch_size, max_len)
                optimizer.zero_grad()
                outputs = model(X, pad) # (batch_size, max_len, n_vocab)
                loss = criterion(outputs.reshape(-1, n_vocab), Y.reshape(-1)) # reshape to: (batch_size * max_len, n_vocab), (batch_size * max_len, )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * masked.sum().item()
                train_total += masked.sum().item()

            model.eval()
            acc_score.reset()
            with torch.no_grad():
                generator.manual_seed(seed)
                for X, Y, pad in val_loader:
                    masked = Y != ignore_idx # bool mask where 1 if not padding or unmasked
                    if masked.sum() == 0:
                        continue # check if mini-batch by chance has no chosen position for masking/corrupting/retention, continue to next mini-batch if so
                    
                    X, Y, pad = X.to(device), Y.to(device), pad.to(device) # (batch_size, max_len, d_model), (batch_size, max_len), (batch_size, max_len)
                    outputs = model(X, pad) # (batch_size, max_len, n_vocab)
                    loss = criterion(outputs.reshape(-1, n_vocab), Y.reshape(-1)) # reshape to: (batch_size * max_len, n_vocab), (batch_size * max_len, )
                    preds = outputs.argmax(-1)
        
                    val_loss += loss.item() * masked.sum().item()
                    val_total += masked.sum().item()
                    acc_score.update(preds[masked], Y[masked])

            epoch_eval_metric = acc_score.compute().item()
            train_epoch_loss = train_loss / train_total
            val_epoch_loss = val_loss / val_total
            train_epoch_losses.append(train_epoch_loss)
            val_epoch_losses.append(val_epoch_loss)

            if (epoch == 0) or ((epoch + 1) % 5 == 0) or (epoch == (epochs - 1)):
                pbar.write(f"Epoch: {epoch + 1}\n\tTrain Loss: {train_epoch_loss:.3f}\n\tVal Loss: {val_epoch_loss:.3f}\n\tMasked Token Accuracy: {epoch_eval_metric:.3f}\n")

            if epoch_eval_metric > best_eval_metric:
                best_eval_metric = epoch_eval_metric
                best_epoch = epoch
                torch.save(model.state_dict(), state_dict_path)

            scheduler.step(epoch_eval_metric)

    model.load_state_dict(torch.load(state_dict_path, weights_only=True))

    out_dict = {
        'model':model,
        'train_epoch_losses':train_epoch_losses,
        'val_epoch_losses':val_epoch_losses,
        'best_eval_metric':best_eval_metric,
        'best_epoch':best_epoch,
        'state_dict_path':state_dict_path
    }

    return out_dict

def cnn_bilstm_objective(trial, train_loader, val_loader, weights_tensor, frozen_encoder, epochs, ignore_index, seed, device, num_classes=3, pack=True):
    try:
        torch.manual_seed(seed)
        
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64])
        conv_layers = trial.suggest_int('conv_layers', 1, 2)
        lr = trial.suggest_float('lr', 1e-4, 2e-3, log=True)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)

        cnn_bilstm_tagger = ss_classes.CNNBiLSTMTagger(frozen_encoder, kernel_size, conv_layers, hidden_size, num_classes, dropout, pack).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=ignore_index)
        optimizer = optim.AdamW(cnn_bilstm_tagger.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

        best_eval_metric = float('-inf')
        mcc_score = MatthewsCorrCoef(task='multiclass', num_classes=num_classes).to(device)

        #for epoch in tqdm(range(epochs), desc='Epochs', leave=False):
        for epoch in range(epochs):
            cnn_bilstm_tagger.train()
            cnn_bilstm_tagger.encoder.eval()
            for X, Y, lengths, is_padding in train_loader:
                X, Y, lengths, is_padding = X.to(device), Y.to(device), lengths.to(device), is_padding.to(device)
                optimizer.zero_grad()
                outputs = cnn_bilstm_tagger(X, lengths, is_padding)
                loss = criterion(outputs.reshape(-1, num_classes), Y.reshape(-1, )) # (batch_size * max_len, num_classes), (batch_size * max_len, )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cnn_bilstm_tagger.parameters(), max_norm=1.0)
                optimizer.step()

            cnn_bilstm_tagger.eval()
            mcc_score.reset()
            with torch.no_grad():
                for X, Y, lengths, is_padding in val_loader:
                    X, Y, lengths, is_padding = X.to(device), Y.to(device), lengths.to(device), is_padding.to(device)
                    outputs = cnn_bilstm_tagger(X, lengths, is_padding)
                    preds = outputs.argmax(-1)
                    scored = (Y != ignore_index)
                    mcc_score.update(preds[scored], Y[scored])

            epoch_eval_metric = mcc_score.compute().item()
            
            if epoch_eval_metric > best_eval_metric:
                best_eval_metric = epoch_eval_metric
            
            scheduler.step(epoch_eval_metric)

            trial.report(epoch_eval_metric, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_eval_metric
    finally:
        if 'cnn_bilstm_tagger' in locals():
            del cnn_bilstm_tagger
        if 'optimizer' in locals():
            del optimizer
        torch.cuda.empty_cache()

def cnn_bilstm_train_val_loop(train_loader, val_loader, weights_tensor, frozen_encoder, epochs, ignore_index, device, kernel_size=3, hidden_size=64, conv_layers=2, lr=2e-3, dropout=0.1, weight_decay=1e-3, num_classes=3, pack=True):
    state_dict_path = MODELS_DIR / './ss_proj_cnn_bilstm.pt'

    cnn_bilstm_tagger = ss_classes.CNNBiLSTMTagger(frozen_encoder, kernel_size, conv_layers, hidden_size, num_classes, dropout, pack).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=ignore_index)
    optimizer = optim.AdamW(cnn_bilstm_tagger.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    best_eval_metric = float('-inf')
    train_epoch_losses = []
    val_epoch_losses = []
    mcc_score = MatthewsCorrCoef(task='multiclass', num_classes=num_classes).to(device)

    with tqdm(range(epochs), desc='Epochs', leave=False) as pbar:
        for epoch in pbar:
            cnn_bilstm_tagger.train()
            cnn_bilstm_tagger.encoder.eval()
            train_loss = 0.0
            val_loss = 0.0
            #train_corrects = 0
            val_corrects = 0
            train_total = 0
            val_total = 0

            for X, Y, lengths, is_padding in train_loader:
                X, Y, lengths, is_padding = X.to(device), Y.to(device), lengths.to(device), is_padding.to(device)
                optimizer.zero_grad()
                outputs = cnn_bilstm_tagger(X, lengths, is_padding)
                loss = criterion(outputs.reshape(-1, num_classes), Y.reshape(-1, )) # (batch_size * max_len, num_classes), (batch_size * max_len, )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cnn_bilstm_tagger.parameters(), max_norm=1.0)
                optimizer.step()

                scored = (Y != ignore_index)
                n = scored.sum().item()
                train_loss += loss.item() * n
                preds = outputs.argmax(-1)
                #train_corrects += preds[scored].eq(Y[scored]).sum().item()
                train_total += n

            cnn_bilstm_tagger.eval()
            mcc_score.reset()
            with torch.no_grad():
                true_list = []
                preds_list = []
                for X, Y, lengths, is_padding in val_loader:
                    X, Y, lengths, is_padding = X.to(device), Y.to(device), lengths.to(device), is_padding.to(device)
                    outputs = cnn_bilstm_tagger(X, lengths, is_padding)
                    loss = criterion(outputs.reshape(-1, num_classes), Y.reshape(-1, )) # (batch_size * max_len, num_classes), (batch_size * max_len, )
                    preds = outputs.argmax(-1)
        
                    scored = (Y != ignore_index)
                    n = scored.sum().item()
                    val_loss += loss.item() * n
                    val_corrects += preds[scored].eq(Y[scored]).sum().item()
                    val_total += n
                    mcc_score.update(preds[scored], Y[scored])

                    true_list.append(Y[scored].detach().cpu().numpy())
                    preds_list.append(preds[scored].detach().cpu().numpy())

            epoch_eval_metric = mcc_score.compute().item()
            train_epoch_loss = train_loss / train_total
            val_epoch_loss = val_loss / val_total
            #train_acc = train_corrects / train_total
            val_acc = val_corrects / val_total
            train_epoch_losses.append(train_epoch_loss)
            val_epoch_losses.append(val_epoch_loss)

            if (epoch == 0) or ((epoch + 1) % 5 == 0) or (epoch == (epochs - 1)):
                pbar.write(f"Epoch: {epoch + 1}\n\tTrain Loss: {train_epoch_loss:.3f}\n\tVal Loss: {val_epoch_loss:.3f}\n\tVal MCC: {epoch_eval_metric:.3f}\n\tVal Acc: {val_acc:.3f}\n")

            if epoch_eval_metric > best_eval_metric:
                best_eval_metric = epoch_eval_metric
                best_epoch_acc = val_acc
                best_true = np.concatenate(true_list)
                best_preds = np.concatenate(preds_list)
                best_epoch = epoch
                torch.save(cnn_bilstm_tagger.state_dict(), state_dict_path)

            scheduler.step(epoch_eval_metric)

    cnn_bilstm_tagger.load_state_dict(torch.load(state_dict_path, weights_only=True))

    out_dict = {
        'model':cnn_bilstm_tagger,
        'train_epoch_losses':train_epoch_losses,
        'val_epoch_losses':val_epoch_losses,
        'best_eval_metric':best_eval_metric,
        'best_epoch_acc':best_epoch_acc,
        'best_true':best_true,
        'best_preds':best_preds,
        'best_epoch':best_epoch,
        'state_dict_path':state_dict_path
    }

    return out_dict