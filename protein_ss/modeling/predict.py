import torch
import numpy as np

from torchmetrics.classification import MatthewsCorrCoef

def cnn_bilstm_eval(model, test_loader, ignore_index, target_tokenizer, device, plot_report_fn, title, num_classes=3):
    mcc_score = MatthewsCorrCoef(task='multiclass', num_classes=num_classes).to(device)

    model.eval()
    test_corrects = 0
    test_total = 0
    mcc_score.reset()
    with torch.no_grad():
        true_list = []
        preds_list = []
        for X, Y, lengths, is_padding in test_loader:
            X, Y, lengths, is_padding = X.to(device), Y.to(device), lengths.to(device), is_padding.to(device)
            outputs = model(X, lengths, is_padding)
            preds = outputs.argmax(-1)

            scored = (Y != ignore_index)
            n = scored.sum().item()
            test_corrects += preds[scored].eq(Y[scored]).sum().item()
            test_total += n
            mcc_score.update(preds[scored], Y[scored])

            true_list.append(Y[scored].detach().cpu().numpy())
            preds_list.append(preds[scored].detach().cpu().numpy())

    eval_metric = mcc_score.compute().item()
    test_acc = test_corrects / test_total
    best_true = np.concatenate(true_list)
    best_preds = np.concatenate(preds_list)

    print(f"Test Set MCC: {eval_metric:.3f}\nTest Set Acc: {test_acc:.3f}\n")

    plot_report_fn(best_true, best_preds, target_tokenizer, title)