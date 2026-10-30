import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report, ConfusionMatrixDisplay

def plot_learning_curves(train_epoch_losses, val_epoch_losses, title: str, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    X = np.arange(1, len(train_epoch_losses) + 1)
    plt.plot(X, val_epoch_losses, label='Val')
    plt.plot(X, train_epoch_losses, label='Train')
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    plt.close()

def cm_and_class_report(best_true, best_preds, target_tokenizer, title):
    ConfusionMatrixDisplay.from_predictions(best_true, best_preds, cmap='Blues', normalize='true', labels=list(target_tokenizer.values()), display_labels=list(target_tokenizer.keys()))
    plt.title(title)
    plt.show()
    plt.close()

    print(classification_report(best_true, best_preds, labels=list(target_tokenizer.values()), target_names=list(target_tokenizer.keys())))