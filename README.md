# Protein Secondary Structure Prediction with MLM Pre-training

Three-state (Q3: helix `H`, strand `E`, coil `C`) protein secondary structure
prediction from amino acid sequence, using self-supervised masked-language-model
(MLM) pre-training followed by a lightweight supervised tagger.

## Approach

1. **Data.** Sequences and DSSP3 labels come from the Hugging Face dataset
   `proteinea/secondary_structure_prediction`. A ~3,000-sequence subset of
   `training_hhblits.csv` (length ≤ 512) is split into 2,000 training and
   1,000 validation sequences due to considerations for computational cost.
   `CB513.csv` was held out as the test set.
2. **MLM pre-training.** A small Transformer encoder (`ProteinMLM`: sinusoidal
   positional encodings, pre-LayerNorm layers) was trained with BERT-style masking.
   15% of residues were selected and then replaced with `[MASK]` (80%), a random
   residue (10%) or left unchanged (10%). Unknown residues (`X`) and padding were
   never chosen as targets. The loss and masked-token accuracy were computed only
   on the selected positions. Training masks were redrawn every epoch, while
   validation masks are fixed by a seeded generator so the metric is comparable
   across epochs and trials.
3. **Supervised tagging.** The pre-trained encoder was frozen and feeds a CNN
   (odd kernel sizes, per-layer LayerNorm, padding positions zeroed) followed by a
   BiLSTM and a linear classification head, which predicts a label for every
   residue. Because the classes were imbalanced, the cross-entropy loss was
   weighted by class frequencies computed on the training split only. The best
   epoch was chosen by validation MCC.
4. **Hyperparameter search.** Both models were tuned with Optuna (TPE sampler,
   median pruner). The MLM search covers model width, heads, depth, learning rate
   and dropout. The tagger search covers kernel size, hidden size, conv depth,
   learning rate, dropout and weight decay (search space defined for architectural
   hyperparameters were narrow around a shallow task head architecture due to 
   observed issues with overfitting).
5. **Evaluation.** The final tagger was evaluated on CB513 with accuracy, MCC, a
   normalised confusion matrix and a per-class classification report.

## Project Organization

```
├── LICENSE            
├── Makefile           
├── README.md          
├── data
│   ├── external       
│   ├── interim        
│   ├── processed      
│   └── raw            
│
├── docs               
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks
│
├── pyproject.toml     
│
├── references         
│
├── reports            
│   └── figures        
│
├── requirements.txt   <- requirements file for reproducing the analysis environment generated with `pip freeze > requirements.txt`
│
├── setup.cfg          
│
└── protein_ss   <- Source code for use in this project.
    │
    ├── __init__.py             
    │
    ├── config.py               <- Store configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

