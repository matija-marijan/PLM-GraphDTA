import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn

from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet

from models.pdc_ginconv import PDC_GINConvNet
from models.vnoc_ginconv import Vnoc_GINConvNet
from models.pdc_vnoc_ginconv import PDC_Vnoc_GINConvNet
from models.plm_ginconv import PLM_GINConvNet
from models.pdconv_ginconv import PDConv_GINConvNet
from models.pdconv_vnoc_ginconv import PDConv_Vnoc_GINConvNet

from models.esm_gat import ESM_GATNet

import wandb
import random
from utils import *
import argparse
from tqdm import tqdm
import json

scaler = torch.cuda.amp.GradScaler()

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, wandb_log=False):
    # print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    # for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, desc=f"Epoch {epoch}"):
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    tqdm.write('Train loss: {:.6f}'.format(loss.item()))
    if wandb_log:
        wandb.log({"loss": loss.item()}, commit=False)

def predicting(model, device, loader):
    model.eval()
    total_preds = []
    total_labels = []
    # print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        # for data in tqdm(loader, total=len(loader), leave=False, desc="Predicting"):
        for data in loader:
            data = data.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                output = model(data)
            total_preds.append(output.cpu())
            total_labels.append(data.y.view(-1, 1).cpu())
            
    total_preds = torch.cat(total_preds, dim=0)
    total_labels = torch.cat(total_labels, dim=0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

datasets = ['davis', 'kiba', 'davis_mutation']

all_models = {
    'GINConvNet': GINConvNet, 
    'GATNet': GATNet, 
    'GAT_GCN': GAT_GCN, 
    'GCNNet': GCNNet,

    'ESM_GATNet': ESM_GATNet,
    'PLM_GINConvNet': PLM_GINConvNet,
    'Vnoc_GINConvNet': Vnoc_GINConvNet,    

    'PDC_GINConvNet': PDC_GINConvNet, 
    'PDC_Vnoc_GINConvNet': PDC_Vnoc_GINConvNet,

    'PDConv_GINConvNet': PDConv_GINConvNet,
    'PDConv_Vnoc_GINConvNet': PDConv_Vnoc_GINConvNet
}

parser = argparse.ArgumentParser(description="Run a specific model on a specific dataset.")

parser.add_argument('--dataset', type=str, choices=datasets, required=True, 
                    help="Dataset name: 'davis' or 'kiba'.")
parser.add_argument('--model', type=str, choices=list(all_models.keys()), required=True, 
                    help="Model name. Choose from: " + ", ".join(all_models.keys()) + ".")

parser.add_argument('--cuda', type=int, default=0, 
                    help="CUDA device index (default: 0).")
parser.add_argument('--seed', type=int, default=None,
                    help="Random seed for reproducibility (default: None).")
parser.add_argument('--wandb', action='store_true', default=False,
                    help="Flag for using wandb logging (default: False).")
parser.add_argument('--validation_fold', type=int, default=0,
                    help="Fold index to use for validation when using k-fold cross-validation (default: 0).")

parser.add_argument('--plm_layers', type=int, nargs='+', default=[256, 192, 128],
                    help="List of layer sizes for the protein language model embedding branch (default: [320, 256, 128]).")
parser.add_argument('--conv_layers', type=int, nargs='+', default=[32, 64, 96],
                    help="List of filter sizes for the convolutional layers in the drug graph channel (default: [32, 64, 96]).")
parser.add_argument('--kernel_size', type=int, default=8,
                    help="Convolution filter kernel size for convolutional models (default: 8)")

parser.add_argument('--description', type=str, default=None,
                    help="Description to add to run and/or group name for logging (default: None).")
parser.add_argument('--protein_embedding_type', type=str, default=None,
                    help="Type of precomputed protein embeddings (default: None).")

args = parser.parse_args()

modeling = all_models[args.model]
model_st = modeling.__name__

dataset = args.dataset
# split_type = args.split_type

# Select CUDA device if applicable
cuda_name = f"cuda:{args.cuda}"
print('cuda_name:', cuda_name)

# Set seed:
if args.seed is not None:
    seed = args.seed
    print("Seed: " + str(seed))
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000
NUM_WORKERS = 24

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

group_name = f"{args.model}_{args.dataset}_plm_{args.plm_layers}_conv{args.conv_layers}_kernel_{args.kernel_size}"
run_name = f"{args.model}_{args.dataset}_plm_{args.plm_layers}_conv{args.conv_layers}_kernel_{args.kernel_size}"
if args.description is not None:
    run_name += f"_desc_{args.description}"
    group_name += f"_desc_{args.description}"
if args.seed is not None:
    run_name += f"_seed_{args.seed}"
    group_name += f"_seed_{args.seed}"
run_name += f"_fold_{args.validation_fold}"

if args.wandb:
    wandb.init(project = 'E-GraphDTA - Validation', config = args, group = group_name, name = run_name)

if args.protein_embedding_type is not None:
    protein_emb_path = f"data/{dataset}/proteins_{args.protein_embedding_type}.json"
    with open(protein_emb_path, "r") as f:
        protein_emb_data = json.load(f)
    first_emb = protein_emb_data[0]["embedding"]
    embed_dim = len(first_emb)
else:
    embed_dim = 128  # default embedding dimension if no precomputed embeddings are used
    
# Main program: Train on specified dataset 
if __name__ == "__main__":
    print('Training ' + model_st + ' on ' + dataset + ' dataset...')
    dta_dataset = DTADataset(root='data', dataset=dataset, protein_embedding_type=args.protein_embedding_type)

    # original k-fold split (hard coded!)
    all_folds = [0, 1, 2, 3, 4]
    val_fold = args.validation_fold
    train_folds = [f for f in all_folds if f != val_fold]
    
    train_mask = torch.isin(dta_dataset._data.fold, torch.tensor(train_folds))
    train_dataset = dta_dataset[train_mask]
    val_dataset = dta_dataset[dta_dataset._data.fold == val_fold]
    test_dataset = dta_dataset[dta_dataset._data.fold == -1]

    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, pin_memory = True, num_workers = NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, pin_memory = True, num_workers = NUM_WORKERS//2)
    val_loader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, pin_memory = True, num_workers = NUM_WORKERS//2)

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling(embed_dim = embed_dim, plm_layers = args.plm_layers, conv_layers = args.conv_layers, kernel_size = args.kernel_size).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_mse = 1000
    best_epoch = -1
    
    model_file_name = 'trained_models/model_' + run_name + '_validation.pt'
    result_file_name = 'trained_models/result_' + run_name + '_validation.csv'
    os.makedirs('trained_models', exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        tqdm.write(f'\nEpoch {epoch+1}')
        train(model, device, train_loader, optimizer, epoch+1, wandb_log=args.wandb)

        G,P = predicting(model, device, val_loader)
        ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P)]
        if args.wandb:
            wandb.log({"rmse": ret[0], "mse": ret[1], "pearson": ret[2], "spearman": ret[3]})

        if ret[1]<best_mse:
            torch.save(model.state_dict(), model_file_name)
            best_epoch = epoch+1
            best_mse = ret[1]
        tqdm.write(f'Validation MSE: {ret[1]:.6f}\nBest MSE: {best_mse:.6f} (epoch {best_epoch})')

    model.load_state_dict(torch.load(model_file_name))

    G,P = predicting(model, device, val_loader)
    val_ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]

    tqdm.write('\nResults on val set:')
    tqdm.write(f"RMSE: {val_ret[0]}")
    tqdm.write(f"MSE: {val_ret[1]}")
    tqdm.write(f"Pearson: {val_ret[2]}")
    tqdm.write(f"Spearman: {val_ret[3]}")
    tqdm.write(f"CI: {val_ret[4]}\n")

    if args.wandb:
        wandb.log({
            "val_rmse": val_ret[0],
            "val_mse": val_ret[1],
            "val_pearson": val_ret[2],
            "val_spearman": val_ret[3],
            "val_ci": val_ret[4]            
        })
        wandb.finish()

    with open(result_file_name, 'w') as f:
        # write header
        f.write("val_rmse,val_mse,val_pearson,val_spearman,val_ci\n")
        # write values
        f.write(','.join(map(str, val_ret)))
        f.write('\n')
    