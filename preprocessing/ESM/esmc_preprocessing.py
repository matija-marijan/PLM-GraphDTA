import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import json
from collections import OrderedDict
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Choose between esmc_300m and esmc_600m
model = ESMC.from_pretrained("esmc_600m").to(device)   
embed_dim = model.embed.embedding_dim

datasets = ['davis', 'kiba', 'davis_mutation']

for dataset in datasets:
    all_embeddings = []
    processed_dataset = 'data/' + dataset + '/proteins_esmc_' + embed_dim + '.json'
    
    if not os.path.isfile(processed_dataset):
        proteins = json.load(open('data/' + dataset + "/proteins.json"), object_pairs_hook=OrderedDict)
        prots = []
        protein_keys = []
        for t in proteins.keys():
            prots.append(proteins[t])
            protein_keys.append(t)
        
        for prot in tqdm(prots, desc=f"Processing {dataset} proteins"):
            esm_prot = ESMProtein(sequence=prot)
            protein_tensor = model.encode(esm_prot)
            logits_output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            
            embeddings = logits_output.embeddings
            protein_embedding = embeddings.mean(dim=1)
            protein_embedding = protein_embedding.squeeze(0)
            
            all_embeddings.append(protein_embedding.cpu().numpy())
            
        all_embeddings = np.asarray(all_embeddings)
        
        df = pd.DataFrame({
            'protein_key': protein_keys,
            'sequence': prots,
            'embedding': [emb.tolist() for emb in all_embeddings]
        })
            
        df.to_json(processed_dataset, orient='records', indent=4)
        
        print(processed_dataset, ' has been created')
    else:
        print(processed_dataset, ' is already created')
