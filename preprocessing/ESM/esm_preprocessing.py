import os
import pandas as pd
import numpy as np
import esm
from rdkit import Chem
import networkx as nx
from utils import *
import argparse
from tqdm import tqdm
import json
from collections import OrderedDict

# model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
final_layer = model.num_layers
embed_dim = model.embed_dim

if torch.cuda.is_available():
    model = model.cuda()
batch_converter = alphabet.get_batch_converter()

datasets = ['davis', 'kiba', 'davis_mutation']
for dataset in datasets:
    embeddings = []
    processed_dataset = 'data/' + dataset + '/proteins_esm_' + str(embed_dim) + '.json'

    if not os.path.isfile(processed_dataset):
        proteins = json.load(open('data/' + dataset + "/proteins.json"), object_pairs_hook=OrderedDict)
        prots = []
        protein_keys = []
        for t in proteins.keys():
            prots.append(proteins[t])
            protein_keys.append(t)

        protein_list = list(zip(protein_keys, prots))
        batch_size = 4 if dataset == 'davis' else 1
        labels = []

        for i in tqdm(range(0, len(protein_list), batch_size), desc=f"Processing {dataset} proteins"):

            batch_prots = protein_list[i : i + batch_size]

            batch_labels, batch_strs, batch_tokens = batch_converter(batch_prots)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            with torch.no_grad():
                if torch.cuda.is_available():
                    results = model(batch_tokens.cuda(), repr_layers=[final_layer])
                else:
                    results = model(batch_tokens, repr_layers = [final_layer])
            token_representations = results["representations"][final_layer]

            sequence_representations = []
            for j, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[j, 1 : tokens_len - 1].mean(0).cpu())

            for j in range(0, len(sequence_representations)):
                embeddings.append(sequence_representations[j])

        embeddings = np.asarray(embeddings)

        df = pd.DataFrame({
            'protein_key': protein_keys,
            'sequence': prots,
            'embedding': [emb.tolist() for emb in embeddings]
        })
        df.to_json(processed_dataset, orient='records', indent=4)

        print(processed_dataset, ' has been created')
    else:
        print(processed_dataset, ' is already created')