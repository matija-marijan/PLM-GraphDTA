import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import json
from .deepfrier.Predictor import Predictor
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from collections import OrderedDict

model_config = 'preprocessing/FRI/trained_models/model_config.json'
ont = 'mf'
emb_layer = 'global_max_pooling1d'

with open(model_config) as json_file:
    params = json.load(json_file)

params = params['cnn']
gcn = params['gcn']
models = params['models']
predictor = Predictor(models[ont], gcn = gcn)

datasets = ['davis', 'kiba', 'davis_mutation']
for dataset in datasets:
    embeddings = []
    processed_dataset = 'data/' + dataset + '/proteins_deepfri_' + ont + '.json'
    if not os.path.isfile(processed_dataset):

        predictor = Predictor(models[ont], gcn=gcn)
        
        proteins = json.load(open('data/' + dataset + "/proteins.json"), object_pairs_hook=OrderedDict)
        prots = []
        protein_keys = []
        for t in proteins.keys():
            prots.append(proteins[t])
            protein_keys.append(t)

        # DeepFRI protein representation
        for i in tqdm(range(0, len(prots)), desc=f"Processing {dataset} proteins"):
            prot = prots[i]
            emb = predictor.predict_embeddings(prot, layer_name = emb_layer)
            embeddings.append(emb)

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