#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:27:44 2023

@author: mheinzinger
"""

import argparse
import time
from pathlib import Path
import os
import json
from collections import OrderedDict
from tqdm import tqdm
import torch
from transformers import T5EncoderModel, T5Tokenizer
import pandas as pd
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print("Using device: {}".format(device))


def get_T5_model(model_dir):
    print("Loading T5 from: {}".format(model_dir))
    model = T5EncoderModel.from_pretrained(model_dir).to(device)
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False )
    return model, vocab


def read_fasta( fasta_path, split_char, id_field, is_3Di ):
    '''
        Reads in fasta file containing multiple sequences.
        Returns dictionary of holding multiple sequences or only single 
        sequence, depending on input file.
    '''
    
    sequences = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                sequences[ uniprot_id ] = ''
            else:
                # repl. all white-space chars and join seqs spanning multiple lines
                if is_3Di:
                    sequences[ uniprot_id ] += ''.join( line.split() ).replace("-","").lower() # drop gaps and cast to upper-case
                else:
                    sequences[ uniprot_id ] += ''.join( line.split() ).replace("-","")
                    
    return sequences


def get_embeddings( seq_path, emb_path, model_dir, split_char, id_field, 
                       per_protein, half_precision, is_3Di,
                       max_residues=4000, max_seq_len=1000, max_batch=100 ):
    
    seq_dict = dict()
    emb_dict = dict()  # maps protein id -> 1D embedding vector (mean pooled)
    protein_keys = []  # collected only for successfully embedded proteins
    prots = []         # original cleaned sequences (match ordering of embeddings)

    # Read in fasta
    seq_dict = read_fasta( seq_path, split_char, id_field, is_3Di )
    prefix = "<fold2AA>" if is_3Di else "<AA2fold>"
    
    model, vocab = get_T5_model(model_dir)
    if half_precision:
        model = model.half()
        print("Using model in half-precision!")

    print('########################################')
    print(f"Input is 3Di: {is_3Di}")
    print('Example sequence: {}\n{}'.format( next(iter(
            seq_dict.keys())), next(iter(seq_dict.values()))) )
    print('########################################')
    print('Total number of sequences: {}'.format(len(seq_dict)))

    avg_length = sum([ len(seq) for _, seq in seq_dict.items()]) / len(seq_dict)
    n_long     = sum([ 1 for _, seq in seq_dict.items() if len(seq)>max_seq_len])
    # sort sequences by length to trigger OOM at the beginning
    seq_dict   = sorted( seq_dict.items(), key=lambda kv: len( seq_dict[kv[0]] ), reverse=True )
    
    print("Average sequence length: {}".format(avg_length))
    print("Number of sequences >{}: {}".format(max_seq_len, n_long))
    
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict,1):
        # replace non-standard AAs
        seq_clean = seq.replace('U','X').replace('Z','X').replace('O','X')
        seq_len = len(seq_clean)
        seq_tokenized = prefix + ' ' + ' '.join(list(seq_clean))
        batch.append((pdb_id,seq_tokenized,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed 
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len 
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            token_encoding = vocab.batch_encode_plus(seqs, 
                                                     add_special_tokens=True, 
                                                     padding="longest", 
                                                     return_tensors='pt' 
                                                     ).to(device)
            try:
                with torch.no_grad():
                    embedding_repr = model(token_encoding.input_ids, 
                                           attention_mask=token_encoding.attention_mask
                                           )
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(
                    pdb_id, seq_len)
                    )
                continue
            
            # batch-size x seq_len x embedding_dim
            # extra token is added at the end of the seq
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                # account for prefix in offset
                emb = embedding_repr.last_hidden_state[batch_idx,1:s_len+1]
                
                # Always mean-pool to create a single per-protein embedding vector
                # (matches JSON format used in other preprocessing scripts)
                emb_vector = emb.mean(dim=0)
                emb_np = emb_vector.detach().cpu().numpy().squeeze()
                emb_dict[ identifier ] = emb_np
                # Record only after success so lengths align
                protein_keys.append(identifier)
                # retrieve original cleaned sequence length s_len corresponds to seq_clean
                # but we need the actual sequence string; reconstruct from tokenized by removing prefix/spaces
                # We have seq_clean earlier before tokenization; rebuild from attention mask isn't straightforward here,
                # so instead store seq_clean by slicing from seqs based on batch index
                # Since we constructed seq_tokenized from seq_clean with prefix + spaced chars, we can reverse:
                seq_clean_recovered = seqs[batch_idx].replace(prefix + ' ', '').replace(' ', '')
                prots.append(seq_clean_recovered)
                if len(emb_dict) == 1:
                    print("Example: embedded protein {} with length {} to emb. of shape: {}".format(
                                identifier, s_len, emb_vector.shape))

    end = time.time()
    
    # Build DataFrame and save as JSON matching format of ESM/DeepFRI preprocessing
    all_embeddings = [emb_dict[k] for k in protein_keys]
    all_embeddings = np.asarray(all_embeddings)
    df = pd.DataFrame({
        'protein_key': protein_keys,
        'sequence': prots,
        'embedding': [emb.tolist() for emb in all_embeddings]
    })
    # Ensure output has .json extension; if not, we still write JSON
    emb_path_json = Path(str(emb_path))
    if emb_path_json.suffix != '.json':
        print(f"Warning: output path {emb_path_json} does not end with .json; writing JSON anyway.")
    df.to_json(emb_path_json, orient='records', indent=4)
    print(f"Saved JSON embeddings to {emb_path_json}")

    print('\n############# STATS #############')
    print('Total number of embeddings: {}'.format(len(emb_dict)))
    print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format( 
            end-start, (end-start)/len(emb_dict), avg_length))
    return True


def create_arg_parser():
    """Deprecated: CLI-based FASTA interface no longer used in this repository pipeline.
    Kept for backward compatibility but not invoked.
    """
    parser = argparse.ArgumentParser(description='Deprecated CLI - use JSON pipeline')
    return parser

def main():
    # Match ESM/ESMC/DeepFRI pipeline: read JSON proteins and write JSON embeddings per dataset
    model_dir = "Rostlab/ProstT5"
    model, vocab = get_T5_model(model_dir)
    model.eval()
    embed_dim = getattr(model.config, 'd_model', None)
    if embed_dim is None:
        # fallback from an example forward pass
        embed_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 1024

    prefix = "<AA2fold>"

    datasets = ['davis', 'kiba', 'davis_mutation']
    for dataset in datasets:
        all_embeddings = []
        processed_dataset = f'data/{dataset}/proteins_prost_{embed_dim}.json'

        if not os.path.isfile(processed_dataset):
            proteins = json.load(open(f'data/{dataset}/proteins.json'), object_pairs_hook=OrderedDict)
            prots = []
            protein_keys = []
            for t in proteins.keys():
                prots.append(proteins[t])
                protein_keys.append(t)

            for prot in tqdm(prots, desc=f"Processing {dataset} proteins (PROST)"):
                # Clean and tokenize
                seq_clean = prot.replace('U','X').replace('Z','X').replace('O','X')
                seq_len = len(seq_clean)
                seq_tokenized = prefix + ' ' + ' '.join(list(seq_clean))

                token_encoding = vocab.batch_encode_plus([seq_tokenized],
                                                         add_special_tokens=True,
                                                         padding="longest",
                                                         return_tensors='pt').to(device)
                with torch.no_grad():
                    outputs = model(token_encoding.input_ids, attention_mask=token_encoding.attention_mask)
                hidden = outputs.last_hidden_state  # [1, L, D]
                emb = hidden[0, 1:seq_len+1]       # drop prefix token representation
                emb_vector = emb.mean(dim=0).squeeze(0)
                all_embeddings.append(emb_vector.detach().cpu().numpy())

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


if __name__ == '__main__':
    main()