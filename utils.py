import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric import data as DATA
import torch
import pandas as pd
from rdkit import Chem
import networkx as nx
from tqdm import tqdm
import json
import pickle
from collections import OrderedDict

class DTADataset(InMemoryDataset):
    def __init__(self, root: str = 'data', dataset: str = 'davis', target_type: str = None): #, mutation: bool = False): #, cluster_type: str = None):

        self.root = root
        self.dataset = dataset
        # self.cluster_type = cluster_type

        self.target_type = target_type
        # self.mutation = mutation

        super().__init__(root, transform=None, pre_transform=None)

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_file_names = os.path.join(self.root, self.dataset)
        # raw_file_names += f'_{self.cluster_type}' if self.cluster_type else ""
        # raw_file_names += f"_{self.target_type}" if self.target_type else ""
        # raw_file_names += '.csv'
        return [raw_file_names]

    @property
    def processed_file_names(self):
        processed_file_names = f"{self.dataset}"
        # processed_file_names += f"_{self.cluster_type}" if self.cluster_type else ""
        processed_file_names += f"_{self.target_type}" if self.target_type else ""
        # processed_file_names += f"_mutation.pt" if self.mutation else ".pt"
        processed_file_names += ".pt"
        return [processed_file_names]

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self):
        data_list = []

        fpath = self.raw_file_names[0] + '/'

        # Load fold information
        train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
        test_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))

        index_to_fold = {}
        for fold_num, fold_indices in enumerate(train_fold):
            for idx in fold_indices:
                index_to_fold[idx] = fold_num
        for idx in test_fold:
            index_to_fold[idx] = -1

        # Load ligands, proteins, protein encodings, and affinity data
        ligands = json.load(open(fpath + "drugs.json"), object_pairs_hook=OrderedDict)
        affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
        # if self.mutation and self.dataset == 'davis':
            # proteins = json.load(open(fpath + "proteins_mutation.json"), object_pairs_hook=OrderedDict)
        # else:
        proteins = json.load(open(fpath + "proteins.json"), object_pairs_hook=OrderedDict)

        # Load precomputed protein embeddings
        if self.target_type == 'deepfri' or self.target_type == 'esm':
            # if self.mutation and self.dataset == 'davis':
                # with open(fpath + f"proteins_{self.target_type}_mutation_.json", 'r') as f:
                    # protein_embeddings_file = {entry['protein_key']: entry for entry in json.load(f)}
            # else:
            with open(fpath + f"proteins_{self.target_type}.json", 'r') as f:
                protein_embeddings_file = {entry['protein_key']: entry for entry in json.load(f)}

        # Prepare lists of drugs, proteins, and their keys
        drugs = []
        prots = []
        ligand_keys = []
        protein_keys = []
        protein_encodings = []
        for d in ligands.keys():
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
            drugs.append(lg)
            ligand_keys.append(d)
        for t in proteins.keys():
            prots.append(proteins[t])
            protein_keys.append(t)
            if self.target_type == 'deepfri' or self.target_type == 'esm':
                entry = protein_embeddings_file[t]
                emb = entry['embedding']
                protein_encodings.append(emb)
        if "davis" in self.dataset:
            affinity = [-np.log10(y/1e9) for y in affinity]
        affinity = np.asarray(affinity)
        rows, cols = np.where(np.isnan(affinity)==False)  

        # Collect all protein-drug pairs for DataFrame
        affinity_rows = []
        for pair_ind in range(len(rows)):
            prot = prots[cols[pair_ind]]
            drug = drugs[rows[pair_ind]]
            prot_key = protein_keys[cols[pair_ind]]
            drug_key = ligand_keys[rows[pair_ind]]
            aff = affinity[rows[pair_ind], cols[pair_ind]]
            fold = index_to_fold.get(pair_ind, 'none')
            if self.target_type == 'deepfri' or self.target_type == 'esm':
                prot_encoding = protein_encodings[cols[pair_ind]]
                affinity_rows.append({
                    'target_sequence': prot,
                    'compound_iso_smiles': drug,
                    'affinity': aff,
                    'drug_key': drug_key,
                    'protein_key': prot_key,
                    'protein_encoding': prot_encoding,
                    'fold': fold
                })
            else:
                affinity_rows.append({
                    'target_sequence': prot,
                    'compound_iso_smiles': drug,
                    'affinity': aff,
                    'drug_key': drug_key,
                    'protein_key': prot_key,
                    'fold': fold
                })
        dataset = pd.DataFrame(affinity_rows)
        
        drug_smiles = dataset['compound_iso_smiles'].values
        target_sequences = dataset['target_sequence'].values
        affinities = dataset['affinity'].values
        folds = dataset['fold'].values
        if self.target_type == 'deepfri' or self.target_type == 'esm':
            target_encodings = np.array(dataset['protein_encoding'])
        elif self.target_type is None:
            target_encodings = np.asarray([seq_cat(t) for t in target_sequences])
        else:   
            raise ValueError(f"Unknown target_type: {self.target_type}. Supported types are 'esm', 'deepfri', or None.")
        # cluster_labels = dataset['cluster_number'].values if self.cluster_type else None

        # Convert SMILES to graph data
        smiles_set = set(drug_smiles)
        graph_list = {}
        for smile in smiles_set:
            graph_list[smile] = smile_to_graph(smile)

        assert (len(drug_smiles) == len(target_encodings) and len(target_encodings) == len(affinities)), \
            "The three lists must be the same length!"

        # Create PyTorch-Geometric data objects
        data_len = len(target_sequences)
        for i in tqdm(range(data_len), desc=f"Processing {self.dataset} data"):
            smiles = drug_smiles[i]
            target = target_encodings[i]
            labels = affinities[i]
            c_size, features, edge_index = graph_list[smiles]
            fold = folds[i]
            # cluster_number = cluster_labels[i] if self.cluster_type else -1
            # protein_sequence = target_sequences[i]
            
            GCNData = DATA.Data(x=torch.Tensor(np.array(features)),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            
            GCNData.target = torch.FloatTensor([target]) \
                if (self.target_type == 'esm' or self.target_type == 'deepfri') \
                else torch.LongTensor(np.array([target]))
            
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            GCNData.__setitem__('fold', torch.LongTensor([fold]))
            # GCNData.__setitem__('cluster_number', torch.LongTensor([cluster_number]))
            # GCNData.__setitem__('target_sequence', protein_sequence)

            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index     

SEQ_VOC = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SEQ_DICT = {v:(i+1) for i,v in enumerate(SEQ_VOC)}
SEQ_DICT_LEN = len(SEQ_DICT)
MAX_SEQ_LEN = 1000

def seq_cat(prot):
    x = np.zeros(MAX_SEQ_LEN)
    for i, ch in enumerate(prot[:MAX_SEQ_LEN]): 
        x[i] = SEQ_DICT[ch]
    return x

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def bce(y,f):
    bce = -(y * np.log(f) + (1 - y) * np.log(1 - f)).mean()
    return bce

def l1(y,f):
    l1 = np.mean(np.abs(f - y))
    return l1