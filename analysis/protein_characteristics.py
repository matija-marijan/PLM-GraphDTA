import pandas as pd
import numpy as np

import os
import sys
import requests
import re
from io import StringIO
import csv

import json
from collections import OrderedDict


def ProtParam(uniprot_id):
    '''
    The aim of this function is to extract protein characteristics, based on their UniProt ID.

    input: 
        - input_csv, which contains UniProt IDs
        - output_csv, which is used to store appended new data to the input dataset
    output:

    '''

    web_url = "https://web.expasy.org/cgi-bin/protparam/protparam_bis.cgi?"
    
    protein_url = web_url+uniprot_id+"@@" # dodala sam @ i onda je radio ahahahh LOLL
    files ={
        'file': uniprot_id
    }
    response = requests.get(protein_url)
    # response = requests.get(web_url)
    if response.status_code==200:
        # print(response.text)
        xml_data = response.text

        # Extract and print the relevant information
        molecular_weight = re.findall(r'<strong>Molecular weight:</strong>\s*(-?\d*\.?\d+)', xml_data)[0]
        theoretical_pI = re.findall(r'<strong>Theoretical pI:</strong>\s*(-?\d*\.?\d+)', xml_data)[0]
        ext_coeff_abs = re.findall(r'Abs 0.1% \(=1 g/l\)\s*(-?\d*\.?\d+)', xml_data)
        instability_index = re.findall(r'The instability index \(II\) is computed to be\s*(-?\d*\.?\d+)', xml_data)[0]
        aliphatic_index = re.findall(r'<strong>Aliphatic index:</strong>\s*(-?\d*\.?\d+)', xml_data)[0]
        hydrophaticity = re.findall(r'<strong>Grand average of hydropathicity \(GRAVY\):</strong>(-?\d*\.?\d+)', xml_data)[0]
        print(f"Molecular weight: {molecular_weight}")
        print('Theoretical pI: ', theoretical_pI)
        print('Ext.coef abs: ',  ext_coeff_abs)
        print('Instability index: ', instability_index)
        print("Aliphatic index: ", aliphatic_index)
        print('GRAVY: ', hydrophaticity)
        
        # You can extract other properties similarly
        # Example:

        # Add more properties as needed
    else:
        print(f"Failed to retrieve data: {response.status_code}")

    return float(molecular_weight), float(theoretical_pI), float(ext_coeff_abs[0]), float(ext_coeff_abs[1]), float(instability_index[0]), float(aliphatic_index[0]), float(hydrophaticity)

def ProtParam_from_sequence(sequence):

    # URL for the ProtParam tool
    url = 'https://web.expasy.org/cgi-bin/protparam/protparam'

    # Prepare the payload for submission
    payload = {
        'sequence': sequence,
        'compute': 'Compute parameters'
    }

    # Submit the sequence to the server
    response = requests.post(url, data=payload)

    # Check if the request was successful
    if response.status_code == 200:
        # print(response.text)
        # Parse the response
        xml_data = response.text
        print(xml_data)

        # Extract and print the relevant information
        molecular_weight = re.findall(r'<strong>Molecular weight:</strong>\s*(-?\d*\.?\d+)', xml_data)[0]
        theoretical_pI = re.findall(r'<strong>Theoretical pI:</strong>\s*(-?\d*\.?\d+)', xml_data)[0]
        ext_coeff_abs = re.findall(r'Abs 0.1% \(=1 g/l\)\s*(-?\d*\.?\d+)', xml_data)
        instability_index = re.findall(r'The instability index \(II\) is computed to be\s*(-?\d*\.?\d+)', xml_data)[0]
        aliphatic_index = re.findall(r'<strong>Aliphatic index:</strong>\s*(-?\d*\.?\d+)', xml_data)[0]
        hydrophaticity = re.findall(r'<strong>Grand average of hydropathicity \(GRAVY\):</strong>(-?\d*\.?\d+)', xml_data)[0]
        num_acids = re.findall(r'<strong>Number of amino acids:</strong>\s*(-?\d*\.?\d+)', xml_data)[0]
        A_c = re.findall(r'Ala \(A\)\s*(-?\d*\.?\d+)', xml_data)[0]
        R_c = re.findall(r'Arg \(R\)\s*(-?\d*\.?\d+)', xml_data)[0]
        N_c = re.findall(r'Asn \(N\)\s*(-?\d*\.?\d+)', xml_data)[0]
        D_c = re.findall(r'Asp \(D\)\s*(-?\d*\.?\d+)', xml_data)[0]
        C_c = re.findall(r'Cys \(C\)\s*(-?\d*\.?\d+)', xml_data)[0]
        Q_c = re.findall(r'Gln \(Q\)\s*(-?\d*\.?\d+)', xml_data)[0]
        E_c = re.findall(r'Glu \(E\)\s*(-?\d*\.?\d+)', xml_data)[0]
        G_c = re.findall(r'Gly \(G\)\s*(-?\d*\.?\d+)', xml_data)[0]
        H_c = re.findall(r'His \(H\)\s*(-?\d*\.?\d+)', xml_data)[0]
        I_c = re.findall(r'Ile \(I\)\s*(-?\d*\.?\d+)', xml_data)[0]
        L_c = re.findall(r'Leu \(L\)\s*(-?\d*\.?\d+)', xml_data)[0]
        K_c = re.findall(r'Lys \(K\)\s*(-?\d*\.?\d+)', xml_data)[0]
        M_c = re.findall(r'Met \(M\)\s*(-?\d*\.?\d+)', xml_data)[0]
        F_c = re.findall(r'Phe \(F\)\s*(-?\d*\.?\d+)', xml_data)[0]
        P_c = re.findall(r'Pro \(P\)\s*(-?\d*\.?\d+)', xml_data)[0]
        S_c = re.findall(r'Ser \(S\)\s*(-?\d*\.?\d+)', xml_data)[0]
        T_c = re.findall(r'Thr \(T\)\s*(-?\d*\.?\d+)', xml_data)[0]
        W_c = re.findall(r'Trp \(W\)\s*(-?\d*\.?\d+)', xml_data)[0]
        Y_c = re.findall(r'Tyr \(Y\)\s*(-?\d*\.?\d+)', xml_data)[0]
        V_c = re.findall(r'Val \(V\)\s*(-?\d*\.?\d+)', xml_data)[0]
        O_c = re.findall(r'Pyl \(O\)\s*(-?\d*\.?\d+)', xml_data)[0]
        U_c = re.findall(r'Sec \(U\)\s*(-?\d*\.?\d+)', xml_data)[0]
        neg = re.findall(r'<strong>Total number of negatively charged residues \(Asp \+ Glu\):</strong>\s*(-?\d*\.?\d+)', xml_data)[0]
        pos = re.findall(r'<strong>Total number of positively charged residues \(Arg \+ Lys\):</strong>\s*(-?\d*\.?\d+)', xml_data)[0]
        # C_a = re.findall(r'Carbon      C\s*(-?\d*\.?\d+)', xml_data)[0]
        # H_a = re.findall(r'Hydrogen    H\s*(-?\d*\.?\d+)', xml_data)[0]
        # N_a = re.findall(r'Nitrogen    N\s*(-?\d*\.?\d+)', xml_data)[0]
        # O_a = re.findall(r'Oxygen      O\s*(-?\d*\.?\d+)', xml_data)[0]
        # S_a = re.findall(r'Sulfur      S\s*(-?\d*\.?\d+)', xml_data)[0]
        # num_atoms = re.findall(r'<strong>Total number of atoms:</strong>\s*(-?\d*\.?\d+)', xml_data)[0]

        print(f"Molecular weight: {molecular_weight}")
        print('Theoretical pI: ', theoretical_pI)
        print('Ext.coef abs: ',  ext_coeff_abs)
        print('Instability index: ', instability_index)
        print("Aliphatic index: ", aliphatic_index)
        print('GRAVY: ', hydrophaticity)
        print('Number of amino acids: ', num_acids)
        print('Alanine: ', A_c)
        print('Arginine: ', R_c)
        print('Asparagine: ', N_c)
        print('Aspartic acid: ', D_c)
        print('Cysteine: ', C_c)
        print('Glutamine: ', Q_c)
        print('Glutamic acid: ', E_c)
        print('Glycine: ', G_c)
        print('Histidine: ', H_c)
        print('Isoleucine: ', I_c)
        print('Leucine: ', L_c)
        print('Lysine: ', K_c)
        print('Methionine: ', M_c)
        print('Phenylalanine: ', F_c)
        print('Proline: ', P_c)
        print('Serine: ', S_c)
        print('Threonine: ', T_c)
        print('Tryptophan: ', W_c)
        print('Tyrosine: ', Y_c)
        print('Valine: ', V_c)
        print('Pyrrolysine: ', O_c)
        print('Selenocysteine: ', U_c)
        print('Positively charged residues: ', pos)
        print('Negatively charged residues: ', neg)
        
        # You can extract other properties similarly
        # Example:

        # Add more properties as needed
    else:
        print(f"Failed to retrieve data: {response.status_code}")

    return float(molecular_weight), float(theoretical_pI), float(ext_coeff_abs[0]), float(ext_coeff_abs[1]), float(instability_index[0]), float(aliphatic_index[0]), float(hydrophaticity), float(num_acids), float(A_c), float(R_c), float(N_c), float(D_c), float(C_c), float(Q_c), float(E_c), float(G_c), float(H_c), float(I_c), float(L_c), float(K_c), float(M_c), float(F_c), float(P_c), float(S_c), float(T_c), float(W_c), float(Y_c), float(V_c), float(O_c), float(U_c), float(pos), float(neg)

def download_GO_for_protein(uniprot_id, output_csv, download_limit = 100):
    """
    Using QuickGO API, extracting GO annotations for a specific protein, using its UniProt id

    input:
        - uniprot_id, identification for a certain protein
        - output_csv where the data will be stored
        - download_limit, the maximal number of annotatins per protein to be extracted. Default number is 100. 
    output:
        - list of GO annotations for a specific protein, based on the UniProt ID.
    """

    requestURL = "https://www.ebi.ac.uk/QuickGO/services/annotation/downloadSearch?includeFields=goName&selectedFields=qualifier,goId&downloadLimit={}&geneProductId={}".format(download_limit, uniprot_id)

    r = requests.get(requestURL, headers={ "Accept" : "text/tsv"})

    if not r.ok:
        r.raise_for_status()
        sys.exit()

    responseBody = r.text
    data = StringIO(responseBody)
    df = pd.read_csv(data, sep='\t')

    # Display the DataFrame
    # print(df)
    GO = df['GO TERM'].to_list()
    QUALIFIER = df['QUALIFIER'].to_list()
    fields = [uniprot_id, GO, QUALIFIER]

    with open(output_csv, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    return GO  

if __name__ == "__main__":
    save_names = ['davis', 'kiba', 'davis_mutation']
    prot_dirs = ['data/davis/proteins.json', 'data/kiba/proteins.json', 'data/davis_mutation/proteins.json']
    for save_name, prot_dir in zip(save_names, prot_dirs):
        proteins = json.load(open(prot_dir), object_pairs_hook=OrderedDict)
        columns = ['GeneID', 'Sequence', 'molecular_weight', 'theoretical_pI', 'ext_coeff_abs', 'ext_coeff_abs_reduced_Cys', 'instability_index', 'aliphatic_index', 'hydropathicity', 
                'num_acids', 'A_c', 'R_c', 'N_c', 'D_c', 'C_c', 'Q_c', 'E_c', 'G_c', 'H_c', 'I_c', 'L_c', 'K_c', 'M_c', 'F_c', 'P_c', 'S_c', 'T_c', 'W_c', 'Y_c', 'V_c', 'O_c', 'U_c', 
                'pos', 'neg']
        # df = pd.DataFrame(columns=columns)
        data = []
        i = 0
        for gene, sequence in proteins.items():
            molwt, th_pi, ext_co_ab0, ext_co_ab1, inst, aliph, hyd, num_acids, A_c, R_c, N_c, D_c, C_c, Q_c, E_c, G_c, H_c, I_c, L_c, K_c, M_c, F_c, P_c, S_c, T_c, W_c, Y_c, V_c, O_c, U_c, pos, neg = ProtParam_from_sequence(sequence)
            row = {
                'GeneID': gene,
                'Sequence': sequence,
                'molecular_weight': molwt,
                'theoretical_pI': th_pi,
                'ext_coeff_abs': ext_co_ab0,
                'ext_coeff_abs_reduced_Cys': ext_co_ab1,
                'instability_index': inst,
                'aliphatic_index': aliph,
                'hydropathicity': hyd,
                'num_acids': num_acids,
                'A_c': A_c,
                'R_c': R_c,
                'N_c': N_c,
                'D_c': D_c,
                'C_c': C_c,
                'Q_c': Q_c,
                'E_c': E_c,
                'G_c': G_c,
                'H_c': H_c,
                'I_c': I_c,
                'L_c': L_c,
                'K_c': K_c,
                'M_c': M_c,
                'F_c': F_c,
                'P_c': P_c,
                'S_c': S_c,
                'T_c': T_c,
                'W_c': W_c,
                'Y_c': Y_c,
                'V_c': V_c,
                'O_c': O_c,
                'U_c': U_c,
                'pos': pos,
                'neg': neg,
            }
            data.append(row)
            i += 1
            print(i)

        df = pd.DataFrame(data, columns=columns)
        output_dir = 'analysis/interpretability/protein_parameters'
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(f'{output_dir}/{save_name}_proteins_ProtParam.csv', index=False)