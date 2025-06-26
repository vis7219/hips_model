import torch
import pandas as pd

from module.models import regmodel
from module.embeddings import esm2, esm3, esmc, morgan

def init_regmodel(prot_embed : str, lig_embed : str, esm2_type : str, dropout : float, fasta : list, smiles : list, device : torch.device):
    
    
    # Create Protein embeddings
    print('Creating protein embeddings')
    match prot_embed:
        case 'esm2':
            p_embedding = esm2(protein_batch = fasta, device = device, model = esm2_type)
            
        case 'esm3':
            p_embedding = esm3(protein_batch = fasta, device = device)
            
        case 'esmc':
            p_embedding = esmc(protein_batch = fasta, device = device)

    p_embed_df = pd.DataFrame({'FASTA' : fasta , 'P_EMBED' : p_embedding})
    p_embed_df.set_index('FASTA', inplace = True)
    
    
    # Create ligand embeddings
    print('Creating ligand embeddings')
    match lig_embed:
        case 'morgan':
            l_embedding = morgan(ligand_batch = smiles)
            
    l_embed_df = pd.DataFrame({'SMILES' : smiles , 'L_EMBED' : l_embedding})
    l_embed_df.set_index('SMILES', inplace = True)
        
    
    # Initializing model
    print('Initializing model')
    lig_embed_size = l_embedding[0].shape[0]
    tar_embed_size = p_embedding[0].shape[0]
    init_node      = lig_embed_size + tar_embed_size # Get initialize input size
    
    model          = regmodel(dropout , init_node).to(device)
    
    return model, p_embed_df, l_embed_df