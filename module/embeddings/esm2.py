from transformers import pipeline
import torch

def esm2(protein_batch : list , device : torch.device, model: str) -> list:
    """
    Get the ESM2 embeddings for a batch of proteins
    
    Parameters:
    -----------
    protein_batch : list
        List of FASTA sequences
        
    device : torch.device
        Device to be used for computation
        
    embed_model : str
        Pre-trained model to be used for embedding
        
        Options: 8M, 35M, 150M, 650M, 3B, 15B   

        
    Returns:
    --------
    embeddings : list
        list of 1D tensors of embeddings
    """
    
    embed_model = { '8M'   : 'esm2_t6_8M_UR50D',
                    '35M'  : 'esm2_t12_35M_UR50D',
                    '150M' : 'esm2_t30_150M_UR50D',
                    '650M' : 'esm2_t33_650M_UR50D',
                    '3B'   : 'esm2_t36_3B_UR50D',
                    '15B'  : 'esm2_t48_15B_UR50D'}
    
    pipe = pipeline('feature-extraction', model = f'facebook/{embed_model[model]}', device = device)
    embeddings = pipe(protein_batch)
    embeddings = [torch.tensor(x).squeeze(0)[1:-1].mean(0) for x in embeddings]
    #embeddings = [torch.tensor(x).mean(1).squeeze(0) for x in embeddings] # Not accounting for tokens
    #embeddings = torch.stack(embeddings).to(device)
    
    return embeddings