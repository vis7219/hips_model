import torch
from transformers import logging

# Set logging level to ERROR (suppresses warnings & info messages)
logging.set_verbosity_error()

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
    embeddings : torch.tensor
        3D tensors of embeddings
    """
    from transformers import pipeline
    from torch import nn
    
    
    embed_model = { '8M'   : 'esm2_t6_8M_UR50D',
                    '35M'  : 'esm2_t12_35M_UR50D',
                    '150M' : 'esm2_t30_150M_UR50D',
                    '650M' : 'esm2_t33_650M_UR50D',
                    '3B'   : 'esm2_t36_3B_UR50D',
                    '15B'  : 'esm2_t48_15B_UR50D'}
    
    pipe = pipeline('feature-extraction', model = f'facebook/{embed_model[model]}', device = device)
    embeddings = pipe(protein_batch)
    embeddings = [torch.tensor(x).squeeze(0) for x in embeddings] #[1:-1] for x in embeddings]
    #embeddings = [torch.tensor(x).mean(1).squeeze(0) for x in embeddings] # Not accounting for tokens
    #embeddings = torch.stack(embeddings).to(device)
    
    # Padding since we are not getting mean along the feature dimension
    max_prot_len = max(t.shape[0] for t in embeddings)
    embeddings = torch.stack([nn.functional.pad(t, (0,0,0,max_prot_len - t.shape[0]), value = 1) for t in embeddings]).to(device)
    
    embeddings = [i[0:550, :] for i in embeddings]
    embeddings = torch.stack([torch.tensor(x).squeeze(0) for x in embeddings])
    
    return embeddings


def esm3(protein_batch : list , device : torch.device) -> list:
    """
    Get the ESM3 embeddings for a batch of proteins.
    Huggingface login token required. Refere to 'https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1'.
    
    Parameters:
    -----------
    protein_batch : list
        List of FASTA sequences
        
    device : torch.device
        Device to be used for computation
        
    Returns:
    --------
    embeddings : list
        list of 1D tensors of embeddings
    """
    
    from huggingface_hub import login
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESM3InferenceClient, ESMProtein, SamplingConfig

    import torch._dynamo # Used to suppress something according to an error given when running pipe
    torch._dynamo.config.suppress_errors = True
    
    login() # HuggingFace Login
    
    pipe: ESM3InferenceClient = ESM3.from_pretrained(f"esm3_sm_open_v1").to(device)
    
    embeddings = []
    for fasta in protein_batch:
        protein = ESMProtein(sequence=fasta)
        protein = pipe.encode(protein)
        protein = pipe.forward_and_sample(protein , SamplingConfig(return_per_residue_embeddings=True))
        #protein = torch.mean(protein.per_residue_embedding , dim = 0).to('cpu') # Not accounting for tokens
        protein = torch.mean(protein.per_residue_embedding , dim = 0).to('cpu')
        
        embeddings.append(protein)
        
    return embeddings


def esmc(protein_batch : list , device : torch.device) -> list:
    """
    Get the ESM-C embeddings for a batch of proteins
    
    Parameters:
    -----------
    protein_batch : list
        List of FASTA sequences
        
    device : torch.device
        Device to be used for computation 
        
    Returns:
    --------
    embeddings : list
        list of 1D tensors of embeddings
    """
    
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    import logging
    logging.getLogger("transformers").setLevel(logging.CRITICAL)
    from tqdm import tqdm

    import torch

    client = ESMC.from_pretrained("esmc_300m").to(device)
            
    embeddings = []
    with tqdm(total=len(protein_batch)) as pbar:
        pbar.set_description(f'ESMC Embeddings')
        for prot in protein_batch:
            #print(prot)
            prot_esm = ESMProtein(sequence = prot)
            prot_encode = client.encode(prot_esm)
            prot_logits = client.logits(prot_encode, LogitsConfig(sequence=True, return_embeddings=True))
            
            embeddings.append(torch.tensor(prot_logits.embeddings).squeeze(0)[1:-1].mean(0).to('cpu'))
            
            # esm_out.append(prot_logits.embeddings)
            pbar.update(1)
        
    # embeddings = [torch.tensor(x).squeeze(0)[1:-1].mean(0) for x in esm_out]
    # embeddings = torch.tensor(prot_embedding)
    
    return embeddings