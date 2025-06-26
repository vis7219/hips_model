from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, SamplingConfig

import torch._dynamo # Used to suppress something according to an error given when running pipe
torch._dynamo.config.suppress_errors = True


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