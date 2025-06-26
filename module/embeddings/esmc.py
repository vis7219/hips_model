from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

import logging
logging.getLogger("transformers").setLevel(logging.CRITICAL)
from tqdm import tqdm

import torch

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