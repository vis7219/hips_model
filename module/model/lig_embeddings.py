import torch
from transformers import logging
from tqdm import tqdm

# Set logging level to ERROR (suppresses warnings & info messages)
logging.set_verbosity_error()

def chembert(ligand_batch : list, device : torch.device):
    """
    Get ChemBERTa embeddings for SMILES batch
    
    Parameters:
    -----------
    ligand_batch : list
        list of SMILE sequences
        
    device : torch.device
    
    Returns:
    --------
    embeddings : torch.tensor
        3D tensor (N, L, F)
            N -> batch size
            L -> length
            F -> Features
    """

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(device)
    
    with torch.no_grad():
        
        for i in tqdm(range(len(ligand_batch))):
            tokens = tokenizer(ligand_batch[i], return_tensors = 'pt', padding = 'max_length', max_length = 150, truncation = True).to(device)
            model_out = model(**tokens)
            
            temp = model_out[0]#.squeeze(0)#[1:-1, :]
            #tokens = tokens['input_ids'].squeeze(0)#[:, 1:-1]
            # temp = temp.mean(2)
            #temp = temp.unsqueeze(-1)
            
            # if i == 0:
            #     embeddings = temp
                
            # else:
            #     embeddings = torch.cat((embeddings,temp), dim = 0)
        #return embeddings
            yield (temp, ligand_batch[i])