from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

import torch

def morgan(ligand_batch : list) -> list:
    """
    Get the Morgan fingerprint embeddings for a batch of ligands
    
    Parameters:
    -----------
    ligand_batch : list
        List of SMILE sequences
        
    Returns:
    --------
    embeddings : tensor
        list of 1D embedding tensors for the given protein list
    """
    
    # Morgan Generator object
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius = 2, fpSize = 2048)
    
    lig_fingerprint = []
    for lig_smiles in ligand_batch:
        lig_mol = Chem.MolFromSmiles(lig_smiles) # SMILES to mol
        lig_fingerprint.append(torch.tensor(mfpgen.GetFingerprintAsNumPy(lig_mol))) # mol to fingerprint
    
    #embeddings = torch.stack(lig_fingerprint).to(device)
    embeddings = lig_fingerprint
    
    return embeddings