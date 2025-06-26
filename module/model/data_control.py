from torch.utils.data import Dataset
import torch
import h5py

class acc_metrics:
    def __init__(self, device : torch.device):
        
        from torchmetrics.regression import R2Score, ConcordanceCorrCoef, MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef
        
        self.device = device
        
        self.r2  = R2Score().to(self.device)
        #self.ccc = ConcordanceCorrCoef().to(self.device)
        #self.mse = MeanSquaredError().to(self.device)
        #self.mae = MeanAbsoluteError().to(self.device)
        #self.pcc = PearsonCorrCoef().to(self.device)
        #self.scc = SpearmanCorrCoef().to(self.device)
        
        self.r2_score = 0
        #self.ccc_score = 0
        #self.mse_score = 0
        #self.mae_score = 0
        #self.pcc_score = 0
        #self.scc_score = 0
        
        self.acc_dict = {'r2':[], 'ccc':[], 'mse':[], 'mae':[], 'pcc':[], 'scc':[]}
        
    def calc_acc_iter(self, pred_val : float , true_val : float):
        """
        Calculates accuracy per batch iteration
        
        Parameters
        ----------
        
        pred_val : float
            Predicted value
            
        true_val : float
            True value
        """
        if pred_val.shape == 0 or true_val.shape == 0:
            pass
        else:
            self.r2_score  += self.r2(pred_val , true_val).item()
            #self.ccc_score += self.ccc(pred_val , true_val).item()
            #self.mse_score += self.mse(pred_val , true_val).item()
            #self.mae_score += self.mae(pred_val , true_val).item()
            #self.pcc_score += self.pcc(pred_val , true_val).item()
            #self.scc_score += self.scc(pred_val , true_val).item()
        
    def calc_acc_avg(self, loader_len : int):
        """
        Finds average accuracy over all batch iteractions
        
        Parameters
        ----------
        
        loader_len : int
            Number of batches
        """
        
        self.acc_dict['r2'].append(self.r2_score/loader_len)
        #self.acc_dict['ccc'].append(self.ccc_score/loader_len)
        #self.acc_dict['mse'].append(self.mse_score/loader_len)
        #self.acc_dict['mae'].append(self.mae_score/loader_len)
        #self.acc_dict['pcc'].append(self.pcc_score/loader_len)
        #self.acc_dict['scc'].append(self.scc_score/loader_len)
        
    def reset_acc(self):
        """
        Reset the metrics for faster processing
        """
        self.r2.reset()
        #self.ccc.reset()
        #self.mse.reset()
        #self.mae.reset()
        #self.pcc.reset()
        #self.scc.reset()
        
    def reset_score(self):
        """
        Reset the score per epoch
        """
        self.r2_score = 0
        #self.ccc_score = 0
        #self.mse_score = 0
        #self.mae_score = 0
        #self.pcc_score = 0
        #self.scc_score = 0
        
    def get_acc(self):
        return self.acc_dict
    
def collate_fn(batch):
    """
    Collate function for the DataLoader
    """
    proteins, ligands, targets = zip(*batch)
    return torch.stack(proteins), torch.stack(ligands), list(targets)

class Dataset(Dataset):
    """
    Converts the pandas DataFrame input into a torch Dataset to be read by the model
    """
    
    def __init__(self, df, lig_path, tar_path):
        """
        Initialization
        
        Parameters:
        -----------
        df : pd.core.frame.DataFrame
            DataFrame containing columns -> ['proteins','ligands','affinity]
        """
        self.proteins = df['GENE_ID'].values
        self.ligands  = df['LIG_ID'].values
        self.affinity = df['affinity'].values
        
        self.lig_path = lig_path
        self.tar_path = tar_path
        

    def __len__(self):
        """
        Length of dataset
        
        Parameters:
        -----------
        
        Returns:
        --------
        len(self.proteins) : int
        """
        return len(self.proteins)

    def __getitem__(self, idx):
        """
        Get the protein, ligand, and affinity of the idx-th sample

        Parameter:
        ----------
        idx : int
            Index of the sample
        
        Returns:
        --------
        (protein,ligand,affinity) : tuple
        """
        protein  = self.proteins[idx]
        with h5py.File(self.tar_path, 'r') as f:
            protein = torch.tensor(f[protein][:])
            protein = protein.squeeze()
        
        ligand   = self.ligands[idx]
        with h5py.File(self.lig_path, 'r') as f:
            ligand = torch.tensor(f[ligand][:])
            ligand = ligand.squeeze()
        
        affinity = self.affinity[idx]
        return protein, ligand, affinity