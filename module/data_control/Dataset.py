from torch.utils.data import Dataset

class Dataset(Dataset):
    """
    Converts the pandas DataFrame input into a torch Dataset to be read by the model
    """
    
    def __init__(self, df):
        """
        Initialization
        
        Parameters:
        -----------
        df : pd.core.frame.DataFrame
            DataFrame containing columns -> ['proteins','ligands','affinity]
        """
        self.proteins = df['proteins'].values
        self.ligands  = df['ligands'].values
        self.affinity = df['affinity'].values
        

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
        ligand   = self.ligands[idx]
        affinity = self.affinity[idx]
        return protein, ligand, affinity