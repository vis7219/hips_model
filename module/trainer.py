from torch.utils.data import Subset , DataLoader
import torch
import torch.nn as nn

from module.data_control import Dataset, collate_fn, acc_metrics
from module.init_models import init_regmodel
from module.get_results import plot_loss, plot_pred_v_true, plot_acc, save_loss, save_pred_true, save_acc

from tqdm import tqdm

import numpy as np
import pandas as pd

class trainer:
    """
    Model trainer
    """
    
    def __init__(self, df : pd.DataFrame, model_type : str, model_params : dict):
        """
        Initializes the trainer class
        
        Parameters:
        -----------
        df : pd.DataFrame
            With columns ['proteins' , 'ligands' , 'affinity' , 'split']
            
            proteins : FASTA
            ligands  : ISOMERIC SMILES
            affinity : Binding affinity values
            split    : ['train' , 'test' , 'val']
            
        model : str
            Model to train
            regmodel/deepdta
            
        model_params : dict
        
            regmodel
            --------
            {
                prot_embedding : str = ['esm2', 'esm3', 'esmc'],
                lig_embedding  : str = ['morgan'],
                esm2_type      : str = ['8M', '35M', '150M', '650M', '3B', '15B' ],
                dropout        : float = [0.1 - 1.0, 0]
            }
            
            deepdta
            -------
            {
                smilelen    : int
                seqlen      : int
                channel     : int
                prot_kernel : int
                lig_kernel  : int
            }

        """
        
        print('Initiallizing data')
        # Initializing data
        df        = df
        fasta     = list(set(df.loc[: , 'proteins']))
        smiles    = list(set(df.loc[: , 'ligands']))
        train_idx = df[df['split'] == 'train'].index.values
        test_idx  = df[df['split'] == 'test'].index.values
        val_idx   = df[df['split'] == 'val'].index.values
        dataset   = Dataset(df)
        
        self.train_dataset = Subset(dataset , train_idx)
        self.test_dataset  = Subset(dataset , test_idx)
        self.val_dataset   = Subset(dataset , val_idx)
        
        # Initializing pytorch variables
        self.device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initializing model
        self.train_done = False
        self.test_done = False
        match model_type:
            
            case 'regmodel':
                # Initializing parameters
                prot_embed = model_params['prot_embedding']
                lig_embed  = model_params['lig_embedding']
                esm2_type  = model_params['esm2_type']
                dropout    = model_params['dropout']
                
                self.model, self.p_embed_df, self.l_embed_df = init_regmodel(prot_embed = prot_embed,
                                                                             lig_embed  = lig_embed,
                                                                             esm2_type  = esm2_type,
                                                                             dropout    = dropout,
                                                                             fasta      = fasta,
                                                                             smiles     = smiles,
                                                                             device     = self.device)
      
    def _shared_step(self, data_batch : DataLoader, mode : str):
        """
        Set of steps common in training/validating & testing
        
        Parameters:
        -----------
        data_batch : DataLoader
        
        mode : str
            ['train', 'valid', 'test']
        """
        
        # Initialization
        loss_value = 0.0
        
        with tqdm(total=len(data_batch)) as pbar:
            match mode:
                case 'train':
                    pbar.set_description(f'Epoch :{self.epoch}. Training')
                case 'valid':
                    pbar.set_description(f'Epoch :{self.epoch}. Validating')
                case 'test':
                    pbar.set_description('Testing')
                    self.model.load_state_dict(self.best_weights)
                    pred_results = []
                    true_results = []
            
            # Iterating through the batches
            for protein , ligand , affinity in data_batch:
                  
                protein = torch.stack(tuple(self.p_embed_df.loc[protein , 'P_EMBED'])).to(self.device)
                ligand  = torch.stack(tuple(self.l_embed_df.loc[ligand , 'L_EMBED'])).to(self.device)
                affinity = torch.tensor(affinity , dtype = torch.float, device = self.device)
                
                # Steps based on what is happening with model
                match mode:
                    case 'train':
                        # Prediction & Backpropogation
                        self.optimizer.zero_grad()
                        predict = self.model(protein , ligand)
                        predict = predict.reshape(1) if predict.ndim == 0 else predict
                        loss = self.criterion(predict , affinity)
                        
                        loss.backward()
                        self.optimizer.step()
                        
                        # Accuracy
                        self.train_acc.reset_acc() # Resetting acc metric
                        self.train_acc.calc_acc_iter(pred_val = predict, true_val = affinity) # Accuracy for a batch
                        
                    case 'valid':
                        with torch.no_grad():
                            # Prediction
                            predict = self.model(protein , ligand)
                            loss = self.criterion(predict , affinity)
                            
                            # Accuracy
                            self.valid_acc.reset_acc()
                            self.valid_acc.calc_acc_iter(pred_val = predict, true_val = affinity)
                    
                    case 'test':
                        with torch.no_grad():
                            # Prediction
                            predict = self.model(protein , ligand)
                            loss = self.criterion(predict , affinity)
                            
                            # Accuracy
                            self.test_acc.reset_acc()
                            self.test_acc.calc_acc_iter(pred_val = predict, true_val = affinity)
                            
                            # Prediction vs True values
                            pred_results.extend(predict.cpu().numpy().tolist())
                            true_results.extend(affinity.cpu().numpy().tolist())
                
                # Loss for a batch
                loss_value += loss.item()
                pbar.update(1)
        
        # Avg loss over all batches this epoch
        final_loss = loss_value / len(data_batch)
        
        match mode:
            case 'train':
                self.train_acc.calc_acc_avg(loader_len = len(data_batch)) # Avg acc over all batches for an epoch
                
                return final_loss
            
            case 'valid':
                self.valid_acc.calc_acc_avg(loader_len = len(data_batch)) # Avg acc over all batches for an epoch
                
                return final_loss
            
            case 'test':
                self.test_acc.calc_acc_avg(loader_len = len(data_batch)) # Avg acc over all batches for an epoch
                final_result = [final_loss , pred_results , true_results]
                
                return final_result
                 
    def train(self,  train_params : dict):
        """
        Training
        
        Parameters:
        ----------
        train_params : dict
        
            {
            epochs     : int,
            lr         : float,
            wt_decay   : float,
            batch_size : float,
            optimizer  : str = ['adam'],
            loss       : str = ['mse']
            }
        """
        # Initializing variables
        print('Initializing Training')
        num_epochs = train_params['epochs']
        lr         = train_params['lr']
        wt_decay   = train_params['wt_decay']
        batch_size = train_params['batch_size']
        
        optimize_type = train_params['optimizer']
        loss_criteria = train_params['loss']
        
        match optimize_type:
            case 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wt_decay)

        match loss_criteria:
            case 'mse':
                 self.criterion = nn.MSELoss()
        
        # Initializing DataLoaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, drop_last = False, collate_fn=collate_fn)
        self.val_loader   = DataLoader(self.val_dataset  , batch_size=batch_size, drop_last = False, collate_fn=collate_fn)
        self.test_loader  = DataLoader(self.test_dataset , batch_size=batch_size, drop_last = False, collate_fn=collate_fn)
        
        # Initializing accuracy metrics
        self.train_acc = acc_metrics(device = self.device)
        self.valid_acc = acc_metrics(device = self.device)
        self.test_acc  = acc_metrics(device = self.device)
        
        self.best_weights  = self.model.state_dict()
        self.best_val_loss = np.inf
        self.best_epoch    = 0
        
        self.train_list = []
        self.valid_list = []
        
        # Begin Training
        print('Beginning Training')
        for self.epoch in range(num_epochs):
            
            # Set model to train
            self.model.train()
            self.train_acc.reset_score() # Reset accuracy score per epoch
            train_loss = self._shared_step(data_batch = self.train_loader, mode = 'train') # Iterating over batches
            self.train_list.append(train_loss) # Avg loss over all batches for an epoch

            
            # Set model to eval
            self.model.eval()
            self.valid_acc.reset_score()
            valid_loss = self._shared_step(data_batch = self.val_loader, mode = 'valid')
            self.valid_list.append(valid_loss)
            
            # Finding parameters for best model
            if valid_loss < self.best_val_loss:
                self.best_weights = self.model.state_dict()
                self.best_val_loss = valid_loss
                self.best_epoch = self.epoch
        
        self.train_done = True
        self.train_acc_dict = self.train_acc.get_acc()
        self.valid_acc_dict = self.valid_acc.get_acc()
        
    def test(self):
        """
        Testing
        """        
        # Begin Testing
        print("Beginning Testing")
        self.test_acc.reset_score()
        test_loss , self.pred_results, self.true_results = self._shared_step(data_batch = self.test_loader, mode = 'test')
        self.test_list = [test_loss]
        self.test_acc_dict = self.test_acc.get_acc()
        self.test_done = True
        
    def get_results(self, save_path : str):
        """
        Creating results files & plots
        
        Parameters:
        save_path : str
            Path to a results folder
        """
        
        plot_loss(loss = self.train_list, loss_type = 'train', best_epoch = self.best_epoch, save_path = save_path)
        plot_loss(loss = self.valid_list, loss_type = 'valid', best_epoch = self.best_epoch, save_path = save_path)
        
        
        save_loss(loss = self.train_list, loss_type = 'train', save_path = save_path)
        save_loss(loss = self.valid_list, loss_type = 'valid', save_path = save_path)
        save_loss(loss = self.test_list, loss_type = 'test', save_path = save_path)
        
        
        plot_acc(acc = self.train_acc_dict['r2'], acc_name = 'r2', acc_type = 'train', best_epoch = self.best_epoch, save_path = save_path)
        plot_acc(acc = self.train_acc_dict['ccc'], acc_name = 'ccc', acc_type = 'train', best_epoch = self.best_epoch, save_path = save_path)
        plot_acc(acc = self.train_acc_dict['mse'], acc_name = 'mse', acc_type = 'train', best_epoch = self.best_epoch, save_path = save_path)
        plot_acc(acc = self.train_acc_dict['mae'], acc_name = 'mae', acc_type = 'train', best_epoch = self.best_epoch, save_path = save_path)
        plot_acc(acc = self.train_acc_dict['pcc'], acc_name = 'pcc', acc_type = 'train', best_epoch = self.best_epoch, save_path = save_path)
        plot_acc(acc = self.train_acc_dict['scc'], acc_name = 'scc', acc_type = 'train', best_epoch = self.best_epoch, save_path = save_path)
        
        plot_acc(acc = self.valid_acc_dict['r2'], acc_name = 'r2', acc_type = 'valid', best_epoch = self.best_epoch, save_path = save_path)
        plot_acc(acc = self.valid_acc_dict['ccc'], acc_name = 'ccc', acc_type = 'valid', best_epoch = self.best_epoch, save_path = save_path)
        plot_acc(acc = self.valid_acc_dict['mse'], acc_name = 'mse', acc_type = 'valid', best_epoch = self.best_epoch, save_path = save_path)
        plot_acc(acc = self.valid_acc_dict['mae'], acc_name = 'mae', acc_type = 'valid', best_epoch = self.best_epoch, save_path = save_path)
        plot_acc(acc = self.valid_acc_dict['pcc'], acc_name = 'pcc', acc_type = 'valid', best_epoch = self.best_epoch, save_path = save_path)
        plot_acc(acc = self.valid_acc_dict['scc'], acc_name = 'scc', acc_type = 'valid', best_epoch = self.best_epoch, save_path = save_path)
        
        
        save_acc(acc = self.train_acc_dict['r2'], acc_type = 'train', acc_name = 'r2', save_path = save_path)
        save_acc(acc = self.train_acc_dict['ccc'], acc_type = 'train', acc_name = 'ccc', save_path = save_path)
        save_acc(acc = self.train_acc_dict['mse'], acc_type = 'train', acc_name = 'mse', save_path = save_path)
        save_acc(acc = self.train_acc_dict['mae'], acc_type = 'train', acc_name = 'mae', save_path = save_path)
        save_acc(acc = self.train_acc_dict['pcc'], acc_type = 'train', acc_name = 'pcc', save_path = save_path)
        save_acc(acc = self.train_acc_dict['scc'], acc_type = 'train', acc_name = 'scc', save_path = save_path)
        
        save_acc(acc = self.valid_acc_dict['r2'], acc_type = 'valid', acc_name = 'r2', save_path = save_path)
        save_acc(acc = self.valid_acc_dict['ccc'], acc_type = 'valid', acc_name = 'ccc', save_path = save_path)
        save_acc(acc = self.valid_acc_dict['mse'], acc_type = 'valid', acc_name = 'mse', save_path = save_path)
        save_acc(acc = self.valid_acc_dict['mae'], acc_type = 'valid', acc_name = 'mae', save_path = save_path)
        save_acc(acc = self.valid_acc_dict['pcc'], acc_type = 'valid', acc_name = 'pcc', save_path = save_path)
        save_acc(acc = self.valid_acc_dict['scc'], acc_type = 'valid', acc_name = 'scc', save_path = save_path)
        
        save_acc(acc = self.test_acc_dict['r2'], acc_type = 'test', acc_name = 'r2', save_path = save_path)
        save_acc(acc = self.test_acc_dict['ccc'], acc_type = 'test', acc_name = 'ccc', save_path = save_path)
        save_acc(acc = self.test_acc_dict['mse'], acc_type = 'test', acc_name = 'mse', save_path = save_path)
        save_acc(acc = self.test_acc_dict['mae'], acc_type = 'test', acc_name = 'mae', save_path = save_path)
        save_acc(acc = self.test_acc_dict['pcc'], acc_type = 'test', acc_name = 'pcc', save_path = save_path)
        save_acc(acc = self.test_acc_dict['scc'], acc_type = 'test', acc_name = 'scc', save_path = save_path)
        
        
        save_pred_true(pred_list= self.pred_results, true_list= self.true_results, save_path = save_path)
        
        
        plot_pred_v_true(pred_list = self.pred_results, true_list = self.true_results, save_path = save_path)
        
        
        self.model.load_state_dict(self.best_weights)
        torch.save(self.model.state_dict() , f'{save_path}/model.pt')