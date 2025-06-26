from torch.utils.data import Subset , DataLoader
import torch
import torch.nn as nn
import wandb

from .model import Dataset, collate_fn, acc_metrics
from .model import attnmodel, attnmodel1, regmodel
from .model import plot_loss, plot_pred_v_true, plot_acc
from .model import save_loss, save_pred_true, save_acc
from .model import chembert
from .model import esm2

from tqdm import tqdm
from pathlib import Path
import h5py

import numpy as np
import pandas as pd

class trainer:
    """
    Model trainer
    """
    
    def __init__(self, df : pd.DataFrame, model_params : dict):
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

        """
        
        print('Initiallizing data')
        # Model parameters
        tarE= model_params['prot_embedding']
        ligE= model_params['lig_embedding']
        
        dropout   = model_params['dropout']
        num_heads = model_params['num_heads']
        E_hidden  = model_params['E_hidden']
        
        self.esm2_type = model_params['esm2_type']
        self.dataname  = model_params['dataset']
        
        self.ligE_path = f'module/lig_{ligE}_{self.dataname}.hdf5'
        self.tarE_path = f'module/prot_{tarE}_{self.dataname}.hdf5'
        
        self.device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        
        # Data
        self.fasta  = list(set(df.loc[: , 'proteins']))
        self.smiles = list(set(df.loc[: , 'ligands']))
        self.lig    = pd.DataFrame(df[['ligands', 'LIG_ID']].set_index('ligands')).drop_duplicates()
        self.tar    = pd.DataFrame(df[['proteins', 'GENE_ID']].set_index('proteins')).drop_duplicates()
        train_idx   = df[df['split'] == 'train'].index.values
        test_idx    = df[df['split'] == 'test'].index.values
        val_idx     = df[df['split'] == 'val'].index.values
        dataset     = Dataset(df, self.ligE_path, self.tarE_path)
        
        self.train_dataset = Subset(dataset , train_idx)
        self.test_dataset  = Subset(dataset , test_idx)
        self.val_dataset   = Subset(dataset , val_idx)
        
        
        # Initializing embeddings
        self._gen_ligembed(lig_type = ligE)
        self._gen_tarembed(tar_type = tarE)
        
        # Affinity value range ## For Cosine Similarity
        # self.max_bind = max(list(set(df.loc[: , 'affinity'])))
        # self.min_bind = min(list(set(df.loc[: ,'affinity'])))
        
        print(f'Train points: {len(train_idx)}')
        print(f'Test points: {len(test_idx)}')
        print(f'Valid points: {len(val_idx)}')


        # Model
        print('Initializing model')
        match model_params['model']:
            
            case 'attnmodel':
                self.model = attnmodel(lig_size= 767, prot_size= 320, E_hidden= E_hidden, dropout= dropout, num_heads= num_heads).to(self.device)
            
            case 'attnmodel1':
                self.model = attnmodel1(lig_size= 767, prot_size= 320, E_hidden= E_hidden, dropout= dropout, num_heads= num_heads).to(self.device)
                
            case 'regmodel':
                self.model = regmodel

    def _gen_ligembed(self, lig_type: str) -> None:
        # LIGAND EMBEDDINGS
        match lig_type:
            case 'chemberta':
                if Path(self.ligE_path).is_file():
                    print(f'Chemberta {self.dataname} embedding found.')
                    
                else:
                    print(f'Creating chemberta {self.dataname} embeddings')
                    
                    with h5py.File(self.ligE_path, 'w') as f:
                        for i in chembert(ligand_batch = self.smiles, device = torch.device('cuda')):
                            embed = i[0]
                            smile = i[1]
                            dset = f.create_dataset(self.lig.loc[smile, 'LIG_ID'], data = embed.to(torch.device('cpu')).numpy())
                        
                        f.attrs['check'] = True
                    print('Done')
    
    def _gen_tarembed(self, tar_type : str) -> None:
        # PROTEIN EMBEDDINGS
        match tar_type:
            case 'esm2':
                if Path(self.tarE_path).is_file():
                    print(f'ESM2 {self.dataname} embedding found.')
                else:
                    print(f'Creating ESM2 {self.dataname} embeddings')
                    
                    protein = esm2(protein_batch = self.fasta, device = torch.device('cuda'), model = self.esm2_type)
                    p_embed_df = {self.tar.loc[self.fasta[i], 'GENE_ID'] : protein[i,:,:].to(torch.device('cpu')).numpy() for i in range(len(self.fasta))}

                    with h5py.File(self.tarE_path, 'w') as f:
                        for key, value in p_embed_df.items():
                            f.create_dataset(key, data = value)
                    print('Done')

    def _step_train(self, data_batch: DataLoader):
        
        self.model.train()
        self.train_acc.reset_score() # Reset accuracy score per epoch
        
        loss_value = 0.0
        with tqdm(total=len(data_batch)) as pbar:
            pbar.set_description(f'Epoch: {self.epoch} - Training')
            
            for protein, ligand, affinity in data_batch:
                affinity = torch.tensor(affinity , dtype = torch.float, device = self.device)

                #ligand = [self.lig.loc[key, 'LIG_ID'] for key in ligand]
                # with h5py.File(f'module/lig_chemberta_{self.model_params['dataset']}.hdf5', 'r') as f:
                #     ligand = torch.stack([torch.tensor(f[self.lig.loc[key, 'LIG_ID']])[:] for key in tqdm(ligand)]).to(torch.device('cuda'))
                #     ligand = ligand.squeeze(1)
                    
                # with h5py.File(f'module/prot_esm2_{self.model_params['dataset']}.hdf5', 'r') as f:
                #     protein = torch.stack([torch.tensor(f[self.tar.loc[key, 'GENE_ID']])[:] for key in tqdm(protein)]).to(torch.device('cuda'))
                #     protein = protein.squeeze(1)
                
                # Step
                self.optimizer.zero_grad(set_to_none=True)
                protein = protein.to(self.device)
                ligand = ligand.to(self.device)
                predict = self.model(protein, ligand)
                #predict = self.model.cosine_to_binding(cosine_val=predict, bind_upper=self.max_bind, bind_lower= self.min_bind)
                loss = self.criterion(predict, affinity)
                loss.backward()
                self.optimizer.step()

                # Accuracy
                self.train_acc.reset_acc() # Resetting acc metric
                self.train_acc.calc_acc_iter(pred_val = predict, true_val = affinity) # Accuracy for a batch
                
                # Loss for a batch
                loss_value += loss.item()
                pbar.update(1)
                
        # Avg loss over all batches this epoch
        final_loss = loss_value / len(data_batch)
        
        # Avg acc over all batches for an epoch
        self.train_acc.calc_acc_avg(loader_len = len(data_batch))
        
        return final_loss
    
    def _step_valid(self, data_batch: DataLoader):
        
        self.model.eval()
        self.valid_acc.reset_score() # Reset accuracy score per epoch
        
        loss_value = 0.0
        with tqdm(total=len(data_batch)) as pbar:
            pbar.set_description(f'Epoch: {self.epoch} - Validating')

            for protein, ligand, affinity in data_batch:
                affinity = torch.tensor(affinity , dtype = torch.float, device = self.device)
                protein = protein.to(self.device)
                ligand = ligand.to(self.device)
                
                # with h5py.File(f'module/lig_chemberta_{self.model_params['dataset']}.hdf5', 'r') as f:
                #     ligand = torch.stack([torch.tensor(f[self.lig.loc[key, 'LIG_ID']][:]) for key in ligand]).to(torch.device('cuda'))
                #     ligand = ligand.squeeze(1)
                
                # with h5py.File(f'module/prot_esm2_{self.model_params['dataset']}.hdf5', 'r') as f:
                #     protein = torch.stack([torch.tensor(f[self.tar.loc[key, 'GENE_ID']][:]) for key in protein]).to(torch.device('cuda'))
                #     protein = protein.squeeze(1)
                
                with torch.no_grad():
                    # Prediction
                    predict = self.model(protein , ligand)
                    #predict = self.model.cosine_to_binding(cosine_val=predict, bind_upper=self.max_bind, bind_lower= self.min_bind)
                    loss = self.criterion(predict , affinity)
                    
                    # Accuracy
                    self.valid_acc.reset_acc()
                    self.valid_acc.calc_acc_iter(pred_val = predict, true_val = affinity)
                    
                    # Loss for a batch
                loss_value += loss.item()
                pbar.update(1)
        
        # Avg loss over all batches this epoch
        final_loss = loss_value / len(data_batch)
        
        # Avg acc over all batches for an epoch
        self.valid_acc.calc_acc_avg(loader_len = len(data_batch))
        
        return final_loss
    
    def _step_test(self, data_batch: DataLoader, test : bool = True):
        
        self.test_acc.reset_score()
        
        loss_value = 0.0
        with tqdm(total=len(data_batch)) as pbar:
            pbar.set_description('Testing')
            
            self.model.load_state_dict(self.best_weights)
            pred_results = []
            true_results = []
            
            for protein , ligand , affinity in data_batch:
                affinity = torch.tensor(affinity , dtype = torch.float, device = self.device)
                protein = protein.to(self.device)
                ligand = ligand.to(self.device)
                
                # with h5py.File(f'module/lig_chemberta_{self.model_params['dataset']}.hdf5', 'r') as f:
                #     ligand = torch.stack([torch.tensor(f[self.lig.loc[key, 'LIG_ID']][:]) for key in ligand]).to(torch.device('cuda'))
                #     ligand = ligand.squeeze(1)
                
                # with h5py.File(f'module/prot_esm2_{self.model_params['dataset']}.hdf5', 'r') as f:
                #     protein = torch.stack([torch.tensor(f[self.tar.loc[key, 'GENE_ID']][:]) for key in protein]).to(torch.device('cuda'))
                #     protein = protein.squeeze(1)
                
                with torch.no_grad():
                    # Prediction
                    predict = self.model(protein , ligand)
                    #predict = self.model.cosine_to_binding(cosine_val=predict, bind_upper=self.max_bind, bind_lower= self.min_bind)
                    loss = self.criterion(predict , affinity)
                    
                    # Accuracy
                    if test:
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
        
        # Avg acc over all batches for an epoch        
        self.test_acc.calc_acc_avg(loader_len = len(data_batch))
        
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
            train_loss = self._step_train(data_batch = self.train_loader)
            self.train_list.append(train_loss) # Avg loss over all batches for an epoch

            
            # Set model to eval
            valid_loss = self._step_valid(data_batch = self.val_loader)
            self.valid_list.append(valid_loss)
            
            # Finding parameters for best model
            if valid_loss < self.best_val_loss:
                self.best_weights = self.model.state_dict()
                self.best_val_loss = valid_loss
                self.best_epoch = self.epoch
                
            # torch.cuda.memory._dump_snapshot(f"cuda/memory_snapshot_epoch_{self.epoch}.pickle")
        
        self.train_done = True
        self.train_acc_dict = self.train_acc.get_acc()
        self.valid_acc_dict = self.valid_acc.get_acc()
        
    def test(self):
        """
        Testing
        """        
        # Begin Testing
        print("Beginning Testing")
        test_loss , self.pred_results, self.true_results = self._step_test(data_batch = self.test_loader, test = True)
        
        self.test_list = [test_loss]
        self.test_acc_dict = self.test_acc.get_acc()
        self.test_done = True
        
        test_train_loss, self.test_train_pred, self.test_train_true = self._step_test(data_batch = self.train_loader, test = False)
        
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
        
        for acc in ['r2', 'ccc', 'mse', 'mae', 'pcc', 'scc']:
            plot_acc(acc = self.train_acc_dict[acc], acc_name = acc, acc_type = 'train', best_epoch = self.best_epoch, save_path = save_path)
            plot_acc(acc = self.valid_acc_dict[acc], acc_name = acc, acc_type = 'valid', best_epoch = self.best_epoch, save_path = save_path)
            
            save_acc(acc = self.train_acc_dict[acc], acc_type = 'train', acc_name = acc, save_path = save_path)
            save_acc(acc = self.valid_acc_dict[acc], acc_type = 'valid', acc_name = acc, save_path = save_path)
            save_acc(acc = self.test_acc_dict[acc], acc_type = 'test', acc_name = acc, save_path = save_path)
        
        
        save_pred_true(pred_list= self.pred_results, true_list= self.true_results, save_path = save_path)
        plot_pred_v_true(pred_list = self.pred_results, true_list = self.true_results, save_path = save_path, save_name = 'Test')
        
        #save_pred_true(pred_list= self.test_train_pred, true_list= self.test_train_true, save_path = save_path)
        plot_pred_v_true(pred_list = self.test_train_pred, true_list = self.test_train_true, save_path = save_path, save_name = 'Train')
        
        
        self.model.load_state_dict(self.best_weights)
        torch.save(self.model.state_dict() , f'{save_path}/model.pt')