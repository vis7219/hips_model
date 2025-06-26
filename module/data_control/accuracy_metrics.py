from torchmetrics.regression import R2Score, ConcordanceCorrCoef, MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef
import torch


class acc_metrics:
    def __init__(self, device : torch.device):
        
        self.device = device
        
        self.r2  = R2Score().to(self.device)
        self.ccc = ConcordanceCorrCoef().to(self.device)
        self.mse = MeanSquaredError().to(self.device)
        self.mae = MeanAbsoluteError().to(self.device)
        self.pcc = PearsonCorrCoef().to(self.device)
        self.scc = SpearmanCorrCoef().to(self.device)
        
        self.r2_score = 0
        self.ccc_score = 0
        self.mse_score = 0
        self.mae_score = 0
        self.pcc_score = 0
        self.scc_score = 0
        
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
        if len(pred_val) <= 1 or len(true_val) <= 1:
            pass
        else:
            self.r2_score  += self.r2(pred_val , true_val).item()
            self.ccc_score += self.ccc(pred_val , true_val).item()
            self.mse_score += self.mse(pred_val , true_val).item()
            self.mae_score += self.mae(pred_val , true_val).item()
            self.pcc_score += self.pcc(pred_val , true_val).item()
            self.scc_score += self.scc(pred_val , true_val).item()
        
    def calc_acc_avg(self, loader_len : int):
        """
        Finds average accuracy over all batch iteractions
        
        Parameters
        ----------
        
        loader_len : int
            Number of batches
        """
        
        self.acc_dict['r2'].append(self.r2_score/loader_len)
        self.acc_dict['ccc'].append(self.ccc_score/loader_len)
        self.acc_dict['mse'].append(self.mse_score/loader_len)
        self.acc_dict['mae'].append(self.mae_score/loader_len)
        self.acc_dict['pcc'].append(self.pcc_score/loader_len)
        self.acc_dict['scc'].append(self.scc_score/loader_len)
        
    def reset_acc(self):
        """
        Reset the metrics for faster processing
        """
        self.r2.reset()
        self.ccc.reset()
        self.mse.reset()
        self.mae.reset()
        self.pcc.reset()
        self.scc.reset()
        
    def reset_score(self):
        """
        Reset the score per epoch
        """
        self.r2_score = 0
        self.ccc_score = 0
        self.mse_score = 0
        self.mae_score = 0
        self.pcc_score = 0
        self.scc_score = 0
        
    def get_acc(self):
        return self.acc_dict