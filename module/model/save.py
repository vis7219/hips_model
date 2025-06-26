import pandas as pd

def save_acc(acc : list, acc_type : str, acc_name : str, save_path : str):
    """
    Parameters
    ----------
    
    loss : list
        list of floating loss values
        
    loss_type : str
        ['train', 'valid', 'test']
    """
    
    acc_dict = {f'{acc_name}' : acc}
    acc_df = pd.DataFrame(acc_dict)
    
    acc_df.to_csv(f'{save_path}/{acc_type}_{acc_name}.tsv', sep = '\t', index = False)
    
def save_loss(loss : list, loss_type : str, save_path : str):
    """
    Parameters
    ----------
    
    loss : list
        list of floating loss values
        
    loss_type : str
        ['train', 'valid', 'test']
    """
    
    loss_dict = {f'{loss_type} loss' : loss}
    loss_df = pd.DataFrame(loss_dict)
    
    loss_df.to_csv(f'{save_path}/{loss_type}_loss.tsv', sep = '\t', index = False)
    
def save_pred_true(pred_list : list, true_list : list, save_path : str):
    
    df = pd.DataFrame({'True_vals' : true_list, 'Pred_vals' : pred_list})
    df.to_csv(f"{save_path}/True_vs_pred.csv", sep = '\t', index = False)