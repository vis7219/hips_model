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