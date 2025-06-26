import pandas as pd

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