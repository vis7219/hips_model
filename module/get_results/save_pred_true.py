import pandas as pd

def save_pred_true(pred_list : list, true_list : list, save_path : str):
    
    df = pd.DataFrame({'True_vals' : true_list, 'Pred_vals' : pred_list})
    df.to_csv(f"{save_path}/True_vs_pred.csv", sep = '\t', index = False)