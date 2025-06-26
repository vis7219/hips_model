import matplotlib.pyplot as plt

def plot_pred_v_true(pred_list : list, true_list : list, save_path : str):
    """
    Parameters
    ----------
    
    pred_list : list
        List of floating predicted values
        
    true_list : list
        List of floating actual values
        
    save_paht : str
    """
    fig , ax = plt.subplots(1,1)
    ax.hexbin(x = true_list, y = pred_list, cmap = 'inferno', gridsize = 50)
    ax.set_xlabel('True values')
    ax.set_ylabel('Predicted values')
    ax.grid(True)
    
    fig.savefig(f'{save_path}/True_v_pred.png' , dpi = 300)
    plt.close()