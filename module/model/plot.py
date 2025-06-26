import matplotlib.pyplot as plt

def plot_acc(acc : list, acc_name : str, acc_type : str, best_epoch : int, save_path : str):
    """
    Parameters
    ----------
    
    acc : list
        List of floating acc values
        
    acc_name : str
        Name of accuracy metric
        
    acc_type : str
        ['train', 'valid', 'test']
        
    best_epoch : int
        Epoch with best valid score
        
    save_path : str
        Path for plot
    """
    
    fig , ax = plt.subplots(1,1)
    ax.plot(range(1, len(acc)+1) , acc)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(f'{acc_name}')
    ax.axvline(x = best_epoch+1, color = 'r', linestyle = '--', label = f'Epoch: {best_epoch}')
    ax.grid(True)
    
    fig.savefig(f'{save_path}/{acc_type}_{acc_name}.png' , dpi = 300)
    plt.close()
    
def plot_loss(loss : list, loss_type : str, best_epoch : int, save_path : str):
    """
    Parameters
    ----------
    
    loss : list
        list of floating loss values
        
    loss_type : str
        ['train', 'valid']
        
    best_epoch : int
        Epoch with best valid score
        
    save_path : str
        Path for plot
    """
    fig , ax = plt.subplots(1,1)
    ax.plot(range(1, len(loss)+1) , loss)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(f'{loss_type} Loss')
    ax.axvline(x = best_epoch+1, color = 'r', linestyle = '--', label = f'Epoch: {best_epoch}')
    ax.grid(True)
    
    fig.savefig(f'{save_path}/{loss_type}_loss.png' , dpi = 300)
    plt.close()
    
def plot_pred_v_true(pred_list : list, true_list : list, save_path : str, save_name : str):
    """
    Parameters
    ----------
    
    pred_list : list
        List of floating predicted values
        
    true_list : list
        List of floating actual values
        
    save_path : str
        Path to saving folder
        
    save_name : str
        ['Train','Test']
    """
    fig , ax = plt.subplots(1,1)
    ax.hexbin(x = true_list, y = pred_list, cmap = 'inferno', gridsize = 50)
    ax.set_xlabel('True values')
    ax.set_ylabel('Predicted values')
    ax.grid(True)
    
    fig.savefig(f'{save_path}/{save_name}_True_v_pred.png' , dpi = 300)
    plt.close()