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