import matplotlib.pyplot as plt

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