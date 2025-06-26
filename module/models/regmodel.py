import torch
import torch.nn as nn


class regmodel(nn.Module):
    """
    Basic 3-Forward layer model.
    """

    def __init__(self, dropout : float , init_node : int):
        """
        Initialization
        
        Parameters:
        -----------  
        dropout : float
            Value for the dropout layer
            
        init_node : int
            No. of nodes to match the size of input
        """
        
        super(regmodel, self).__init__()
        
        # Feed-Forward Neural Network
        self.fc1 = nn.Linear(init_node, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(dropout)
        self.dropout_null = dropout == 0
        self.relu = nn.ReLU()

    def forward(self, protein_batch , ligand_batch):
        
        # Concatenate embeddings
        x = torch.cat((protein_batch , ligand_batch), dim = 1)
        
        # Forward Neural Network
        x = self.fc1(x)
        x = self.relu(x)
        if not self.dropout_null:
            x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        if not self.dropout_null:
            x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        if not self.dropout_null:
            x = self.dropout(x)
            
        x = self.fc4(x)
        
        return x.squeeze()