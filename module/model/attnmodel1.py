
import torch
import torch.nn as nn


class attnmodel1(nn.Module):
    """
    Basic 3-Forward layer model.
    """

    def __init__(self, lig_size: int, prot_size: int, E_hidden: int, dropout: float = 0.0, num_heads: int = 8):
        """
        Initialization
        
        Parameters:
        -----------  
        dropout : float
            Value for the dropout layer
            
        init_node : int
            No. of nodes to match the size of input
        """
        
        super(attnmodel1, self).__init__()
        
        #self.cosine = nn.CosineSimilarity()
        
        # Shared space
        self.P_shared = nn.Linear(prot_size, E_hidden)
        # self.L_shared = nn.Linear(lig_size, E_hidden)
        
        # Protein cross attention block
        # E_q = prot_size
        # E_k = E_v = lig_size
        self.Qp_proj = nn.Linear(E_hidden, E_hidden)
        self.Kl_proj = nn.Linear(E_hidden, E_hidden)
        self.Vl_proj = nn.Linear(E_hidden, E_hidden)
        self.MHAp = nn.MultiheadAttention(embed_dim= E_hidden, num_heads= num_heads, batch_first= True, dropout= dropout)
        
        # Protein self-attention block
        #self.MHA_self = nn.MultiheadAttention(embed_dim= E_hidden, num_heads= num_heads, batch_first= True, dropout= dropout)
        
        # MLP
        self.fc1 = nn.Linear(176000, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Ligand Cross attention block
        # E_q = lig_size
        # E_k = E_v = prot_size
        #self.l_linear = nn.Linear(1, E_hidden)
        # self.Ql_proj = nn.Linear(E_hidden, E_hidden)
        # self.Kp_proj = nn.Linear(E_hidden, E_hidden)
        # self.Vp_proj = nn.Linear(E_hidden, E_hidden)
        # self.MHAl = nn.MultiheadAttention(embed_dim= E_hidden, num_heads= num_heads, batch_first = True, dropout = dropout)

    @staticmethod
    def cosine_to_binding(cosine_val : float, bind_upper : float, bind_lower : float):
        pkd_range = bind_upper - bind_lower
        
        new_cosine = (cosine_val + 1) / 2 * pkd_range + bind_lower
        
        return new_cosine
        
    def forward(self, protein_batch , ligand_batch):
        
        # Shared space
        # protein_batch = self.P_shared(protein_batch)
        # # ligand_batch = self.L_shared(ligand_batch)
        
        # # Protein Query
        # Qp = self.Qp_proj(protein_batch)
        # Kl = self.Kl_proj(ligand_batch)
        # Vl = self.Vl_proj(ligand_batch)
        
        # # ligand Query
        # # Ql = self.Ql_proj(ligand_batch)
        # # Kp = self.Kp_proj(protein_batch)
        # # Vp = self.Vp_proj(protein_batch)
        # #l_lin = self.l_linear(ligand_batch)
        
        # # Cross Attention
        # attn_p, _ = self.MHAp(Qp, Kl, Vl)
        
        # Self Attention
        #attn_p, _ = self.MHA_self(attn_p, attn_p, attn_p)
        
        
        #attn_l, _ = self.MHAl(Ql, Kp, Vp)
        #print(f'attn_p : {attn_p.shape}')

        # Same space
        # protein = attn_p.mean(dim=1)
        # ligand = attn_l.mean(dim=1)
        # ligand = l_lin.mean(dim=1)
        
        # Cosine similarity
        #cosine_val = self.cosine(protein, ligand)
        
        # MLP
        #x = torch.cat((attn_p, attn_l), dim = 1)
        x = torch.flatten(protein_batch, start_dim=1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
            
        x = self.fc4(x)    

        return x.squeeze()