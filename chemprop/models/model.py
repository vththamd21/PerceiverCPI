from typing import List, Union, Tuple

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
#from .mpn import MPN
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import get_activation_function, initialize_weights
# from .CAB import CrossAttentionBlock as CAB 

class GINLayer(nn.Module):
    def __init__(self, hidden_size, epsilon = 0):
        super(GINLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.epsilon = nn.Parameter(torch.Tensor([epsilon]))

    def forward(self, x, edge_index):
        source_idx = edge_index[0]
        target_idx = edge_index[1]

        out = torch.zeros_like(x)
        out.index_add_(0, target_idx, x[source_idx])

        out = out + (1 + self.epsilon) * x
        out = self.mlp(out)
        return out
class GINEncoder(nn.Module):
    def __init__(self, args):
        super(GINEncoder, self).__init__()
        self.atom_fdim = args.atom_features_size
        self.hidden_size = args.hidden_size
        self.depth = args.depth
        self.dropout = nn.Dropout(args.dropout)

        self.W_in = nn.Linear(self.atom_fdim, self.hidden_size)

        self.layers = nn.ModuleList()
        for _ in range (self.depth):
            self.layers.append(GINLayer(self.hidden_size))
    
    def foward(self, batch, feature_batch=None, atom_descriptors_batch=None, atom_features_batch=None, bond_features_batch=None):
        components = batch.get_components(atom_messages=True)
        f_atoms = components.f_atoms
        a2a = components.a2a
        a_scope = components.a_scope

        f_atoms = f_atoms.cuda()

        source_indices = []
        target_indices =[]
        for i, neighbors in enumerate(a2a):
            for neighbor in neighbors:
                source_indices.append(neighbor)
                target_indices.append(i)
            
        if len(source_indices) > 0:
            edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long).cuda()
        else:
            edge_index = torch.zeros((2,0), dtype = torch).long.cuda()

        x = self.W_in(f_atoms)
        x = F.relu(x)

        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.dropout(x)

        mol_vecs = []
        for i, (a_start, a_len) in enumerate(a_scope):
            if a_len == 0:
                mol_vecs.append(torch.zeros(self.hidden_size).cuda())
            else: 
                cur_atoms = x.narrow(0, a_start, a_len)
                mol_vec = cur_atoms.mean(dim = 0)
                mol_vecs.append(mol_vec)
        return torch.stack(mol_vecs, dim = 0)


class InteractionModel(nn.Module):
    """A :class:`InteractionNet` is a model which contains a D-MPNN and 1DCNN following by Concatenation"""

    def __init__(self, args: TrainArgs, featurizer: bool = False):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param featurizer: Whether the model should act as a featurizer.
        """
        super(InteractionModel, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.featurizer = featurizer

        # Protein embedding layers
        self.embedding_xt = nn.Embedding(args.vocab_size, args.prot_hidden)
        self.conv_in = nn.Conv1d(in_channels=args.sequence_length, out_channels=args.prot_1d_out, kernel_size=1)
        self.convs = nn.ModuleList([nn.Conv1d(args.prot_hidden, 2*args.prot_hidden, args.kernel_size, padding=args.kernel_size//2) for _ in range(args.prot_1dcnn_num)])   
        self.rnns = nn.ModuleList([nn.GRU(args.prot_1d_out,args.prot_1d_out, num_layers=1, bidirectional=True,  batch_first=True) for _ in range(args.prot_1dcnn_num)])
        
        # Projection for protein to hidden_size
        self.fc1_xt = nn.Linear(args.prot_hidden*args.prot_1d_out, args.hidden_size)
        
        self.fc_residual_connection = nn.Linear(args.prot_hidden,args.prot_1d_out)
        self.scale = torch.sqrt(torch.FloatTensor([args.alpha])).cuda()
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(args.prot_1d_out)
        self.do = nn.Dropout(args.dropout)
        # self.CAB = CAB(args)

        self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        self.create_encoder(args)
        self.create_ffn(args)

        initialize_weights(self)

    def create_encoder(self, args: TrainArgs) -> None:
        """
        Creates the message passing encoder for the model.
        """
        self.encoder = GINEncoder(args)                   
                        
    def create_ffn(self, args: TrainArgs) -> None:
        """
        Creates the feed-forward layers for the model.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size * args.number_of_molecules

            if args.use_input_features:
                first_linear_dim += args.features_size
        
        first_linear_dim += args.hidden_size

        if args.atom_descriptors == 'descriptor':
            first_linear_dim += args.atom_descriptors_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, self.output_size),
            ])
            

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def featurize(self, batch, features_batch=None, atom_descriptors_batch=None, atom_features_batch=None, bond_features_batch=None) -> torch.FloatTensor:
        return self.ffn[:-1](self.encoder(batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch))

    def normalization(self, vector_present, threshold=0.1):
        vector_present_clone = vector_present.clone()
        num = vector_present_clone - vector_present_clone.min(1,keepdim = True)[0]
        de = vector_present_clone.max(1,keepdim = True)[0] - vector_present_clone.min(1,keepdim = True)[0]
        return num / de

    def forward(self, batch, sequence_tensor=None, add_feature=None, features_batch=None, atom_descriptors_batch=None, atom_features_batch=None, bond_features_batch=None) -> torch.FloatTensor:
        if self.featurizer:
            return self.featurize(batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch)
        
        # 1. Compound Features (GIN)
        mpnn_out = self.normalization(self.encoder(batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch))

        # 2. Protein Features (CNN)
        sequence = sequence_tensor.cuda()
        embedded_xt = self.embedding_xt(sequence)
        input_nn = self.conv_in(embedded_xt)
        conv_input = input_nn.permute(0, 2, 1)

        for i, conv in enumerate(self.convs):
            conved = self.norm(conv(conv_input))
            conved = F.glu(conved, dim=1)
            conved = conved + self.scale*conv_input
            conv_input = conved

        out_conv = self.relu(conved)
        protein_tensor = out_conv.view(out_conv.size(0),out_conv.size(1)*out_conv.size(2))
        protein_tensor = self.do(self.relu(self.fc1_xt(self.normalization(protein_tensor))))
        
        # 3. Concatenation (Compound || Protein)
        output = torch.cat([mpnn_out, protein_tensor], dim=1)
        
        # 4. Prediction
        output = self.ffn(output)

        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))
            if not self.training:
                output = self.multiclass_softmax(output)

        return output