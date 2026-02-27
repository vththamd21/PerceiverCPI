import torch
import torch.nn as nn
from typing import List, Union, Tuple
import numpy as np
from rdkit import Chem
from .mpn import MPN
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import get_activation_function, initialize_weights
from .CAB import CrossAttentionBlock as CAB

class InteractionModel(nn.Module):
    def __init__(self, args: TrainArgs, featurizer: bool = False):
        super(InteractionModel, self).__init__()
        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.featurizer = featurizer

        # Mạng 2D-CNN xử lý ma trận không gian 3D PDB
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.bn_1 = nn.BatchNorm2d(32)
        self.conv2d_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.bn_2 = nn.BatchNorm2d(64)
        self.conv2d_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn_3 = nn.BatchNorm2d(128)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_protein_3d = nn.Linear(128 * 4 * 4, args.hidden_size)

        self.fc_mg = nn.Linear(2048, args.hidden_size)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(args.dropout)

        self.CAB = CAB(args)
        self.output_size = args.num_tasks
        if self.classification: self.sigmoid = nn.Sigmoid()

        self.create_encoder(args)
        self.create_ffn(args)
        initialize_weights(self)

    def create_encoder(self, args: TrainArgs) -> None:
        self.encoder = MPN(args)
              
    def create_ffn(self, args: TrainArgs) -> None:
        first_linear_dim = args.hidden_size * args.number_of_molecules
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        ffn = [dropout, nn.Linear(first_linear_dim, args.ffn_hidden_size)]
        for _ in range(args.ffn_num_layers - 2):
            ffn.extend([activation, dropout, nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size)])
        ffn.extend([activation, dropout, nn.Linear(args.ffn_hidden_size, self.output_size)])
        self.ffn = nn.Sequential(*ffn)

    def normalization(self, vector_present):
        vector_present_clone = vector_present.clone()
        num = vector_present_clone - vector_present_clone.min(1,keepdim=True)[0]
        de = vector_present_clone.max(1,keepdim=True)[0] - vector_present_clone.min(1,keepdim=True)[0] + 1e-9
        return num / de

    def forward(self, batch, distance_matrices: torch.Tensor, add_feature: torch.Tensor = None, 
                features_batch=None, atom_descriptors_batch=None, atom_features_batch=None, bond_features_batch=None):
        
        mpnn_out = self.normalization(self.encoder(batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch))

        x_3d = distance_matrices.unsqueeze(1) 
        x_3d = self.relu(self.bn_1(self.conv2d_1(x_3d)))
        x_3d = self.relu(self.bn_2(self.conv2d_2(x_3d)))
        x_3d = self.relu(self.bn_3(self.conv2d_3(x_3d)))
        x_3d = self.adaptive_pool(x_3d) 
        x_3d = torch.flatten(x_3d, 1)   
        
        protein_tensor = self.normalization(self.do(self.relu(self.fc_protein_3d(x_3d))))

        if add_feature is not None:
            add_feature = self.do(self.relu(self.fc_mg(add_feature)))

        output = self.CAB(mpnn_out, add_feature, protein_tensor)
        output = self.ffn(output)

        if self.classification and not self.training:
            output = self.sigmoid(output)

        return output