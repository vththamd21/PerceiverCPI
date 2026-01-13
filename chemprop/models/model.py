from typing import List, Union, Tuple

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mpn import MPN
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import get_activation_function, initialize_weights
# ĐÃ SỬA: Bỏ import CAB vì không dùng nữa
# from .CAB import CrossAttentionBlock as CAB 


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

        # ĐÃ SỬA: Bỏ khởi tạo self.CAB
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
        self.encoder = MPN(args)
              
        if args.checkpoint_frzn is not None:
            if args.freeze_first_only:
                for param in list(self.encoder.encoder.children())[0].parameters():
                    param.requires_grad=False
            else:
                for param in self.encoder.parameters():
                    param.requires_grad=False                   
                        
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
        
        # --- ĐÃ SỬA: Cập nhật kích thước input cho FFN ---
        # Vì ta nối (concat) mpnn_out (size: hidden_size) và protein_tensor (size: hidden_size)
        # Nên kích thước đầu vào của FFN phải cộng thêm args.hidden_size
        first_linear_dim += args.hidden_size
        # -------------------------------------------------

        if args.atom_descriptors == 'descriptor':
            first_linear_dim += args.atom_descriptors_size

        first_linear_dim = first_linear_dim
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
        
        if args.checkpoint_frzn is not None:
            if args.frzn_ffn_layers >0:
                for param in list(self.ffn.parameters())[0:2*args.frzn_ffn_layers]:
                    param.requires_grad=False


    def featurize(self, batch, features_batch=None, atom_descriptors_batch=None, atom_features_batch=None, bond_features_batch=None) -> torch.FloatTensor:
        return self.ffn[:-1](self.encoder(batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch))

    def fingerprint(self, batch, features_batch=None, atom_descriptors_batch=None) -> torch.FloatTensor:
        return self.encoder(batch, features_batch, atom_descriptors_batch)

    def normalization(self, vector_present, threshold=0.1):
        vector_present_clone = vector_present.clone()
        num = vector_present_clone - vector_present_clone.min(1,keepdim = True)[0]
        de = vector_present_clone.max(1,keepdim = True)[0] - vector_present_clone.min(1,keepdim = True)[0]
        return num / de

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                sequence_tensor: List[np.ndarray] = None,
                add_feature: List[np.ndarray] = None,
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Runs the :class:`InteractionNet` on input.
        """
        if self.featurizer:
            return self.featurize(batch, features_batch, atom_descriptors_batch,
                                  atom_features_batch, bond_features_batch)
        
        # 1. Calculate Compound Features (D-MPNN)
        mpnn_out = self.normalization(self.encoder(batch, features_batch, atom_descriptors_batch,
                                       atom_features_batch, bond_features_batch))

        # 2. Calculate Protein Features (CNN + FC)
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
        
        # --- ĐÃ SỬA: Thay thế Cross-Attention bằng Concat ---
        # Nối vector thuốc và vector protein lại với nhau
        output = torch.cat([mpnn_out, protein_tensor], dim=1)
        # ----------------------------------------------------
        
        # Output layers (FFN)
        output = self.ffn(output)

        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))
            if not self.training:
                output = self.multiclass_softmax(output)

        return output