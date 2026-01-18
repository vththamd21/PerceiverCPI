from typing import List, Union, Tuple
from functools import reduce

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F

from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function


class GATLayer(nn.Module):
    """
    A GAT layer (Graph Attention Network) that handles edge features.
    
    Formula:
    alpha_ij = softmax_j( LeakyReLU( a^T [Wh_i || Wh_j || We_ij] ) )
    h_i' = sum_j( alpha_ij * (Wh_j + We_ij) )
    """
    def __init__(self, input_dim: int, output_dim: int, bond_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super(GATLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Dimensions per head
        if output_dim % num_heads != 0:
            raise ValueError(f"Output dim {output_dim} must be divisible by num_heads {num_heads}")
        self.head_dim = output_dim // num_heads

        # Linear projections
        self.W_atom = nn.Linear(input_dim, output_dim, bias=bias)
        self.W_bond = nn.Linear(bond_dim, output_dim, bias=bias)

        # Attention mechanism parameters
        # We use separate vectors for source, target, and edge to implement: a^T [Wh_i || Wh_j || We_ij]
        # as (Wh_i * a_src) + (Wh_j * a_dst) + (We_ij * a_edge)
        self.att_src = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        self.att_edge = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_atom.weight)
        nn.init.xavier_uniform_(self.W_bond.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_edge)

    def forward(self, 
                x: torch.Tensor, 
                edge_attr: torch.Tensor, 
                a2a: torch.LongTensor, 
                a2b: torch.LongTensor) -> torch.Tensor:
        """
        x: [num_atoms, input_dim]
        edge_attr: [num_bonds, bond_dim]
        a2a: [num_atoms, max_degree] - Indices of neighbor atoms
        a2b: [num_atoms, max_degree] - Indices of incident bonds
        """
        num_atoms = x.size(0)
        
        # 1. Project features -> [N, num_heads * head_dim]
        # Reshape to [N, num_heads, head_dim]
        h_atoms = self.W_atom(x).view(num_atoms, self.num_heads, self.head_dim)
        h_bonds = self.W_bond(edge_attr) # [num_bonds, output_dim]
        
        # 2. Prepare neighbor features (Gathering)
        # Neighbor atoms: [num_atoms, max_degree, num_heads, head_dim]
        neigh_atoms = index_select_ND(h_atoms, a2a) 
        
        # Incident bonds: [num_atoms, max_degree, num_heads, head_dim]
        # Need to reshape h_bonds first to be gathered correctly across heads
        h_bonds_reshaped = h_bonds.view(-1, self.num_heads, self.head_dim)
        neigh_bonds = index_select_ND(h_bonds_reshaped, a2b)

        # 3. Calculate Attention Scores
        # Score = LeakyReLU( (h_i * a_src) + (h_j * a_dst) + (e_ij * a_edge) )
        
        # Source contribution: [N, 1, heads, dim] * [1, heads, dim] -> sum dim -> [N, 1, heads]
        score_src = (h_atoms * self.att_src).sum(dim=-1).unsqueeze(1) 
        
        # Target contribution: [N, deg, heads, dim] * [1, heads, dim] -> sum dim -> [N, deg, heads]
        score_dst = (neigh_atoms * self.att_dst).sum(dim=-1)
        
        # Edge contribution: [N, deg, heads, dim] * [1, heads, dim] -> sum dim -> [N, deg, heads]
        score_edge = (neigh_bonds * self.att_edge).sum(dim=-1)
        
        # Combined score: [N, deg, heads]
        scores = self.leaky_relu(score_src + score_dst + score_edge)
        
        # 4. Masking (Handling padding in a2a)
        # a2a has 0 where there is no neighbor. We must mask these out for Softmax.
        # mask: [N, deg, 1]
        mask = (a2a != 0).float().unsqueeze(-1)
        
        # Apply mask: Set scores of padding neighbors to -infinity
        scores = scores.masked_fill(mask == 0, -1e9)
        
        # 5. Softmax -> Attention Weights
        alpha = F.softmax(scores, dim=1) # [N, deg, heads]
        alpha = self.dropout_layer(alpha)
        
        # 6. Aggregation
        # Value to aggregate: h_j + e_ij (Standard GraphConv-like combination, weighted by alpha)
        # [N, deg, heads, dim]
        value = neigh_atoms + neigh_bonds
        
        # Weighted sum: sum_j( alpha_ij * value_ij )
        # alpha: [N, deg, heads] -> unsqueeze -> [N, deg, heads, 1]
        # value: [N, deg, heads, dim]
        # result: [N, heads, dim]
        out = (alpha.unsqueeze(-1) * value).sum(dim=1)
        
        # Flatten heads: [N, output_dim]
        out = out.view(num_atoms, self.output_dim)
        
        return out


class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule.
    Modified to use GAT (Graph Attention Network).
    """

    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.undirected = args.undirected # Not used in GAT directly but kept for compatibility
        self.device = args.device
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm

        # GAT Hyperparameters
        # Chemprop doesn't have a 'num_heads' arg by default, we infer or set a default.
        # To make it robust without changing args.py, we default to 4 heads.
        self.num_heads = 4 

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input projection layer
        self.W_i = nn.Linear(self.atom_fdim, self.hidden_size, bias=self.bias)

        # GAT Layers
        self.layers = nn.ModuleList([
            GATLayer(
                input_dim=self.hidden_size, 
                output_dim=self.hidden_size, 
                bond_dim=self.bond_fdim, 
                num_heads=self.num_heads, 
                dropout=self.dropout, 
                bias=self.bias
            ) for _ in range(self.depth)
        ])

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Output / Readout layers
        self.W_o = nn.Linear(self.hidden_size, self.hidden_size)

        # layer after concatenating the descriptors if args.atom_descriptors == descriptors
        if args.atom_descriptors == 'descriptor':
            self.atom_descriptors_size = args.atom_descriptors_size
            self.atom_descriptors_layer = nn.Linear(self.hidden_size + self.atom_descriptors_size,
                                                    self.hidden_size + self.atom_descriptors_size,)

    def forward(self,
                mol_graph: BatchMolGraph,
                atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs using GAT.
        """
        if atom_descriptors_batch is not None:
            atom_descriptors_batch = [np.zeros([1, atom_descriptors_batch[0].shape[1]])] + atom_descriptors_batch
            atom_descriptors_batch = torch.from_numpy(np.concatenate(atom_descriptors_batch, axis=0)).float().to(self.device)

        # Get components.
        # atom_messages=False gives us the standard b2a, but we mainly need a2a (atom neighbors) and a2b (incident bonds)
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(atom_messages=False)
        f_atoms, f_bonds, a2b = f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(self.device)
        
        # Get atom-to-atom connectivity
        a2a = mol_graph.get_a2a().to(self.device)

        # Input Projection
        x = self.W_i(f_atoms)  # num_atoms x hidden_size
        x = self.act_func(x)
        x = self.dropout_layer(x)

        # GAT Loop
        for layer in self.layers:
            # Save previous x for skip connection
            x_in = x
            
            # GAT Forward
            x = layer(x, f_bonds, a2a, a2b)
            
            # Skip connection + Activation + Norm/Dropout
            x = x + x_in
            x = self.act_func(x)
            x = self.dropout_layer(x)

        # Output processing
        atom_hiddens = x
        atom_hiddens = self.W_o(atom_hiddens)
        atom_hiddens = self.dropout_layer(atom_hiddens)

        # concatenate the atom descriptors
        if atom_descriptors_batch is not None:
            if len(atom_hiddens) != len(atom_descriptors_batch):
                raise ValueError(f'The number of atoms is different from the length of the extra atom features')

            atom_hiddens = torch.cat([atom_hiddens, atom_descriptors_batch], dim=1)
            atom_hiddens = self.atom_descriptors_layer(atom_hiddens)
            atom_hiddens = self.dropout_layer(atom_hiddens)

        # Readout (Pooling)
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                if self.aggregation == 'mean':
                    mol_vec = mol_vec.sum(dim=0) / a_size
                elif self.aggregation == 'sum':
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation == 'norm':
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs


class MPN(nn.Module):
    """An :class:`MPN` is a wrapper around :class:`MPNEncoder` which featurizes input as needed."""

    def __init__(self,
                 args: TrainArgs,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPN, self).__init__()
        self.atom_fdim = atom_fdim or get_atom_fdim(overwrite_default_atom=args.overwrite_default_atom_features)
        self.bond_fdim = bond_fdim or get_bond_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                    overwrite_default_bond=args.overwrite_default_bond_features,
                                                    atom_messages=args.atom_messages)

        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.atom_descriptors = args.atom_descriptors
        self.overwrite_default_atom_features = args.overwrite_default_atom_features
        self.overwrite_default_bond_features = args.overwrite_default_bond_features

        if self.features_only:
            return

        if args.mpn_shared:
            self.encoder = nn.ModuleList([MPNEncoder(args, self.atom_fdim, self.bond_fdim)] * args.number_of_molecules)
        else:
            self.encoder = nn.ModuleList([MPNEncoder(args, self.atom_fdim, self.bond_fdim)
                                          for _ in range(args.number_of_molecules)])

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecules.
        """
        if type(batch[0]) != BatchMolGraph:
            batch = [[mols[i] for mols in batch] for i in range(len(batch[0]))]

            if self.atom_descriptors == 'feature':
                if len(batch) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')

                batch = [
                    mol2graph(
                        mols=b,
                        atom_features_batch=atom_features_batch,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features
                    )
                    for b in batch
                ]
            elif bond_features_batch is not None:
                if len(batch) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')

                batch = [
                    mol2graph(
                        mols=b,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features
                    )
                    for b in batch
                ]
            else:
                batch = [mol2graph(b) for b in batch]

        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float().to(self.device)

            if self.features_only:
                return features_batch

        if self.atom_descriptors == 'descriptor':
            if len(batch) > 1:
                raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                          'per input (i.e., number_of_molecules = 1).')

            encodings = [enc(ba, atom_descriptors_batch) for enc, ba in zip(self.encoder, batch)]
        else:
            encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]

        output = reduce(lambda x, y: torch.cat((x, y), dim=1), encodings)

        if self.use_input_features:
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view(1, -1)
            output = torch.cat([output, features_batch], dim=1)
        
        return output
