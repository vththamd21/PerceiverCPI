from typing import List, Union, Tuple

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from chemprop.args import TrainArgs
# Import các hàm cần thiết
from chemprop.features import BatchMolGraph, mol2graph, get_atom_fdim
from chemprop.nn_utils import get_activation_function, initialize_weights
# Import CAB cho Cross-Attention
from .CAB import CrossAttentionBlock as CAB

# --- GAT IMPLEMENTATION ---
class GATLayer(nn.Module):
    """
    Graph Attention Layer thủ công (không cần thư viện rời).
    """
    def __init__(self, in_features, out_features, dropout, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # W: Biến đổi tuyến tính input features
        self.W = nn.Linear(in_features, out_features, bias=False)
        # a: Vector trọng số cho cơ chế attention
        self.a = nn.Linear(2*out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, edge_index):
        # x: (num_atoms, in_features)
        # edge_index: (2, num_edges)
        
        h = self.W(x) # (num_atoms, out_features)
        N = h.size(0)

        source_idx = edge_index[0]
        target_idx = edge_index[1]

        # Tính Attention Score (e_ij)
        # Nối vector của node nguồn và node đích
        a_input = torch.cat([h[source_idx], h[target_idx]], dim=1) # (num_edges, 2*out_features)
        e = self.leakyrelu(self.a(a_input)).squeeze(1) # (num_edges)

        # Tính Softmax thủ công cho sparse graph
        # 1. exp(e)
        attention = torch.exp(e)
        # 2. Tổng exp(e) cho mỗi node đích (target)
        attention_sum = torch.zeros(N, device=x.device)
        attention_sum.index_add_(0, target_idx, attention)
        # 3. Chuẩn hóa (chia cho tổng) + epsilon tránh chia 0
        attention_norm = attention / (attention_sum[target_idx] + 1e-8)
        
        # Dropout trên attention weights
        attention_norm = F.dropout(attention_norm, self.dropout, training=self.training)

        # Tổng hợp thông tin từ hàng xóm theo trọng số attention
        h_prime = torch.zeros_like(h)
        h_prime.index_add_(0, target_idx, h[source_idx] * attention_norm.unsqueeze(1))
        
        # Residual connection (nếu kích thước khớp)
        if self.in_features == self.out_features:
            h_prime = h_prime + x
            
        return F.elu(h_prime)

class GATEncoder(nn.Module):
    def __init__(self, args):
        super(GATEncoder, self).__init__()
        self.args = args 
        
        # Lấy kích thước feature đầu vào chính xác
        self.atom_fdim = get_atom_fdim(overwrite_default_atom=args.overwrite_default_atom_features)
        if args.atom_features_size > 0:
            self.atom_fdim += args.atom_features_size

        self.hidden_size = args.hidden_size
        self.depth = args.depth
        self.dropout = nn.Dropout(args.dropout)
        
        # Layer GAT đầu tiên: atom_fdim -> hidden_size
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(self.atom_fdim, self.hidden_size, args.dropout))
        
        # Các layer GAT tiếp theo: hidden_size -> hidden_size
        for _ in range(self.depth - 1):
            self.layers.append(GATLayer(self.hidden_size, self.hidden_size, args.dropout))

    def forward(self, batch, features_batch=None, atom_descriptors_batch=None, atom_features_batch=None, bond_features_batch=None):
        # 1. Xử lý Batch List (fix lỗi DataLoader trả về list)
        if isinstance(batch, list) and len(batch) > 0 and hasattr(batch[0], 'get_components'):
            batch = batch[0]
            
        # 2. Xử lý chuyển đổi mol2graph (nếu cần)
        if not hasattr(batch, 'get_components'):
            af_batch = atom_features_batch if atom_features_batch is not None else (None,)
            bf_batch = bond_features_batch if bond_features_batch is not None else (None,)
            batch = mol2graph(batch, af_batch, bf_batch)
        
        # 3. Lấy dữ liệu đồ thị
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = batch.get_components(atom_messages=True)
        a2a = batch.get_a2a() # Lấy danh sách kề dạng node-to-node

        f_atoms = f_atoms.cuda()
        a2a = a2a.cuda()
        
        # 4. Tạo edge_index từ a2a
        num_atoms = a2a.size(0)
        target_indices = torch.arange(num_atoms, device=a2a.device).unsqueeze(1).expand_as(a2a)
        
        source_flat = a2a.flatten()
        target_flat = target_indices.flatten()
        
        # Loại bỏ padding (index 0)
        mask = source_flat != 0
        
        if mask.sum() > 0:
            edge_index = torch.stack([source_flat[mask], target_flat[mask]], dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=a2a.device)

        # 5. Chạy qua các lớp GAT
        x = f_atoms
        for layer in self.layers:
            x = layer(x, edge_index)
            # x = self.dropout(x) # Dropout đã được xử lý bên trong GATLayer

        # 6. Readout (Pooling)
        mol_vecs = []
        for i, (a_start, a_len) in enumerate(a_scope):
            if a_len == 0:
                mol_vecs.append(torch.zeros(self.hidden_size).cuda())
            else:
                cur_atoms = x.narrow(0, a_start, a_len)
                # Mean Pooling
                mol_vec = cur_atoms.mean(dim=0)
                mol_vecs.append(mol_vec)

        return torch.stack(mol_vecs, dim=0)
# ------------------------------

class InteractionModel(nn.Module):
    """
    Model Architecture:
    - Compound Encoder: GAT (Graph Attention Network)
    - Protein Encoder: 1D-CNN
    - Interaction: Cross Attention Block (CAB)
    - No ECFP
    """

    def __init__(self, args: TrainArgs, featurizer: bool = False):
        super(InteractionModel, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.featurizer = featurizer

        # Protein layers
        self.embedding_xt = nn.Embedding(args.vocab_size, args.prot_hidden)
        self.conv_in = nn.Conv1d(in_channels=args.sequence_length, out_channels=args.prot_1d_out, kernel_size=1)
        self.convs = nn.ModuleList([nn.Conv1d(args.prot_hidden, 2*args.prot_hidden, args.kernel_size, padding=args.kernel_size//2) for _ in range(args.prot_1dcnn_num)])   
        self.rnns = nn.ModuleList([nn.GRU(args.prot_1d_out,args.prot_1d_out, num_layers=1, bidirectional=True,  batch_first=True) for _ in range(args.prot_1dcnn_num)])
        self.fc1_xt = nn.Linear(args.prot_hidden*args.prot_1d_out, args.hidden_size)
        self.fc_residual_connection = nn.Linear(args.prot_hidden,args.prot_1d_out)
        self.scale = torch.sqrt(torch.FloatTensor([args.alpha])).cuda()
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(args.prot_1d_out)
        self.do = nn.Dropout(args.dropout)
        
        # --- CAB Layer (Cross Attention) ---
        self.CAB = CAB(args)
        
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
        # --- SỬA: Dùng GATEncoder ---
        self.encoder = GATEncoder(args)
                                
    def create_ffn(self, args: TrainArgs) -> None:
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            # Input của FFN là output của CAB (kích thước = hidden_size)
            first_linear_dim = args.hidden_size * args.number_of_molecules

            if args.use_input_features:
                first_linear_dim += args.features_size
        
        # Không cộng thêm hidden_size vì không dùng concat

        if args.atom_descriptors == 'descriptor':
            first_linear_dim += args.atom_descriptors_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        if args.ffn_num_layers == 1:
            ffn = [dropout, nn.Linear(first_linear_dim, self.output_size)]
        else:
            ffn = [dropout, nn.Linear(first_linear_dim, args.ffn_hidden_size)]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([activation, dropout, nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size)])
            ffn.extend([activation, dropout, nn.Linear(args.ffn_hidden_size, self.output_size)])

        self.ffn = nn.Sequential(*ffn)
        
        if args.checkpoint_frzn is not None:
             if args.frzn_ffn_layers >0:
                for param in list(self.ffn.parameters())[0:2*args.frzn_ffn_layers]: 
                    param.requires_grad=False

    def featurize(self, batch, features_batch=None, atom_descriptors_batch=None, atom_features_batch=None, bond_features_batch=None) -> torch.FloatTensor:
        return self.ffn[:-1](self.encoder(batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch))

    def fingerprint(self, batch, features_batch=None, atom_descriptors_batch=None) -> torch.FloatTensor:
        return self.encoder(batch, features_batch, atom_descriptors_batch)

    # Hàm normalization có epsilon chống lỗi NaN
    def normalization(self, vector_present, threshold=0.1):
        vector_present_clone = vector_present.clone()
        min_v = vector_present_clone.min(1, keepdim=True)[0]
        max_v = vector_present_clone.max(1, keepdim=True)[0]
        
        num = vector_present_clone - min_v
        de = (max_v - min_v) + 1e-8 
        
        return num / de

    def forward(self, batch, sequence_tensor=None, add_feature=None, features_batch=None, atom_descriptors_batch=None, atom_features_batch=None, bond_features_batch=None) -> torch.FloatTensor:
        if self.featurizer:
            return self.featurize(batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch)
        
        # 1. Compound Features (GAT) -> (Batch, Hidden Size)
        mpnn_out = self.normalization(self.encoder(batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch))

        # 2. Protein Features (CNN) -> (Batch, Hidden Size)
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
        
        # 3. Cross Attention (CAB)
        # Sử dụng CAB để tương tác giữa GAT Features và Protein Features
        output = self.CAB(mpnn_out, protein_tensor)
        
        # 4. Prediction
        output = self.ffn(output)

        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))
            if not self.training:
                output = self.multiclass_softmax(output)

        return output