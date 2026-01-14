from typing import List, Union, Tuple

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from chemprop.args import TrainArgs
# --- SỬA: Import thêm get_atom_fdim ---
from chemprop.features import BatchMolGraph, mol2graph, get_atom_fdim
from chemprop.nn_utils import get_activation_function, initialize_weights

# --- GIN IMPLEMENTATION ---
class GINLayer(nn.Module):
    def __init__(self, hidden_size, epsilon=0):
        super(GINLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
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
        # Cộng dồn feature từ hàng xóm (source) vào node đích (target)
        out.index_add_(0, target_idx, x[source_idx]) 
        
        out = out + (1 + self.epsilon) * x 
        out = self.mlp(out)
        return out

class GINEncoder(nn.Module):
    def __init__(self, args):
        super(GINEncoder, self).__init__()
        self.args = args 
        
        # --- SỬA LỖI SIZE TẠI ĐÂY ---
        # Lấy kích thước feature mặc định (thường là 133) + feature bổ sung (nếu có)
        self.atom_fdim = get_atom_fdim(overwrite_default_atom=args.overwrite_default_atom_features)
        if args.atom_features_size > 0:
            self.atom_fdim += args.atom_features_size
        # -----------------------------

        self.hidden_size = args.hidden_size
        self.depth = args.depth
        self.dropout = nn.Dropout(args.dropout)
        
        # Bây giờ self.atom_fdim sẽ là 133 (hoặc lớn hơn), khớp với dữ liệu input
        self.W_in = nn.Linear(self.atom_fdim, self.hidden_size)

        self.layers = nn.ModuleList()
        for _ in range(self.depth):
            self.layers.append(GINLayer(self.hidden_size))

    def forward(self, batch, features_batch=None, atom_descriptors_batch=None, atom_features_batch=None, bond_features_batch=None):
        # 1. Xử lý đầu vào: Nếu là list (do DataLoader trả về), lấy phần tử đầu tiên
        if isinstance(batch, list) and len(batch) > 0 and hasattr(batch[0], 'get_components'):
            batch = batch[0]
            
        # 2. Nếu chưa phải BatchMolGraph, gọi mol2graph
        if not hasattr(batch, 'get_components'):
            af_batch = atom_features_batch if atom_features_batch is not None else (None,)
            bf_batch = bond_features_batch if bond_features_batch is not None else (None,)
            batch = mol2graph(batch, af_batch, bf_batch)
        
        # 3. Lấy dữ liệu từ đồ thị
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = batch.get_components(atom_messages=True)
        
        # Lấy cấu trúc liên kết atom-to-atom (cho GIN)
        a2a = batch.get_a2a() 

        f_atoms = f_atoms.cuda()
        a2a = a2a.cuda()
        
        # 4. Tạo edge_index từ a2a
        num_atoms = a2a.size(0)
        
        # Tạo index cho target (node nhận tin): [0, 0...], [1, 1...], ...
        target_indices = torch.arange(num_atoms, device=a2a.device).unsqueeze(1).expand_as(a2a)
        
        # Flatten để tạo danh sách cạnh
        source_flat = a2a.flatten()
        target_flat = target_indices.flatten()
        
        # Lọc bỏ padding (index 0 là padding/dummy node, không phải atom thật)
        mask = source_flat != 0
        
        if mask.sum() > 0:
            edge_index = torch.stack([source_flat[mask], target_flat[mask]], dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=a2a.device)

        # 5. Chạy GIN
        x = self.W_in(f_atoms) # Lúc này shape sẽ khớp: (N, 133) x (133, hidden)
        x = F.relu(x)
        
        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.dropout(x)

        # 6. Pooling (Readout)
        mol_vecs = []
        for i, (a_start, a_len) in enumerate(a_scope):
            if a_len == 0:
                mol_vecs.append(torch.zeros(self.hidden_size).cuda())
            else:
                cur_atoms = x.narrow(0, a_start, a_len)
                mol_vec = cur_atoms.mean(dim=0)
                mol_vecs.append(mol_vec)

        return torch.stack(mol_vecs, dim=0)
# --------------------------

class InteractionModel(nn.Module):
    """A :class:`InteractionNet` using GIN for Compound and CNN for Protein, with Concatenation."""

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
        self.encoder = GINEncoder(args)
                                
    def create_ffn(self, args: TrainArgs) -> None:
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

    def normalization(self, vector_present, threshold=0.1):
        vector_present_clone = vector_present.clone()
        num = vector_present_clone - vector_present_clone.min(1,keepdim = True)[0]
        de = vector_present_clone.max(1,keepdim = True)[0] - vector_present_clone.min(1,keepdim = True)[0]
        return num / de

    def forward(self, batch, sequence_tensor=None, add_feature=None, features_batch=None, atom_descriptors_batch=None, atom_features_batch=None, bond_features_batch=None) -> torch.FloatTensor:
        if self.featurizer:
            return self.featurize(batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch)
        
        mpnn_out = self.normalization(self.encoder(batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch))

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
        
        output = torch.cat([mpnn_out, protein_tensor], dim=1)
        
        output = self.ffn(output)

        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))
            if not self.training:
                output = self.multiclass_softmax(output)

        return output