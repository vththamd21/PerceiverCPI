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
from .CAB import CrossAttentionBlock as CAB
from transformers import AutoModel, AutoTokenizer

class InteractionModel(nn.Module):
    """A :class:`InteractionNet` is a model which contains a D-MPNN, MLP and ESM-2 Transformer following by Cross attention Block"""

    def __init__(self, args: TrainArgs, featurizer: bool = False):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param featurizer: Whether the model should act as a featurizer.
        """
        super(InteractionModel, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.featurizer = featurizer

        # --- CẤU HÌNH ESM-2 ---
        # Sử dụng mô hình ESM-2 nhỏ (8M tham số) để chạy nhanh. 
        # Nếu có GPU mạnh (A100/V100), bạn có thể đổi thành 'facebook/esm2_t33_650M_UR50D'
        self.esm_model_name = 'facebook/esm2_t6_8M_UR50D' 
        
        # Các lớp xử lý Compound (Giữ nguyên)
        self.fc_mg = nn.Linear(2048, args.hidden_size)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(args.dropout)

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

        # Khởi tạo trọng số cho các lớp MPNN/MLP trước
        initialize_weights(self)

        # --- KHỞI TẠO ESM-2 (Load Pre-trained) ---
        # Load sau initialize_weights để tránh bị reset trọng số pre-trained
        print(f"Loading Protein Language Model: {self.esm_model_name}...")
        self.esm_tokenizer = AutoTokenizer.from_pretrained(self.esm_model_name)
        self.esm_model = AutoModel.from_pretrained(self.esm_model_name)

        # ĐÓNG BĂNG (FREEZE) ESM-2
        # Giúp train nhanh hơn và tránh lỗi OOM (Out of Memory)
        for param in self.esm_model.parameters():
            param.requires_grad = False
        
        # Lớp chiếu: Chuyển vector ESM (vd: 320 hoặc 1280) về hidden_size của model (vd: 128)
        self.esm_hidden_dim = self.esm_model.config.hidden_size
        self.fc_esm_project = nn.Linear(self.esm_hidden_dim, args.hidden_size)
        self.esm_dropout = nn.Dropout(args.dropout)

    def create_encoder(self, args: TrainArgs) -> None:
        """Creates the message passing encoder for the model."""
        self.encoder = MPN(args)
              
        if args.checkpoint_frzn is not None:
            if args.freeze_first_only: 
                for param in list(self.encoder.encoder.children())[0].parameters():
                    param.requires_grad=False
            else: 
                for param in self.encoder.parameters():
                    param.requires_grad=False                   
                        
    def create_ffn(self, args: TrainArgs) -> None:
        """Creates the feed-forward layers for the model."""
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size * args.number_of_molecules

            if args.use_input_features:
                first_linear_dim += args.features_size

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
            
        self.ffn = nn.Sequential(*ffn)
        
        if args.checkpoint_frzn is not None:
            if args.frzn_ffn_layers >0:
                for param in list(self.ffn.parameters())[0:2*args.frzn_ffn_layers]:
                    param.requires_grad=False

    def featurize(self, batch, features_batch=None, atom_descriptors_batch=None, atom_features_batch=None, bond_features_batch=None):
        return self.ffn[:-1](self.encoder(batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch))

    def normalization(self, vector_present, threshold=0.1):
        vector_present_clone = vector_present.clone()
        num = vector_present_clone - vector_present_clone.min(1,keepdim = True)[0]
        de = vector_present_clone.max(1,keepdim = True)[0] - vector_present_clone.min(1,keepdim = True)[0] + 1e-9
        return num / de

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                sequence_inputs: dict, # Thay đổi: nhận dict từ tokenizer thay vì tensor
                add_feature: List[np.ndarray] = None,
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        
        if self.featurizer:
            return self.featurize(batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch)
        
        # 1. Trích xuất đặc trưng Graph (MPNN)
        mpnn_out = self.normalization(self.encoder(batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch))

        # 2. Trích xuất đặc trưng Protein (ESM-2)
        # Lấy input_ids và attention_mask từ batch đã tokenize
        input_ids = sequence_inputs['input_ids'].to(mpnn_out.device)
        attention_mask = sequence_inputs['attention_mask'].to(mpnn_out.device)

        # Chạy ESM-2 (không tính gradient cho backbone)
        with torch.no_grad():
            esm_output = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)
        
        last_hidden_state = esm_output.last_hidden_state # (Batch, Seq_Len, ESM_Dim)

        # Mean Pooling: Tạo 1 vector đại diện cho cả chuỗi protein
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        protein_embedding = sum_embeddings / sum_mask # (Batch, ESM_Dim)

        # Chiếu và chuẩn hóa
        protein_tensor = self.fc_esm_project(protein_embedding)
        protein_tensor = self.esm_dropout(self.relu(protein_tensor))
        protein_tensor = self.normalization(protein_tensor) # Chuẩn hóa MinMax như logic cũ

        # 3. Trích xuất đặc trưng Morgan Fingerprint
        add_feature = self.do(self.relu(self.fc_mg(add_feature.cuda())))

        # 4. Cross Attention Blocks
        output = self.CAB(mpnn_out, add_feature, protein_tensor)

        # 5. Output Prediction
        output = self.ffn(output)

        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))
            if not self.training:
                output = self.multiclass_softmax(output)

        return output