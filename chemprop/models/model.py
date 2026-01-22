from typing import List, Union, Tuple

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer  # <--- Thêm mới

from .mpn import MPN
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import get_activation_function, initialize_weights
from .CAB import CrossAttentionBlock as CAB


class InteractionModel(nn.Module):
    """
    InteractionModel modified to use ESM-2 for protein encoding 
    instead of CNN/RNN.
    """

    def __init__(self, args: TrainArgs, featurizer: bool = False):
        super(InteractionModel, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.featurizer = featurizer

        # -----------------------------------------------------------
        # PHẦN THAY ĐỔI: Cấu hình ESM-2
        # Bạn có thể chọn các phiên bản khác nhau:
        # 'facebook/esm2_t6_8M_UR50D' (Nhẹ nhất, 8M params - KHUYẾN NGHỊ ĐỂ TEST)
        # 'facebook/esm2_t12_35M_UR50D' (35M params)
        # 'facebook/esm2_t33_650M_UR50D' (Nặng, cần GPU mạnh)
        self.esm_checkpoint = "facebook/esm2_t6_8M_UR50D" 
        
        print(f"Loading ESM-2 model from {self.esm_checkpoint}...")
        self.esm_tokenizer = AutoTokenizer.from_pretrained(self.esm_checkpoint)
        self.esm_model = AutoModel.from_pretrained(self.esm_checkpoint)

        # Đóng băng (Freeze) ESM-2 để không train lại (tiết kiệm bộ nhớ)
        # Nếu muốn fine-tune thì comment đoạn này lại
        for param in self.esm_model.parameters():
            param.requires_grad = False
        
        # Lấy kích thước hidden state của ESM (ví dụ: 320 cho bản 8M)
        self.esm_hidden_dim = self.esm_model.config.hidden_size

        # Lớp chiếu (Projection) từ ESM dimension sang hidden_size của model chung
        self.prot_projector = nn.Linear(self.esm_hidden_dim, args.hidden_size)
        # -----------------------------------------------------------

        # Giữ lại các thành phần xử lý phân tử (MPNN) và CAB
        self.fc_mg = nn.Linear(2048, args.hidden_size)
        self.CAB = CAB(args)
        
        self.relu = nn.ReLU()
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
        self.encoder = MPN(args)
        if args.checkpoint_frzn is not None:
            if args.freeze_first_only: 
                for param in list(self.encoder.encoder.children())[0].parameters():
                    param.requires_grad=False
            else: 
                for param in self.encoder.parameters():
                    param.requires_grad=False                   
                        
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

    def featurize(self, batch, features_batch=None, atom_descriptors_batch=None, atom_features_batch=None, bond_features_batch=None):
        return self.ffn[:-1](self.encoder(batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch))

    def fingerprint(self, batch, features_batch=None, atom_descriptors_batch=None):
        return self.encoder(batch, features_batch, atom_descriptors_batch)

    def normalization(self, vector_present, threshold=0.1):
        vector_present_clone = vector_present.clone()
        num = vector_present_clone - vector_present_clone.min(1, keepdim=True)[0]
        de = vector_present_clone.max(1, keepdim=True)[0] - vector_present_clone.min(1, keepdim=True)[0] + 1e-9
        return num / de

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[BatchMolGraph]],
                sequence_tensor: torch.Tensor = None, # Lưu ý: Cái này bây giờ phải chứa Token IDs của ESM
                add_feature: List[np.ndarray] = None,
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        
        if self.featurizer:
            return self.featurize(batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch)
        
        # 1. Molecule Branch (Giữ nguyên)
        mpnn_out = self.normalization(self.encoder(batch, features_batch, atom_descriptors_batch,
                                       atom_features_batch, bond_features_batch))

        # 2. Protein Branch (THAY ĐỔI: Dùng ESM-2)
        # sequence_tensor lúc này được kỳ vọng là tensor chứa Input IDs của tokenizer ESM
        # Shape: [Batch_Size, Seq_Length]
        input_ids = sequence_tensor.long().cuda()
        
        # Tạo attention mask tự động (nếu padding token id của ESM khác 1, cần điều chỉnh)
        # Tokenizer của ESM thường có pad_token_id là 1
        attention_mask = (input_ids != self.esm_tokenizer.pad_token_id).long().cuda()

        # Chạy qua ESM Model
        with torch.no_grad(): # Tắt gradient cho ESM để tiết kiệm bộ nhớ
            esm_output = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)
            
        # Lấy hidden state: [Batch, Seq, Hidden]
        last_hidden_state = esm_output.last_hidden_state
        
        # Chiến lược Pooling: Lấy token đầu tiên (<CLS>) làm đại diện cho cả protein
        # Hoặc bạn có thể dùng mean pooling: last_hidden_state.mean(dim=1)
        cls_embedding = last_hidden_state[:, 0, :] # Shape: [Batch, ESM_Hidden]

        # Chiếu về không gian chung (Project)
        protein_tensor = self.prot_projector(cls_embedding) # [Batch, Hidden_Size]
        protein_tensor = self.do(self.relu(protein_tensor))

        # 3. Morgan Fingerprint (Giữ nguyên)
        add_feature = self.do(self.relu(self.fc_mg(add_feature.cuda())))

        # 4. Cross Attention Block (Giữ nguyên)
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