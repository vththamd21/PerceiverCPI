import logging
from typing import Callable

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.models import InteractionModel
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR
import numpy as np

def train(model: InteractionModel,
          data_loader: MoleculeDataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None,
          tokenizer= None) -> int: # tokenizer ở đây không còn dùng, sẽ dùng model.esm_tokenizer
    """
    Trains a model for an epoch.
    """
    debug = logger.debug if logger is not None else print

    model.train()
    loss_sum = iter_count = 0

    # Lấy tokenizer từ model
    esm_tokenizer = model.esm_tokenizer

    for batch in tqdm(data_loader, total=len(data_loader), leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, target_batch, protein_sequence_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch, data_weights_batch, add_feature = \
            batch.batch_graph(), batch.features(), batch.targets(), batch.sequences(), batch.atom_descriptors(), \
            batch.atom_features(), batch.bond_features(), batch.data_weights(), batch.add_features()
        
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        mask_weight = torch.Tensor([[args.alpha if list(args.tau)[0]<=x<= list(args.tau)[1] else args.beta for x in tb] for tb in target_batch])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])
        
        if args.target_weights is not None:
            target_weights = torch.Tensor(args.target_weights)
        else:
            target_weights = torch.ones_like(targets)
        data_weights = torch.Tensor(data_weights_batch).unsqueeze(1)
        
        model.zero_grad()

        # --- XỬ LÝ PROTEIN MỚI ---
        # Lấy danh sách chuỗi raw string
        # protein_sequence_batch là list các tuple (sequence, ...), ta lấy phần tử đầu
        raw_sequences = [t[0] for t in protein_sequence_batch]

        # Tokenize bằng ESM Tokenizer
        # max_length=1024 là giới hạn hợp lý cho ESM-2 (tránh OOM)
        encoded_inputs = esm_tokenizer(
            raw_sequences,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        # -------------------------

        add_feature = torch.Tensor(add_feature)

        # Gọi model với encoded_inputs thay vì sequence_tensor cũ
        preds = model(
            batch=mol_batch, 
            sequence_inputs=encoded_inputs, 
            add_feature=add_feature,
            features_batch=features_batch, 
            atom_descriptors_batch=atom_descriptors_batch, 
            atom_features_batch=atom_features_batch, 
            bond_features_batch=bond_features_batch
        )

        # Move tensors to correct device
        mask = mask.to(preds.device)
        mask_weight = mask_weight.to(preds.device)
        targets = targets.to(preds.device)
        target_weights = target_weights.to(preds.device)
        data_weights = data_weights.to(preds.device)

        if args.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * target_weights * data_weights * mask
        else:
            loss = loss_func(preds, targets) * target_weights * data_weights * mask_weight
        
        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += 1

        loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        n_iter += len(batch)
        
    return n_iter