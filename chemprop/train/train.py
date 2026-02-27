import logging
from typing import Callable
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from tqdm import tqdm
from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader
from chemprop.models import InteractionModel

def train(model: InteractionModel, data_loader: MoleculeDataLoader, loss_func: Callable,
          optimizer, scheduler, args: TrainArgs, n_iter: int = 0, logger=None, writer=None, tokenizer=None) -> int:
    
    model.train()
    loss_sum = iter_count = 0
    device = next(model.parameters()).device # Tự động lấy device (cpu, cuda, hoặc mps)

    for batch in tqdm(data_loader, total=len(data_loader), leave=False):
        mol_batch, features_batch, target_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch, data_weights_batch, add_feature = \
            batch.batch_graph(), batch.features(), batch.targets(), batch.atom_descriptors(), \
            batch.atom_features(), batch.bond_features(), batch.data_weights(), batch.add_features()
        
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch]).to(device)
        mask_weight = torch.Tensor([[args.alpha if list(args.tau)[0]<=x<= list(args.tau)[1] else args.beta for x in tb] for tb in target_batch]).to(device)
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch]).to(device)
        
        target_weights = torch.Tensor(args.target_weights).to(device) if args.target_weights is not None else torch.ones_like(targets).to(device)
        data_weights = torch.Tensor(data_weights_batch).unsqueeze(1).to(device)
        
        model.zero_grad()

        # Ma trận 3D và Feature
        dist_matrix_batch = batch.distance_matrices()
        dist_matrices_tensor = torch.FloatTensor(dist_matrix_batch).to(device)
        add_feature = torch.Tensor(add_feature).to(device)

        preds = model(
            batch=mol_batch, 
            distance_matrices=dist_matrices_tensor, 
            add_feature=add_feature,
            features_batch=features_batch, atom_descriptors_batch=atom_descriptors_batch, 
            atom_features_batch=atom_features_batch, bond_features_batch=bond_features_batch
        )

        loss = loss_func(preds, targets) * target_weights * data_weights * mask_weight
        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += 1

        loss.backward()
        if args.grad_clip: nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        n_iter += len(batch)
        
    return n_iter