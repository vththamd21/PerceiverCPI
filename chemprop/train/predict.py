from typing import List
import torch
from tqdm import tqdm
from chemprop.data import MoleculeDataLoader, StandardScaler
from chemprop.models import InteractionModel

def predict(model: InteractionModel, data_loader: MoleculeDataLoader, disable_progress_bar: bool = False, scaler: StandardScaler = None) -> List[List[float]]:
    model.eval()
    preds = []
    device = next(model.parameters()).device

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch, add_feature = \
            batch.batch_graph(), batch.features(), batch.atom_descriptors(), \
            batch.atom_features(), batch.bond_features(), batch.add_features()

        dist_matrix_batch = batch.distance_matrices()
        dist_matrices_tensor = torch.FloatTensor(dist_matrix_batch).to(device)
        add_feature = torch.Tensor(add_feature).to(device)

        with torch.no_grad():
            batch_preds = model(
                batch=mol_batch, distance_matrices=dist_matrices_tensor, add_feature=add_feature,
                features_batch=features_batch, atom_descriptors_batch=atom_descriptors_batch,
                atom_features_batch=atom_features_batch, bond_features_batch=bond_features_batch
            )

        batch_preds = batch_preds.data.cpu().numpy()
        if scaler is not None: batch_preds = scaler.inverse_transform(batch_preds)
        preds.extend(batch_preds.tolist())

    return preds
