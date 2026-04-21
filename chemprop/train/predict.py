from typing import List
from tqdm import tqdm
import torch
import numpy as np
from chemprop.data import MoleculeDataset
from tqdm import tqdm
from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.models import InteractionModel
import numpy as np
from chemprop.args import TrainArgs

# def predict(model: InteractionModel,
#             data_loader: MoleculeDataLoader,
#             args: TrainArgs,
#             disable_progress_bar: bool = False,
#             scaler: StandardScaler = None,
#             tokenizer=None) -> List[List[float]]:
#     """
#     Makes predictions on a dataset using an ensemble of models.

#     :param model: A :class:`~chemprop.models.model.InteractionModel`.
#     :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
#     :param disable_progress_bar: Whether to disable the progress bar.
#     :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
#     :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks.
#     """
#     model.eval()

#     preds = []

#     for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
#         # Prepare batch
#         batch: MoleculeDataset
#         mol_batch, features_batch, protein_sequence_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch, add_feature = \
#             batch.batch_graph(), batch.features(),batch.sequences(), batch.atom_descriptors(), batch.atom_features(), batch.bond_features(), batch.add_features()

#         dummy_array = [0]*args.sequence_length
#         sequence_2_ar = [list(tokenizer.encode(list(t[0]))) + dummy_array for t in protein_sequence_batch]
        
#         new_ar = []

#         for arr in sequence_2_ar:
#             while len(arr)>args.sequence_length:
#                 arr.pop(len(arr)-1)
#             new_ar.append(np.zeros(args.sequence_length)+np.array(arr))
        
#         # convert list_sequence to tensor
#         sequence_tensor = torch.LongTensor(new_ar)
#         add_feature = torch.Tensor(add_feature)

#         # Make predictions
#         with torch.no_grad():
#             batch_preds  = model(mol_batch,sequence_tensor, add_feature, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch)
#         batch_preds = batch_preds.data.cpu().numpy()

#         # Inverse scale if regression
#         if scaler is not None:
#             batch_preds = scaler.inverse_transform(batch_preds)

#         # Collect vectors
#         batch_preds = batch_preds.tolist()
#         preds.extend(batch_preds)

#     return preds

def predict(model: InteractionModel,
            data_loader: MoleculeDataLoader,
            disable_tqdm: bool = False,
            scaler: StandardScaler = None,
            tokenizer = None,
            args: TrainArgs = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in tqdm(data_loader, disable=disable_tqdm, leave=False):
            # --- BRIDGE TRICK CHO GRAPHSAGE ---
            indices = batch.idx.tolist()
            original_data = data_loader.dataset.molecule_dataset._data
            chemprop_batch = MoleculeDataset([original_data[i] for i in indices])

            mol_batch = batch.to(args.device)
            features_batch, protein_sequence_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch, add_feature = \
                chemprop_batch.features(), chemprop_batch.sequences(), chemprop_batch.atom_descriptors(), \
                chemprop_batch.atom_features(), chemprop_batch.bond_features(), chemprop_batch.add_features()
            # ----------------------------------

            dummy_array = [0]*args.sequence_length
            sequence_2_ar = [list(tokenizer.encode(list(t[0]))) + dummy_array for t in protein_sequence_batch]
            new_ar = []
            for arr in sequence_2_ar:
                while len(arr) > args.sequence_length:
                    arr.pop(len(arr)-1)
                new_ar.append(np.zeros(args.sequence_length) + np.array(arr))
            
            sequence_tensor = torch.LongTensor(new_ar)
            add_feature = torch.Tensor(add_feature)

            batch_preds = model(mol_batch, sequence_tensor, add_feature, features_batch, 
                                atom_descriptors_batch, atom_features_batch, bond_features_batch)
            
            batch_preds = batch_preds.data.cpu().numpy()
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)

            preds.extend(batch_preds.tolist())

    return preds