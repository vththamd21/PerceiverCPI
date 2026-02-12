from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.models import InteractionModel

def predict(model: InteractionModel,
            data_loader: MoleculeDataLoader,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~chemprop.models.model.InteractionModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~chemprop.data.scaler.StandardScaler` object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples, while the inner list is tasks.
    """
    model.eval()

    preds = []

    # Lấy tokenizer từ model (được khởi tạo trong InteractionModel)
    # Lưu ý: Đảm bảo model đã load ESM-2 thành công
    if hasattr(model, 'module'):
        esm_tokenizer = model.module.esm_tokenizer
    else:
        esm_tokenizer = model.esm_tokenizer

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, protein_sequence_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch, add_feature = \
            batch.batch_graph(), batch.features(), batch.sequences(), batch.atom_descriptors(), \
            batch.atom_features(), batch.bond_features(), batch.add_features()

        # --- XỬ LÝ PROTEIN (FIXED FOR ESM-2) ---
        # Lấy danh sách chuỗi raw string từ batch
        # protein_sequence_batch là list các tuple (sequence, ...), lấy phần tử đầu tiên
        raw_sequences = [t[0] for t in protein_sequence_batch]

        # Tokenize bằng ESM Tokenizer
        encoded_inputs = esm_tokenizer(
            raw_sequences,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        # ---------------------------------------

        # Run model
        with torch.no_grad():
            batch_preds = model(
                batch=mol_batch,
                sequence_inputs=encoded_inputs, # Truyền dict từ tokenizer
                add_feature=torch.Tensor(add_feature),
                features_batch=features_batch,
                atom_descriptors_batch=atom_descriptors_batch,
                atom_features_batch=atom_features_batch,
                bond_features_batch=bond_features_batch
            )

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if applicable
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        preds.extend(batch_preds.tolist())

    return preds
