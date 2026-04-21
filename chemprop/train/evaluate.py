import torch
import numpy as np
from chemprop.data import MoleculeDataset
from collections import defaultdict
import logging
from typing import Dict, List
from tqdm import tqdm
from .predict import predict
from chemprop.data import MoleculeDataLoader, StandardScaler
from chemprop.models import InteractionModel
from chemprop.utils import get_metric_func
from chemprop.args import TrainArgs

def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         num_tasks: int,
                         metrics: List[str],
                         dataset_type: str,
                         logger: logging.Logger = None) -> Dict[str, List[float]]:
    """
    Evaluates predictions using a metric function after filtering out invalid targets.

    :param preds: A list of lists of shape :code:`(data_size, num_tasks)` with model predictions.
    :param targets: A list of lists of shape :code:`(data_size, num_tasks)` with targets.
    :param num_tasks: Number of tasks.
    :param metrics: A list of names of metric functions.
    :param dataset_type: Dataset type.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.
    """
    info = logger.info if logger is not None else print

    metric_to_func = {metric: get_metric_func(metric) for metric in metrics}

    if len(preds) == 0:
        return {metric: [float('nan')] * num_tasks for metric in metrics}

    # Filter out empty targets
    # valid_preds and valid_targets have shape (num_tasks, data_size)
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(len(preds)):
            if targets[j][i] is not None:  # Skip those without targets
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])

    # Compute metric
    results = defaultdict(list)
    for i in range(num_tasks):
        # # Skip if all targets or preds are identical, otherwise we'll crash during classification
        if dataset_type == 'classification':
            nan = False
            if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                nan = True
                info('Warning: Found a task with targets all 0s or all 1s')
            if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                nan = True
                info('Warning: Found a task with predictions all 0s or all 1s')

            if nan:
                for metric in metrics:
                    results[metric].append(float('nan'))
                continue

        if len(valid_targets[i]) == 0:
            continue

        for metric, metric_func in metric_to_func.items():
            if dataset_type == 'multiclass' and metric == 'cross_entropy':
                results[metric].append(metric_func(valid_targets[i], valid_preds[i],
                                                   labels=list(range(len(valid_preds[i][0])))))
            else:
                results[metric].append(metric_func(valid_targets[i], valid_preds[i]))

    results = dict(results)

    return results


# def evaluate(model: InteractionModel,
#              data_loader: MoleculeDataLoader,
#              num_tasks: int,
#              metrics: List[str],
#              dataset_type: str,
#              args:TrainArgs,
#              scaler: StandardScaler = None,
#              logger: logging.Logger = None, tokenizer = None) -> Dict[str, List[float]]:
#     """
#     Evaluates an ensemble of models on a dataset by making predictions and then evaluating the predictions.

#     :param model: A :class:`~chemprop.models.model.InteractionModel`.
#     :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
#     :param num_tasks: Number of tasks.
#     :param metrics: A list of names of metric functions.
#     :param dataset_type: Dataset type.
#     :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
#     :param logger: A logger to record output.
#     :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.

#     """
#     preds = predict(
#         model=model,
#         data_loader=data_loader,
#         args = args,
#         scaler=scaler,
#         tokenizer = tokenizer
#     )

#     results = evaluate_predictions(
#         preds=preds,
#         targets=data_loader.targets,
#         num_tasks=num_tasks,
#         metrics=metrics,
#         dataset_type=dataset_type,
#         logger=logger
#     )

#     return results

def evaluate(model: InteractionModel,
             data_loader: MoleculeDataLoader,
             num_tasks: int,
             metrics: List[str],
             dataset_type: str,
             args: TrainArgs,
             scaler: StandardScaler = None,
             logger: logging.Logger = None,
             tokenizer = None) -> Dict[str, List[float]]:
    """
    Evaluates an ensemble of models on a dataset.
    """
    model.eval()
    preds = []
    all_targets = [] # Thêm mảng này để lưu toàn bộ nhãn

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), leave=False):
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

            # Cộng dồn dự đoán và nhãn
            preds.extend(batch_preds.tolist())
            all_targets.extend(chemprop_batch.targets()) # <--- SỬA LỖI Ở ĐÂY

    return evaluate_predictions(
        preds=preds,
        targets=all_targets, # <--- TRUYỀN TOÀN BỘ NHÃN VÀO ĐÂY
        num_tasks=num_tasks,
        metrics=metrics,
        dataset_type=dataset_type,
        logger=logger
    )

    # return evaluate_predictions(
    #     preds=preds,
    #     targets=chemprop_batch.targets(), # Đảm bảo lấy targets đúng từ dữ liệu gốc
    #     num_tasks=num_tasks,
    #     metrics=metrics,
    #     dataset_type=dataset_type,
    #     logger=logger
    )