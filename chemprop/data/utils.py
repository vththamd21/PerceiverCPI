import csv
from logging import Logger
from typing import List, Tuple
import numpy as np

from .data import MoleculeDatapoint, MoleculeDataset
from .scaffold import scaffold_split
from chemprop.args import TrainArgs
from chemprop.features import load_features

def get_data(path: str,
             skip_invalid_smiles: bool = True,
             args: TrainArgs = None,
             features_path: List[str] = None,
             max_data_size: int = None,
             use_compound_names: bool = None) -> MoleculeDataset:
    """
    Gets smiles string and target values (and optionally compound names if provided) from a CSV file.
    """
    if args is not None:
        features_path = args.features_path
        max_data_size = args.max_data_size
        use_compound_names = args.use_compound_names

    if features_path is not None:
        features_data = []
        for feat_path in features_path:
            features_data.append(load_features(feat_path))
        features_data = np.concatenate(features_data, axis=1)
    else:
        features_data = None

    skip_smiles = set()
    
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # --- Tự động tìm cột pdb_path ---
        try:
            pdb_path_index = header.index('pdb_path')
        except ValueError:
            pdb_path_index = None
        
        data = []
        for i, line in enumerate(reader):
            if skip_invalid_smiles and line[0] in skip_smiles: 
                continue

            features = features_data[i] if type(features_data) != type(None) else None

            datapoint = MoleculeDatapoint(
                line=line,
                args=args,
                features=features,
                use_compound_names=use_compound_names,
                pdb_path_index=pdb_path_index
            )
            data.append(datapoint)

            if max_data_size and len(data) >= max_data_size: 
                break

    return MoleculeDataset(data)

def get_data_from_smiles(smiles: List[str], skip_invalid_smiles: bool = True, logger: Logger = None) -> MoleculeDataset:
    """
    Converts SMILES to a MoleculeDataset.
    """
    data = [MoleculeDatapoint([smile]) for smile in smiles]
    return MoleculeDataset(data)

def split_data(data: MoleculeDataset,
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0,
               args: TrainArgs = None,
               logger: Logger = None) -> Tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset]:
    """
    Splits data into train, validation, and test splits.
    """
    np.random.seed(seed)
    
    if split_type == 'scaffold_balanced':
        return scaffold_split(data, sizes=sizes, balanced=True, seed=seed, logger=logger)
    elif split_type == 'random':
        indices = list(range(len(data)))
        np.random.shuffle(indices)
        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))
        train = [data[i] for i in indices[:train_size]]
        val = [data[i] for i in indices[train_size:train_val_size]]
        test = [data[i] for i in indices[train_val_size:]]
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)
    else:
        raise ValueError(f'split_type "{split_type}" not supported.')

def get_class_sizes(data: MoleculeDataset) -> List[List[float]]:
    """
    Determines the proportions of the different classes in the classification dataset.
    """
    targets = data.targets()
    num_tasks = len(targets[0])
    class_sizes = []
    for i in range(num_tasks):
        valid_targets = [t[i] for t in targets if t[i] is not None]
        num_pos = sum(valid_targets)
        num_neg = len(valid_targets) - num_pos
        class_sizes.append([num_neg, num_pos])
    return class_sizes

def validate_dataset_type(data: MoleculeDataset, dataset_type: str) -> None:
    """
    Validates the dataset type to ensure the data matches the provided type.
    """
    if dataset_type == 'classification':
        targets = data.targets()
        for i in range(len(targets[0])):
            valid_targets = [t[i] for t in targets if t[i] is not None]
            unique_targets = set(valid_targets)
            if not (unique_targets <= {0, 1}):
                raise ValueError('Classification dataset targets must be 0 or 1.')