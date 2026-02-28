from collections import defaultdict
import csv
from typing import Callable, ClassVar, Dict, List, Set, Tuple, Union
import numpy as np
from torch.utils.data.dataset import Dataset
from rdkit import Chem

#from chemprop.args import TrainArgs
from chemprop.features import get_features_generator
from chemprop.features.pdb_features import get_calpha_distance_matrix

class MoleculeDatapoint:
    def __init__(self, line: List[str], args = None, features: np.ndarray = None, use_compound_names: bool = False, pdb_path_index: int = None):
        if args is not None:
            self.features_generator = args.features_generator
            self.args = args
        else:
            self.features_generator = self.args = self.atom_descriptors_generator = None

        if features is not None and self.features_generator is not None:
            raise ValueError('Cannot provide both loaded features and a features generator.')

        self.features = features
        self.atom_descriptors = None
        self.atom_features = None
        self.bond_features = None

        if use_compound_names:
            self.compound_name = line[0]
            line = line[1:]
        else:
            self.compound_name = None

        self.smiles = line[0]
        self.sequence = line[1]
        self.targets = [float(x) if x != '' else None for x in line[2:3]]
        
        # Đường dẫn tới file PDB
        self.pdb_path = line[pdb_path_index] if pdb_path_index is not None and len(line) > pdb_path_index else None
        self.distance_matrix = None

        if self.args is not None and self.args.data_weights_path is not None:
            self.data_weight = float(line[-1])
        else:
            self.data_weight = 1.0

        self.mol = Chem.MolFromSmiles(self.smiles)

        if self.features_generator is not None:
            self.features = []
            for fg in self.features_generator:
                features_generator = get_features_generator(fg)
                for m in self.mol:
                    if m is not None and m.GetNumHeavyAtoms() > 0:
                        self.features.extend(features_generator(m))
            self.features = np.array(self.features)

    def get_distance_matrix(self, max_length=1024) -> np.ndarray:
        if self.distance_matrix is None and self.pdb_path is not None:
            self.distance_matrix = get_calpha_distance_matrix(self.pdb_path, max_length)
        elif self.distance_matrix is None:
            self.distance_matrix = np.zeros((max_length, max_length))
        return self.distance_matrix

    def set_features(self, features: np.ndarray) -> None:
        self.features = features

class MoleculeDataset(Dataset):
    def __init__(self, data: List[MoleculeDatapoint]):
        self.data = data

    def distance_matrices(self) -> List[np.ndarray]:
        return [dp.get_distance_matrix() for dp in self.data]

    def add_features(self) -> List[np.ndarray]:
        if len(self.data) == 0 or self.data[0].features is None: return None
        return [dp.features for dp in self.data]

    def smiles(self) -> List[str]:
        return [dp.smiles for dp in self.data]

    def sequences(self) -> List[str]:
        return [dp.sequence for dp in self.data]

    def mols(self) -> List[Chem.Mol]:
        return [dp.mol for dp in self.data]

    def batch_graph(self) -> List[Chem.Mol]:
        return self.mols()

    def features(self) -> List[np.ndarray]:
        if len(self.data) == 0 or self.data[0].features is None: return None
        return [dp.features for dp in self.data]

    def atom_descriptors(self) -> List[np.ndarray]:
        if len(self.data) == 0 or self.data[0].atom_descriptors is None: return None
        return [dp.atom_descriptors for dp in self.data]

    def atom_features(self) -> List[np.ndarray]:
        if len(self.data) == 0 or self.data[0].atom_features is None: return None
        return [dp.atom_features for dp in self.data]

    def bond_features(self) -> List[np.ndarray]:
        if len(self.data) == 0 or self.data[0].bond_features is None: return None
        return [dp.bond_features for dp in self.data]

    def data_weights(self) -> List[float]:
        return [dp.data_weight for dp in self.data]

    def targets(self) -> List[List[float]]:
        return [dp.targets for dp in self.data]

    def num_tasks(self) -> int:
        return len(self.data[0].targets) if len(self.data) > 0 else 0

    def normalize_features(self, scaler) -> None:
        if len(self.data) == 0 or self.data[0].features is None: return None
        for dp in self.data:
            dp.set_features(scaler.transform(dp.features.reshape(1, -1))[0])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        return self.data[item]
    
    CACHE_GRAPH = False

    def set_cache_graph(set_cache: bool) -> None:
    global CACHE_GRAPH
    CACHE_GRAPH = set_cache

    def empty_cache() -> None:
        pass
    def cache_graph() -> bool:
        global CACHE_GRAPH
        return CACHE_GRAPH