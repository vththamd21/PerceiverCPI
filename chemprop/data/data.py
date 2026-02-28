from collections import defaultdict
import csv
from typing import Callable, ClassVar, Dict, List, Set, Tuple, Union
import numpy as np
from torch.utils.data.dataset import Dataset
from rdkit import Chem

from chemprop.features import get_features_generator
from chemprop.features import get_smiles_from_name
from chemprop.features.pdb_features import get_calpha_distance_matrix

# =====================================================================
# CÁC HÀM CACHE GIẢ (DUMMY FUNCTIONS) ĐỂ TRÁNH LỖI IMPORT CỦA CHEM-PROP
# =====================================================================
CACHE_GRAPH = False
CACHE_MOL = False

def set_cache_graph(set_cache: bool) -> None:
    global CACHE_GRAPH
    CACHE_GRAPH = set_cache

def set_cache_mol(set_cache: bool) -> None:
    global CACHE_MOL
    CACHE_MOL = set_cache

def empty_cache() -> None:
    pass

def cache_graph() -> bool:
    global CACHE_GRAPH
    return CACHE_GRAPH

def cache_mol() -> bool:
    global CACHE_MOL
    return CACHE_MOL
# =====================================================================

class MoleculeDatapoint:
    """
    Chứa một phân tử (molecule), protein 3D, các đặc trưng (features) và nhãn (targets).
    """
    def __init__(self, line: List[str], args = None, features: np.ndarray = None, use_compound_names: bool = False, pdb_path_index: int = None):
        if args is not None:
            self.features_generator = getattr(args, 'features_generator', None)
            self.args = args
        else:
            self.features_generator = self.args = None

        if features is not None and self.features_generator is not None:
            raise ValueError('Cannot provide both loaded features and a features generator.')

        self.features = features
        self.atom_descriptors = None
        self.atom_features = None
        self.bond_features = None

        # Xử lý nếu CSV có cột tên hợp chất ở đầu
        if use_compound_names:
            self.compound_name = line[0]
            line = line[1:]
        else:
            self.compound_name = None

        self.smiles = line[0]
        self.sequence = line[1]
        self.targets = [float(x) if x != '' else None for x in line[2:3]]
        
        # Đường dẫn file 3D PDB
        self.pdb_path = line[pdb_path_index] if pdb_path_index is not None and len(line) > pdb_path_index else None
        self.distance_matrix = None

        if self.args is not None and getattr(self.args, 'data_weights_path', None) is not None:
            self.data_weight = float(line[-1])
        else:
            self.data_weight = 1.0

        # Khởi tạo phân tử bằng RDKit
        self.mol = Chem.MolFromSmiles(self.smiles)

        if self.mol is None:
            if self.compound_name is not None:
                self.smiles = get_smiles_from_name(self.compound_name)
                self.mol = Chem.MolFromSmiles(self.smiles)

            if self.mol is None:
                raise ValueError(f'Invalid SMILES or molecule name: {self.smiles}')

        # Tạo features nếu có
        if self.features_generator is not None:
            self.features = []
            for fg in self.features_generator:
                features_generator_func = get_features_generator(fg)
                for m in self.mol:
                    if m is not None and m.GetNumHeavyAtoms() > 0:
                        self.features.extend(features_generator_func(m))
            self.features = np.array(self.features)

    def get_distance_matrix(self, max_length=1024) -> np.ndarray:
        """Đọc file PDB và trả về ma trận không gian 3D"""
        if self.distance_matrix is None and self.pdb_path is not None:
            self.distance_matrix = get_calpha_distance_matrix(self.pdb_path, max_length)
        elif self.distance_matrix is None:
            self.distance_matrix = np.zeros((max_length, max_length))
        return self.distance_matrix

    def set_features(self, features: np.ndarray) -> None:
        self.features = features

class MoleculeDataset(Dataset):
    """
    Dataset chứa danh sách các MoleculeDatapoint.
    """
    def __init__(self, data: List[MoleculeDatapoint]):
        self.data = data

    def distance_matrices(self) -> List[np.ndarray]:
        return [dp.get_distance_matrix() for dp in self.data]

    def add_features(self) -> List[np.ndarray]:
        if len(self.data) == 0 or getattr(self.data[0], 'features', None) is None: return None
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
        if len(self.data) == 0 or getattr(self.data[0], 'features', None) is None: return None
        return [dp.features for dp in self.data]

    def atom_descriptors(self) -> List[np.ndarray]:
        if len(self.data) == 0 or getattr(self.data[0], 'atom_descriptors', None) is None: return None
        return [dp.atom_descriptors for dp in self.data]

    def atom_features(self) -> List[np.ndarray]:
        if len(self.data) == 0 or getattr(self.data[0], 'atom_features', None) is None: return None
        return [dp.atom_features for dp in self.data]

    def bond_features(self) -> List[np.ndarray]:
        if len(self.data) == 0 or getattr(self.data[0], 'bond_features', None) is None: return None
        return [dp.bond_features for dp in self.data]

    def data_weights(self) -> List[float]:
        return [dp.data_weight for dp in self.data]

    def targets(self) -> List[List[float]]:
        return [dp.targets for dp in self.data]

    def num_tasks(self) -> int:
        return len(self.data[0].targets) if len(self.data) > 0 else 0

    def normalize_features(self, scaler) -> None:
        if len(self.data) == 0 or getattr(self.data[0], 'features', None) is None: return None
        for dp in self.data:
            dp.set_features(scaler.transform(dp.features.reshape(1, -1))[0])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        return self.data[item]