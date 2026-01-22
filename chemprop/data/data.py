import threading
from collections import OrderedDict
from random import Random
from typing import Dict, Iterator, List, Optional, Union, Tuple

import numpy as np
import torch  # <--- Quan trọng: Cần import torch
from transformers import AutoTokenizer # <--- Quan trọng: Cần import transformers
from torch.utils.data import DataLoader, Dataset, Sampler
from rdkit import Chem

from .scaler import StandardScaler
from chemprop.features import get_features_generator
from chemprop.features import BatchMolGraph, MolGraph
from chemprop.features import is_explicit_h, is_reaction
from chemprop.rdkit import make_mol
from rdkit.Chem import AllChem

# --- [START] ESM CONFIGURATION ---
# Khai báo Tokenizer toàn cục để tránh lỗi NameError
ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
ESM_MAX_LEN = 1024

try:
    print(f"Loading ESM Tokenizer: {ESM_MODEL_NAME}...")
    GLOBAL_ESM_TOKENIZER = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
except Exception as e:
    print(f"Warning: Could not load ESM Tokenizer. Error: {e}")
    GLOBAL_ESM_TOKENIZER = None
# --- [END] ESM CONFIGURATION ---


# Cache of graph featurizations
CACHE_GRAPH = True
SMILES_TO_GRAPH: Dict[str, MolGraph] = {}


def cache_graph() -> bool:
    r"""Returns whether :class:`~chemprop.features.MolGraph`\ s will be cached."""
    return CACHE_GRAPH


def set_cache_graph(cache_graph: bool) -> None:
    r"""Sets whether :class:`~chemprop.features.MolGraph`\ s will be cached."""
    global CACHE_GRAPH
    CACHE_GRAPH = cache_graph


def empty_cache():
    r"""Empties the cache of :class:`~chemprop.features.MolGraph` and RDKit molecules."""
    SMILES_TO_GRAPH.clear()
    SMILES_TO_MOL.clear()


# Cache of RDKit molecules
CACHE_MOL = True
SMILES_TO_MOL: Dict[str, Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]] = {}


def cache_mol() -> bool:
    r"""Returns whether RDKit molecules will be cached."""
    return CACHE_MOL


def set_cache_mol(cache_mol: bool) -> None:
    r"""Sets whether RDKit molecules will be cached."""
    global CACHE_MOL
    CACHE_MOL = cache_mol


class MoleculeDatapoint:
    """A :class:`MoleculeDatapoint` contains a single molecule and its associated features and targets."""

    def __init__(self,
                 smiles: List[str],
                 sequences: List[str],
                 targets: List[Optional[float]] = None,
                 row: OrderedDict = None,
                 data_weight: float = 1,
                 features: np.ndarray = None,
                 features_generator: List[str] = None,
                 atom_features: np.ndarray = None,
                 atom_descriptors: np.ndarray = None,
                 bond_features: np.ndarray = None,
                 overwrite_default_atom_features: bool = False,
                 overwrite_default_bond_features: bool = False):
        """
        :param smiles: A list of the SMILES strings for the molecules.
        :param sequences: A list of protein sequences.
        :param targets: A list of targets for the molecule.
        ...
        """
        if features is not None and features_generator is not None:
            raise ValueError('Cannot provide both loaded features and a features generator.')

        self.smiles = smiles
        self.sequences = sequences
        self.targets = targets
        self.row = row
        self.data_weight = data_weight
        self.features = features
        self.features_generator = features_generator
        self.atom_descriptors = atom_descriptors
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.overwrite_default_atom_features = overwrite_default_atom_features
        self.overwrite_default_bond_features = overwrite_default_bond_features
        self.is_reaction = is_reaction()
        self.is_explicit_h = is_explicit_h()

        # --- [START] ESM TOKENIZATION LOGIC ---
        # Xử lý Protein Sequence thành Tensor ngay khi khởi tạo Datapoint
        if self.sequences and len(self.sequences) > 0 and GLOBAL_ESM_TOKENIZER is not None:
            try:
                # Lấy sequence đầu tiên (giả định 1 protein/datapoint)
                prot_seq = self.sequences[0]
                
                # Tokenize
                tokenized = GLOBAL_ESM_TOKENIZER(
                    prot_seq,
                    padding='max_length',
                    truncation=True,
                    max_length=ESM_MAX_LEN,
                    return_tensors='pt'
                )
                # Squeeze để bỏ dimension batch (1, seq_len) -> (seq_len)
                self.sequence_tensor = tokenized['input_ids'].squeeze(0)
            except Exception as e:
                print(f"Error tokenizing sequence: {e}")
                self.sequence_tensor = torch.zeros(ESM_MAX_LEN, dtype=torch.long)
        else:
            # Fallback nếu không có sequence hoặc tokenizer lỗi
            self.sequence_tensor = torch.zeros(ESM_MAX_LEN, dtype=torch.long)
        # --- [END] ESM TOKENIZATION LOGIC ---

        # Generate additional features if given a generator
        if self.features_generator is not None:
            self.features = []
            for fg in self.features_generator:
                features_generator = get_features_generator(fg)
                for m in self.mol:
                    if not self.is_reaction:
                        if m is not None and m.GetNumHeavyAtoms() > 0:
                            self.features.extend(features_generator(m))
                        elif m is not None and m.GetNumHeavyAtoms() == 0:
                            self.features.extend(np.zeros(len(features_generator(Chem.MolFromSmiles('C')))))                           
                    else:
                        if m[0] is not None and m[1] is not None and m[0].GetNumHeavyAtoms() > 0:
                            self.features.extend(features_generator(m[0]))
                        elif m[0] is not None and m[1] is not None and m[0].GetNumHeavyAtoms() == 0:
                            self.features.extend(np.zeros(len(features_generator(Chem.MolFromSmiles('C')))))   
            self.features = np.array(self.features)

        # Fix nans in features
        replace_token = 0
        if self.features is not None:
            self.features = np.where(np.isnan(self.features), replace_token, self.features)

        # Fix nans in atom/bond descriptors
        if self.atom_descriptors is not None:
            self.atom_descriptors = np.where(np.isnan(self.atom_descriptors), replace_token, self.atom_descriptors)
        if self.atom_features is not None:
            self.atom_features = np.where(np.isnan(self.atom_features), replace_token, self.atom_features)
        if self.bond_features is not None:
            self.bond_features = np.where(np.isnan(self.bond_features), replace_token, self.bond_features)

        self.raw_features, self.raw_targets = self.features, self.targets
        self.raw_atom_descriptors, self.raw_atom_features, self.raw_bond_features = \
            self.atom_descriptors, self.atom_features, self.bond_features

    @property
    def mol(self) -> Union[List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]]:
        """Gets the corresponding list of RDKit molecules for the corresponding SMILES list."""
        mol = make_mols(self.smiles, self.is_reaction, self.is_explicit_h)
        if cache_mol():
            for s, m in zip(self.smiles, mol):
                SMILES_TO_MOL[s] = m
        return mol

    @property
    def number_of_molecules(self) -> int:
        return len(self.smiles)

    def set_features(self, features: np.ndarray) -> None:
        self.features = features

    def set_atom_descriptors(self, atom_descriptors: np.ndarray) -> None:
        self.atom_descriptors = atom_descriptors

    def set_atom_features(self, atom_features: np.ndarray) -> None:
        self.atom_features = atom_features

    def set_bond_features(self, bond_features: np.ndarray) -> None:
        self.bond_features = bond_features

    def extend_features(self, features: np.ndarray) -> None:
        self.features = np.append(self.features, features) if self.features is not None else features

    def num_tasks(self) -> int:
        return len(self.targets)

    def set_targets(self, targets: List[Optional[float]]):
        self.targets = targets

    def add_features(self) -> List[np.ndarray]:
        if len(self._data) == 0 or self._data[0].features is None:
            return None
        # Note: Logic here seems customized in original file, keeping somewhat generic placeholder or original
        # Original file logic for add_features inside MoleculeDatapoint seemed broken (referencing self._data which belongs to Dataset)
        # Assuming standard getter for features:
        return self.features

    def reset_features_and_targets(self) -> None:
        self.features, self.targets = self.raw_features, self.raw_targets
        self.atom_descriptors, self.atom_features, self.bond_features = \
            self.raw_atom_descriptors, self.raw_atom_features, self.raw_bond_features


class MoleculeDataset(Dataset):
    r"""A :class:`MoleculeDataset` contains a list of :class:`MoleculeDatapoint`\ s."""

    def __init__(self, data: List[MoleculeDatapoint]):
        self._data = data
        self._scaler = None
        self._batch_graph = None
        self._random = Random()
        
        # Placeholder for batch sequence tensor
        self.batch_sequence_tensor = None

    def smiles(self, flatten: bool = False) -> Union[List[str], List[List[str]]]:
        if flatten:
            return [smiles for d in self._data for smiles in d.smiles]
        return [d.smiles for d in self._data]

    def sequences(self, flatten: bool = False) -> Union[List[str], List[List[str]]]:
        if flatten:
            return [sequences for d in self._data for sequences in d.sequences]
        return [d.sequences for d in self._data]

    def mols(self, flatten: bool = False):
        if flatten:
            return [mol for d in self._data for mol in d.mol]
        return [d.mol for d in self._data]

    def add_features(self) -> List[np.ndarray]:
        # Original implementation seemed to use Morgan Fingerprint on the fly here
        # We preserve the logic if it was intended to return the pre-calculated features
        if len(self._data) == 0:
            return None
        return [d.features for d in self._data]

    @property
    def number_of_molecules(self) -> int:
        return self._data[0].number_of_molecules if len(self._data) > 0 else None

    def batch_graph(self) -> List[BatchMolGraph]:
        if self._batch_graph is None:
            self._batch_graph = []
            mol_graphs = []
            for d in self._data:
                mol_graphs_list = []
                for s, m in zip(d.smiles, d.mol):
                    if s in SMILES_TO_GRAPH:
                        mol_graph = SMILES_TO_GRAPH[s]
                    else:
                        if len(d.smiles) > 1 and (d.atom_features is not None or d.bond_features is not None):
                            raise NotImplementedError('Atom descriptors are currently only supported with one molecule per input.')
                        mol_graph = MolGraph(m, d.atom_features, d.bond_features,
                                             overwrite_default_atom_features=d.overwrite_default_atom_features,
                                             overwrite_default_bond_features=d.overwrite_default_bond_features)
                        if cache_graph():
                            SMILES_TO_GRAPH[s] = mol_graph
                    mol_graphs_list.append(mol_graph)
                mol_graphs.append(mol_graphs_list)
            self._batch_graph = [BatchMolGraph([g[i] for g in mol_graphs]) for i in range(len(mol_graphs[0]))]
        return self._batch_graph

    def features(self) -> List[np.ndarray]:
        if len(self._data) == 0 or self._data[0].features is None:
            return None
        return [d.features for d in self._data]

    def atom_features(self) -> List[np.ndarray]:
        if len(self._data) == 0 or self._data[0].atom_features is None:
            return None
        return [d.atom_features for d in self._data]

    def atom_descriptors(self) -> List[np.ndarray]:
        if len(self._data) == 0 or self._data[0].atom_descriptors is None:
            return None
        return [d.atom_descriptors for d in self._data]

    def bond_features(self) -> List[np.ndarray]:
        if len(self._data) == 0 or self._data[0].bond_features is None:
            return None
        return [d.bond_features for d in self._data]

    def data_weights(self) -> List[float]:
        return [d.data_weight for d in self._data]

    def targets(self) -> List[List[Optional[float]]]:
        return [d.targets for d in self._data]

    def num_tasks(self) -> int:
        return self._data[0].num_tasks() if len(self._data) > 0 else None

    def features_size(self) -> int:
        return len(self._data[0].features) if len(self._data) > 0 and self._data[0].features is not None else None

    def atom_descriptors_size(self) -> int:
        return len(self._data[0].atom_descriptors[0]) if len(self._data) > 0 and self._data[0].atom_descriptors is not None else None

    def atom_features_size(self) -> int:
        return len(self._data[0].atom_features[0]) if len(self._data) > 0 and self._data[0].atom_features is not None else None

    def bond_features_size(self) -> int:
        return len(self._data[0].bond_features[0]) if len(self._data) > 0 and self._data[0].bond_features is not None else None

    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0,
                           scale_atom_descriptors: bool = False, scale_bond_features: bool = False) -> StandardScaler:
        if len(self._data) == 0 or \
                (self._data[0].features is None and not scale_bond_features and not scale_atom_descriptors):
            return None

        if scaler is not None:
            self._scaler = scaler
        elif self._scaler is None:
            if scale_atom_descriptors and not self._data[0].atom_descriptors is None:
                features = np.vstack([d.raw_atom_descriptors for d in self._data])
            elif scale_atom_descriptors and not self._data[0].atom_features is None:
                features = np.vstack([d.raw_atom_features for d in self._data])
            elif scale_bond_features:
                features = np.vstack([d.raw_bond_features for d in self._data])
            else:
                features = np.vstack([d.raw_features for d in self._data])
            self._scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self._scaler.fit(features)

        if scale_atom_descriptors and not self._data[0].atom_descriptors is None:
            for d in self._data:
                d.set_atom_descriptors(self._scaler.transform(d.raw_atom_descriptors))
        elif scale_atom_descriptors and not self._data[0].atom_features is None:
            for d in self._data:
                d.set_atom_features(self._scaler.transform(d.raw_atom_features))
        elif scale_bond_features:
            for d in self._data:
                d.set_bond_features(self._scaler.transform(d.raw_bond_features))
        else:
            for d in self._data:
                d.set_features(self._scaler.transform(d.raw_features.reshape(1, -1))[0])
        return self._scaler

    def normalize_targets(self) -> StandardScaler:
        targets = [d.raw_targets for d in self._data]
        scaler = StandardScaler().fit(targets)
        scaled_targets = scaler.transform(targets).tolist()
        self.set_targets(scaled_targets)
        return scaler

    def set_targets(self, targets: List[List[Optional[float]]]) -> None:
        assert len(self._data) == len(targets)
        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i])

    def reset_features_and_targets(self) -> None:
        for d in self._data:
            d.reset_features_and_targets()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        return self._data[item]


class MoleculeSampler(Sampler):
    """A :class:`MoleculeSampler` samples data from a :class:`MoleculeDataset` for a :class:`MoleculeDataLoader`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle
        self._random = Random(seed)

        if self.class_balance:
            indices = np.arange(len(dataset))
            has_active = np.array([any(target == 1 for target in datapoint.targets) for datapoint in dataset])
            self.positive_indices = indices[has_active].tolist()
            self.negative_indices = indices[~has_active].tolist()
            self.length = 2 * min(len(self.positive_indices), len(self.negative_indices))
        else:
            self.positive_indices = self.negative_indices = None
            self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        if self.class_balance:
            if self.shuffle:
                self._random.shuffle(self.positive_indices)
                self._random.shuffle(self.negative_indices)
            indices = [index for pair in zip(self.positive_indices, self.negative_indices) for index in pair]
        else:
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                self._random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self.length


def construct_molecule_batch(data: List[MoleculeDatapoint]) -> MoleculeDataset:
    r"""
    Constructs a :class:`MoleculeDataset` from a list of :class:`MoleculeDatapoint`\ s.
    Precomputes the BatchMolGraph and batches the ESM sequence tensors.
    """
    # 1. Tạo Dataset
    dataset = MoleculeDataset(data)
    
    # 2. Tạo Graph Batch
    dataset.batch_graph() 

    # 3. [NEW] Gom Sequence Tensor (ESM)
    # Lấy sequence_tensor từ mỗi datapoint
    if hasattr(data[0], 'sequence_tensor'):
        seq_tensors = [d.sequence_tensor for d in data]
        try:
            # Stack thành [Batch_Size, Max_Len]
            dataset.batch_sequence_tensor = torch.stack(seq_tensors)
        except Exception as e:
            print(f"Warning: Failed to stack sequence tensors: {e}")
            dataset.batch_sequence_tensor = None
    else:
        dataset.batch_sequence_tensor = None

    return dataset


class MoleculeDataLoader(DataLoader):
    """A :class:`MoleculeDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`MoleculeDataset`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 batch_size: int = 50,
                 num_workers: int = 8,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._class_balance = class_balance
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._context = 'forkserver'
            self._timeout = 3600

        self._sampler = MoleculeSampler(
            dataset=self._dataset,
            class_balance=self._class_balance,
            shuffle=self._shuffle,
            seed=self._seed
        )

        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=construct_molecule_batch,
            multiprocessing_context=self._context,
            timeout=self._timeout
        )

    @property
    def targets(self) -> List[List[Optional[float]]]:
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')
        return [self._dataset[index].targets for index in self._sampler]

    @property
    def iter_size(self) -> int:
        return len(self._sampler)

    def __iter__(self) -> Iterator[MoleculeDataset]:
        return super(MoleculeDataLoader, self).__iter__()

    
def make_mols(smiles: List[str], reaction: bool, keep_h: bool):
    """Builds a list of RDKit molecules."""
    if reaction:
        mol = [SMILES_TO_MOL[s] if s in SMILES_TO_MOL else (make_mol(s.split(">")[0], keep_h), make_mol(s.split(">")[-1], keep_h)) for s in smiles]
    else:
        mol = [SMILES_TO_MOL[s] if s in SMILES_TO_MOL else make_mol(s, keep_h) for s in smiles]
    return mol
