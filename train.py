"""Trains a chemprop model on a dataset."""

from chemprop.train import chemprop_train
from rdkit import RDLogger
import warnings
from torch_geometric.loader import DataLoader as PyGDataLoader
train_loader = PyGDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
# Disable RDKit logging (C++ level)
RDLogger.DisableLog('rdApp.*')

# Disable Python warnings (optional, for other library warnings)
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    chemprop_train()
