"""Trains a chemprop model on a dataset."""

from chemprop.train import chemprop_train
from rdkit import RDLogger
import warnings

# Disable RDKit logging (C++ level)
RDLogger.DisableLog('rdApp.*')

# Disable Python warnings (optional, for other library warnings)
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    chemprop_train()
