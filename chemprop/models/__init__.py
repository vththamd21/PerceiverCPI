# from .model import InteractionModel
# from .mpn import MPN, MPNEncoder
from .model import InteractionModel
from .mpn import MPN, SageEncoder  # Đổi MPNEncoder thành SageEncoder
__all__ = [
    'InteractionModel',
    'MPN',
    'MPNEncoder'
]
