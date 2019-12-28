# import torch

# import warnings
# warnings.filterwarnings("once")
#
# from torch.optim.optimizer import Optimizer
# import math
# import itertools as it
# import torch.optim as optim
#
#
# def get_scheduler(scheduler: str = 'ReduceLROnPlateau'):
#     # https://github.com/lonePatient/lookahead_pytorch/blob/master/run.py
#
#     if scheduler == 'ReduceLROnPlateau':
#         scheduler = ReduceLROnPlateau(optimizer, factor=0.8, patience=3)
#
#
#     return scheduler
