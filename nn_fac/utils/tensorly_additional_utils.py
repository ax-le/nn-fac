""" Additional utils for tensorly."""

import tensorly as tl
import numpy as np
if tl.get_backend() == "pytorch":
    import torch # The goal here is to avoid loading pytorch if not needed, especially for venv with only numpy.
# Maybe add a check to see if module is imported inside the functions.

def set_random_state(seed):
    match tl.get_backend():
        case "numpy":
            np.random.seed(seed)
        case "pytorch":
            np.random.seed(seed)
            torch.manual_seed(seed)
        case _:
            raise NotImplementedError("Backend not supported.")

def log(x):
    match tl.get_backend():
        case "numpy":
            return np.log(x, where=x!=0, out=1e-12*np.ones_like(x))
        case "pytorch":
            return torch.where(x!=0, torch.log(x), 1e-12*torch.ones_like(x))
        case _:
            raise NotImplementedError("Backend not supported.")