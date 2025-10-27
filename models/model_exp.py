import pandas as pd
import numpy as np
import torch
import pickle
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from models.model_ata import *
from models.preprocessing import convert_to_int

import seaborn as sns
import matplotlib.pyplot as plt


def read_pickle(f_name):
    with open(f_name, 'rb') as f:
        return pickle.load(f)

def write_pickle(f_name, data):
    with open(f_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        return True

def map_one_hot(x, pnA, pnB, sel=0):
    try:
        idx_A = x.index(1)
        value = pnA[idx_A]
    except (ValueError, IndexError):
        result = [0] * len(pnB)
        return result, False

    if value in pnB:
        idx_B = pnB.index(value)
        result = [0] * len(pnB)
        result[idx_B] = 1
        return result, True
    else:
        result = [0] * len(pnB)
        result[sel] = 1
        return result, False

## Mapping same, but different ordered ATAs
def map_ata_list(x, ataA, ataB):
    result = [0] * len(ataB)
    for idx_A, value in enumerate(ataA):
        if value in ataB:
            idx_B = ataB.index(value)
            result[idx_B] = x[idx_A]
    return result
