import numpy as np

from numpy.typing import NDArray
from typing import Tuple, Dict


def save_data(data: Tuple[NDArray] | Dict,
              out_file: str):

    header = ""
    if isinstance(data, dict):
        first_key = next(iter(data))
        n_rows = len(data[first_key])
        n_cols = len(data)

        data_mx = np.zeros((n_rows, n_cols))
        for i, key in enumerate(data):
            header += key + " "
            if data[key]:
                col = np.array(data[key])
                data_mx[:, i] = col
    else:
        data_mx = np.column_stack(data)

    np.savetxt(out_file, data_mx, header=header)