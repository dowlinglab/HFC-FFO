import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import seaborn

sys.path.append("../")

from fffit.utils import values_scaled_to_real
from fffit.utils import values_real_to_scaled
from utils.r14 import R14Constants
from matplotlib import ticker

R14 = R14Constants()

NM_TO_ANGSTROM = 10
K_B = 0.008314 # J/MOL K
KJMOL_TO_K = 1.0 / K_B


def main():
    # ID the top ten by lowest average MAPE
    dff = pd.read_csv("r14-final-iter2.csv", index_col=0)

    data_f = dff[list(R14.param_names)].values
    param_bounds = R14.param_bounds
    param_bounds[:2] = param_bounds[:2] * NM_TO_ANGSTROM
    param_bounds[2:] = param_bounds[2:] * KJMOL_TO_K
    final_f = values_scaled_to_real(data_f, param_bounds)
    print(final_f)
    final = pd.DataFrame(final_f)
    final.columns = ['sigma_C1(A)','sigma_F1','epsilon_C1(K)','epsilon_F1']
    final.to_csv('r14-finalff.csv',index=False)

if __name__ == "__main__":
    main()

