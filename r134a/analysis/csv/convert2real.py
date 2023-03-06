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
from utils.r134a import R134aConstants
from matplotlib import ticker

R134a = R134aConstants()

NM_TO_ANGSTROM = 10
K_B = 0.008314 # J/MOL K
KJMOL_TO_K = 1.0 / K_B


def main():
    # ID the top ten by lowest average MAPE
    dff = pd.read_csv("r134a-final-iter2.csv", index_col=0)

    data_f = dff[list(R134a.param_names)].values
    param_bounds = R134a.param_bounds
    param_bounds[:5] = param_bounds[:5] * NM_TO_ANGSTROM
    param_bounds[5:] = param_bounds[5:] * KJMOL_TO_K
    final_f = values_scaled_to_real(data_f, param_bounds)
    print(final_f)
    final = pd.DataFrame(final_f)
    final.columns = ['sigma_C1(A)','sigma_C2','sigma_F1','sigma_F2','sigma_H1','epsilon_C1(K)','epsilon_C2','epsilon_F1','epsilon_F2','epsilon_H1']
    final.to_csv('finalff.csv',index=False)

if __name__ == "__main__":
    main()

