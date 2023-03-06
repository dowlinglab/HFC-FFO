import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

from fffit.utils import (
    values_real_to_scaled,
    values_scaled_to_real,
)

sys.path.append("../")

from utils.r143a import R143aConstants
from utils.id_new_samples import prepare_df_vle
from utils.analyze_samples import prepare_df_vle_errors
from utils.plot import plot_property, render_mpl_table

from fffit.pareto import find_pareto_set, is_pareto_efficient

R143a = R143aConstants()

import matplotlib._color_data as mcd

############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum = 4

##############################################################################
##############################################################################

csv_path = "/scratch365/nwang2/ff_development/HFC_143a_FFO_FF/r143a/analysis/csv/"
in_csv_names = [
    "r143a-vle-iter" + str(i) + "-results.csv" for i in range(1, iternum + 1)
]
out_csv_name = "r143a-all-vle.csv"

# Read files
df_csvs = [
    pd.read_csv(csv_path + in_csv_name, index_col=0)
    for in_csv_name in in_csv_names
]
df_csv = pd.concat(df_csvs)
df_all = prepare_df_vle(df_csv, R143a)
print(df_all)
print(df_all.shape)

def main():

    # Create a dataframe with one row per parameter set
    df_paramsets = prepare_df_vle_errors(df_all, R143a)
    print(df_paramsets)
    print(df_paramsets)

    '''# ID pareto points
    result, pareto_points, dominated_points = find_pareto_set(
        df_paramsets.filter(["mse_liq_density", "mse_vap_density", "mse_Pvap", "mse_Hvap", "mse_Tc", "mse_rhoc"]).values,
        is_pareto_efficient
    )
    df_paramsets = df_paramsets.join(pd.DataFrame(result, columns=["is_pareto"]))

    df_paramsets[df_paramsets["is_pareto"]==True].to_csv(csv_path + "/" + out_csv_name)'''


if __name__ == "__main__":
    main()
