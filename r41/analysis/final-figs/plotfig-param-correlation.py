#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import seaborn
from fffit.utils import (
    values_real_to_scaled,
    values_scaled_to_real,
)


matplotlib.rc("font", family="sans-serif")
matplotlib.rc("font", serif="Arial")

sys.path.append("../")

from utils.r32 import R32Constants
from utils.id_new_samples import prepare_df_vle
from utils.analyze_samples import prepare_df_vle_errors
from utils.plot import plot_property, render_mpl_table

R32 = R32Constants()

iternum = 3
csv_path = "../csv/"
in_csv_names = [
    "r32-vle-iter" + str(i) + "-results.csv" for i in range(1, iternum + 1)
]

# Read files
df_csvs = [
    pd.read_csv(csv_path + in_csv_name, index_col=0)
    for in_csv_name in in_csv_names
]
df_csv = pd.concat(df_csvs)
df_all = prepare_df_vle(df_csv, R32)

def main():

    # Create a dataframe with one row per parameter set
    df_paramsets = prepare_df_vle_errors(df_all, R32)
    
    best_liqdens = df_paramsets.sort_values('mape_liq_density').iloc[0]
    best_vapdens = df_paramsets.sort_values('mape_vap_density').iloc[0]
    best_pvap = df_paramsets.sort_values('mape_Pvap').iloc[0]
    best_hvap = df_paramsets.sort_values('mape_Hvap').iloc[0]

    #Calculate l1 norm between parameter set values
    n_paramsets = len(df_paramsets)
    pairs = np.zeros((n_paramsets, 4, 2))
    count = 0
    for i in range(n_paramsets):
        params = df_paramsets[list(R32.param_names)].iloc[i].values
        params_liqdens = best_liqdens[list(R32.param_names)].values
        params_vapdens = best_vapdens[list(R32.param_names)].values
        params_pvap = best_pvap[list(R32.param_names)].values
        params_hvap = best_hvap[list(R32.param_names)].values

        dist_liqdens = np.sum(np.abs(params_liqdens - params))
        dist_vapdens = np.sum(np.abs(params_vapdens - params))
        dist_pvap = np.sum(np.abs(params_pvap - params))
        dist_hvap = np.sum(np.abs(params_hvap - params))

        err_liqdens = df_paramsets['mape_liq_density'].iloc[i]
        err_vapdens = df_paramsets['mape_vap_density'].iloc[i]
        err_pvap = df_paramsets['mape_Pvap'].iloc[i]
        err_hvap = df_paramsets['mape_Hvap'].iloc[i]

        pairs[count, 0, :] = [dist_liqdens, err_liqdens]
        pairs[count, 1, :] = [dist_vapdens, err_vapdens]
        pairs[count, 2, :] = [dist_pvap, err_pvap]
        pairs[count, 3, :] = [dist_hvap, err_hvap]

        count += 1
    
    fig, axs = plt.subplots(2,2)
    axs[0,0].scatter(
        pairs[:, 0, 0],
        pairs[:, 0, 1],
        alpha=0.5,
        s=20,
        c="C3"
    )
    axs[0,1].scatter(
        pairs[:, 1, 0],
        pairs[:, 1, 1],
        alpha=0.5,
        s=20,
        c="C3"
    )
    axs[1,0].scatter(
        pairs[:, 2, 0],
        pairs[:, 2, 1],
        alpha=0.5,
        s=20,
        c="C3"
    )
    axs[1,1].scatter(
        pairs[:, 3, 0],
        pairs[:, 3, 1],
        alpha=0.5,
        s=20,
        c="C3"
    )
    axs[0,0].text(0.78, 0.8, "(a)", fontsize=18, transform=axs[0,0].transAxes)
    axs[0,1].text(0.78, 0.8, "(b)", fontsize=18, transform=axs[0,1].transAxes)
    axs[1,0].text(0.78, 0.8, "(c)", fontsize=18, transform=axs[1,0].transAxes)
    axs[1,1].text(0.78, 0.8, "(d)", fontsize=18, transform=axs[1,1].transAxes)

    axs[0,0].set_xlim(-0.1,2.5)
    axs[0,0].xaxis.set_major_locator(MultipleLocator(1))
    axs[0,0].xaxis.set_minor_locator(AutoMinorLocator(2))
    axs[0,0].yaxis.set_minor_locator(AutoMinorLocator(2))
    axs[0,0].tick_params("both", direction="in", which="both", length=2, labelsize=12, pad=5)
    axs[0,0].tick_params("both", which="major", length=4)
    axs[0,0].xaxis.set_ticks_position("both")
    axs[0,0].yaxis.set_ticks_position("both")

    axs[0,0].set_xlabel(r"$L_1$ norm", fontsize=12)
    axs[0,0].set_ylabel(r"MAPE $\rho^l_\mathrm{sat}$", fontsize=12)

    axs[0,1].set_xlim(-0.1,2.5)
    axs[0,1].xaxis.set_major_locator(MultipleLocator(1))
    axs[0,1].xaxis.set_minor_locator(AutoMinorLocator(2))
    axs[0,1].yaxis.set_minor_locator(AutoMinorLocator(2))
    axs[0,1].tick_params("both", direction="in", which="both", length=2, labelsize=12, pad=5)
    axs[0,1].tick_params("both", which="major", length=4)
    axs[0,1].xaxis.set_ticks_position("both")
    axs[0,1].yaxis.set_ticks_position("both")

    axs[0,1].set_xlabel(r"$L_1$ norm", fontsize=12)
    axs[0,1].set_ylabel(r"MAPE $\rho^v_\mathrm{sat}$", fontsize=12)

    axs[1,0].set_xlim(-0.1,2.5)
    axs[1,0].xaxis.set_major_locator(MultipleLocator(1))
    axs[1,0].xaxis.set_minor_locator(AutoMinorLocator(2))
    axs[1,0].yaxis.set_minor_locator(AutoMinorLocator(2))
    axs[1,0].tick_params("both", direction="in", which="both", length=2, labelsize=12, pad=5)
    axs[1,0].tick_params("both", which="major", length=4)
    axs[1,0].xaxis.set_ticks_position("both")
    axs[1,0].yaxis.set_ticks_position("both")

    axs[1,0].set_xlabel(r"$L_1$ norm", fontsize=12)
    axs[1,0].set_ylabel(r"MAPE $P_\mathrm{vap}$", fontsize=12)

    axs[1,1].set_xlim(-0.1,2.5)
    axs[1,1].xaxis.set_major_locator(MultipleLocator(1))
    axs[1,1].xaxis.set_minor_locator(AutoMinorLocator(2))
    axs[1,1].yaxis.set_minor_locator(AutoMinorLocator(2))
    axs[1,1].tick_params("both", direction="in", which="both", length=2, labelsize=12, pad=5)
    axs[1,1].tick_params("both", which="major", length=4)
    axs[1,1].xaxis.set_ticks_position("both")
    axs[1,1].yaxis.set_ticks_position("both")

    axs[1,1].set_xlabel(r"$L_1$ norm", fontsize=12)
    axs[1,1].set_ylabel(r"MAPE $H_\mathrm{vap}$", fontsize=12)



    fig.subplots_adjust(bottom=0.15, top=0.88, left=0.15, right=0.95, wspace=0.55, hspace=0.5)
    fig.savefig("pdfs/fig_paramcorrelation.pdf")

if __name__ == "__main__":
    main()
