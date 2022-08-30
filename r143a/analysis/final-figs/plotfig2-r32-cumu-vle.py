import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import seaborn

sys.path.append("../")

from utils.r32 import R32Constants
from utils.id_new_samples import prepare_df_vle
from utils.analyze_samples import prepare_df_vle_errors
from utils.analyze_samples import prepare_df_density_errors

R32 = R32Constants()

matplotlib.rc("font", family="sans-serif")
matplotlib.rc("font", serif="Arial")

############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum = 3

##############################################################################
##############################################################################

csv_path = "../csv/"
in_csv_names = [
    "r32-vle-iter" + str(i) + "-results.csv" for i in range(1, iternum + 1)
]

# Read files
df_csvs = [
    pd.read_csv(csv_path + in_csv_name, index_col=0)
    for in_csv_name in in_csv_names
]
dfs = [prepare_df_vle(df_csv, R32) for df_csv in df_csvs]


def main():

    # Create a dataframe with one row per parameter set
    dfs_paramsets = [prepare_df_vle_errors(df, R32) for df in dfs]

    names = {
        "mape_liq_density": "Liquid density",
        "mape_vap_density": "Vapor density",
        "mape_Pvap": "Vapor pressure",
        "mape_Hvap": "Enthalpy of vaporization",
        "mape_Tc": "Critical temperature",
        "mape_rhoc": "Critical density",
    }

    # Plot MAPE sorted by each property
    fig, axes = plt.subplots(6, 1, figsize=(4,7))
    piter=0
    for name, label in names.items():
        axes[piter].plot(
            dfs_paramsets[0].sort_values(name)[name],
            np.arange(1, 26,1),
            '-o',
            markersize=6,
            alpha=0.8,
            label="VLE-1",
        )
        axes[piter].plot(
            dfs_paramsets[1].sort_values(name)[name],
            np.arange(1, 26, 1),
            '-o',
            markersize=6,
            alpha=0.8,
            label="VLE-2",
        )
        axes[piter].plot(
            dfs_paramsets[2].sort_values(name)[name],
            np.arange(1, 26, 1),
            '-o',
            markersize=6,
            alpha=0.8,
            label="VLE-3",
        )
        axes[piter].text(
            9.7, 2.5,
            label,
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=16,
        )
        axes[piter].set_ylim(0,28)
        axes[piter].set_xlim(0,10)
        axes[piter].set_yticks([0,10,20])
        axes[piter].set_yticks([5,15], minor=True)
        axes[piter].xaxis.set_major_locator(MultipleLocator(2))
        axes[piter].xaxis.set_minor_locator(AutoMinorLocator(2))
        axes[piter].tick_params("both", direction="in", which="both", labelbottom=False, length=2, labelsize=16)
        axes[piter].tick_params("both", which="major", length=4)
        axes[piter].xaxis.set_ticks_position("both")
        axes[piter].yaxis.set_ticks_position("both")
        piter +=1

    text = axes[5].set_ylabel(r"$N_\mathrm{cumu.}$ parameter sets", fontsize=20, labelpad=14)
    text.set_y(3)
    axes[5].set_xlabel("Property MAPE", fontsize=20, labelpad=10)
    axes[5].tick_params(labelbottom=True)
    plt.subplots_adjust(hspace=.0)
    plt.subplots_adjust(left = 0.18, right=0.92, top=0.84, bottom=0.15)
    axes[0].legend(fontsize=16, loc=(-0.07,1.07), ncol=3, columnspacing=1, handletextpad=0.5)
    #print(fig.get_size_inches())
    fig.set_size_inches(5, 6)
    fig.savefig("pdfs/fig2_r32-cumu-vle.pdf")

if __name__ == "__main__":
    main()
