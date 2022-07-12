import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import seaborn

sys.path.append("../")

from utils.r125 import R125Constants
from utils.id_new_samples import prepare_df_density
from utils.analyze_samples import prepare_df_density_errors

R125 = R125Constants()

matplotlib.rc("font", family="sans-serif")
matplotlib.rc("font", serif="Arial")

############################# QUANTITIES TO EDIT #############################
##############################################################################

liquid_density_threshold=500 #kg/m3
iternum = 4

##############################################################################
##############################################################################

csv_path = "../csv/"
in_csv_names = [
    "r125-density-iter" + str(i) + "-results.csv" for i in range(1, iternum + 1)
]

# Read files
df_csvs = [
    pd.read_csv(csv_path + in_csv_name, index_col=0)
    for in_csv_name in in_csv_names
]
dfs = [prepare_df_density(df_csv, R125, liquid_density_threshold)[0] for df_csv in df_csvs]


def main():

    #seaborn.set_palette("Set2")
    seaborn.set_palette("colorblind")
    # Create a dataframe with one row per parameter set
    dfs_paramsets = [prepare_df_density_errors(df, R125) for df in dfs]

    name = "mape_liq_density"
    fig, ax = plt.subplots()
    axins = inset_axes(ax, width="100%", height="100%",
            bbox_to_anchor=(0.65, 0.10, 0.25, 0.45),
            bbox_transform=ax.transAxes, loc=3)
    ax.set_box_aspect(1.2)
    ax.plot(
        dfs_paramsets[0].sort_values(name)[name],
        np.arange(1, 201,1),
        '-s',
        markersize=6,
        linewidth=3,
        alpha=0.4,
        label="LD-1",
    )
    ax.plot(
        dfs_paramsets[1].sort_values(name)[name],
        np.arange(1, 201,1),
        '-s',
        markersize=6,
        linewidth=3,
        alpha=0.4,
        label="LD-2",
    )
    ax.plot(
        dfs_paramsets[2].sort_values(name)[name],
        np.arange(1, 201,1),
        '-s',
        markersize=6,
        linewidth=3,
        alpha=0.4,
        label="LD-3",
    )
    ax.plot(
        dfs_paramsets[3].sort_values(name)[name],
        np.arange(1, 201,1),
        '-s',
        markersize=6,
        linewidth=3,
        alpha=0.4,
        label="LD-4",
    )

    ax.set_ylim(0,205)
    ax.set_xlim(0,100)
    ax.set_yticks([0,50,100,150,200])
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params("both", direction="in", which="both", length=4, labelsize=16, pad=10)
    ax.tick_params("both", which="major", length=8)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")

    ax.set_ylabel(r"$N_\mathrm{cumu.}$ parameter sets", fontsize=20, labelpad=20)
    ax.set_xlabel("Liquid density MAPE", fontsize=20, labelpad=15)
    ax.legend(fontsize=16, loc=(-0.06,1.05), ncol=2, columnspacing=1, handletextpad=0.5)

    axins.plot(
        dfs_paramsets[0].sort_values(name)[name],
        np.arange(1, 201,1),
        '-s',
        markersize=4,
        linewidth=2,
        alpha=0.4,
        label="LD-1",
    )
    axins.plot(
        dfs_paramsets[1].sort_values(name)[name],
        np.arange(1, 201,1),
        '-s',
        markersize=4,
        linewidth=2,
        alpha=0.4,
        label="LD-2",
    )
    axins.plot(
        dfs_paramsets[2].sort_values(name)[name],
        np.arange(1, 201,1),
        '-s',
        markersize=4,
        linewidth=2,
        alpha=0.4,
        label="LD-3",
    )
    axins.plot(
        dfs_paramsets[3].sort_values(name)[name],
        np.arange(1, 201,1),
        '-s',
        markersize=4,
        linewidth=2,
        alpha=0.4,
        label="LD-4",
    )

    axins.set_xlim(0,2.5)
    axins.set_ylim(0,200)
    axins.tick_params("both", direction="in", which="both", length=3, labelsize=12)
    axins.tick_params("both", which="major", length=6)
    axins.xaxis.set_major_locator(MultipleLocator(1))
    axins.xaxis.set_minor_locator(AutoMinorLocator(2))
    axins.yaxis.set_major_locator(MultipleLocator(50))
    axins.yaxis.set_minor_locator(AutoMinorLocator(2))
    axins.xaxis.set_ticks_position("both")
    axins.yaxis.set_ticks_position("both")

    fig.tight_layout()
    fig.savefig("pdfs/fig2_r125-density-cumu.pdf")

if __name__ == "__main__":
    main()
