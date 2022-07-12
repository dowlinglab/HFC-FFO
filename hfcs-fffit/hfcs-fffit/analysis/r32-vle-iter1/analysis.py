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

from utils.r32 import R32Constants
from utils.id_new_samples import prepare_df_vle
from utils.analyze_samples import prepare_df_vle_errors
from utils.plot import plot_property, render_mpl_table

R32 = R32Constants()

import matplotlib._color_data as mcd

############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum = 1

##############################################################################
##############################################################################

csv_path = "/scratch365/rdefever/hfcs-fffit/hfcs-fffit/analysis/csv/"
in_csv_names = [
    "r32-vle-iter" + str(i) + "-results.csv" for i in range(1, iternum + 1)
]
out_csv_name = "r32-vle-iter" + str(iternum + 1) + "-params.csv"

# Read files
df_csvs = [
    pd.read_csv(csv_path + in_csv_name, index_col=0)
    for in_csv_name in in_csv_names
]
df_csv = pd.concat(df_csvs)
df_all = prepare_df_vle(df_csv, R32)


def main():

    # Basic plots to view output
    plot_property(
        df_all,
        "liq_density",
        R32.liq_density_bounds,
        axis_name="Liquid Density [kg/m$^3$]",
    )
    plot_property(
        df_all,
        "vap_density",
        R32.vap_density_bounds,
        axis_name="Vapor Density [kg/m$^3$]",
    )
    plot_property(
        df_all, "Pvap", R32.Pvap_bounds, axis_name="Vapor Pressure [bar]"
    )
    plot_property(
        df_all,
        "Hvap",
        R32.Hvap_bounds,
        axis_name="Enthalpy of vaporization [kJ/kg]",
    )

    # Create a dataframe with one row per parameter set
    df_paramsets = prepare_df_vle_errors(df_all, R32)

    # Plot MSE for each property
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, figsize=(5, 10))
    fig.suptitle("Mean square errors", y=1.05, x=0.6)
    ax1.plot(
        range(df_paramsets.shape[0]),
        df_paramsets["mse_liq_density"],
        label="Liquid Density",
        color="black",
    )
    ax1.set_title("Liquid density")
    ax1.set_ylabel("kg$^2$/m$^2$")
    ax2.plot(
        range(df_paramsets.shape[0]),
        df_paramsets["mse_vap_density"],
        label="Vapor Density",
        color="black",
    )
    ax2.set_title("Vapor density", x=0.75, y=0.8)
    ax2.set_ylabel("kg$^2$/m$^2$")
    ax3.plot(
        range(df_paramsets.shape[0]),
        df_paramsets["mse_Pvap"],
        label="Vapor Pressure",
        color="black",
    )
    ax3.set_title("Vapor pressure")
    ax3.set_ylabel("bar$^2$")
    ax4.plot(
        range(df_paramsets.shape[0]),
        df_paramsets["mse_Hvap"],
        label="Enthalpy of Vaporization",
        color="black",
    )
    ax4.set_title("Enthalpy of vaporization")
    ax4.set_ylabel("kJ$^2$/mol$^2$")
    ax5.plot(
        range(df_paramsets.shape[0]),
        df_paramsets["mse_Tc"],
        label="Critical Temperature",
        color="black",
    )
    ax5.set_title("Critical temperature")
    ax5.set_ylabel("K$^2$")
    ax6.plot(
        range(df_paramsets.shape[0]),
        df_paramsets["mse_rhoc"],
        label="Critical Density",
        color="black",
    )
    ax6.set_title("Critical density")
    ax6.set_ylabel("kg$^2$/m$^2$")
    ax6.set_xlabel("Index (Unsorted)")
    fig.tight_layout()
    fig.savefig("figs/mse.png", dpi=300)

    fig, ax = plt.subplots()
    # Plot MAPE with all properties
    names = {
        "mape_liq_density": "Liquid density",
        "mape_vap_density": "Vapor density",
        "mape_Pvap": "Vapor pressure",
        "mape_Hvap": "Enthalpy of vaporization",
        "mape_Tc": "Critical temperature",
        "mape_rhoc": "Critical density",
    }
    for name, label in names.items():
        ax.plot(
            range(df_paramsets.shape[0]), df_paramsets[name], label=label,
        )
    ax.set_xlabel("Index (Unsorted)", fontsize=16, labelpad=15)
    ax.set_ylabel("Mean abs. % error", fontsize=16, labelpad=15)
    ax.tick_params(axis="both", labelsize=12)
    fig.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig("figs/mape_unsorted.png", dpi=300)

    fig, ax = plt.subplots()
    # Plot MAPE sorted by vapor density MAPE
    for name, label in names.items():
        ax.plot(
            range(df_paramsets.shape[0]),
            df_paramsets.sort_values("mape_vap_density")[name],
            label=label,
        )
    ax.set_xlabel(
        "Index (Sorted by vapor density MAPE)", fontsize=16, labelpad=15
    )
    ax.set_ylabel("Mean abs. % error", fontsize=16, labelpad=15)
    ax.tick_params(axis="both", labelsize=12)
    fig.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig("figs/mape_sorted.png", dpi=300)

    # Plot VLE envelopes
    # Generate colors for param sets
    phi = np.linspace(0, 2 * np.pi, len(df_paramsets))
    rgb_cycle = np.vstack(
        (  # Three sinusoids
            0.5 * (1.0 + np.cos(phi)),  # scaled to [0,1]
            0.5 * (1.0 + np.cos(phi + 2 * np.pi / 3)),  # 120° phase shifted.
            0.5 * (1.0 + np.cos(phi - 2 * np.pi / 3)),
        )
    ).T  # Shape = (df_paramsets,3)

    fig, ax = plt.subplots()
    ax.set_xlabel("Density [kg/m$^3$]", fontsize=16, labelpad=15)
    ax.set_ylabel("Temperature [K]", fontsize=16, labelpad=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(0, 1250)
    ax.set_ylim(230, 370)

    temps = R32.expt_liq_density.keys()
    for temp in temps:
        ax.scatter(
            df_paramsets.filter(regex=(f"liq_density_{float(temp):.0f}K")),
            np.tile(temp, len(df_paramsets)),
            c=rgb_cycle,
            s=60,
            alpha=0.5,
        )
        ax.scatter(
            df_paramsets.filter(regex=(f"vap_density_{float(temp):.0f}K")),
            np.tile(temp, len(df_paramsets)),
            c=rgb_cycle,
            s=60,
            alpha=0.5,
        )
    ax.scatter(
        df_paramsets.filter(regex=("sim_rhoc")),
        df_paramsets.filter(regex=("sim_Tc")),
        c=rgb_cycle,
        s=60,
        alpha=0.5,
    )
    ax.scatter(
        R32.expt_liq_density.values(),
        R32.expt_liq_density.keys(),
        color="black",
        marker="x",
        s=80,
    )
    ax.scatter(
        R32.expt_vap_density.values(),
        R32.expt_vap_density.keys(),
        color="black",
        marker="x",
        s=80,
    )
    ax.scatter(R32.expt_rhoc, R32.expt_Tc, color="black", marker="x", s=80)

    fig.tight_layout()
    fig.savefig("figs/vle-envelope.png", dpi=300)

    # Plot Vapor Pressure
    fig, ax = plt.subplots()
    ax.set_xlabel("Temperature [K]", fontsize=16, labelpad=15)
    ax.set_ylabel("Vapor Pressure [bar]", fontsize=16, labelpad=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(230, 330)
    ax.set_ylim(0, 40)

    for temp in temps:
        ax.scatter(
            np.tile(temp, len(df_paramsets)),
            df_paramsets.filter(regex=(f"Pvap_{float(temp):.0f}K")),
            c=rgb_cycle,
            s=60,
            alpha=0.5,
        )
    ax.scatter(
        R32.expt_Pvap.keys(),
        R32.expt_Pvap.values(),
        color="black",
        marker="x",
        s=80,
    )
    fig.tight_layout()
    fig.savefig("figs/Pvap.png", dpi=300)

    # Plot Enthalpy of Vaporization
    fig, ax = plt.subplots()

    ax.set_xlabel("Temperature [K]", fontsize=16, labelpad=15)
    ax.set_ylabel("Enthalpy of Vaporization [kJ/kg]", fontsize=16, labelpad=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(230, 330)

    for temp in temps:
        ax.scatter(
            np.tile(temp, len(df_paramsets)),
            df_paramsets.filter(regex=(f"Hvap_{float(temp):.0f}K")),
            c=rgb_cycle,
            s=60,
            alpha=0.5,
        )
    ax.scatter(
        R32.expt_Hvap.keys(),
        R32.expt_Hvap.values(),
        color="black",
        marker="x",
        s=80,
    )
    fig.tight_layout()
    fig.savefig("figs/Hvap.png", dpi=300)

    # Save tables of data sorted by sum of squares results
    render_mpl_table(
        df_paramsets.sort_values("mape_liq_density").head(),
        out_name="table_sort_liq_density",
    )
    render_mpl_table(
        df_paramsets.sort_values("mape_vap_density").head(),
        out_name="table_sort_vap_density",
    )
    render_mpl_table(
        df_paramsets.sort_values("mape_Pvap").head(),
        out_name="table_sort_pvap",
    )
    render_mpl_table(
        df_paramsets.sort_values("mape_Hvap").head(),
        out_name="table_sort_hvap",
    )

    # Pair plots to look at correlation of errors
    grid = seaborn.PairGrid(
        data=df_paramsets,
        vars=[
            "mape_liq_density",
            "mape_vap_density",
            "mape_Pvap",
            "mape_Hvap",
            "mape_Tc",
            "mape_rhoc",
        ],
    )
    grid = grid.map_lower(plt.scatter)
    grid = grid.map_diag(plt.hist, bins=10, edgecolor="k")
    for i, j in zip(*np.triu_indices_from(grid.axes, 1)):
        grid.axes[i, j].set_visible(False)

    grid.savefig("figs/mape_pairs.png")

if __name__ == "__main__":
    main()
