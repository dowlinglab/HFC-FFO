import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn

sys.path.append("../")

from scipy.stats import linregress
from utils.r32 import R32Constants
from utils.r125 import R125Constants
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

R32 = R32Constants()
R125 = R125Constants()

matplotlib.rc("font", family="sans-serif")
matplotlib.rc("font", serif="Arial")

df_r32 = pd.read_csv("../csv/r32-pareto.csv", index_col=0)
df_r32_gaff = pd.read_csv("../csv/r32-gaff.csv", index_col=0)
df_r32_raabe = pd.read_csv("../csv/r32-raabe.csv", index_col=0)

df_r125 = pd.read_csv("../csv/r125-pareto.csv", index_col=0)
df_r125_gaff = pd.read_csv("../csv/r125-gaff.csv", index_col=0)


def main():

    # Plot VLE envelopes
    clrs = seaborn.color_palette('bright', n_colors=len(df_r32))
    np.random.seed(11)
    np.random.shuffle(clrs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    temps_r32 = R32.expt_liq_density.keys()
    temps_r125 = R125.expt_liq_density.keys()
    for temp in temps_r32:
        ax1.scatter(
            df_r32.filter(regex=(f"liq_density_{float(temp):.0f}K")),
            np.tile(temp, len(df_r32)),
            c=clrs,
            s=160,
            alpha=0.2,
        )
        ax1.scatter(
            df_r32.filter(regex=(f"vap_density_{float(temp):.0f}K")),
            np.tile(temp, len(df_r32)),
            c=clrs,
            s=160,
            alpha=0.2,
        )
    ax1.scatter(
        df_r32.filter(regex=("sim_rhoc")),
        df_r32.filter(regex=("sim_Tc")),
        c=clrs,
        s=160,
        alpha=0.2,
    )

    tc, rhoc = calc_critical(df_r32_gaff)
    ax1.scatter(
        df_r32_gaff["rholiq_kgm3"],
        df_r32_gaff["T_K"],
        c='gray',
        s=120,
        alpha=0.7,
        label="GAFF",
        marker='s',
    )
    ax1.scatter(
        df_r32_gaff["rhovap_kgm3"],
        df_r32_gaff["T_K"],
        c='gray',
        s=120,
        alpha=0.7,
        marker='s',
    )
    ax1.scatter(
        rhoc,
        tc,
        c='gray',
        s=120,
        alpha=0.7,
        marker='s',
    )

    tc, rhoc = calc_critical(df_r32_raabe)
    ax1.scatter(
        df_r32_raabe["rholiq_kgm3"],
        df_r32_raabe["T_K"],
        c='#0989d9',
        s=140,
        alpha=0.7,
        label="Raabe",
        marker='^',
    )
    ax1.scatter(
        df_r32_raabe["rhovap_kgm3"],
        df_r32_raabe["T_K"],
        c='#0989d9',
        s=140,
        alpha=0.7,
        marker='^',
    )
    ax1.scatter(
        rhoc,
        tc,
        c='#0989d9',
        s=140,
        alpha=0.7,
        marker='^',
    )

    ax1.scatter(
        R32.expt_liq_density.values(),
        R32.expt_liq_density.keys(),
        color="black",
        marker="x",
        linewidths=2,
        s=200,
        label="Experiment  "
    )
    ax1.scatter(
        R32.expt_vap_density.values(),
        R32.expt_vap_density.keys(),
        color="black",
        marker="x",
        linewidths=2,
        s=200,
    )
    ax1.scatter(R32.expt_rhoc, R32.expt_Tc, color="black", marker="x", linewidths=2, s=200)

    ax1.set_xlim(-100, 1550)
    ax1.xaxis.set_major_locator(MultipleLocator(400))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
    
    ax1.set_ylim(220,410)
    ax1.yaxis.set_major_locator(MultipleLocator(40))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
    
    ax1.tick_params("both", direction="in", which="both", length=4, labelsize=26, pad=10)
    ax1.tick_params("both", which="major", length=8)
    ax1.xaxis.set_ticks_position("both")
    ax1.yaxis.set_ticks_position("both")

    ax1.set_ylabel("T (K)", fontsize=32, labelpad=10)
    ax1.set_xlabel(r"$\mathregular{\rho}$ (kg/m$\mathregular{^3}$)", fontsize=32, labelpad=15)
    #for axis in ['top','bottom','left','right']:
    #    ax.spines[axis].set_linewidth(2.0)

    clrs = seaborn.color_palette('bright', n_colors=len(df_r125))
    np.random.seed(11)
    np.random.shuffle(clrs)

    for temp in temps_r125:
        ax2.scatter(
            df_r125.filter(regex=(f"liq_density_{float(temp):.0f}K")),
            np.tile(temp, len(df_r125)),
            c=clrs,
            s=160,
            alpha=0.2,
        )
        ax2.scatter(
            df_r125.filter(regex=(f"vap_density_{float(temp):.0f}K")),
            np.tile(temp, len(df_r125)),
            c=clrs,
            s=160,
            alpha=0.2,
        )
    ax2.scatter(
        df_r125.filter(regex=("sim_rhoc")),
        df_r125.filter(regex=("sim_Tc")),
        c=clrs,
        s=160,
        alpha=0.2,
    )

    tc, rhoc = calc_critical(df_r125_gaff)
    ax2.scatter(
        df_r125_gaff["rholiq_kgm3"],
        df_r125_gaff["T_K"],
        c='gray',
        s=120,
        alpha=0.7,
        marker='s',
    )
    ax2.scatter(
        df_r125_gaff["rhovap_kgm3"],
        df_r125_gaff["T_K"],
        c='gray',
        s=120,
        alpha=0.7,
        marker='s',
    )
    ax2.scatter(
        rhoc,
        tc,
        c='gray',
        s=120,
        alpha=0.7,
        marker='s',
    )

    ax2.scatter(
        R125.expt_liq_density.values(),
        R125.expt_liq_density.keys(),
        color="black",
        marker="x",
        linewidths=2,
        s=200,
    )
    ax2.scatter(
        R125.expt_vap_density.values(),
        R125.expt_vap_density.keys(),
        color="black",
        marker="x",
        linewidths=2,
        s=200,
    )
    ax2.scatter(R125.expt_rhoc, R125.expt_Tc, color="black", marker="x", linewidths=2, s=200)

    ax2.set_xlim(-100, 1550)
    ax2.xaxis.set_major_locator(MultipleLocator(400))
    ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
    
    ax2.set_ylim(220,410)
    ax2.yaxis.set_major_locator(MultipleLocator(40))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
    
    ax2.tick_params("both", direction="in", which="both", length=4, labelsize=26, pad=10)
    ax2.tick_params("both", which="major", length=8)
    ax2.xaxis.set_ticks_position("both")
    ax2.yaxis.set_ticks_position("both")

    ax2.set_ylabel("T (K)", fontsize=32, labelpad=10)
    ax2.set_xlabel(r"$\mathregular{\rho}$ (kg/m$\mathregular{^3}$)", fontsize=32, labelpad=15)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2.0)
        ax2.spines[axis].set_linewidth(2.0)

    ax1.legend(loc="lower left", bbox_to_anchor=(0.28, 1.03), ncol=3, fontsize=22, handletextpad=0.1, markerscale=0.9, edgecolor="dimgrey")
    ax1.text(0.08, 0.82, "a", fontsize=40, transform=ax1.transAxes)
    ax1.text(0.5, 0.82, "HFC-32", fontsize=34, transform=ax1.transAxes)
    ax2.text(0.08, 0.82, "b", fontsize=40, transform=ax2.transAxes)
    ax2.text(0.4,  0.82, "HFC-125", fontsize=36, transform=ax2.transAxes)
    fig.subplots_adjust(bottom=0.2, top=0.75, left=0.15, right=0.95, wspace=0.55)

    fig.savefig("pdfs/fig3_results-vle.pdf")

    # Plot Pvap / Hvap
    fig, axs = plt.subplots(2, 2)

    clrs = seaborn.color_palette('bright', n_colors=len(df_r32))
    np.random.seed(11)
    np.random.shuffle(clrs)

    for temp in temps_r32:
        axs[0,0].scatter(
            np.tile(temp, len(df_r32)),
            df_r32.filter(regex=(f"Pvap_{float(temp):.0f}K")),
            c=clrs,
            s=70,
            alpha=0.2,
        )
    axs[0,0].scatter(
        df_r32_gaff["T_K"],
        df_r32_gaff["pvap_bar"],
        c='gray',
        s=50,
        alpha=0.7,
        label="GAFF",
        marker='s',
    )
    axs[0,0].scatter(
        df_r32_raabe["T_K"],
        df_r32_raabe["pvap_bar"],
        c='#0989d9',
        s=70,
        alpha=0.7,
        label="Raabe",
        marker='^',
    )
    axs[0,0].scatter(
        R32.expt_Pvap.keys(),
        R32.expt_Pvap.values(),
        color="black",
        marker="x",
        label="Experiment  ",
        s=80,
    )

    axs[0,0].set_xlim(220,330)
    axs[0,0].xaxis.set_major_locator(MultipleLocator(40))
    axs[0,0].xaxis.set_minor_locator(AutoMinorLocator(4))

    axs[0,0].set_ylim(0,40)
    axs[0,0].yaxis.set_major_locator(MultipleLocator(10))
    axs[0,0].yaxis.set_minor_locator(AutoMinorLocator(5))

    axs[0,0].tick_params("both", direction="in", which="both", length=2, labelsize=12, pad=5)
    axs[0,0].tick_params("both", which="major", length=4)
    axs[0,0].xaxis.set_ticks_position("both")
    axs[0,0].yaxis.set_ticks_position("both")

    axs[0,0].set_xlabel("T (K)", fontsize=16, labelpad=8)
    axs[0,0].set_ylabel(r"$\mathregular{P_{vap}}$ (bar)", fontsize=16, labelpad=8)
    #for axis in ['top','bottom','left','right']:
    #    axs[0,0].spines[axis].set_linewidth(2.0)
    #    axs[0,1].spines[axis].set_linewidth(2.0)
    #    axs[1,0].spines[axis].set_linewidth(2.0)
    #    axs[1,1].spines[axis].set_linewidth(2.0)

    # Plot Enthalpy of Vaporization
    for temp in temps_r32:
        axs[1,0].scatter(
            np.tile(temp, len(df_r32)),
            df_r32.filter(regex=(f"Hvap_{float(temp):.0f}K")),
            c=clrs,
            s=70,
            alpha=0.2,
        )
    axs[1,0].scatter(
        df_r32_gaff["T_K"],
        df_r32_gaff["hvap_kJmol"] / R32.molecular_weight * 1000.0,
        c='gray',
        s=50,
        alpha=0.7,
        marker='s',
    )
    axs[1,0].scatter(
        df_r32_raabe["T_K"],
        df_r32_raabe["hvap_kJmol"] / R32.molecular_weight * 1000.0,
        c='#0989d9',
        s=70,
        alpha=0.7,
        marker='^',
    )
    axs[1,0].scatter(
        R32.expt_Hvap.keys(),
        R32.expt_Hvap.values(),
        color="black",
        marker="x",
        s=80,
    )

    axs[1,0].set_xlim(220,330)
    axs[1,0].xaxis.set_major_locator(MultipleLocator(40))
    axs[1,0].xaxis.set_minor_locator(AutoMinorLocator(4))

    axs[1,0].set_ylim(-10,410)
    axs[1,0].yaxis.set_major_locator(MultipleLocator(100))
    axs[1,0].yaxis.set_minor_locator(AutoMinorLocator(5))

    axs[1,0].tick_params("both", direction="in", which="both", length=2, labelsize=12, pad=5)
    axs[1,0].tick_params("both", which="major", length=4)
    axs[1,0].xaxis.set_ticks_position("both")
    axs[1,0].yaxis.set_ticks_position("both")

    axs[1,0].set_xlabel("T (K)", fontsize=16, labelpad=8)
    axs[1,0].set_ylabel(r"$\mathregular{\Delta H_{vap}}$ (kJ/kg)", fontsize=16, labelpad=8)

    clrs = seaborn.color_palette('bright', n_colors=len(df_r125))
    np.random.seed(11)
    np.random.shuffle(clrs)

    for temp in temps_r125:
        axs[0,1].scatter(
            np.tile(temp, len(df_r125)),
            df_r125.filter(regex=(f"Pvap_{float(temp):.0f}K")),
            c=clrs,
            s=70,
            alpha=0.2,
        )
    axs[0,1].scatter(
        df_r125_gaff["T_K"],
        df_r125_gaff["pvap_bar"],
        c='gray',
        s=50,
        alpha=0.7,
        marker='s',
    )
    axs[0,1].scatter(
        R125.expt_Pvap.keys(),
        R125.expt_Pvap.values(),
        color="black",
        marker="x",
        s=80,
    )

    axs[0,1].set_xlim(220,330)
    axs[0,1].xaxis.set_major_locator(MultipleLocator(40))
    axs[0,1].xaxis.set_minor_locator(AutoMinorLocator(4))

    axs[0,1].set_ylim(0,40)
    axs[0,1].yaxis.set_major_locator(MultipleLocator(10))
    axs[0,1].yaxis.set_minor_locator(AutoMinorLocator(5))

    axs[0,1].tick_params("both", direction="in", which="both", length=2, labelsize=12, pad=5)
    axs[0,1].tick_params("both", which="major", length=4)
    axs[0,1].xaxis.set_ticks_position("both")
    axs[0,1].yaxis.set_ticks_position("both")

    axs[0,1].set_xlabel("T (K)", fontsize=16, labelpad=8)
    axs[0,1].set_ylabel(r"$\mathregular{P_{vap}}$ (bar)", fontsize=16, labelpad=8)

    # Plot Enthalpy of Vaporization
    for temp in temps_r125:
        axs[1,1].scatter(
            np.tile(temp, len(df_r125)),
            df_r125.filter(regex=(f"Hvap_{float(temp):.0f}K")),
            c=clrs,
            s=70,
            alpha=0.2,
        )
    axs[1,1].scatter(
        df_r125_gaff["T_K"],
        df_r125_gaff["hvap_kJmol"] / R125.molecular_weight * 1000.0,
        c='gray',
        s=50,
        alpha=0.7,
        marker='s',
    )
    axs[1,1].scatter(
        R125.expt_Hvap.keys(),
        R125.expt_Hvap.values(),
        color="black",
        marker="x",
        s=80,
    )

    axs[1,1].set_xlim(220,330)
    axs[1,1].xaxis.set_major_locator(MultipleLocator(40))
    axs[1,1].xaxis.set_minor_locator(AutoMinorLocator(4))

    axs[1,1].set_ylim(-10,410)
    axs[1,1].yaxis.set_major_locator(MultipleLocator(100))
    axs[1,1].yaxis.set_minor_locator(AutoMinorLocator(5))

    axs[1,1].tick_params("both", direction="in", which="both", length=2, labelsize=12, pad=5)
    axs[1,1].tick_params("both", which="major", length=4)
    axs[1,1].xaxis.set_ticks_position("both")
    axs[1,1].yaxis.set_ticks_position("both")

    axs[1,1].set_xlabel("T (K)", fontsize=16, labelpad=8)
    axs[1,1].set_ylabel(r"$\mathregular{\Delta H_{vap}}$ (kJ/kg)", fontsize=16, labelpad=8)


    axs[0,0].text(0.08, 0.8, "a", fontsize=20, transform=axs[0,0].transAxes)
    axs[0,0].text(0.56, 0.08, "HFC-32", fontsize=16, transform=axs[0,0].transAxes)

    axs[0,1].text(0.08, 0.8, "b", fontsize=20, transform=axs[0,1].transAxes)
    axs[0,1].text(0.5, 0.8, "HFC-125", fontsize=16, transform=axs[0,1].transAxes)

    axs[1,0].text(0.08, 0.08, "c", fontsize=20, transform=axs[1,0].transAxes)
    axs[1,0].text(0.56, 0.08, "HFC-32", fontsize=16, transform=axs[1,0].transAxes)

    axs[1,1].text(0.08, 0.8, "d", fontsize=20, transform=axs[1,1].transAxes)
    axs[1,1].text(0.5, 0.8, "HFC-125", fontsize=16, transform=axs[1,1].transAxes)


    axs[0,0].legend(loc="lower left", bbox_to_anchor=(0.25, 1.05), ncol=3, fontsize=12, handletextpad=0.1, markerscale=0.8, edgecolor="dimgrey")

    fig.subplots_adjust(bottom=0.15, top=0.88, left=0.15, right=0.95, wspace=0.55, hspace=0.5)
    fig.savefig("pdfs/fig3_result-si.pdf")


def calc_critical(df):
    """Compute the critical temperature and density

    Accepts a dataframe with "T_K", "rholiq_kgm3" and "rhovap_kgm3"
    Returns the critical temperature (K) and density (kg/m3)

    Computes the critical properties with the law of rectilinear diameters
    """
    temps = df["T_K"].values
    liq_density = df["rholiq_kgm3"].values
    vap_density = df["rhovap_kgm3"].values
    # Critical Point (Law of rectilinear diameters)
    slope1, intercept1, r_value1, p_value1, std_err1 = linregress(
        temps,
        (liq_density + vap_density) / 2.0,
    )

    slope2, intercept2, r_value2, p_value2, std_err2 = linregress(
        temps,
        (liq_density - vap_density) ** (1 / 0.32),
    )

    Tc = np.abs(intercept2 / slope2)
    rhoc = intercept1 + slope1 * Tc

    return Tc, rhoc

if __name__ == "__main__":
    main()
