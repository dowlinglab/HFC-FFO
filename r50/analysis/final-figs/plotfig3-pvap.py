import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn

sys.path.append("../")

from scipy.stats import linregress
from utils.r50 import R50Constants
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

R50 = R50Constants()

matplotlib.rc("font", family="sans-serif")
matplotlib.rc("font", serif="Arial")

df_r50 = pd.read_csv("../csv/r50-pareto-iter3.csv", index_col=0)
df_r50_gaff = pd.read_csv("../../run/gaff/results.csv", index_col=False)
# Add https://doi.org/10.1021/jp9072137
df_r50_lit = pd.read_csv("../csv/r50-lit.csv", index_col=False)
df_r50_trappe = pd.read_csv("../csv/r50-trappe.csv", index_col=False)

def main():

    #fig, ax2 = plt.subplots(1, 1, figsize=(6,6))
    temps_r50 = R50.expt_liq_density.keys()

    '''clrs = seaborn.color_palette('bright', n_colors=len(df_r50))
    np.random.seed(11)
    np.random.shuffle(clrs)

    for temp in temps_r50:
        ax2.scatter(
            df_r50.filter(regex=(f"liq_density_{float(temp):.0f}K")),
            np.tile(temp, len(df_r50)),
            c=clrs,
            s=160,
            alpha=0.2,
        )
        ax2.scatter(
            df_r50.filter(regex=(f"vap_density_{float(temp):.0f}K")),
            np.tile(temp, len(df_r50)),
            c=clrs,
            s=160,
            alpha=0.2,
        )
    ax2.scatter(
        df_r50.filter(regex=("sim_rhoc")),
        df_r50.filter(regex=("sim_Tc")),
        c=clrs,
        s=160,
        alpha=0.2,
    )

    tc, rhoc = calc_critical(df_r50_gaff)
    ax2.scatter(
        df_r50_gaff["liq_density"],
        df_r50_gaff["temperature"],
        c='gray',
        s=120,
        alpha=0.7,
        marker='s',
        label="GAFF",
    )
    ax2.scatter(
        df_r50_gaff["vap_density"],
        df_r50_gaff["temperature"],
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

    tc, rhoc = calc_critical(df_r50_lit)
    ax2.scatter(
        df_r50_lit["liq_density"],
        df_r50_lit["temperature"],
        c='#0989d9',
        s=160,
        alpha=0.7,
        marker='^',
        label="Potoff et al.",
    )
    ax2.scatter(
        df_r50_lit["vap_density"],
        df_r50_lit["temperature"],
        c='#0989d9',
        s=160,
        alpha=0.7,
        marker='^',
    )
    ax2.scatter(
        rhoc,
        tc,
        c='#0989d9',
        s=160,
        alpha=0.7,
        marker='^',
    )

    ax2.scatter(
        R50.expt_liq_density.values(),
        R50.expt_liq_density.keys(),
        color="black",
        marker="x",
        linewidths=2,
        s=200,
        label="Experiment",
    )
    ax2.scatter(
        R50.expt_vap_density.values(),
        R50.expt_vap_density.keys(),
        color="black",
        marker="x",
        linewidths=2,
        s=200,
    )
    ax2.scatter(R50.expt_rhoc, R50.expt_Tc, color="black", marker="x", linewidths=2, s=200)

    ax2.set_xlim(-50, 1850)
    ax2.xaxis.set_major_locator(MultipleLocator(500))
    ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
    
    ax2.set_ylim(125,255)
    ax2.yaxis.set_major_locator(MultipleLocator(20))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
    
    ax2.tick_params("both", direction="in", which="both", length=4, labelsize=26, pad=10)
    ax2.tick_params("both", which="major", length=8)
    ax2.xaxis.set_ticks_position("both")
    ax2.yaxis.set_ticks_position("both")

    ax2.set_ylabel("T (K)", fontsize=32, labelpad=10)
    ax2.set_xlabel(r"$\mathregular{\rho}$ (kg/m$\mathregular{^3}$)", fontsize=32, labelpad=15)
    for axis in ['top','bottom','left','right']:
    #    ax1.spines[axis].set_linewidth(2.0)
        ax2.spines[axis].set_linewidth(2.0)

    ax2.legend(loc="lower left", bbox_to_anchor=(-0.16, 1.03), ncol=2, fontsize=22, handletextpad=0.1, markerscale=0.9, edgecolor="dimgrey")
    ax2.text(0.7,  0.82, "R-50", fontsize=30, transform=ax2.transAxes)
    fig.subplots_adjust(bottom=0.2, top=0.75, left=0.15, right=0.95, wspace=0.55)

    fig.savefig("pdfs/fig3_r50-results-vle.png",dpi=300)'''


    # Plot Pvap / Hvap
    fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(12,6))
    #fig, ax1 = plt.subplots(1, 1, figsize=(6,6))
    clrs = seaborn.color_palette('bright', n_colors=len(df_r50))
    np.random.seed(11)
    np.random.shuffle(clrs)

    for temp in temps_r50:
        axs[0].scatter(
            np.tile(temp, len(df_r50)),
            df_r50.filter(regex=(f"Pvap_{float(temp):.0f}K")),
            c=clrs,
            s=70,
            alpha=0.2,
        )
    axs[0].scatter(
        df_r50_gaff["temperature"],
        df_r50_gaff["Pvap"],
        c='gray',
        s=70,
        alpha=0.7,
        label="GAFF",
        marker='s',
    )
    axs[0].scatter(
        df_r50_lit["temperature"],
        df_r50_lit["Pvap"],
        c='#0989d9',
        s=70,
        alpha=0.7,
        label="Potoff et al.",
        marker='^',
    )
    axs[0].scatter(
        df_r50_trappe["temperature"],
        df_r50_trappe["Pvap"],
        c='red',
        s=70,
        alpha=0.7,
        label="TraPPE",
        marker='*',
    )

    axs[0].scatter(
        R50.expt_Pvap.keys(),
        R50.expt_Pvap.values(),
        color="black",
        marker="x",
        label="Experiment",
        s=80,
    )

    axs[0].set_xlim(110,180)
    axs[0].xaxis.set_major_locator(MultipleLocator(40))
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(4))

    axs[0].set_ylim(0,50)
    axs[0].yaxis.set_major_locator(MultipleLocator(10))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(5))

    axs[0].tick_params("both", direction="in", which="both", length=2, labelsize=16, pad=10)
    axs[0].tick_params("both", which="major", length=4)
    axs[0].xaxis.set_ticks_position("both")
    axs[0].yaxis.set_ticks_position("both")

    axs[0].set_xlabel("T (K)", fontsize=16, labelpad=8)
    axs[0].set_ylabel(r"$\mathregular{P_{vap}}$ (bar)", fontsize=16, labelpad=8)
    #for axis in ['top','bottom','left','right']:
    #    axs[0,0].spines[axis].set_linewidth(2.0)
    #    axs[0,1].spines[axis].set_linewidth(2.0)
    #    axs[1,0].spines[axis].set_linewidth(2.0)
    #    axs[1,1].spines[axis].set_linewidth(2.0)

    # Plot Enthalpy of Vaporization
    for temp in temps_r50:
        axs[1].scatter(
            np.tile(temp, len(df_r50)),
            df_r50.filter(regex=(f"Hvap_{float(temp):.0f}K")),
            c=clrs,
            s=70,
            alpha=0.2,
        )
    axs[1].scatter(
        df_r50_gaff["temperature"],
        df_r50_gaff["Hvap"] / R50.molecular_weight * 1000.0,
        c='gray',
        s=70,
        alpha=0.7,
        marker='s',
    )
    axs[1].scatter(
        df_r50_lit["temperature"],
        df_r50_lit["Hvap"] ,#kj/kg
        c='#0989d9',
        s=70,
        alpha=0.7,
        marker='^',
    )
    axs[1].scatter(
        df_r50_trappe["temperature"],
        df_r50_trappe["Hvap"] ,#kj/kg
        c='red',
        s=70,
        alpha=0.7,
        marker='*',
    )
    axs[1].scatter(
        R50.expt_Hvap.keys(),
        R50.expt_Hvap.values(),
        color="black",
        marker="x",
        s=80,
    )

    axs[1].set_xlim(110,180)
    axs[1].xaxis.set_major_locator(MultipleLocator(40))
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(4))

    axs[1].set_ylim(200,550)
    axs[1].yaxis.set_major_locator(MultipleLocator(100))
    axs[1].yaxis.set_minor_locator(AutoMinorLocator(5))

    axs[1].tick_params("both", direction="in", which="both", length=2, labelsize=16, pad=10)
    axs[1].tick_params("both", which="major", length=4)
    axs[1].xaxis.set_ticks_position("both")
    axs[1].yaxis.set_ticks_position("both")

    axs[1].set_xlabel("T (K)", fontsize=16, labelpad=8)
    axs[1].set_ylabel(r"$\mathregular{\Delta H_{vap}}$ (kJ/kg)", fontsize=16, labelpad=8)

    '''clrs = seaborn.color_palette('bright', n_colors=len(df_r50))
    np.random.seed(11)
    np.random.shuffle(clrs)

    for temp in temps_r50:
        axs[0,1].scatter(
            np.tile(temp, len(df_r50)),
            df_r50.filter(regex=(f"Pvap_{float(temp):.0f}K")),
            c=clrs,
            s=70,
            alpha=0.2,
        )
    axs[0,1].scatter(
        df_r50_gaff["T_K"],
        df_r50_gaff["pvap_bar"],
        c='gray',
        s=50,
        alpha=0.7,
        marker='s',
    )
    axs[0,1].scatter(
        R50.expt_Pvap.keys(),
        R50.expt_Pvap.values(),
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
    for temp in temps_r50:
        axs[1,1].scatter(
            np.tile(temp, len(df_r50)),
            df_r50.filter(regex=(f"Hvap_{float(temp):.0f}K")),
            c=clrs,
            s=70,
            alpha=0.2,
        )
    axs[1,1].scatter(
        df_r50_gaff["T_K"],
        df_r50_gaff["hvap_kJmol"] / R50.molecular_weight * 1000.0,
        c='gray',
        s=50,
        alpha=0.7,
        marker='s',
    )
    axs[1,1].scatter(
        R50.expt_Hvap.keys(),
        R50.expt_Hvap.values(),
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
    axs[1,1].set_ylabel(r"$\mathregular{\Delta H_{vap}}$ (kJ/kg)", fontsize=16, labelpad=8)'''


    axs[0].text(0.08, 0.8, "R-50", fontsize=20, transform=axs[0].transAxes)
    #axs[0].text(0.56, 0.08, r"$\mathregular{P_{vap}}$ (bar)", fontsize=16, transform=axs[0].transAxes)

    #axs[0,1].text(0.08, 0.8, "b", fontsize=20, transform=axs[0,1].transAxes)
    #axs[1].text(0.5, 0.8, r"$\mathregular{\Delta H_{vap}}$ (kJ/mol)", fontsize=16, transform=axs[1].transAxes)

    '''axs[1,0].text(0.08, 0.08, "c", fontsize=20, transform=axs[1,0].transAxes)
    axs[1,0].text(0.56, 0.08, "R-50", fontsize=16, transform=axs[1,0].transAxes)

    axs[1,1].text(0.08, 0.8, "d", fontsize=20, transform=axs[1,1].transAxes)
    axs[1,1].text(0.5, 0.8, "HFC-50", fontsize=16, transform=axs[1,1].transAxes)'''


    axs[0].legend(loc="lower left", bbox_to_anchor=(0.2, 1.05), ncol=4, fontsize=16, handletextpad=0.1, markerscale=0.8, edgecolor="dimgrey")

    fig.subplots_adjust(bottom=0.15, top=0.85, left=0.15, right=0.85, wspace=0.55, hspace=0.5)
    fig.savefig("pdfs/fig3-p-h-png",dpi=300)


def calc_critical(df):
    """Compute the critical temperature and density

    Accepts a dataframe with "T_K", "rholiq_kgm3" and "rhovap_kgm3"
    Returns the critical temperature (K) and density (kg/m3)

    Computes the critical properties with the law of rectilinear diameters
    """
    temps = df["temperature"].values
    liq_density = df["liq_density"].values
    vap_density = df["vap_density"].values
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
