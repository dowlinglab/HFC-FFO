import os
import six
import numpy as np
import matplotlib.pyplot as plt

from fffit.utils import values_scaled_to_real

def plot_property(df, prp_name, bounds, axis_name=None):
    """Plot a comparison of the simulated and experimental property

    Parameters
    ----------
    df : pandas.Dataframe
        dataframe with information
    prp_name : string
        property name to plot from df
    bounds : np.ndarray
        bounds of the property
    axis_name : string
        name to use on the plot axis, with units

    Returns
    -------
    None

    Notes
    -----
    Saves a plt with name 'figs/expt_v_sim_{prp_name}.png'
    """

    if axis_name is None:
        axis_name = prp_name

    # Basic plots to view output
    yeqx = np.arange(bounds[0] - 10, bounds[1] + 20, 10)

    fig, ax = plt.subplots()

    ax.plot(yeqx, yeqx, color="black")
    ax.scatter(
        values_scaled_to_real(df["expt_" + prp_name], bounds),
        values_scaled_to_real(df["sim_" + prp_name], bounds),
        alpha=0.2,
        color="black",
    )
    ax.set_xlabel("Expt. " + axis_name, fontsize=16, labelpad=15)
    ax.set_ylabel("Sim. " + axis_name, fontsize=16, labelpad=15)
    ax.tick_params(axis="both", labelsize=12)

    ax.set_xlim(yeqx[0], yeqx[-1])
    ax.set_ylim(yeqx[0], yeqx[-1])
    ax.set_aspect("equal", "box")

    fig.tight_layout()
    try:
        fig.savefig("figs/expt_v_sim_" + prp_name + ".png", dpi=300)
    except FileNotFoundError:
        os.mkdir("figs")
        fig.savefig("figs/expt_v_sim_" + prp_name + ".png", dpi=300)

def render_mpl_table(
    data,
    out_name="table",
    col_width=3.0,
    row_height=0.625,
    font_size=14,
    header_color="#40466e",
    row_colors=["#f1f1f2", "w"],
    edge_color="w",
    bbox=[0, 0, 1, 1],
    header_columns=0,
    ax=None,
    **kwargs,
):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array(
            [col_width, row_height]
        )
        fig, ax = plt.subplots(figsize=size)
        ax.axis("off")

    mpl_table = ax.table(
        cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs
    )

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight="bold", color="w")
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    try:
        fig.savefig("figs/" + out_name + ".png")
    except FileNotFoundError:
        os.mkdir('figs')
        fig.savefig("figs/" + out_name + ".png")
