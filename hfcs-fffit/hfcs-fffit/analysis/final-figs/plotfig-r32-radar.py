import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import seaborn

sys.path.append("../")

from utils.r32 import R32Constants
from radar_chart import radar_factory

R32 = R32Constants()

matplotlib.rc("font", family="sans-serif")
matplotlib.rc("font", serif="Arial")

csv_path = "../csv/r32-pareto.csv"
df = pd.read_csv(csv_path, index_col=0)

def main():

    # ID the top ten by lowest average MAPE
    top10 = df.loc[df.filter(regex="mape*").mean(axis=1).sort_values()[:10].index]
    data = top10[list(R32.param_names)].values

    N = 6
    #theta = radar_factory(N, frame='polygon')
    theta = radar_factory(N, frame='circle')

    spoke_labels = [
        "$\mathregular{\sigma_C}$",
        "$\mathregular{\sigma_F}$",
        "$\mathregular{\sigma_H}$",
        "$\mathregular{\epsilon_C}$",
        "$\mathregular{\epsilon_F}$",
        "$\mathregular{\epsilon_H}$",
    ]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))

    colors = seaborn.color_palette('bright', n_colors=len(data))
    # Plot the four cases from the example data on separate axes
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
    for d, color in zip(data, colors):
        ax.plot(theta, d, '-o', markersize=12, color=color, linewidth=3, zorder=-5, alpha=0.65, markeredgewidth=4)
        ax.fill(theta, d, facecolor=color, alpha=0.11, zorder=-5)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_ylim(0,1.0)
    ax.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0],fontsize=22)
    for label in ax.get_yticklabels():
        label.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.25, boxstyle=mpatch.BoxStyle("round4")))


    ax.set_varlabels(spoke_labels, fontsize=50)
    ax.set_rlabel_position(180/N)

    for label in ax.get_xticklabels():
        x,y = label.get_position()
        label.set_position((x,y-0.1))
    ax.yaxis.grid(linewidth=3, alpha=0.75)
    ax.spines['polar'].set_linewidth(4)

    fig.savefig("pdfs/fig_r32-top10-radar.pdf")

    # Plot all
    data = df[list(R32.param_names)].values

    N = 6
    #theta = radar_factory(N, frame='polygon')
    theta = radar_factory(N, frame='circle')

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))

    colors = seaborn.color_palette('bright', n_colors=len(data))
    # Plot the four cases from the example data on separate axes
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
    for d, color in zip(data, colors):
        ax.plot(theta, d, '-o', markersize=10, color=color, linewidth=3, zorder=-5, alpha=0.65, markeredgewidth=2)
        ax.fill(theta, d, facecolor=color, alpha=0.06)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_ylim(0,1.0)
    ax.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0],fontsize=26)
    for label in ax.get_yticklabels():
        label.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.25, boxstyle=mpatch.BoxStyle("round4")))

    ax.set_varlabels(spoke_labels, fontsize=50)
    ax.set_rlabel_position(180/N)

    for label in ax.get_xticklabels():
        x,y = label.get_position()
        label.set_position((x,y-0.1))

    ax.yaxis.grid(linewidth=3, alpha=0.65)
    ax.spines['polar'].set_linewidth(4)

    fig.savefig("pdfs/fig_r32-all-radar.pdf")



if __name__ == "__main__":
    main()
