import sys
import gpflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import scipy.optimize as optimize

from fffit.utils import (
    shuffle_and_split,
    values_scaled_to_real,
)


from fffit.models import run_gpflow_scipy
from fffit.pareto import find_pareto_set, is_pareto_efficient

sys.path.append("../")

from utils.r143a import R143aConstants
from utils.id_new_samples import (
    prepare_df_density,
    prepare_df_vle,
    rank_samples,
)

R143a = R143aConstants()

############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum = 1
##############################################################################
##############################################################################

distance_seed = 10
liquid_density_threshold = 500  # kg/m^3


# Read VLE files
csv_path = "/scratch365/nwang2/ff_development/HFC_143a_FFO_FF/r143a/analysis/csv/"
out_csv_name = "r143a-vle-iter" + str(iternum + 1) + "-params.csv"

vle_mses = pd.read_csv(csv_path + "vle_mses.csv",index_col=0)

# Find new points for next iteration
pareto_points = vle_mses[vle_mses["is_pareto"] == True]
print(f"A total of {len(pareto_points)} pareto efficient points were found.")
new_points = pareto_points.sort_values("mse_liq_density").iloc[[0]]
new_points = new_points.append(
    pareto_points.sort_values("mse_vap_density").iloc[[0]]
)
new_points = new_points.append(pareto_points.sort_values("mse_Hvap").iloc[[0]])
new_points = new_points.append(pareto_points.sort_values("mse_Pvap").iloc[[0]])
pareto_points.drop(index=new_points.index, inplace=True)

distance = 1.5
discarded_points = pd.DataFrame(columns=pareto_points.columns)

np.random.seed(distance_seed)

while len(pareto_points > 0):
    # Shuffle the pareto points
    pareto_points = pareto_points.sample(frac=1)
    new_points = new_points.append(pareto_points.iloc[[0]])
    # Remove anything within distance
    l1_norm = np.sum(
        np.abs(
            pareto_points[list(R143a.param_names)].values
            - new_points[list(R143a.param_names)].iloc[[-1]].values
        ),
        axis=1,
    )
    points_to_remove = np.where(l1_norm < distance)[0]
    discarded_points = discarded_points.append(
        pareto_points.iloc[points_to_remove]
    )
    pareto_points.drop(
        index=pareto_points.index[points_to_remove], inplace=True
    )
print(
    f"After removing similar points, we are left with {len(new_points)} pareto efficient points."
)


'''# Plot new points
g = seaborn.pairplot(next_iter_points, vars=list(R143a.param_names))
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("figs/R143a-new-points.pdf")'''

# Save to CSV
new_points.drop(
    columns=[
        "mse_liq_density",
        "mse_vap_density",
        "mse_Pvap",
        "mse_Hvap",
        "is_pareto",
    ],
    inplace=True,
)
new_points.to_csv(csv_path + out_csv_name)
