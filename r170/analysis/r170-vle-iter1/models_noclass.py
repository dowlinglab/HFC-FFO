import sys
import gpflow
import numpy as np
import tensorflow as tf
from gpflow.kernels import RBF
from gpflow.mean_functions import MeanFunction
from gpflow import Parameter
from gpflow.base import TensorType
from gpflow.utilities import print_summary
from gpflow.config import default_float 

sys.path.append("../")

from fffit.utils import values_scaled_to_real
from fffit.utils import values_real_to_scaled
from utils.r170 import R170Constants

R170 = R170Constants()

def run_gpflow_scipy(x_train, y_train, kernel, mean_function="linear", fmt="notebook"):
    """Create and train a GPFlow model

    Parameters
    ----------
    x_train : np.ndarray, shape=(n_samples, n_parameters)
        The x training data
    y_train : np.ndarray, shape=(n_samples, 1)
        The y training data
    kernel : string
        Kernel to use for the GP model
    mean_function: string or None, default = "linear"
        Type of mean function for the GP model
        Options are "linear", or None
    fmt : string, optional, default="notebook"
        The formatting type for the GPFlow print_summary
    """

    if mean_function is not None:
        if mean_function == "linear":
            x_train_modify = x_train
            mean_function = gpflow.mean_functions.Linear(
                A=np.zeros(x_train_modify.shape[1]).reshape(-1, 1)
            )
        elif mean_function.lower() == "none":
            x_train_modify = x_train
            mean_function = None
        elif mean_function.lower() == "sigma_epsilon_nonlinear":
            x_train_modify = np.concatenate((x_train[:,0:1]*x_train[:,2:3],x_train[:,1:2]*x_train[:,3:4],x_train[:,4:5]),axis=1)
            mean_function = gpflow.mean_functions.Linear(
                A=np.zeros(x_train_modify.shape[1]).reshape(-1, 1)
            )
        elif mean_function.lower() == "vap_nonlinear":
            T_train_modify = values_scaled_to_real(x_train[:,4], R170.temperature_bounds)
            x_train_modify = np.concatenate((x_train[:,0:4],np.exp(1/T_train_modify)),axis=1)
            mean_function = gpflow.mean_functions.Linear(
                A=np.zeros(x_train_modify.shape[1]).reshape(-1,1)
            )
        elif mean_function.lower() == "enthalpy_nonlinear":
            x_train_modify = np.concatenate((x_train[:,0:5],np.log(x_train[:,4:5])*x_train[:,4:5]),axis=1)
            mean_function = gpflow.mean_functions.Linear(
                A=np.zeros(x_train_modify.shape[1]).reshape(-1,1)
            )
        else:
            raise ValueError(
                "Only supported mean functions are 'linear' and 'none'"
            )

    # Create the model
    model = gpflow.models.GPR(
        data=(x_train_modify, y_train.reshape(-1, 1)),
        kernel=kernel,
        mean_function=mean_function
    )

    # Print initial values
    print_summary(model, fmt=fmt)

    # Optimize model with scipy
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(model.training_loss, model.trainable_variables)

    # Print the optimized values
    print_summary(model, fmt="notebook")

    # Return the model
    return model
