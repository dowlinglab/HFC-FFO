import gpflow
import numpy as np
import tensorflow as tf
from gpflow.kernels import RBF
from gpflow.mean_functions import MeanFunction
from gpflow import Parameter
from gpflow.base import TensorType
from gpflow.utilities import print_summary
from gpflow.config import default_float 

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
            mean_function = gpflow.mean_functions.Linear(
                A=np.zeros(x_train.shape[1]).reshape(-1, 1)
            )
        elif mean_function.lower() == "none":
            mean_function = None
        elif mean_function.lower() == "sigma_epsilon_nonlinear":
                mean_function = sigma_epsilon_nonlinear(
                A=np.zeros(3).reshape(-1, 1)
            )
        elif mean_function.lower() == "vap_nonlinear":
                mean_function = vap_nonlinear(
                A=np.zeros(5).reshape(-1,1)
            )
        elif mean_function.lower() == "enthalpy_nonlinear":
                mean_function = enthalpy_nonlinear(
                A=np.zeros(6).reshape(-1,1)
            )
        else:
            raise ValueError(
                "Only supported mean functions are 'linear' and 'none'"
            )

    # Create the model
    model = gpflow.models.GPR(
        data=(x_train, y_train.reshape(-1, 1)),
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


class sigma_epsilon_nonlinear(MeanFunction):
    """
    y_i = A x_i + b
    x_1 = x1*x3
    x_2 = x2*x4
    x_3 = x5
    only allowed 5 columns input for x
    A requests 3 coefficients
    """
    
    def __init__(self, A: TensorType = None, b: TensorType = None) -> None:
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.
        """
        MeanFunction.__init__(self)
        A = np.ones((1, 1), dtype=default_float()) if A is None else A
        b = np.zeros(1, dtype=default_float()) if b is None else b
        self.A = Parameter(np.atleast_2d(A))
        self.b = Parameter(b)

    def __call__(self, X) -> tf.Tensor:
        x_train_reshape = tf.concat([X[:,0:1]*X[:,2:3],X[:,1:2]*X[:,3:4]],axis=1)
        x_train_reshape = tf.concat([x_train_reshape,X[:,4:5]],axis=1)
        return tf.tensordot(x_train_reshape, self.A, [[-1], [0]]) + self.b
    

class vap_nonlinear(MeanFunction):
    """
    y_i = A x_i + b
    x_5 = exp(1/x5)
    only allowed 5 columns input for x
    A reuqests 5 coefficients
    """

    def __init__(self, A: TensorType = None, b: TensorType = None) -> None:
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.
        """
        MeanFunction.__init__(self)
        A = np.ones((1, 1), dtype=default_float()) if A is None else A
        b = np.zeros(1, dtype=default_float()) if b is None else b
        self.A = Parameter(np.atleast_2d(A))
        self.b = Parameter(b)

    def __call__(self, X) -> tf.Tensor:
        x_train_reshape = tf.concat([X[:,0:4],tf.math.exp(1/X[:,4:5])],axis=1)
        return tf.tensordot(x_train_reshape, self.A, [[-1], [0]]) + self.b
    
    
class enthalpy_nonlinear(MeanFunction):
    """
    y_i = A x_i + b
    x_6 = x5*ln(x5)
    only allowed 5 columns input for x
    notice that A here request 6 coefficients
    """
    
    def __init__(self, A: TensorType = None, b: TensorType = None) -> None:
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.
        """
        MeanFunction.__init__(self)
        A = np.ones((1, 1), dtype=default_float()) if A is None else A
        b = np.zeros(1, dtype=default_float()) if b is None else b
        self.A = Parameter(np.atleast_2d(A))
        self.b = Parameter(b)

    def __call__(self, X) -> tf.Tensor:
        x_train_reshape = tf.concat([X[:,0:5],tf.math.log(X[:,4:5])*X[:,4:5]],axis=1)
        return tf.tensordot(x_train_reshape, self.A, [[-1], [0]]) + self.b
