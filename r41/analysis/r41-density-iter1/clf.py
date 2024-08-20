import sys
import gpflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from sklearn import svm
import scipy.optimize as optimize

from fffit.utils import (
    shuffle_and_split,
    values_real_to_scaled,
    values_scaled_to_real,
    variances_scaled_to_real,
)

from fffit.plot import (
    plot_model_performance,
    plot_slices_temperature,
    plot_slices_params,
    plot_model_vs_test,
)

from fffit.models import run_gpflow_scipy
from skmultilearn.model_selection import iterative_train_test_split

sys.path.append("../")

from utils.r41 import R41Constants
from utils.id_new_samples import (
    prepare_df_density,
    classify_samples,
    rank_samples,
)

R41 = R41Constants()

def stratifyvector(Y):
    """
    Creates a stratified vector based on the label data Y

    Parameters:
    Y : numpy array
        label data
    Returns:
    stratifyVector : numpy array
        Stratified vector
    """
    # Iterate over number of bins, trying to find the larger number of bins that
    # guarantees at least 5 values per bin
    for n in range(1,100):
        # Bin Y using n bins
        stratifyVector=pd.cut(Y,n,labels=False)
        # Define isValid (all bins have at least 5 values)
        isValid=True
        # Check that all bins have at least 5 values
        for k in range(n):
            if np.count_nonzero(stratifyVector==k)<5:
                isValid=False
        #If isValid is false, n is too large; nBins must be the previous iteration
        if not isValid:
            nBins=n-1
            break
    # Generate vector for stratified splitting based on labels
    stratifyVector=pd.cut(Y,nBins,labels=False)
    return stratifyVector

def shuffle_split_strat(df, param_names, property_name, fraction_train=0.8, shuffle_seed=None):
    """Randomly shuffle the DataFrame and extracts the train and test sets

    Parameters
    ----------
    df : pandas.DataFrame
        The pandas dataframe with the samples
    param_names : list-like
        names of the parameters to extract from the dataframe (x_data)
    property_name : string
        Name of the property to extract from the dataframe (y_data)
    fraction_train : float, optional, default = 0.8
        Fraction of sample to use as training data. Remainder is test data.
    shuffle_seed : int, optional, default = None
        seed for random number generator for shuffle

    Returns
    -------
    x_train : np.ndarray
        Training inputs
    y_train : np.ndarray
        Training results
    x_test : np.ndarray
        Testing inputs
    y_test : np.ndarray
        Testing results
    """
    if fraction_train < 0.0 or fraction_train > 1.0:
        raise ValueError("`fraction_train` must be between 0 and 1.")
    else:
        fraction_test = 1.0 - fraction_train

    try:
        prp_idx = df.columns.get_loc(property_name)
    except KeyError:
        raise ValueError(
            "`property_name` does not match any headers of `df`"
        )
    if type(param_names) not in (list, tuple):
        raise TypeError("`param_names` must be a list or tuple")
    else:
        param_names = list(param_names)

    data = df[param_names + [property_name]].values
    # total_entries = data.shape[0]
    # train_entries = int(total_entries * fraction_train)
    #Shuffle the data before splitting train/test sets
    # if shuffle_seed is not None:
    #     np.random.seed(shuffle_seed)
    # np.random.shuffle(data)

    # x_train = data[:train_entries, :-1].astype(np.float64)
    # y_train = data[:train_entries, -1].astype(np.float64)
    # x_test = data[train_entries:, :-1].astype(np.float64)
    # y_test = data[train_entries:, -1].astype(np.float64)
    strat_vec = stratifyvector(data[:,-1])
    x_train, x_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1], train_size=fraction_train, random_state=shuffle_seed, shuffle=True, stratify=strat_vec)
    x_train = x_train.astype(np.float64)
    x_test = x_test.astype(np.float64)
    y_train = y_train.astype(np.float64)
    y_test = y_test.astype(np.float64)
    # x_train, x_test, y_train, y_test = iterative_train_test_split(data[:,:-1], data[:,-1], test_size = 1-fraction_train)

    return x_train, y_train, x_test, y_test

############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum =1
cl_shuffle_seed1 = 2  #classifier #Use seed 97 for class2 and seed 2 for class1
cl_shuffle_seed2 = 97 
cl_shuffle_seed = 1

##############################################################################
##############################################################################

liquid_density_threshold = 400 # kg/m^3  ##>500 is liquid; <500 is gas. used for classifier

csv_path = "../csv/"
in_csv_name = "r41-density-iter" + str(iternum) + "-results.csv" 

# Read file
df_csv = pd.read_csv(csv_path + in_csv_name, index_col=0)

df_all, df_liquid, df_vapor = prepare_df_density(
    df_csv, R41, liquid_density_threshold
)
print("There are ",df_liquid.shape[0], " liquid simulations.")
print("There are ",df_vapor.shape[0]," vapor simulations.")
print("Total number of simulations: ",df_all.shape[0])

### Step 2: Fit classifier and GP models

# Create training/test set
param_names = list(R41.param_names) + ["temperature"]
property_name = "is_liquid"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all, param_names, property_name, shuffle_seed=cl_shuffle_seed
)

# Create and fit classifier
classifier = svm.SVC(kernel="rbf", class_weight="balanced")
classifier.fit(x_train, y_train)
test_score = classifier.score(x_test, y_test)
print(f"Classifer is {test_score*100.0}% accurate on the test set.")
ConfusionMatrixDisplay.from_estimator(classifier, x_test, y_test)  
plt.savefig("classifier.pdf")
latin_hypercube = np.genfromtxt("../../LHS_500000_x_6.csv",delimiter=",",skip_header=1,)[:, 1:]
liquid_samples, vapor_samples = classify_samples(latin_hypercube, classifier)




#df_csv = pd.concat(df_csvs)
# df_all, df_liquid, df_vapor = prepare_df_density(
#     df_csv, R41, liquid_density_threshold
# )
# print("There are ",df_liquid.shape[0], " liquid simulations.")
# print("There are ",df_vapor.shape[0]," vapor simulations.")
# print("Total number of simulations: ",df_all.shape[0])

# ### Step 2: Fit classifier and GP models

# # Create training/test set
# param_names = list(R41.param_names) + ["temperature"]
# property_name = "is_liquid"

# #Stratified sampling
# x_train, y_train, x_test, y_test = shuffle_split_strat(
#     df_all, param_names, property_name, shuffle_seed=cl_shuffle_seed1
# )
# pd.DataFrame(y_train).to_csv("y_train_strat.csv")
# classifier = svm.SVC(kernel="rbf")
# classifier.fit(x_train, y_train)
# test_score = classifier.score(x_test, y_test)
# print(f"Classifer is {test_score*100.0}% accurate on the test set.")
# ConfusionMatrixDisplay.from_estimator(classifier, x_test, y_test)  
# plt.savefig("classifier_strat.pdf")
# liquid_samples, vapor_samples = classify_samples(latin_hypercube, classifier)

# #Shuffle and split
# x_train2, y_train2, x_test2, y_test2 = shuffle_and_split(
#     df_all, param_names, property_name, shuffle_seed=cl_shuffle_seed2
# )
# pd.DataFrame(y_train2).to_csv("y_train_reg.csv")
# classifier2 = svm.SVC(kernel="rbf")
# classifier2.fit(x_train2, y_train2)
# test_score2 = classifier2.score(x_test2, y_test2)
# print(f"Classifer is {test_score2*100.0}% accurate on the test set.")
# ConfusionMatrixDisplay.from_estimator(classifier2, x_test2, y_test2)  
# plt.savefig("classifier_shuff.pdf")
# liquid_samples, vapor_samples = classify_samples(latin_hypercube, classifier2)