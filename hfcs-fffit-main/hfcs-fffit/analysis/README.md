## hfcs-fffit/analysis

The directory naming scheme is ``[molname]-[itertype]-iter[iternum]`` where ``molname`` is ``r32`` or ``r125`` and ``itertype`` is either ``density`` or ``vle``. 

Each directory contains the files necessary to fit the support vector machine (SVM) and Gaussian processs (GP) surrogate models, and then use the surrogate models to identify the parameter sets to use for the next iteration of molecular simulations.

The ``utils`` directory contains contains the following:

* ``r32.py`` and ``r125.py``: parameter bounds and experimental reference data for each molecule
*  ``analyze_samples.py`` and ``id_new_samples.py``: helper functions for preparing/editing ``pandas`` Dataframes with simulation results
* ``plot.py``: helper functions for creating plots

The ``final-analysis`` directory contains a script that extracts the top-performing parameter sets, the ``csv`` directory contains CSV files storing parameter sets and simulation results for each iteration, and the ``final-figs`` directory contains scripts and PDFs for the HFC-related figures found in the manuscript.

