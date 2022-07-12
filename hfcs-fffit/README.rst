hfcs-fffit
==========

**hfcs-fffit** is a repository used for a case study of our
machine learning directed force field optimization workflow
outlined in `this preprint <https://arxiv.org/abs/2103.03208>`_.
Here we use the procedure to generate force fields for two
hydrofluorocarbons (HFCs), HFC-32 and HFC-125.

Citation
~~~~~~~~
This work has been submitted for review. In the meantime, you
may cite the `preprint <https://arxiv.org/abs/2103.03208>`_ as:

.. code-block:: bash

    BJ Befort, RS DeFever, GM Tow, AW Dowling, and EJ Maginn. Machine learning
    directed optimization of classical molecular modeling force fields. arXiv
    (2021), https://arxiv.org/abs/2103.03208


Available Data
~~~~~~~~~~~~~~

HFC Parameter Sets
##################
The non-dominated and "top four" parameter sets for each HFC are
provided under ``hfcs-fffit/analysis/csv/``. The non-dominated
sets are found in ``r32-pareto.csv`` and ``r125-pareto.csv``, and
the "top four" sets are found in ``r32-final-4.csv`` and
``r125-final-4.csv``. The parameter values in the csv files are
normalized between 0 and 1 based upon the parameter bounds for each
atom type (see manuscript, or ``hfcs-fffit/analysis/utils/r32.py``
and ``hfcs-fffit/analysis/utils/r125.py`` for definitions of
the upper and lower parameter bounds).

Molecular Simulations Inputs
############################
All molecular simulations were performed under ``hfcs-fffit/runs``.
Each iteration was managed with ``signac-flow``. Inside of each
directory in ``runs``, you will find all the necessary files to
run the simulations. Note that you may not get the exact same simulation
results due to differences in software versions, random seeds, etc.
Nonetheless, all of the results from our molecular simulations are saved
under ``hfcs-fffit/analysis/csv/rXX-YY-iterZZ-results.csv``, where ``XX``
is the molecule, ``YY`` is the stage (liquid density or VLE), and
``ZZ`` is the iteration number.

Surrogate Modeling Analysis
############################
All of the scripts for the surrogate modeling are provided in
``hfcs-fffit/analysis``, following the same naming structure as
the csv files.

Figures
#######
All scripts required to generate the primary figures in the
manuscript are reported under ``hfcs-fffit/analysis/final-figs`` and the
associated PDF files are located under
``hfcs-fffit/analysis/final-figs/pdfs``.

Using This Package
~~~~~~~~~~~~~~~~~~

Installation
############

This package has a number of requirements that can be installed in
different ways. We recommend using a conda environment to manage
most of the installation and dependencies. However, some items will
need to be installed from source or pip.

We recommend starting with a fresh conda environment, then installing
the packages listed under ``requirements-pip.txt`` with pip, then
installing the packages listed under ``requirements-conda.txt`` with
conda, and finally installing a few items from source
``requirements-other.txt``. We recommend ``python3.7`` and
taking packages from ``conda-forge``.

Running the simulations will also require an installation of GROMACS.
This can be installed separately (see installation instructions
`here <https://manual.gromacs.org/documentation/2021.2/install-guide/index.html>`_).

**WARNING**: Cloning the ``hfcs-fffit`` repository will take some time
and ~1.5 GB of disk space since it contains the Latin hypercube
that have ~1e6 parameter sets each.

An example of the procedure is provided below:

.. code-block:: bash

    # First clone hfcs-fffit and install pip/conda available dependencies
    # with a new conda environment named hfcs-fffit
    git clone git@github.com:dowlinglab/hfcs-fffit.git
    cd hfcs-fffit/
    conda create --name hfcs-fffit python=3.7 -c conda-forge
    conda activate hfcs-fffit
    python3 -m pip install -r requirements-pip.txt
    conda install --file requirements-conda.txt -c conda-forge
    cd ../

    # Now clone and install  other dependencies
    git clone git@github.com:dowlinglab/fffit.git
    # Checkout the v0.1 release of fffit and install
    cd fffit/
    git checkout tags/v0.1
    pip install .
    cd ../
    # Checkout the v0.1 release of block average and install
    git clone git@github.com:rsdefever/block_average.git
    cd block_average/
    git checkout tags/v0.1
    pip install .
    cd ../

HFC-32 Liquid Density Optimization
##################################

**NOTE**: We use signac and signac flow (`<https://signac.io/>`_)
to manage the setup and execution of the molecular simulations. These
instructions assume a working knowledge of that software.

The first iteration of the liquid density simulations were
performed under the ``hfcs-fffit/runs/r32-density-iter1/``.
A Latin hypercube with 200 parameter sets exists under
``hfcs-fffit/runs/r32-density-iter1/data/lh_samples_200_r32.txt``.
The signac workspace is created by ``hfcs-fffit/runs/r32-density-iter1/init.py``.

.. code-block:: bash

    cd hfcs-fffit/runs/r32-density-iter1/
    python init.py

The thermodynamic conditions for the simulations and the bounds for each parameter
(LJ sigma and epsilon for C, F, and H) are defined inside ``init.py``.

The simulation workflow is
defined in ``hfcs-fffit/runs/r32-density-iter1/project.py``. The flow operations
defined therein create the simulation input files, perform the simulations,
and run the analysis (calculating the average density). In order to run
these flow operations on a cluster with a job scheduler, it will be
necessary to edit the files under
``hfcs-fffit/runs/r32-density-iter1/templates/`` to be compatible with
your cluster. The signac documentation contains the necessary details.

Once the first iteration of simulations have completed (i.e., all the flow
operations are done), you can perform analysis. The necessary files are located
under ``hfcs-fffit/runs/analysis`` and ``hfcs-fffit/runs/analysis/r32-density-iter1``.
The first step is to extract the results from your signac project into a CSV file
so they can be stored and accessed more easily in the future. This step is
performed by ``extract_r32_density.py``. The script requires the iteration number
as a command line argument.

**WARNING**: Running this script will overwrite your local copy of our simulation
results (stored as CSV files) with the results from your simulations.

To extract the results for iteration 1 run the following:

.. code-block:: bash

    cd hfcs-fffit/analysis/
    python extract_r32_density.py 1


The CSV file with the results is saved under
``hfcs-fffit/analysis/csv/rXX-YY-iterZZ-results.csv`` where ``XX``
is the molecule, ``YY`` is the stage (liquid density or VLE), and
``ZZ`` is the iteration number.

The analysis is performed within a separate directory for each iteration.
For example, for the first iteration, it is performed under
``hfcs-fffit/analysis/r32-density-iter1``. The script ``id-new-samples.py``
loads the results from the CSV file, fits the SVM classifier and GP surrogate
models, loads the Latin hypercube with 1e6 prospective parameter sets,
and identifies the 200 new parameter sets to use for molecular simulations in
iteration 2. These parameter sets are saved to a CSV file:
``hfcs-fffit/analysis/csv/r32-density-iter2-params.csv``.

The second iteration of the liquid density simulations were
performed under the ``hfcs-fffit/runs/r32-density-iter2/``. The procedure
is the same as for iteration 1, but this time the force field parameters
are taken from: ``hfcs-fffit/analysis/csv/r32-density-iter2-params.csv``.
The procedure for analysis is likewise analogous to iteration 1, however,
note that in training the surrogate models,
``hfcs-fffit/runs/analysis/r32-density-iter2/id-new-samples.py`` now uses
the simulation results from both iterations 1 and 2.

HFC-32 VLE Optimization
#######################

The optimization for the vapor-liquid equilibrium iterations is very similar.
The final (iteration 4) liquid density analysis saves the parameters as
``hfcs-fffit/analysis/csv/r32-vle-iter1-params.csv``. The first VLE iteration
begins with these parameters. Once again, the simulations are performed under:
``hfcs-fffit/runs/r32-vle-iter1``. The ``init.py`` is used to set up the
signac workspace, and ``project.py`` defines the simulation workflow
(create the inputs, perform the molecular simulations, and run the analysis).
The results are extracted into the CSV file with
``hfcs-fffit/analysis/extract_r32_vle.py``. Once again the iteration number
is a command line argument, and the results are saved to a CSV file
``hfcs-fffit/analysis/csv/r32-vle-iter1-results.csv``.

Each VLE iteration has a folder with the analysis
scripts (e.g., ``hfcs-fffit/analysis/r32-vle-iter1``). The ``analysis.py``
and ``evaluate-gps.py`` perform basic analysis and create figures evaluating
the performance of the GP models. The ``id-new-samples.py`` loads a Latin
hypercube with 1e6 prospective parameter sets, and identifies the top-performing
parameter sets which will be evaluated with molecular simulations during the
subsequent iteration. For example, the parameter sets to be used for the second
VLE iteration are saved to ``hfcs-fffit/analysis/csv/r32-vle-iter2-params.csv``.
Each subsequent VLE iteration is performed in the same manner.

Credits
~~~~~~~

This work was supported by the National Science Foundation
under grant NSF Award Number OAC-1835630 and NSF Award Number CBET-1917474.
Any opinions, findings, and conclusions or recommendations expressed
in this material are those of the author(s) and do not necessarily
reflect the views of the National Science Foundation.
