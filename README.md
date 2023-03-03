# HFC Force Field Optimization Using a Machine Learning Based Method
<!-- Introduction: Provide a brief introduction to the project, including its purpose, goals, and any key features or benefits. -->
## Introduction
**HFC-FFO** is a repository used to rapidly calibrate the LJ parameters of HFC forcefields given experimental data. The key feature of this work is that is using machine learning tools in the form of Gaussian processes (GPs) which allow us to cheaply estimate the resuls of a molecular simulation given temperature state points and thermophysical property data.

## Citation
This work has been submitted for review. In the meantime, you
may cite the `preprint <LINK HERE>`_ as:

.. code-block:: bash

    FILL IN LATER
   
## Available Data

### HFC Parameter Sets
The non-dominated and best parameter sets for each HFC are
provided under ``HFC-FFO/rXX/analysis/csv/``. Where XX represents a different HFC. For example r-143a, r-14, r-170. The non-dominated
sets are found in ``rXX-pareto.csv``, and
the best sets are found in ``rXX-final.csv``. The parameter values in the csv files are
normalized between 0 and 1 based upon the parameter bounds for each
atom type (see manuscript, or ``HFC-FFO/rXX/analysis/utils/rXX.py`` for definitions of
the upper and lower parameter bounds for each refrigerant).

### Molecular Simulations Inputs
All molecular simulations were performed inside ``HFC-FFO/r##/analysis`` or ``HFC-FFO/r##/runs`` where it exists.
Each iteration was managed with ``signac-flow``. Inside of each
directory in ``runs`` or "analysis", you will find all the necessary files to
run the simulations. Note that you may not get the exact same simulation
results due to differences in software versions, random seeds, etc.
Nonetheless, all of the results from our molecular simulations are saved
under ``HFC-FFO/analysis/csv/rXX-YY-iterZZ-results.csv``, where ``XX``
is the molecule, ``YY`` is the stage (liquid density or VLE), and
``ZZ`` is the iteration number.

### Surrogate Modeling Analysis
All of the scripts for the surrogate modeling are provided in
``HFC-FFO/r##/analysis``, following the same naming structure as
the csv files.

### Figures
All scripts required to generate the primary figures in the
manuscript are reported under ``HFC-FFO/r##/analysis/final-figs`` and the
associated PDF files are located under
``HFC-FFO/r##/analysis/final-figs/pdfs``

## Installation
To run this software, you must have access to all packages in the hfcs-fffit environment which can be installed using the instructions in the next section.
<!-- Installation: Provide instructions on how to install and set up the project, including any dependencies that need to be installed. -->
This package has a number of requirements that can be installed in
different ways. We recommend using a conda environment to manage
most of the installation and dependencies. However, some items will
need to be installed from source or pip.

**Note: Can we ask Eliseo to update this part? I'm not really sure how to do it. The stuff below is just copied from Bridgette's readme**
We recommend starting with a fresh conda environment, then installing
the packages listed under ``requirements-pip.txt`` with pip, then
installing the packages listed under ``requirements-conda.txt`` with
conda, and finally installing a few items from source
``requirements-other.txt``. We recommend ``python3.7`` and
taking packages from ``conda-forge``.

Running the simulations will also require an installation of GROMACS.
This can be installed separately (see installation instructions
`here <https://manual.gromacs.org/documentation/2021.2/install-guide/index.html>`_).

**WARNING**: Cloning the ``HFC-FFO`` repository will take some time
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

<!-- Usage: Provide instructions on how to use the project, including any configuration or customization options. Examples of usage scenarios can also be added. -->
## Usage
**Ning, Please update the repo so that files are organized nicely. This is super hard to write at present because there's no file organization**
### Liquid Density Optimization

**NOTE**: We use signac and signac flow (`<https://signac.io/>`_)
to manage the setup and execution of the molecular simulations. These
instructions assume a working knowledge of that software.

The first iteration of the liquid density simulations were
performed under ``HFC-FFO/r##/runs/rXX-density-iter1/``.

To run liquid density iterations, follow the following steps:
1. Create The Initial Configuration
   - Prepare rXX_gaff.xml
   - Go to the data folder and use the run.sh file
        - .. code-block:: bash
                conda activate hfcs-fffit
                cd HFC-FFO/rXX/run/rXX-density-iter1/data
                python create_system.py  ff.xml  run.sh
              
   - Leave ''HFC-FFO/rXX/run/rXX-density-iter1/data'' untouched
   - Initialize files for simulation use
        - .. code-block:: bash
                cd HFC-FFO/rXX/run/rXX-density-iter1/
                python init.py
             
        - (Ning, what does signac.rc and workspace/ do? What order do they go in?)
2. Check status a few times throughout the process
    - .. code-block:: bash

          python project.py status -d
          
3. Create Force Fields
    - **Ning, where to cd to?**
    - .. code-block:: bash
    
            python project.py run -o create_forcefield
            python project.py run -o generate_inputs
4. Create Systems
    - **Note: rm -r workspace/ signac_project_document.json signac.rc will remove everything and allow you to start fresh if you mess up**
    - **Ning, where to cd to?**
    - .. code-block:: bash
    
            python project.py run -o create_system
5. Fix Topology
    - **Ning, where to cd to?**
    - .. code-block:: bash
    
            python project.py run -o fix_topology    
6. Run Simulation
    - **Ning, where to cd to?**
    - .. code-block:: bash
    
            python python project.py submit -o simulate --bundle=24 --parallel
7. Prepare rXX.py in HFC_FFO/rXX/analysis/utils
    - Ensure that experimental data is correct and that the class name has been updated
8. Calculate Density
    - .. code-block:: bash
    
            cd HFC-FFO/rXX/run/rXX-density-iter1
            python project.py submit -o calculate_density --bundle=24 --parallel
    - Extract density using the following after each LD iteration
       .. code-block:: bash
       
           python extract_rXX_density.py ZZ
9. Run GP Validation
    - **Ning, where to cd to?**
    - .. code-block:: bash
   
           module load gcc/11.1.0
           python id-new-samples.py
           python plotfig_gp_examples.py 
           python r134a-cumu-density.py
           python clf.py



**Ning, I copied and commented out the LD and VLE iter guide that Bridgette wrote here. Feel free to add anything from here that you think is necessary**
<!-- **Ning, where can I find these for each sample. It's not consistent?** A Latin hypercube with 500000 parameter sets exists under
``HFC-FFO/analysis/rXX-density-iter1/LHS_500000_x_10.csv``.
The signac workspace is created by ``hfcs-fffit/runs/rXX-density-iter1/init.py``.

.. code-block:: bash

    cd HFC-FFO/rXX/runs/rXX-density-iter1/
    python init.py

The thermodynamic conditions for the simulations and the bounds for each parameter
(LJ sigma and epsilon for C, F, and H) are defined inside "HFC-FFO/rXX/analysis/r##.py".

The simulation workflow is
defined in ``HFC-FFO/rXX/run/rXX-density-iter1/project.py``. The flow operations
defined therein create the simulation input files, perform the simulations,
and run the analysis (calculating the average density). In order to run
these flow operations on a cluster with a job scheduler, it will be
necessary to edit the files under
``HFC-FFO/rXX/run/r32-density-iter1/templates/`` to be compatible with
your cluster. The signac documentation contains the necessary details.

Once the first iteration of simulations have completed (i.e., all the flow
operations are done), you can perform analysis. The necessary files are located
under ``HFC-FFO/rXX/runs/analysis`` and ``hfcs-fffit/runs/analysis/r32-density-iter1``.
The first step is to extract the results from your signac project into a CSV file
so they can be stored and accessed more easily in the future. This step is
performed by ``extract_rXX_density.py``. The script requires the iteration number
as a command line argument.
 -->
<!-- **WARNING**: Running this script will overwrite your local copy of our simulation
results (stored as CSV files) with the results from your simulations.

To extract the results for iteration 1 run the following:

.. code-block:: bash

    cd HFC-FFO/rXX/analysis/
    python extract_rXX_density.py 1


The CSV file with the results is saved under
``HFC-FFO/rXX/analysis/csv/rXX-YY-iterZZ-results.csv`` where ``XX``
is the molecule, ``YY`` is the stage (liquid density or VLE), and
``ZZ`` is the iteration number.

The analysis is performed within a separate directory for each iteration.
For example, for the first iteration, it is performed under
``HFC-FFO/rXX/analysis/r32-density-iter1``. The script ``id-new-samples.py``
loads the results from the CSV file, fits the SVM classifier and GP surrogate
models, loads the Latin hypercube with 1e6 prospective parameter sets,
and identifies the 200 new parameter sets to use for molecular simulations in
iteration 2. These parameter sets are saved to a CSV file:
``HFC-FFO/rXX/analysis/csv/r32-density-iter2-params.csv``.

The second iteration of the liquid density simulations were
performed under the ``hfcs-fffit/runs/r32-density-iter2/``. The procedure
is the same as for iteration 1, but this time the force field parameters
are taken from: ``hfcs-fffit/analysis/csv/r32-density-iter2-params.csv``.
The procedure for analysis is likewise analogous to iteration 1, however,
note that in training the surrogate models,
``hfcs-fffit/runs/analysis/r32-density-iter2/id-new-samples.py`` now uses
the simulation results from both iterations 1 and 2. -->

### VLE Optimization

To run vapor-liquid-equilbrium iterations, follow the following steps:
1. Get the parameters
   - Go to the data folder and use the run.sh file
        - Note: JJ represents the density iteration you want to run
        - .. code-block:: bash
                conda activate hfcs-fffit
                cd HFC-FFO/rXX/run/rXX-density-iterJJ/
                module load gcc/11.2.0
                python id-new-samples.py
              
   - Initialize files for simulation use
        - .. code-block:: bash
                cd HFC-FFO/rXX/run/rXX-vle-iter1/
                python init.py
             
        - (Ning, what does signac.rc and workspace/ do? What order do they go in?)
2. Check status a few times throughout the process
    - **Ning, where to cd to? Does it matter?**
    - .. code-block:: bash

          python project.py status -d
          
3. Create Force Fields
    - **Ning, where to cd to?**
    - .. code-block:: bash
            cd HFC-FFO/rXX/run/rXX-vle-iter1
            python project.py run -o create_forcefield
            
4. Create Systems
    - .. code-block:: bash
    
            cd HFC-FFO/rXX/run/rXX-vle-iter1
            python project.py run -o calc_vapboxl
            python project.py run -o calc_liqboxl
            
5. Run Simulation
    - **Ning, where to cd to?**
    - .. code-block:: bash
    
6. Calculate VLE Properties
    - .. code-block:: bash
    
            cd HFC-FFO/rXX/run/rXX-vle-iter1        
            python project.py submit -o equilibrate_liqbox --bundle=12 --parallel
            python project.py run -o extract_final_liqbox
            python project.py submit -o run_gemc --bundle=12 --parallel
            python project.py run -o calculate_props

    - Extract VLE properties using the following after each vle iteration
       .. code-block:: bash
       
           python extract_rXX_vle.py ZZ
7. Analyze Data
    - .. code-block:: bash
           cd HFC-FFO/rXX/run/rXX-vle-iter1
           module load gcc/11.1.0
           python id-new-samples.py
           python get-new-samples.py 
           python analysis.py -> figs/ 


### Final Analysis
The pareto optimal parameter sets and final processing steps can be ran using the following:
1. Summarize data
    - **Ning, where to cd to?**
    - KK represents final VLE iteration
    - .. code-block:: bash
            cd HFC-FFO/rXX/run/rXX-vle-iterKK
            python id-pareto.py
            cd HFC-FFO/rXX/analysis/final-analysis
            python select_final.py
            cd HFC-FFO/rXX/analysis/final-figs
            python plotfig-r143a-parallel.py
            python  plotfig-r143a-parallel-zoomed.py
            python plotfig-r143a-radar.py

<!-- The optimization for the vapor-liquid equilibrium iterations is very similar.
The final (iteration 4) liquid density analysis saves the parameters as
``HFC-FFO/rXX/analysis/csv/rXX-vle-iter1-params.csv``. The first VLE iteration
begins with these parameters. Once again, the simulations are performed under:
``HFC-FFO/rXX/runs/rXX-vle-iter1``. The ``init.py`` is used to set up the
signac workspace, and ``project.py`` defines the simulation workflow
(create the inputs, perform the molecular simulations, and run the analysis).
The results are extracted into the CSV file with
``HFC-FFO/rXX/analysis/extract_rXX_vle.py``. Once again the iteration number
is a command line argument, and the results are saved to a CSV file
``HFC-FFO/rXX/analysis/csv/rXX-vle-iter1-results.csv``.

Each VLE iteration has a folder with the analysis
scripts (e.g., ``hfcs-fffit/analysis/r32-vle-iter1``). The ``analysis.py``
and ``evaluate-gps.py`` perform basic analysis and create figures evaluating
the performance of the GP models. The ``id-new-samples.py`` loads a Latin
hypercube with 1e6 prospective parameter sets, and identifies the top-performing
parameter sets which will be evaluated with molecular simulations during the
subsequent iteration. For example, the parameter sets to be used for the second
VLE iteration are saved to ``HFC-FFO/rXX/analysis/csv/rXX-vle-iter2-params.csv``.
Each subsequent VLE iteration is performed in the same manner. -->

<!-- Contributing: Explain how people can contribute to the project, such as reporting issues or submitting pull requests. Also provide guidelines for contributions, such as coding conventions, commit message formatting, and branch naming conventions. -->

<!-- Credits: Acknowledge any contributors or sources of inspiration for the project. -->

## Credits
This work is funnded by the National Science Foundation, EFRI DChem: Next-generation Low Global Warming Refrigerants, Award no. 2029354 and uses the computing resources provided by the Center for Research Computing (CRC) at the University of Notre Dame. The authors would like to thank Bridgette Befort as her work is used as the basis of this method.

<!-- License: Specify the license under which the project is released, and include any relevant copyright notices. -->

<!-- Contact: Provide contact information for the project maintainer(s) in case of questions, suggestions or issues. -->
## Contact
Please contact Ning Wang (nwang2@nd.edu), Eliseo Marin Rimoldi (emarinri@nd.edu) or Montana Carlozo (mcarlozo@nd.edu) with any questions, suggestions, or issues
