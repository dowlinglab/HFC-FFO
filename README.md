# Machine Learning-Enabled Development of Accurate Force Fields for Refrigerants
Authors: Ning Wang, Montana Carlozo, Eliseo Marin-Rimoldi, Bridgette Belfort, Alexander W. Dowling, and Edward J. Maginn
<!-- Introduction: Provide a brief introduction to the project, including its purpose, goals, and any key features or benefits. -->
## Introduction
**HFC-FFO** is a repository used to rapidly calibrate the LJ parameters of HFC forcefields given experimental data. The key feature of this work is using machine learning tools in the form of Gaussian processes (GPs) which allow us to cheaply estimate the results of a molecular simulation given temperature state points and thermophysical property data.

## Citation
This work has been published on JCTC, whose link is `https://doi.org/10.1021/acs.jctc.3c00338`. Please cite as:

Ning Wang, Montana N. Carlozo, Eliseo Marin-Rimoldi, Bridgette J. Befort, Alexander W. Dowling, and Edward J. Maginn*, “Machine Learning-Enabled Development of Accurate Force Fields for Refrigerants”, J. Chem. Theory Comput., 2023, 19, 14, 4546–4558
   
## Available Data

### HFC Parameter Sets
The non-dominated and best parameter sets for each HFC are
provided under ``HFC-FFO/rXX/analysis/csv/``. Where XX represents a different HFC. For example r-143a, r-14, r-170. The non-dominated
sets are found in ``rXX-pareto.csv``, and
the best sets are found in ``rXX-final.csv``. The parameter values in the CSV files are
normalized between 0 and 1 based on the parameter bounds for each
atom type (see manuscript, or ``HFC-FFO/rXX/analysis/utils/rXX.py`` for definitions of
the upper and lower parameter bounds for each refrigerant).

### Molecular Simulations Inputs
All molecular simulations were performed inside ``HFC-FFO/r##/runs`` where it exists.
Each iteration was managed with ``signac-flow``. Inside of each
directory in ``runs``, you will find all the necessary files to
run the simulations. Note that you may not get the exact same simulation
results due to differences in software versions, random seeds, etc.
Nonetheless, all of the results from our molecular simulations are saved
under ``HFC-FFO/analysis/csv/rXX-YY-iterZZ-results.csv``, where ``XX``
is the molecule, ``YY`` is the stage (liquid density or VLE), and
``ZZ`` is the iteration number.

### Surrogate Modeling Analysis
All of the scripts for the surrogate modeling are provided in
``HFC-FFO/r##/analysis``, following the same naming structure as
the CSV files.

### Figures
All scripts required to generate the primary figures in the
manuscript are reported under ``HFC-FFO/r##/analysis/final-figs`` and the
associated PDF files are located under
``HFC-FFO/r##/analysis/final-figs/pdfs``

## Installation
To run this software, you must have access to all packages in the hfcs-fffit environment (hfcs-fffit.yml) which can be installed using the instructions in the next section.
<!-- Installation: Provide instructions on how to install and set up the project, including any dependencies that need to be installed. -->
This package has a number of requirements that can be installed in
different ways. We recommend using a conda environment to manage
most of the installation and dependencies. However, some items will
need to be installed from source or pip.

Running the simulations will also require an installation of GROMACS.
This can be installed separately (see installation instructions
`here <https://manual.gromacs.org/documentation/2021.2/install-guide/index.html>`).

An example of the procedure is provided below:

    # First clone hfcs-fffit and install pip/conda available dependencies
    # with a new conda environment named hfcs-fffit
    git clone https://github.com/emarinri/hfcs-fffit.git
    git switch -c update-dependencies origin/update-dependencies
    cd hfcs-fffit
    conda env create -f ./devtools/conda-envs/hfcs-fffit.yaml
    conda activate hfcs-fffit
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

### Liquid Density Optimization

**NOTE**: We use Signac and signac flow (`<https://signac.io/>`)
to manage the setup and execution of the molecular simulations. These
instructions assume a working knowledge of that software.

The first iteration of the liquid density simulations was
performed under ``HFC-FFO/r##/runs/rXX-density-iter1/``.

To run liquid density iterations, follow the following steps:
1. Create the initial configuration
   - Prepare rXX_gaff.xml
   - Go to the data folder and use the run.sh file
   ```
     conda activate hfcs-fffit
     cd HFC-FFO/rXX/run/rXX-density-iter1/data
     source run.sh
   ```
2. Initialize Signac workflow              
   - Leave ''HFC-FFO/rXX/run/rXX-density-iter1/data'' untouched
   - Initialize files for simulation use
    ```   
     cd HFC-FFO/rXX/run/rXX-density-iter1/
     python init.py
    ```        
3. Check status a few times throughout the process
   ```  
     python project.py status
   ```       
4. Create force fields and generate inputs
   ```
     python project.py run -o create_forcefield
     python project.py run -o generate_inputs
   ```
5. Create systems
    - **Note: rm -r workspace/ signac_project_document.json signac.rc will remove everything and allow you to start fresh if you mess up**
   ```
     python project.py run -o create_system
   ```
6. Fix topology
   ```
     python project.py run -o fix_topology    
   ```
7. Run simulation
   ```
     python python project.py submit -o simulate --bundle=24 --parallel
   ```  
8. Calculate density
   ```
     python project.py submit -o calculate_density --bundle=24 --parallel
9. Extract density using the following after each LD iteration in the analysis/ folder
   ```
     python extract_rXX_density.py ZZ
   ```
10. Run GP optimization and get samples for the next iteration in the analysis/ folder
   ```
     module load gcc/11.2.0
     python id-new-samples.py
     python plotfig_gp_examples.py 
   ```       

### VLE Optimization

To run vapor-liquid-equilibrium iterations, follow the following steps:
1. Use analysis/csv/rXX-vle-iter1-params.csv to initialize files for simulation use
   ```
     cd HFC-FFO/rXX/run/rXX-vle-iter1/
     python init.py
   ```          
2. Check status a few times throughout the process
   ```
     python project.py status 
   ```       
3. Create force fields
   ```
     python project.py run -o create_forcefield
   ```         
4. Calculate vapor/liquid box size
   ```
     python project.py run -o calc_vapboxl
     python project.py run -o calc_liqboxl
   ```         
5. Run simulation
   ```
     python project.py submit -o equilibrate_liqbox --bundle=12 --parallel
     python project.py run -o extract_final_liqbox
     python project.py submit -o run_gemc --bundle=12 --parallel
   ```   
6. Calculate VLE Properties
   ```
     python project.py run -o calculate_props
   ```
7. Extract VLE properties using the following after each VLE iteration in the analysis/ folder
   ```
     python extract_rXX_vle.py ZZ
   ```
8. Analyze Data
   ```
     module load gcc/11.2.0
     python id-new-samples.py
     python get-new-samples.py 
     python analysis.py 
   ```

### Final Analysis
The nondominated parameter sets and final processing steps can be run using the following:
1. Summarize data
   ```
     cd HFC-FFO/rXX/run/rXX-vle-iterKK
     python id-pareto.py
     cd HFC-FFO/rXX/analysis/final-analysis
     python select_final.py
2. Plots in the paper were generated by codes in HFC-FFO/rXX/analysis/final-figs/

### Known Issues
The instructions outlined above seem to be system-dependent. In some cases, users have the following error:
```
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
```
If you observe this, please try the following in the terminal
```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
which should fix the problem. This is not an optimal solution and is something we would like to address. We found that related projects (`1<https://github.com/openmm/openmm/issues/3943>`_)(`2 <https://github.com/conda/conda/issues/12410>`_) have similar issues.
If you are aware of a robust solution to this issue, please let us know by raising an issue or sending an email!

## Credits
This work is funded by the National Science Foundation, EFRI DChem: Next-generation Low Global Warming Refrigerants, Award no. 2029354, and uses the computing resources provided by the Center for Research Computing (CRC) at the University of Notre Dame. The authors would like to thank Bridgette Befort as her work is used as the basis of this method.

## Contact
Please contact Ning Wang (nwang2@nd.edu), Eliseo Marin Rimoldi (emarinri@nd.edu), or Montana Carlozo (mcarlozo@nd.edu) with any questions, suggestions, or issues.
