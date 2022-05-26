## hfcs-fffit/runs

The directory naming scheme is ``[molname]-[itertype]-iter[iternum]`` where ``molname`` is ``r32`` or ``r125`` and ``itertype`` is either ``density`` or ``vle``. 

Each directory contains the files necessary to run the molecular simulations associated with the given molecule/iteration. The ``init.py`` defines the ``signac`` workspace and the ``project.py`` defines the flow operations (i.e., performs the system setup, creates the molecular simulation inputs, performs the simulations, and performs any analysis).

