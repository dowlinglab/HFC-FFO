import sys
import signac
import numpy as np
import unyt as u

from parameters import simulation_temperatures, project_name


def init_project():

    # Initialize project
    project = signac.init_project(project_name)

    for t in simulation_temperatures: 

        state_point = {
            "T": float(t.in_units(u.K).to_value()),
        }

        job = project.open_job(state_point)
        job.init()
    

if __name__ == "__main__":
    init_project()
