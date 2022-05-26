import numpy as np
import signac
import pandas as pd


def save_signac_results(project, param_names, property_names, csv_name):
    """Save the signac results to a CSV file.

    Parameters
    ----------
    project : signac.Project
        signac project to load
    param_names : set
        set of parameter names (excluding temperature)
    property_names : set
        set of property names
    csv_name : string
        name of csv file to save results
    """
    if type(param_names) not in (list, tuple):
        raise TypeError("param_names must be a list or tuple")
    if type(property_names) not in (list, tuple):
        raise TypeError("property_names must be a list or tuple")

    job_groupby = tuple(param_names)
    property_names = tuple(property_names)

    print(f"Extracting the following properties: {property_names}")

    # Store data here before converting to dataframe
    data = []

    # Loop over all jobs in project and group by parameter sets
    for params, job_group in project.groupby(job_groupby):

        for job in job_group:
            # Extract the parameters into a dict
            new_row = {
                name: param for (name, param) in zip(job_groupby, params)
            }

            # Extract the temperature for each job.
            # Assumes temperature increments >= 1 K
            temperature = round(job.sp.T)
            new_row["temperature"] = temperature

            # Extract property values. Insert N/A if not found
            for property_name in property_names:
                try:
                    property_ = job.doc[property_name]
                    new_row[property_name] = property_
                except KeyError:
                    print(f"Job failed: {job.id}")
                    new_row[property_name] = np.nan

            data.append(new_row)

    # Save to csv file for record-keeping
    df = pd.DataFrame(data)
    df.to_csv(csv_name)
