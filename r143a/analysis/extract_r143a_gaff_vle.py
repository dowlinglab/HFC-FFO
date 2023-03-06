import signac
import sys

from fffit.signac import save_signac_results
from utils.r143a import R143aConstants


def main():

    R143a = R143aConstants()

    run_path = "/scratch365/nwang2/ff_development/HFC_143a_FFO_FF/r143a/gaff/"
    #itername = "r143a-vle-iter" + str(iternum)
    project_path = run_path #+ itername
    csv_name =  "csv/gaff-results.csv"

    property_names = [
        "liq_density",
        "vap_density",
        "Hvap",
        "Pvap",
        "liq_enthalpy",
        "vap_enthalpy",
    ]

    project = signac.get_project(project_path)
    print(project)
    temperature = round(project.sp.T)
    new_row["temperature"] = temperature
    data=[]
    # Extract property values. Insert N/A if not found
    for property_name in property_names:
        try:
            property_ = project.doc[property_name]
            new_row[property_name] = property_
        except KeyError:
            print(f"Job failed: {job.id}")
            new_row[property_name] = np.nan

    data.append(new_row)
    print(data)
    #save_signac_results(project, R143a.param_names, property_names, csv_name)


if __name__ == "__main__":
    main()
