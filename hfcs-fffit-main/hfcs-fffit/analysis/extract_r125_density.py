import signac
import sys

from fffit.signac import save_signac_results
from utils.r125 import R125Constants


def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_r125_density.py [iteration number]")
        exit(1)
    else:
        iternum = sys.argv[1]

    R125 = R125Constants()

    run_path = "/scratch365/rdefever/hfcs-fffit/hfcs-fffit/runs/"
    itername = "r125-density-iter" + str(iternum)
    project_path = run_path + itername
    csv_name = "csv/" + itername + "-results.csv"

    property_names = ["density"]
    project = signac.get_project(project_path)

    save_signac_results(project, R125.param_names, property_names, csv_name)


if __name__ == "__main__":
    main()
