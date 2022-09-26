import signac
import sys

from fffit.signac import save_signac_results
from utils.r134a import R134aConstants


def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_r134a_density.py [iteration number]")
        exit(1)
    else:
        iternum = sys.argv[1]

    R134a = R134aConstants()

    run_path = "/scratch365/nwang2/ff_development/HFC_143a_FFO_FF/r134a/run/"
    itername = "r134a-density-iter" + str(iternum)
    project_path = run_path + itername
    csv_name = "csv/" + itername + "-results.csv"

    property_names = ["density"]
    project = signac.get_project(project_path)

    save_signac_results(project, R134a.param_names, property_names, csv_name)


if __name__ == "__main__":
    main()
