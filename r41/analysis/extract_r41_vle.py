import signac
import sys

from fffit.signac import save_signac_results
from utils.r41 import R41Constants


def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_r41_vle.py [iteration number]")
        exit(1)
    else:
        iternum = sys.argv[1]

    R41 = R41Constants()

    run_path = "/scratch365/mcarlozo/HFC-FFO/r41/run/"
    itername = "r41-vle-iter" + str(iternum)
    project_path = run_path + itername
    csv_name = "csv/" + itername + "-results.csv"

    property_names = [
        "liq_density",
        "vap_density",
        "Hvap",
        "Pvap",
        "liq_enthalpy",
        "vap_enthalpy",
    ]

    project = signac.get_project(project_path)

    save_signac_results(project, R41.param_names, property_names, csv_name)


if __name__ == "__main__":
    main()
