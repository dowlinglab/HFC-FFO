import numpy as np
import signac
import pandas as pd


def save_signac_results():

    thermo_props=["liq_density", "vap_density", "Pvap", "Hvap", "liq_enthalpy",
    "vap_enthalpy", "nmols_liq" , "nmols_vap", "Pvap_unc", "Hvap_unc", "liq_enthalpy_unc",
    "vap_enthalpy_unc"]

    project = signac.get_project(".")
    temps = []
    props = []
    # Loop over all jobs in project and group by parameter sets

    for T, b in project.groupby("T"):
        for job in b:
            tmp = []
            temps.append(T)
            try:
                for prop in thermo_props:
                    tmp.append(job.doc[prop])
                props.append(tmp)
            except:
                print(f"Job failed: T: {T}, job ID: {job.id}")
                props.append(np.nan)

   
    df = pd.DataFrame(props, index=temps, columns=thermo_props)
    df["Hvap"]= (df["vap_enthalpy"]/df["nmols_vap"]) - (df["liq_enthalpy"]/df["nmols_liq"])
    print(df)

    df.to_csv("results.csv", index_label="temperature")
if __name__ == "__main__":
    save_signac_results()
