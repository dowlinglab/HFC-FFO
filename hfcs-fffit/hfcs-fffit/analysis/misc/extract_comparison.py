import signac
import pandas as pd

projects = {
    "r32-gaff" : signac.get_project("/scratch365/bbefort/nsf-hfcs/comparison/r32-gaff-vle-updatedTemps/"),
    "r32-raabe" : signac.get_project("/scratch365/bbefort/nsf-hfcs/comparison/r32-raabe-vle-updatedTemps/"),
    "r125-gaff" : signac.get_project("/scratch365/bbefort/nsf-hfcs/comparison/r125-gaff-vle-updatedTemps/"),
}

for name, project in projects.items():

    df = pd.DataFrame()

    temp = []
    pres = []
    liq_density = []
    vap_density = []
    pvap = []
    hvap = []

    liq_density_unc = []
    vap_density_unc = []
    pvap_unc = []
    hvap_unc = []

    for job in project:
        try:
            liq_density.append(job.doc.liq_density)
            liq_density_unc.append(job.doc.liq_density_unc)
            vap_density.append(job.doc.vap_density)
            vap_density_unc.append(job.doc.vap_density_unc)
            pvap.append(job.doc.Pvap)
            pvap_unc.append(job.doc.Pvap_unc)
            hvap.append(job.doc.Hvap)
            hvap_unc.append(job.doc.Hvap_unc)
            temp.append(job.sp.T)
            pres.append(job.sp.P)
        except AttributeError:
            pass


    df["T_K"] = temp
    df["P_bar"] = pres

    df["rholiq_kgm3"] = liq_density
    df["rholiq_unc_kgm3"] = liq_density_unc
    df["rhovap_kgm3"] = vap_density
    df["rhovap_unc_kgm3"] = vap_density_unc

    df["pvap_bar"] = pvap
    df["pvap_unc_bar"] = pvap_unc
    df["hvap_kJmol"] = hvap
    df["hvap_unc_kJmol"] = hvap_unc

    df.to_csv(f"../csv/{name}.csv")


