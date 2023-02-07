import pandas as pd


def main():

    #r32 = pd.read_csv("../csv/r32-pareto.csv",index_col=0)
    r50 = pd.read_csv("../csv/r50-pareto-iter3.csv",index_col=0)

    r50_final = r50.drop(
        columns = [
            "sim_liq_density_130K", "sim_liq_density_140K", "sim_liq_density_150K", "sim_liq_density_160K", "sim_liq_density_170K",
            "sim_vap_density_130K", "sim_vap_density_140K", "sim_vap_density_150K", "sim_vap_density_160K", "sim_vap_density_170K",
            "sim_Pvap_130K", "sim_Pvap_140K", "sim_Pvap_150K", "sim_Pvap_160K", "sim_Pvap_170K",
            "sim_Hvap_130K", "sim_Hvap_140K", "sim_Hvap_150K", "sim_Hvap_160K", "sim_Hvap_170K",
            "mse_liq_density", "mse_vap_density", "mse_Pvap", "mse_Hvap", "mse_Tc", "mse_rhoc", "is_pareto",
        ]
    )

    ### Choosing Final Parameter Sets (R-32)
    # Filter for parameter sets with less than 3 % error in all properties
    
    r50_final = r50_final[
                    (r50_final["mape_Pvap"]<=3.5) &
                    (r50_final["mape_Hvap"]<=3.5) &
                    (r50_final["mape_liq_density"]<=3.5) &
                    (r50_final["mape_vap_density"]<=3.5) &
                    (r50_final["mape_Tc"]<=3.5) &
                    (r50_final["mape_rhoc"]<=3.5)
    ]
   
    # Save CSV files
    #r32_final.to_csv("../csv/r32-final-4.csv")
    r50_final.to_csv("../csv/r50-final-iter3.csv")


if __name__ == "__main__":
    main()

