import pandas as pd


def main():

    r14 = pd.read_csv("../csv/r14-pareto-iter2.csv",index_col=0)

    r14_final = r14.drop(
        columns = [
            "sim_liq_density_130K", "sim_liq_density_150K", "sim_liq_density_170K", "sim_liq_density_190K", "sim_liq_density_210K",
            "sim_vap_density_130K", "sim_vap_density_150K", "sim_vap_density_170K", "sim_vap_density_190K", "sim_vap_density_210K",
            "sim_Pvap_130K", "sim_Pvap_150K", "sim_Pvap_170K", "sim_Pvap_190K", "sim_Pvap_210K",
            "sim_Hvap_130K", "sim_Hvap_150K", "sim_Hvap_170K", "sim_Hvap_190K", "sim_Hvap_210K",
            "mse_liq_density", "mse_vap_density", "mse_Pvap", "mse_Hvap", "mse_Tc", "mse_rhoc", "is_pareto",
        ]
    )

    ### Choosing Final Parameter Sets (R-32)
    # Filter for parameter sets with less than 3 % error in all properties
    
    r14_final = r14_final[
                    (r14_final["mape_Pvap"]<=2.0) &
                    (r14_final["mape_Hvap"]<=2.0) &
                    (r14_final["mape_liq_density"]<=2.0) &
                    (r14_final["mape_vap_density"]<=2.0) &
                    (r14_final["mape_Tc"]<=2.0) &
                    (r14_final["mape_rhoc"]<=2.0)
    ]
   
    # Save CSV files
    r14_final.to_csv("../csv/r14-final-iter2.csv")


if __name__ == "__main__":
    main()

