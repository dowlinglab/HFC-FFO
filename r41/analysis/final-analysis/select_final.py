import pandas as pd


def main():

    r41 = pd.read_csv("../csv/r41-pareto-iter2.csv",index_col=0)

    r41_final = r41.drop(
        columns = [
            "sim_liq_density_210K", "sim_liq_density_230K", "sim_liq_density_250K", "sim_liq_density_270K", "sim_liq_density_290K",
            "sim_vap_density_210K", "sim_vap_density_230K", "sim_vap_density_250K", "sim_vap_density_270K", "sim_vap_density_290K",
            "sim_Pvap_210K", "sim_Pvap_230K", "sim_Pvap_250K", "sim_Pvap_270K", "sim_Pvap_290K",
            "sim_Hvap_210K", "sim_Hvap_230K", "sim_Hvap_250K", "sim_Hvap_270K", "sim_Hvap_290K",
            "mse_liq_density", "mse_vap_density", "mse_Pvap", "mse_Hvap", "mse_Tc", "mse_rhoc", "is_pareto",
        ]
    )

    ### Choosing Final Parameter Sets (R-32)
    # Filter for parameter sets with less than 3 % error in all properties
    
    r41_final = r41_final[
                    (r41_final["mape_Pvap"]<=2.0) &
                    (r41_final["mape_Hvap"]<=2.0) &
                    (r41_final["mape_liq_density"]<=2.0) &
                    (r41_final["mape_vap_density"]<=2.0) &
                    (r41_final["mape_Tc"]<=2.0) &
                    (r41_final["mape_rhoc"]<=2.0)
    ]
   
    # Save CSV files
    r41_final.to_csv("../csv/r41-final-iter2.csv")


if __name__ == "__main__":
    main()

