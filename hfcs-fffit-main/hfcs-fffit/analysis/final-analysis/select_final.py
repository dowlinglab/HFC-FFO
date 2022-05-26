import pandas as pd


def main():

    r32 = pd.read_csv("../csv/r32-pareto.csv",index_col=0)
    r125 = pd.read_csv("../csv/r125-pareto.csv",index_col=0)

    r32_final = r32.drop(
        columns = [
            "sim_liq_density_241K", "sim_liq_density_261K", "sim_liq_density_281K", "sim_liq_density_301K", "sim_liq_density_321K",
            "sim_vap_density_241K", "sim_vap_density_261K", "sim_vap_density_281K", "sim_vap_density_301K","sim_vap_density_321K",
            "sim_Pvap_241K", "sim_Pvap_261K", "sim_Pvap_281K", "sim_Pvap_301K", "sim_Pvap_321K",
            "sim_Hvap_241K", "sim_Hvap_261K", "sim_Hvap_281K", "sim_Hvap_301K", "sim_Hvap_321K",
            "mse_liq_density", "mse_vap_density", "mse_Pvap", "mse_Hvap", "mse_Tc", "mse_rhoc", "is_pareto",
        ]
    )

    r125_final = r125.drop(
        columns = [
            "sim_liq_density_229K", "sim_liq_density_249K", "sim_liq_density_269K", "sim_liq_density_289K", "sim_liq_density_309K",
            "sim_vap_density_229K", "sim_vap_density_249K", "sim_vap_density_269K", "sim_vap_density_289K", "sim_vap_density_309K",
            "sim_Pvap_229K", "sim_Pvap_249K", "sim_Pvap_269K", "sim_Pvap_289K", "sim_Pvap_309K",
            "sim_Hvap_229K", "sim_Hvap_249K", "sim_Hvap_269K", "sim_Hvap_289K", "sim_Hvap_309K",
            "mse_liq_density", "mse_vap_density", "mse_Pvap", "mse_Hvap", "mse_Tc", "mse_rhoc", "is_pareto",
        ]
    )

    ### Choosing Final Parameter Sets (R-32)
    # Filter for parameter sets with less than 1.5 % error in all properties
    
    r32_final = r32_final[
            (r32_final["mape_Pvap"]<=1.5) &
            (r32_final["mape_Hvap"]<=1.5) &
            (r32_final["mape_liq_density"]<=1.5) &
            (r32_final["mape_vap_density"]<=1.5) &
            (r32_final["mape_Tc"]<=1.5) &
            (r32_final["mape_rhoc"]<=1.5)
    ]
    
    
    ### Choosing Final Parameter Sets (R-32)
    # Filter for parameter sets with less than 2.5 % error in all properties
    
    r125_final = r125_final[
                    (r125_final["mape_Pvap"]<=2.5) &
                    (r125_final["mape_Hvap"]<=2.5) &
                    (r125_final["mape_liq_density"]<=2.5) &
                    (r125_final["mape_vap_density"]<=2.5) &
                    (r125_final["mape_Tc"]<=2.5) &
                    (r125_final["mape_rhoc"]<=2.5)
    ]
   
    # Save CSV files
    r32_final.to_csv("../csv/r32-final-4.csv")
    r125_final.to_csv("../csv/r125-final-4.csv")


if __name__ == "__main__":
    main()

