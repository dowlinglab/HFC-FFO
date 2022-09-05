import pandas as pd


def main():

    #r32 = pd.read_csv("../csv/r32-pareto.csv",index_col=0)
    r143a = pd.read_csv("../csv/r143a-pareto.csv",index_col=0)

    '''r32_final = r32.drop(
        columns = [
            "sim_liq_density_241K", "sim_liq_density_261K", "sim_liq_density_281K", "sim_liq_density_301K", "sim_liq_density_321K",
            "sim_vap_density_241K", "sim_vap_density_261K", "sim_vap_density_281K", "sim_vap_density_301K","sim_vap_density_321K",
            "sim_Pvap_241K", "sim_Pvap_261K", "sim_Pvap_281K", "sim_Pvap_301K", "sim_Pvap_321K",
            "sim_Hvap_241K", "sim_Hvap_261K", "sim_Hvap_281K", "sim_Hvap_301K", "sim_Hvap_321K",
            "mse_liq_density", "mse_vap_density", "mse_Pvap", "mse_Hvap", "mse_Tc", "mse_rhoc", "is_pareto",
        ]
    )'''

    r143a_final = r143a.drop(
        columns = [
            "sim_liq_density_240K", "sim_liq_density_260K", "sim_liq_density_280K", "sim_liq_density_300K", "sim_liq_density_320K",
            "sim_vap_density_240K", "sim_vap_density_260K", "sim_vap_density_280K", "sim_vap_density_300K", "sim_vap_density_320K",
            "sim_Pvap_240K", "sim_Pvap_260K", "sim_Pvap_280K", "sim_Pvap_300K", "sim_Pvap_320K",
            "sim_Hvap_240K", "sim_Hvap_260K", "sim_Hvap_280K", "sim_Hvap_300K", "sim_Hvap_320K",
            "mse_liq_density", "mse_vap_density", "mse_Pvap", "mse_Hvap", "mse_Tc", "mse_rhoc", "is_pareto",
        ]
    )

    ### Choosing Final Parameter Sets (R-32)
    # Filter for parameter sets with less than 1.5 % error in all properties
    
    '''r32_final = r32_final[
            (r32_final["mape_Pvap"]<=1.5) &
            (r32_final["mape_Hvap"]<=1.5) &
            (r32_final["mape_liq_density"]<=1.5) &
            (r32_final["mape_vap_density"]<=1.5) &
            (r32_final["mape_Tc"]<=1.5) &
            (r32_final["mape_rhoc"]<=1.5)
    ]'''
    
    
    ### Choosing Final Parameter Sets (R-32)
    # Filter for parameter sets with less than 3 % error in all properties
    
    r143a_final = r143a_final[
                    (r143a_final["mape_Pvap"]<=3) &
                    (r143a_final["mape_Hvap"]<=3) &
                    (r143a_final["mape_liq_density"]<=3) &
                    (r143a_final["mape_vap_density"]<=3) &
                    (r143a_final["mape_Tc"]<=3) &
                    (r143a_final["mape_rhoc"]<=3)
    ]
   
    # Save CSV files
    #r32_final.to_csv("../csv/r32-final-4.csv")
    r143a_final.to_csv("../csv/r143a-final.csv")


if __name__ == "__main__":
    main()

