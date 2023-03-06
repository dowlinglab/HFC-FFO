import pandas as pd


def main():

    #r32 = pd.read_csv("../csv/r32-pareto.csv",index_col=0)
    r170 = pd.read_csv("../csv/r170-pareto-iter2.csv",index_col=0)

    '''r32_final = r32.drop(
        columns = [
            "sim_liq_density_241K", "sim_liq_density_261K", "sim_liq_density_281K", "sim_liq_density_301K", "sim_liq_density_321K",
            "sim_vap_density_241K", "sim_vap_density_261K", "sim_vap_density_281K", "sim_vap_density_301K","sim_vap_density_321K",
            "sim_Pvap_241K", "sim_Pvap_261K", "sim_Pvap_281K", "sim_Pvap_301K", "sim_Pvap_321K",
            "sim_Hvap_241K", "sim_Hvap_261K", "sim_Hvap_281K", "sim_Hvap_301K", "sim_Hvap_321K",
            "mse_liq_density", "mse_vap_density", "mse_Pvap", "mse_Hvap", "mse_Tc", "mse_rhoc", "is_pareto",
        ]
    )'''

    r170_final = r170.drop(
        columns = [
            "sim_liq_density_210K", "sim_liq_density_230K", "sim_liq_density_250K", "sim_liq_density_270K", "sim_liq_density_290K",
            "sim_vap_density_210K", "sim_vap_density_230K", "sim_vap_density_250K", "sim_vap_density_270K", "sim_vap_density_290K",
            "sim_Pvap_210K", "sim_Pvap_230K", "sim_Pvap_250K", "sim_Pvap_270K", "sim_Pvap_290K",
            "sim_Hvap_210K", "sim_Hvap_230K", "sim_Hvap_250K", "sim_Hvap_270K", "sim_Hvap_290K",
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
    
    r170_final = r170_final[
                    (r170_final["mape_Pvap"]<=2.5) &
                    (r170_final["mape_Hvap"]<=2.5) &
                    (r170_final["mape_liq_density"]<=2.5) &
                    (r170_final["mape_vap_density"]<=2.5) &
                    (r170_final["mape_Tc"]<=2.5) &
                    (r170_final["mape_rhoc"]<=2.5)
    ]
   
    # Save CSV files
    #r32_final.to_csv("../csv/r32-final-4.csv")
    r170_final.to_csv("../csv/r170-final-iter2.csv")


if __name__ == "__main__":
    main()

