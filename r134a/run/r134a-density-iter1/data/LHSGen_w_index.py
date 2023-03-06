from scipy.stats import qmc
import pandas as pd
import csv

sample=pd.read_csv("LHS_200_x_10.csv",header=None)
print(sample)
sample.columns = ['sigma_C1','sigma_C2','sigma_F1','sigma_F2','sigma_H1','epsilon_C1','epsilon_C2','epsilon_F1','epsilon_F2','epsilon_H1'] #change to sigma and epsilon name of different atom types
sample.set_index('sigma_C1')

filename = 'r134a-density-iter1-params.csv'
sample.to_csv(filename, index = True)
