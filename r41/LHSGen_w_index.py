from scipy.stats import qmc
import pandas as pd
import csv

d= 6 #Number of dimensions
n= int(5e5) ## of samples
seed = 7
n=200
sampler = qmc.LatinHypercube(d, seed = seed)
sample = sampler.random(n)
sample = pd.DataFrame(sample)
sample.columns = ['sigma_C1','sigma_F1', 'sigma_H1', 'epsilon_C1','epsilon_F1', 'epsilon_H1'] #change to sigma and epsilon name of different atom types
#sample.set_index('sigma_C1')

filename = 'LHS_'+str(n)+'_x_'+str(d)+'.csv'
sample.to_csv(filename, index = True)
