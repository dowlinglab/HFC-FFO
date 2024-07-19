from scipy.stats import qmc
import pandas as pd
import csv

d= 4 #Number of dimensions
#n= int(5e5) ## of samples
n=200
sampler = qmc.LatinHypercube(d)
sample = sampler.random(n)
sample = pd.DataFrame(sample)
sample.columns = ['sigma_C1','sigma_F1','epsilon_C1','epsilon_F1'] #change to sigma and epsilon name of different atom types
#sample.set_index('sigma_C1')

filename = 'LHS_'+str(n)+'_x_'+str(d)+'.csv'
sample.to_csv(filename, index = True)
