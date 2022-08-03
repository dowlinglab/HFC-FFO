from scipy.stats import qmc
import pandas as pd
import csv

d= 8 #Number of dimensions
n= int(5e5) ## of samples
sampler = qmc.LatinHypercube(d)
sample = sampler.random(n)
sample = pd.DataFrame(sample)
sample.columns = ['1','2','3','4','5','6','7','8']
sample.set_index('1')

filename = 'LHS_'+str(n)+'_x_'+str(d)+'I_True.csv'
sample.to_csv(filename, index = True)