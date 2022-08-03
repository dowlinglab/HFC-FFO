from scipy.stats import qmc
import csv

d= 8 #Number of dimensions
n= int(5e5) ## of samples
sampler = qmc.LatinHypercube(d)
sample = sampler.random(n)
print(sample)


# with open('LHS_'+str(n)+'_x_'+str(d)+'TEST.csv', 'w', newline='') as file:
with open('LHS_'+str(n)+'_x_'+str(d)+'.csv', 'w', newline='') as file:
    mywriter = csv.writer(file, delimiter=',')
    mywriter.writerows(sample)
