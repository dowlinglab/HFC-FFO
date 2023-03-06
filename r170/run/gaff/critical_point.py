import numpy as np
import unyt as u
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import pandas as pd


#vle_data = [
#(360.0 * u.K, 874.80 * u.kilogram/(u.meter)**3, 190.1 * u.kilogram/(u.meter)**3, 31.0 * u.bar),
#(340.0 * u.K, 1007.3 * u.kilogram/(u.meter)**3, 109.7 * u.kilogram/(u.meter)**3, 19.9 * u.bar),
#(320.0 * u.K, 1113.4 * u.kilogram/(u.meter)**3, 60.80 * u.kilogram/(u.meter)**3, 12.1 * u.bar),
#(300.0 * u.K, 1195.5 * u.kilogram/(u.meter)**3, 33.70 * u.kilogram/(u.meter)**3, 6.90 * u.bar),
#(280.0 * u.K, 1265.0 * u.kilogram/(u.meter)**3, 17.70 * u.kilogram/(u.meter)**3, 3.60 * u.bar),
#(260.0 * u.K, 1326.9 * u.kilogram/(u.meter)**3, 8.500 * u.kilogram/(u.meter)**3, 1.70 * u.bar),
#(240.0 * u.K, 1384.3 * u.kilogram/(u.meter)**3, 3.600 * u.kilogram/(u.meter)**3, 0.70 * u.bar),
#(220.0 * u.K, 1444.1 * u.kilogram/(u.meter)**3, 1.300 * u.kilogram/(u.meter)**3, 0.30 * u.bar),
#(200.0 * u.K, 1494.9 * u.kilogram/(u.meter)**3, 0.300 * u.kilogram/(u.meter)**3, 0.10 * u.bar),
#]

vle_data = pd.read_csv("results.csv")

print(vle_data)
#vle_data = np.array(vle_data)
temp = vle_data["temperature"] 
rho_liq = vle_data["liq_density"]
rho_vap = vle_data["vap_density"]
p_vap = vle_data["Pvap"]

beta = 0.325

x = (rho_liq - rho_vap) ** (1/beta)
y = rho_liq + rho_vap

res = stats.linregress(x, y)
m = res.slope
b = res.intercept

rho_c = b / 2.0

x = temp
y = (rho_liq + rho_vap) / 2.0 - rho_c

res = stats.linregress(x, y)
m = res.slope
b = res.intercept

A = m
temp_c = -b / A
B = (2 * A / m)**beta

## Reidel equation
#
#def logP_reidel(T, A, B, C, D):
#    return A + B / T + C * np.log(T) + D * T ** 2 
#
## Antoine equation
#
#def logP_antoine(T, A, B, C):
#    return A - B / (C + T)
#
#popt, pcov = curve_fit(logP_reidel, temp, np.log(p_vap))
#popt, pcov = curve_fit(logP_antoine, temp, np.log10(p_vap))

#p_c = np.exp(logP_reidel(temp_c, *popt))
#p_c = 10**(logP_antoine(temp_c, *popt))

print(f"Critical temperature: {temp_c}")
print(f"Critical density: {rho_c}")
#print(f"Critical pressure: {p_c}")

fig = plt.figure()
plt.xlabel(r"$(\rho_L - \rho_V)^{1/\beta}$")
plt.ylabel(r"$\rho_L + \rho_V$")
plt.plot(x, y, 'o', linewidth=3, label="Original data")
plt.plot(x, m * x + b, 'r', linewidth=3, label="Fitted line")
plt.legend()
plt.show()

fig = plt.figure()
plt.xlabel(r"$T$")
plt.ylabel(r"$0.5 * (\rho_L + \rho_V) - \rho_c$")
plt.plot(x, y, 'o', linewidth=3, label="Original data")
plt.plot(x, m * x + b, 'r', linewidth=3, label="Fitted line")
plt.legend()
plt.show()

#fig = plt.figure()
#plt.xlabel(r"$T$")
#plt.ylabel(r"$logP$")
#plt.plot(temp, np.log(p_vap), 'o', linewidth=3, label="Original data")
#plt.plot(temp, logP_reidel(temp, *popt), 'r', linewidth=3, label="Fitted line")
#plt.legend()
#plt.show()
