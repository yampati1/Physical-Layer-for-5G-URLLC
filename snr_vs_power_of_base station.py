#Import libraries
import math 
import numpy as np
import matplotlib.pyplot as plt 
import random as random

#Initilize the power of base station

P_dBm = ([1,2,5,10,15,18,20])
P_linear = []
for p in (P_dBm):
    P_linear.append(10** ((p - 30)/10))

# Import the parameters to calculate the SNR

sigma2 = 10**-11
d_u = 300
alpha = 3
F = 100

# To calculate SNR with respect to power
# Run 1000 monte-carlo simulations

snr= []
snr_total = []
for k in range (0,1000):
    h = random.uniform(0,1) + 1j * random.uniform(0,1)
    snr1 = p*(abs(h) **2)*((d_u)**(-alpha))/ (F * sigma2)
    snr.append(snr1)
    snr_avg = sum(snr)/k
    snr_total.append(snr_avg)    

# To calculate SNR in dB

numpy_array= np.array(snr_total)
mul_array = 10 * np.log10(numpy_array)
snr_db = mul_array.tolist()

#Plot the SNR with respect to power of base station
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
plt.plot(P_dBm, snr_db)
plt.grid()
plt.xlabel("Base station Power (dBm)")
plt.ylabel("SNR (dB)")
fig.savefig('power_snr.svg', format='svg', dpi=1200)