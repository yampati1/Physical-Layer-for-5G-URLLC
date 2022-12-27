import numpy as np
import math
import matplotlib.pyplot as plt 
import random
from math import comb

N = np.arange(2,40,5)
P_ci = 10 **-3
K = [2,4,8]
lamda = 1.25 * (10 ** -4)

resk = []
for k in K:
    res = []
    for n in N:
        a = (np.exp(-k*lamda) - np.exp(-lamda))
        b = ((1 - P_ci) ** (1/(n-1))) - 1
        M1 = round(a/b)
        res.append(M1)
    resk.append(res)
    
fig = plt.figure(dpi=200)
ax = fig.add_subplot(111)
plt.plot(N, resk[0], label = 'repK = 2')
plt.plot(N, resk[0], 'o')
plt.plot(N, resk[1], label = 'repK = 4')
plt.plot(N, resk[1], 'x')
plt.plot(N, resk[2], label = 'repK = 8')
plt.plot(N, resk[2], 'v')
plt.grid()
plt.legend()
#plt.ylim(10**-4, 10**-2)
#plt.title("Colision probability of the first reserved resource")
plt.xlabel("URLLC Users (U)")
plt.ylabel("Resource Blocks (M)")
#fig.savefig('M_U_K.svg', format='svg', dpi=1200)