import gurobipy as grb
import math 
import numpy as np
import matplotlib.pyplot as plt 
import random as random
from array import array

T1 = (np.arange(1,21,1))
#print(len(T))
F = (np.arange(1,101,1))
#print(len(F))
U = np.arange(1,46,1)
#U = [10, 20, 30, 40]

sigma2 = 10**-11
d_u = 300
B = 180 * (10**3)
alpha = 3
T_max  = 0.5 *(10**-3)
u_ber = 10** -3
e_ber = 10**-1


#P_max = 10** ((10 - 30)/10)
P_dBm = ([5, 6,7,9,10,12,14,16,18,20,23,25,27,29,31,32,34,37,40,45,47,50])
P_linear = []
for p in (P_dBm):
    P_linear.append(10** ((p - 30)/10))


h_e = []
for u in range(U.shape[0]):
    #print(u)
    for t in range(T1.shape[0]):
        for f in range(F.shape[0]):
            h_c = (random.uniform(0,1) + 1j * random.uniform(0,1))/np.sqrt(2)
            h_e.append(h_c)

h_e = np.array(h_e)
h_e = h_e.reshape(U.shape[0], T1.shape[0], F.shape[0])                   
          
def snr_calculation(p):
    snr_lin_e = []
    snr = []
    for u in range(U.shape[0]):
        for t in range(T1.shape[0]):
            for f in range(F.shape[0]):
                for k in range (0,1):
                    snr1 = p*(abs(h_e[u,t,f]) **2)*((d_u)**(-alpha))/ (len(F) * sigma2)
                    snr_lin_e.append(snr1)
                snr2 = sum(snr_lin_e)/len(snr_lin_e)
                snr.append(snr2)
    snr = np.array(snr)
    snr = snr.reshape(U.shape[0], T1.shape[0], F.shape[0])
    return snr
%%time
def r_u(p):

    snr = snr_calculation(p)
    gamma_fun_u = -(math.log(5*u_ber))/1.5
    #se_e_power = []
    se_e = []
    for u in range(U.shape[0]):
        for t in range(T1.shape[0]):
            for f in range(F.shape[0]):
                
                se_e1 = B *T_max * math.log(1 + (snr[u,t,f]/gamma_fun_u),2)
                se_e.append(se_e1)
    se_e = np.array(se_e)
    se_e = (se_e.reshape(U.shape[0], T1.shape[0], F.shape[0]))
    return se_e

R_min = [32, 64, 128, 256]

obj_fun_u = []
for p in P_linear:
    print(p)
    data_u = r_u(p)
    assignment_model = grb.Model('Assignment')
    x =  assignment_model.addVars(U.shape[0], T1.shape[0], F.shape[0], vtype = grb.GRB.CONTINUOUS, lb = 0, ub = 1,name = 'x')

    assignment_model.addConstrs((sum(x[u, t, f]  for u in range(U.shape[0])) <= 1 for t in range(T1.shape[0]) for f in range(F.shape[0])), name = 'one RB allocation')
    assignment_model.addConstrs((sum(x[u,t,f] for t in range(T1.shape[0]) for f in range(F.shape[0])) >= 1 for u in range(U.shape[0])), name = 'latency requirement') 
    #assignment_model.addConstrs((sum((x[u,t,f] * data_u[u,t,f] - x[u,t,f] * R_min[1])) >= 0 for u in range(U.shape[0]) for t in range(T.shape[0]) for f in range(F.shape[0])), name = 'URLLC')
    obj_fun = sum(data_u[u,t,f] * x[u,t,f] for u in range(U.shape[0]) for t in range(T1.shape[0]) for f in range(F.shape[0])) 
    assignment_model.setObjective(obj_fun, grb.GRB.MAXIMIZE)
    assignment_model.setParam('OutputFlag', False)
    assignment_model.optimize()
    #print('Optimization is done. Objective function value: %.2f' % assignment_model.objVal)
    value = assignment_model.objVal
    obj_fun_u.append(value)

value_kbits_u = []
for x in range(0,len(obj_fun_u)):
    valu_kbits = float(obj_fun_u[x])/1000
    value_kbits_u.append(valu_kbits)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
plt.plot(P_dBm,valu_kbits_u,label = 'URLLC Users')
plt.plot(P_dBm,value_kbits_u, 'x')
plt.legend()
#plt.title('')
plt.xlabel('Power (in dB)')
plt.ylabel('Rate of URLLC Users[Kbps]')
plt.grid()
plt.show()
#fig.savefig('embb.svg', format='svg', dpi=300)
