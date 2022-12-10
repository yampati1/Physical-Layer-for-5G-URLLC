import gurobipy as grb
import numpy as np
import math
import random
import matplotlib.pyplot as plt 

URLLC_resources = np.arange(1, 100,10)

worst = []
for n in range(0, len(URLLC_resources)):
    worst.append((200 - (URLLC_resources[n])))
worst

T = (np.arange(1,41,1))
print(len(T))
#F = (np.arange(1, worst[0] +1 ,1))
#print(len(F))
#M = np.arange(1,11,1)
#print(len(M))

#E = [40]
E = (np.arange(1,10,1))
print((E))
#U = [45,40,35,30]
#U = (np.arange(1,6,1))
#print(len(U))

#P_max = 10** ((10 - 30)/10)
P_dBm = [10, 12, 14]
P_linear = []
for p in (P_dBm):
    P_linear.append(10** ((p - 30)/10))


def h_ee(f1):
    h_e = np.zeros([E.shape[0], T.shape[0], f1])
    for u in range(E.shape[0]):
    #print(u)
        for t in range(T.shape[0]):
            for f in range(f1):
                h_e[u,t,f] = (abs(random.uniform(0,1) + 1j * random.uniform(0,1))/np.sqrt(2))**2
    return h_e
sigma2 = 10**-11
d_u = 300
B = 720 * (10**3)
alpha = 3
T_max  = 0.25 *(10**-3)
u_ber = 10** -3
e_ber = 10**-1

def snr_calculation(p,f1):
    snr = np.zeros([E.shape[0], T.shape[0], f1])
    h_e = h_ee(f1)
    for u in range(E.shape[0]):
        for t in range(T.shape[0]):
            for f in range(f1):
                snr[u,t,f] = p*((h_e[u,t,f]))*((d_u)**(-alpha))/ ((f1) * sigma2)
    return snr
    
def r_e(p,f1):

    snr = snr_calculation(p,f1)
    se_e = snr.copy()
    gamma_fun_u = -(math.log(5*e_ber))/1.5
    #se_e_power = []
    for u in range(E.shape[0]):
        for t in range(T.shape[0]):
            for f in range(f1):
                
                se_e[u,t,f] = B *T_max * math.log(1 + (snr[u,t,f]/gamma_fun_u),2)
                
    return se_e
value_power_opt = []                  

for p in range(0, len(P_dBm)):
    print(P_linear[p])
    value1 = np.zeros([len(worst)])
    for i in range(0, len(worst)):
        print(worst[i])

        #data_u = r_u(P_linear[p])
        data_e = r_e(P_linear[p], worst[i])
        F = np.arange(1, worst[i]+1, 1)
        assignment_model = grb.Model('Assignment')
        x =  assignment_model.addVars(E.shape[0], T.shape[0], F.shape[0], vtype = grb.GRB.CONTINUOUS, lb = 0, ub = 1,name = 'x')
        assignment_model.addConstrs((sum(x[u, t, f]  for u in range(E.shape[0])) <= 1 for t in range(T.shape[0]) for f in range(F.shape[0])), name = 'one RB allocation')
        #assignment_model.addConstrs((sum(x[u,t,f] for t in range(T.shape[0]) for f in range(F.shape[0])) >= 1 for u in range(E.shape[0])), name = 'latency requirement') 
        #assignment_model.addConstrs((sum((x[u,t,f] * data_u[u,t,f] - x[u,t,f] * i) for t in range(T.shape[0]) for f in range(F.shape[0])) >= 0 for u in range(U.shape[0])), name = 'URLLC')
        #obj_fun1 = sum(data_u[u,t,f] * x[u,t,f] for u in range(U.shape[0]) for t in range(T.shape[0]) for f in range(50))
        #assignment_model.addConstrs((sum(x[u,t,f] * data_e[u,t,f] for t in range(T.shape[0]) for f in range(F.shape[0])) >= R_th for u in range(U.shape[0], M.shape[0])), name = 'eMBB')
        obj_fun2 = sum(data_e[u,t,f] * x[u,t,f]  for u in range(E.shape[0]) for t in range(T.shape[0]) for f in range(F.shape[0])) 
        #obj_fun = sum(se_e[u,t,f] * x[u,t,f] for u in range(U.shape[0]) for t in range(T.shape[0]) for f in range(F.shape[0])) 
        #obj_fun2 = sum(se_u_power[p][u,t,f] * x[u,t,f] for u in range(U.shape[0],M.shape[0]) for t in range(T.shape[0]) for f in range(F.shape[0]))
        obj_fun =  obj_fun2
        assignment_model.setObjective(obj_fun, grb.GRB.MAXIMIZE)
        assignment_model.setParam('OutputFlag', False)
        assignment_model.optimize()
        #print('Optimization is done. Objective function value: %.2f' % assignment_model.objVal)
        #print(i)
        value1[i] = assignment_model.objVal
        #obj_fun_e_u.append(value)
        #obj_fun_e_u
    value_power_opt.append(value1)


value_kbits_opt = []
for x in range(0,len(value_power_opt)):
    value_int = []
    for y in range(0, len(value_power_opt[x])):
        valu_kbits = float(value_power_opt[x][y])/1000
        value_int.append(valu_kbits)
    value_kbits_opt.append(value_int)
    
fig = plt.figure(dpi=150)
ax = fig.add_subplot(111)
plt.plot(URLLC_resources, value_kbits_opt[0], label = 'P (dBm) = 10')
plt.plot(URLLC_resources, value_kbits_opt[0], 'o')
plt.plot(URLLC_resources, value_kbits_opt[1], label = 'P (dBm) = 12')
plt.plot(URLLC_resources, value_kbits_opt[1], 'v')
plt.plot(URLLC_resources, value_kbits_opt[2], label = 'P (dBm) = 14')
plt.plot(URLLC_resources, value_kbits_opt[2], 'x')
plt.ylim(0,1100)
#plt.plot(k, R4, label = 'N = 70')
#plt.plot(k, R4, 'x')
#plt.plot(k, R5, label = 'N = 100')
#plt.plot(k, R5, 'x')
plt.grid()
plt.xlabel("Resources for URLLC")
plt.ylabel("eMBB Sum rate (kbps)")
ax.legend(fontsize = 8)
plt.show()
#fig.savefig('embb_urllc__ul.svg', format='svg', dpi=1200)


