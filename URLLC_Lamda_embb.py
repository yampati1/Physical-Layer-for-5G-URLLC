import gurobipy as grb
import numpy as np
import math
import random
import matplotlib.pyplot as plt 

N = np.arange(2,20,5)
P_ci = 10 **-3
K = 4
lamda = [1.25 * (10 ** -5), 1.25 * (10 ** -4.5), 1.25 * (10 ** -3.5)]

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
P_dBm = [20]
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


def snr_calculation(f1):
    snr = np.zeros([E.shape[0], T.shape[0], f1])
    h_e = h_ee(f1)
    for u in range(E.shape[0]):
        for t in range(T.shape[0]):
            for f in range(f1):
                snr[u,t,f] = P_linear[0]*((h_e[u,t,f]))*((d_u)**(-alpha))/ ((f1) * sigma2)
    return snr
    
def r_e(f1):

    snr = snr_calculation(f1)
    se_e = snr.copy()
    gamma_fun_u = -(math.log(5*e_ber))/1.5
    #se_e_power = []
    for u in range(E.shape[0]):
        for t in range(T.shape[0]):
            for f in range(f1):
                
                se_e[u,t,f] = B *T_max * math.log(1 + (snr[u,t,f]/gamma_fun_u),2)
                
    return se_e
    
def urllc_res(l, n):
    a = (np.exp(-K*l) - np.exp(-l))
    b = ((1 - P_ci) ** (1/(n-1))) - 1
    M1 = round(a/b)
    a = (np.exp(-K*l) - np.exp(-2*l))
    b = ((1 - P_ci) ** (1/(n-1))) - 1
    M2 = round(a/b)
    a = (np.exp(-K*l) - np.exp(-3*l))
    b = ((1 - P_ci) ** (1/(n-1))) - 1
    M3 = round(a/b)
    M = M1+M2+M3
    return M
    
value_power_opt = []                  

for l in range(0, len(lamda)):
    print(lamda[l])
    value= []
    for n in range(0,len(N)):
        print(N[n])
        URLLC_res = urllc_res(lamda[l], N[n])
        eMBB_res = 200 - URLLC_res
    

        #data_u = r_u(P_linear[p])
        data_e = r_e(eMBB_res)
        F = np.arange(1, eMBB_res+1, 1)
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
        value.append(assignment_model.objVal)
        #obj_fun_e_u.append(value)
        #obj_fun_e_u
    value_power_opt.append(value)


value_kbits_opt = []
for x in range(0,len(value_power_opt)):
    value_int = []
    for y in range(0, len(value_power_opt[x])):
        valu_kbits = float(value_power_opt[x][y])/1000
        value_int.append(valu_kbits)
    value_kbits_opt.append(value_int)

fig = plt.figure(dpi=150)
ax = fig.add_subplot(111)
plt.plot(N, value_kbits_opt[0], label = r'$\lambda$ = 1.25 * 10^-5')
plt.plot(N, value_kbits_opt[0], 'o')
plt.plot(N, value_kbits_opt[1], label = r'$\lambda$ = 1.25 * 10^-4')
plt.plot(N, value_kbits_opt[1], 'v')
plt.plot(N, value_kbits_opt[2], label = r'$\lambda$ = 1.25 * 10^-3')
plt.plot(N, value_kbits_opt[2], 'x')
#plt.ylim(2000,2680)
#plt.plot(k, R4, label = 'N = 70')
#plt.plot(k, R4, 'x')
#plt.plot(k, R5, label = 'N = 100')
#plt.plot(k, R5, 'x')
plt.grid()
plt.xlabel("URLLC Users (U)")
plt.ylabel("eMBB data sum rate (Kbits)")
ax.legend(fontsize = 8)
plt.show()
#fig.savefig('embb_ul.svg', format='svg', dpi=1200)


