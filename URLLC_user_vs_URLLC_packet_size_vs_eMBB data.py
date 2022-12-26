import gurobipy as grb
import math 
import numpy as np
import matplotlib.pyplot as plt 
import random as random
from array import array

T = (np.arange(1,21,1))
#print(len(T))
F = (np.arange(1,101,1))
#print(len(F))
#M = np.arange(1,31,1)
E = np.arange(1,11,1)
#U = np.arange(1,21,1)
sigma2 = 10**-11
d_u = 300
B = 180 * (10**3)
alpha = 3
T_max  = 0.5 *(10**-3)
u_ber = 10** -3
e_ber = 10**-1
#P_max = 10** ((10 - 30)/10)
#P_dBm = np.arange(5,25,0.1)
P_dBm = ([20])
P_linear = []
for p in (P_dBm):
    P_linear.append(10** ((p - 30)/10))
    
def h_ee(m):         
    h_e = np.zeros([m, T.shape[0], F.shape[0]])
    for u in range(m):
        #print(u)
        for t in range(T.shape[0]):
            for f in range(F.shape[0]):
                h_e[u,t,f] = (abs(random.uniform(0,1) + 1j * random.uniform(0,1))/np.sqrt(2))**2
                #h_e.append(h_c)
    return h_e
                   
#h_e = np.array(h_e)
#h_e = h_e.reshape(M.shape[0], T.shape[0], F.shape[0])                   
def snr_calculation(p,m):
    snr = np.zeros([m, T.shape[0], F.shape[0]])
    h_e = h_ee(m)
    for u in range(m):
        for t in range(T.shape[0]):
            for f in range(F.shape[0]):
                snr[u,t,f] = p*((h_e[u,t,f]))*((d_u)**(-alpha))/ (len(F) * sigma2)
                    #snr_lin_e.append(snr1)
                #snr2 = sum(snr_lin_e)/len(snr_lin_e)
                #snr.append(snr2)
    #snr = np.array(snr)
    #snr = snr.reshape(M.shape[0], T.shape[0], F.shape[0])
    return snr

def r_u(p,u):
    snr_u = snr_calculation(p,u)
    se_u = np.zeros([u, T.shape[0], F.shape[0]])
    gamma_fun_u = -(math.log(5*u_ber))/1.5
    #se_e_power = []
    #se_e = []
    for u in range(u):
        for t in range(T.shape[0]):
            for f in range(F.shape[0]):
                se_u[u,t,f] = B *T_max * math.log(1 + (snr_u[u,t,f]/gamma_fun_u),2)
                #se_e.append(se_e1)
    #se_e = np.array(se_e)
    #se_u = (se_e.reshape(U.shape[0], T.shape[0], F.shape[0]))
    return se_u
def r_e(p,m):
    snr_e = snr_calculation(p,m)
    se_e = np.zeros([m, T.shape[0], F.shape[0]])
    gamma_fun_e = -(math.log(5*e_ber))/1.5
    #se_e_power = []
    #se_e = []
    for u in range(m):
        for t in range(T.shape[0]):
            for f in range(F.shape[0]):
                se_e[u,t,f] = B *T_max * math.log(1 + (snr_e[u,t,f]/gamma_fun_e),2)
                #se_e.append(se_e1)
    #se_e = np.array(se_e)
    #se_e = (se_e.reshape(M.shape[0], T.shape[0], F.shape[0]))
    return se_e
R_th1 = [32, 64,128,256]
R_th = []
for r in (R_th1):
    R_th.append(8* r)

value_rth_eu = []
U1 = [10,20,30,40]
M1 = [20,30,40,50]
for th in range(0, len(R_th)):
    
    value_opt_40eu = []
    for i in range(0, len(U1)):
        print(U1[i])
        for p in range(0, len(P_linear)):
            print(P_linear[p])

            data_u = r_u(P_linear[p], U1[i])
            data_e = r_e(P_linear[p],M1[i])
            M = np.arange(1, M1[i]+1, 1)
            U = np.arange(1, U1[i]+1, 1)
            assignment_model = grb.Model('Assignment')
            x =  assignment_model.addVars(M.shape[0], T.shape[0], F.shape[0], vtype = grb.GRB.CONTINUOUS, lb = 0, ub = 1,name = 'x')
            assignment_model.addConstrs((sum(x[u, t, f]  for u in range(M.shape[0])) <= 1 for t in range(T.shape[0]) for f in range(F.shape[0])), name = 'one RB allocation')
            assignment_model.addConstrs((sum(x[u,t,f] for t in range(T.shape[0]) for f in range(F.shape[0])) >= 1/5 for u in range(U.shape[0])), name = 'latency requirement') 
            #assignment_model.addConstrs((sum((x[u,t,f] * data_u[u,t,f] - x[u,t,f] * i) for t in range(T.shape[0]) for f in range(F.shape[0])) >= 0 for u in range(U.shape[0])), name = 'URLLC')
            obj_fun1 = sum(data_u[u,t,f] * x[u,t,f] for u in range(U.shape[0]) for t in range(T.shape[0]) for f in range(F.shape[0]))
            assignment_model.addConstrs((sum(x[u,t,f] * data_u[u,t,f] for f in range(F.shape[0]) for t in range(T.shape[0])) >= R_th[th] for u in range(U.shape[0])), name = 'URLLC')
            obj_fun2 = sum(data_e[u,t,f] * x[u,t,f]  for u in range(U.shape[0], M.shape[0]) for t in range(T.shape[0]) for f in range(F.shape[0])) 
            #obj_fun = sum(se_e[u,t,f] * x[u,t,f] for u in range(U.shape[0]) for t in range(T.shape[0]) for f in range(F.shape[0])) 
            #obj_fun2 = sum(se_u_power[p][u,t,f] * x[u,t,f] for u in range(U.shape[0],M.shape[0]) for t in range(T.shape[0]) for f in range(F.shape[0]))
            obj_fun =  obj_fun1 + obj_fun2
            assignment_model.setObjective(obj_fun, grb.GRB.MAXIMIZE)
            assignment_model.setParam('OutputFlag', False)
            assignment_model.optimize()
            #print('Optimization is done. Objective function value: %.2f' % assignment_model.objVal)
        value_opt_40eu.append(assignment_model.objVal)
            #obj_fun_e_u.append(value)
            #obj_fun_e_u
    value_rth_eu.append(value_opt_40eu)

T = (np.arange(1,21,1))
#print(len(T))
F = (np.arange(1,101,1))
#print(len(F))
#M = np.arange(1,21,1)
#E = np.arange(1,11,1)
#U = np.arange(1,11,1)
sigma2 = 10**-11
d_u = 300
B = 180 * (10**3)
alpha = 3
T_max  = 0.5 *(10**-3)
u_ber = 10** -3
e_ber = 10**-1
#P_max = 10** ((10 - 30)/10)
#P_dBm = np.arange(10,40,5)
P_dBm = ([20])
P_linear = []
for p in (P_dBm):
    P_linear.append(10** ((p - 30)/10))

def h_ee(m):
    
    h_e = np.zeros([m, T.shape[0], F.shape[0]])
    for u in range(m):
        #print(u)
        for t in range(T.shape[0]):
            for f in range(F.shape[0]):
                h_e[u,t,f] = (abs(random.uniform(0,1) + 1j * random.uniform(0,1))/np.sqrt(2))**2
    return h_e
                   
#h_e = np.array(h_e)
#h_e = h_e.reshape(M.shape[0], T.shape[0], F.shape[0])                   
def snr_calculation(p, m):
    snr = np.zeros([m, T.shape[0], F.shape[0]])
    h_e = h_ee(m)
    for u in range(m):
        for t in range(T.shape[0]):
            for f in range(F.shape[0]):
                snr[u,t,f] = p*((h_e[u,t,f]))*((d_u)**(-alpha))/ (len(F) * sigma2)
                    #snr_lin_e.append(snr1)
                #snr2 = sum(snr_lin_e)/len(snr_lin_e)
                #snr.append(snr2)
    #snr = np.array(snr)
    #snr = snr.reshape(M.shape[0], T.shape[0], F.shape[0])
    return snr

def r_u(p, m):
    snr_u = snr_calculation(p,m)
    se_u = np.zeros([m, T.shape[0], F.shape[0]])
    gamma_fun_u = -(math.log(5*u_ber))/1.5
    #se_e_power = []
    #se_e = []
    for u in range(m):
        for t in range(T.shape[0]):
            for f in range(F.shape[0]):
                se_u[u,t,f] = B *T_max * math.log(1 + (snr_u[u,t,f]/gamma_fun_u),2)
                #se_e.append(se_e1)
    #se_e = np.array(se_e)
    #se_u = (se_e.reshape(U.shape[0], T.shape[0], F.shape[0]))
    return se_u
R_th1 = [32, 64,128,256]
R_th = []
for r in (R_th1):
    R_th.append(8* r)

value_rth_u = []
M1 = [10, 20, 30, 40]
for th in range(0,len(R_th)):
    value_opt1_10u = []
    for i in range(0, len(M1)):
        print(M1[i])
        
        
        for p in range(0, len(P_linear)):
            print(P_linear[p])

            data_u = r_u(P_linear[p], M1[i])
            M = np.arange(1, M1[i]+1, 1)
            #data_e = r_e(P_linear[p])
            assignment_model = grb.Model('Assignment')
            x =  assignment_model.addVars(M.shape[0], T.shape[0], F.shape[0], vtype = grb.GRB.CONTINUOUS, lb = 0, ub = 1,name = 'x')
            assignment_model.addConstrs((sum(x[u, t, f]  for u in range(M.shape[0])) <= 1 for t in range(T.shape[0]) for f in range(F.shape[0])), name = 'one RB allocation')
            assignment_model.addConstrs((sum(x[u,t,f] for t in range(T.shape[0]) for f in range(F.shape[0])) >= 1 for u in range(M.shape[0])), name = 'latency requirement') 
            #assignment_model.addConstrs((sum((x[u,t,f] * data_u[u,t,f] - x[u,t,f] * i) for t in range(T.shape[0]) for f in range(F.shape[0])) >= 0 for u in range(U.shape[0])), name = 'URLLC')
            obj_fun1 = sum(data_u[u,t,f] * x[u,t,f] for u in range(M.shape[0]) for t in range(T.shape[0]) for f in range(F.shape[0]))
            assignment_model.addConstrs((sum(x[u,t,f] * data_u[u,t,f] for f in range(F.shape[0]) for t in range(T.shape[0])) >= R_th[th] for u in range(M.shape[0])), name = 'URLLC')
            #obj_fun2 = sum(data_e[u,t,f] * x[u,t,f]  for u in range(U.shape[0], M.shape[0]) for t in range(T.shape[0]) for f in range(F.shape[0])) 
            #obj_fun = sum(se_e[u,t,f] * x[u,t,f] for u in range(U.shape[0]) for t in range(T.shape[0]) for f in range(F.shape[0])) 
            #obj_fun2 = sum(se_u_power[p][u,t,f] * x[u,t,f] for u in range(U.shape[0],M.shape[0]) for t in range(T.shape[0]) for f in range(F.shape[0]))
            obj_fun =  obj_fun1
            assignment_model.setObjective(obj_fun, grb.GRB.MAXIMIZE)
            assignment_model.setParam('OutputFlag', False)
            assignment_model.optimize()
            #print('Optimization is done. Objective function value: %.2f' % assignment_model.objVal)
        value_opt1_10u.append(assignment_model.objVal)
    value_rth_u.append(value_opt1_10u)

value_kbits = []
for x in range(0, len(value_rth_u)):
    value_int = []
    for y in range(0, len(value_rth_u[x])):
        value = (value_rth_eu[x][y] - value_rth_u[x][y])/1000
        value_int.append(value)
    value_kbits.append(value_int)
    
fig = plt.figure(dpi=150)
ax = fig.add_subplot(111)
plt.plot(U1, value_kbits[0], label = 'URLLC Size = 32 Bytes')
plt.plot(U1, value_kbits[0], 'o')
plt.plot(U1, value_kbits[1], label = 'URLLC Size = 64 Bytes')
plt.plot(U1, value_kbits[1], 'v')
plt.plot(U1, value_kbits[2], label = 'URLLC Size = 128 Bytes')
plt.plot(U1, value_kbits[2], 'x')
plt.plot(U1, value_kbits[3], label = 'URLLC Size = 256 Bytes')
plt.plot(U1, value_kbits[3], '^')
#plt.ylim(0,350)
#plt.plot(k, R4, label = 'N = 70')
#plt.plot(k, R4, 'x')
#plt.plot(k, R5, label = 'N = 100')
#plt.plot(k, R5, 'x')
plt.grid()
plt.xlabel("URLLC Users (U)")
plt.ylabel("eMBB Sum rate (Kbits)")
ax.legend(fontsize = 8)
plt.show()
#fig.savefig('embb_urllc__dl.svg', format='svg', dpi=1200)


