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
M = np.arange(1,21,1)
E = np.arange(1,11,1)
U = np.arange(1,11,1)
sigma2 = 10**-11
d_u = 300
B = 180 * (10**3)
alpha = 3
T_max  = 0.5 *(10**-3)
u_ber = 10** -3
e_ber = 10**-1
P_dBm = np.arange(5,25,1)

P_linear = []
for p in (P_dBm):
    P_linear.append(10** ((p - 30)/10))

h_e = np.zeros([M.shape[0], T.shape[0], F.shape[0]])
for u in range(M.shape[0]):
    #print(u)
    for t in range(T.shape[0]):
        for f in range(F.shape[0]):
            h_e[u,t,f] = (abs(random.uniform(0,1) + 1j * random.uniform(0,1))/np.sqrt(2))**2
            #h_e.append(h_c)
def snr_calculation(p):
    snr = np.zeros([M.shape[0], T.shape[0], F.shape[0]])
    for u in range(M.shape[0]):
        for t in range(T.shape[0]):
            for f in range(F.shape[0]):
                snr[u,t,f] = p*((h_e[u,t,f]))*((d_u)**(-alpha))/ (len(F) * sigma2)
                    #snr_lin_e.append(snr1)
                #snr2 = sum(snr_lin_e)/len(snr_lin_e)
                #snr.append(snr2)
    #snr = np.array(snr)
    #snr = snr.reshape(M.shape[0], T.shape[0], F.shape[0])
    return snr

def r_u(p):
    snr_u = snr_calculation(p)
    se_u = np.zeros([U.shape[0], T.shape[0], F.shape[0]])
    gamma_fun_u = -(math.log(5*u_ber))/1.5
    #se_e_power = []
    #se_e = []
    for u in range(U.shape[0]):
        for t in range(T.shape[0]):
            for f in range(F.shape[0]):
                se_u[u,t,f] = B *T_max * math.log(1 + (snr_u[u,t,f]/gamma_fun_u),2)
                #se_e.append(se_e1)
    #se_e = np.array(se_e)
    #se_u = (se_e.reshape(U.shape[0], T.shape[0], F.shape[0]))
    return se_u

def r_e(p):

    snr = snr_calculation(p)
    se_e = snr.copy()
    gamma_fun_e = -(math.log(5*e_ber))/1.5
    #se_e_power = []
    for u in range(M.shape[0]):
        for t in range(T.shape[0]):
            for f in range(F.shape[0]):
                
                se_e[u,t,f] = B *T_max * math.log(1 + (snr[u,t,f]/gamma_fun_e),2)
                
    return se_e
    
beta = 0.2

def R_u(r_u1):
    R_u1 = r_u1.copy()
#     for u in range(U.shape[0]):
#         for f in range(F.shape[0]):
#             R_u1[u,0,f] = 1
    #avg = 0
    for u in range(U.shape[0]):
        for t in range(1, T.shape[0]):
            avg = sum(r_u1[u,t-1,:])/ F.shape[0]
            for f in range(F.shape[0]):
                R_u1[u,t,f] = ((1- beta) * R_u1[u,t-1,f]) + (beta * r_u1[u,t-1,f])/avg
    return R_u1
def R_e(r_e1):
    R_e1 = r_e1.copy()
#     for u in range(M.shape[0]):
#         for f in range(F.shape[0]):
#             R_e1[u,0,f] = 1
    #avg = 0
    for u in range(M.shape[0]):
        for t in range(1, T.shape[0]):
            avg = sum(r_u1[u,t-1,:])/ F.shape[0]
            for f in range(F.shape[0]):
                R_e1[u,t,f] = ((1- beta) * R_e1[u,t-1,f]) + (beta * r_e1[u,t-1,f])/avg
    return R_e1

value1 = np.zeros([len(P_linear)])
for p in range(0, len(P_linear)):
    print(P_linear[p])
    
    data_u = r_u(P_linear[p])
    data_e = r_e(P_linear[p])
    r_data_u = R_u(data_u)
    r_data_e = R_u(data_e)
    assignment_model = grb.Model('Assignment')
    x =  assignment_model.addVars(M.shape[0], T.shape[0], F.shape[0], vtype = grb.GRB.CONTINUOUS, lb = 0, ub = 1,name = 'x')
    assignment_model.addConstrs((sum(x[u, t, f]  for u in range(M.shape[0])) <= 1 for t in range(T.shape[0]) for f in range(F.shape[0])), name = 'one RB allocation')
    #assignment_model.addConstrs((sum(x[u,t,f] for t in range(T.shape[0]) for f in range(F.shape[0])) >= 1 for u in range(U.shape[0])), name = 'latency requirement') 
    #assignment_model.addConstrs((sum((x[u,t,f] * data_u[u,t,f] - x[u,t,f] * i) for t in range(T.shape[0]) for f in range(F.shape[0])) >= 0 for u in range(U.shape[0])), name = 'URLLC')
    obj_fun1 = sum((data_u[u,t,f]/r_data_u[u,t,f]) * x[u,t,f] for u in range(U.shape[0]) for t in range(T.shape[0]) for f in range(F.shape[0]))
    #assignment_model.addConstrs((sum(x[u,t,f] * data_e[u,t,f] for t in range(T.shape[0]) for f in range(F.shape[0])) >= R_th for u in range(U.shape[0], M.shape[0])), name = 'eMBB')
    obj_fun2 = sum((data_e[u,t,f]/r_data_e[u,t,f]) * x[u,t,f]  for u in range(U.shape[0], M.shape[0]) for t in range(T.shape[0]) for f in range(F.shape[0])) 
    #obj_fun = sum(se_e[u,t,f] * x[u,t,f] for u in range(U.shape[0]) for t in range(T.shape[0]) for f in range(F.shape[0])) 
    #obj_fun2 = sum(se_u_power[p][u,t,f] * x[u,t,f] for u in range(U.shape[0],M.shape[0]) for t in range(T.shape[0]) for f in range(F.shape[0]))
    obj_fun =  obj_fun1 + obj_fun2
    assignment_model.setObjective(obj_fun, grb.GRB.MAXIMIZE)
    assignment_model.setParam('OutputFlag', False)
    assignment_model.optimize()
    #print('Optimization is done. Objective function value: %.2f' % assignment_model.objVal)
    value1[p] = assignment_model.objVal
    #obj_fun_e_u.append(value)
    #obj_fun_e_u

