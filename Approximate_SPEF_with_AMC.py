#Import libraries

import math 
import numpy as np
import matplotlib.pyplot as plt 

#Intialize the SNR values from AMC table

snr_db_e= np.array([-6.5, -4.0, -2.6, -1.0, 1.0, 3.0, 6.6, 10.0, 11.4, 11.8, 13.0, 13.8, 15.6, 16.8, 17.6])
snr_db_u =np.array([-2.5,  0.0, 1.4, 3.0, 5.0, 7.0, 10.6, 14, 15.4, 15.8, 17, 17.8, 19.6, 20.8, 21.6]) 
#snr_db = np.arange(-10,25,1)
mul_array1 = 10**(snr_db_e/10)
mul_array2 = 10**(snr_db_u/10)
snr_e = mul_array1.tolist()
snr_u = mul_array2.tolist()

# Intialize the parameters

B = 180 * (10**3)
T_max  = 0.5 *(10**-3)
u_ber = 10** -5
e_ber = 10**-1

#Calculate the data rate of eMBB and URLLC service

r_tf_E = []
r_tf_U = []
for i in range(0,len(snr_e)):
    
    gamma_fun_e = -(math.log(5*e_ber))/0.15
    se_e = math.log(1 + (snr_e[i]/gamma_fun_e),2)
    #se_e.append(se_e)
    gamma_fun_u = -(math.log(5*u_ber))/1.25
    se_u = math.log(1 + (snr_u[i]/gamma_fun_u),2)
    #se_u.append(se_u)
    r_tf_e = B * T_max * se_e
    r_tf_u = B * T_max * se_u
    r_tf_E.append(r_tf_e)
    r_tf_U.append(r_tf_u)

value_kbits_e = []
for x in range(0,len(r_tf_E)):
    valu_kbits = float(r_tf_E[x])/1000
    value_kbits_e.append(valu_kbits)
    
value_kbits_u = []
for x in range(0,len(r_tf_U)):
    valu_kbits = float(r_tf_U[x])/1000
    value_kbits_u.append(valu_kbits)


#Intialize the SNR values to plot the data rate with resepct to SNR from AMC table
 
snr1= [-6.5, -4.0, -2.6, -1.0, 1.0, 3.0, 6.6, 10.0, 11.4, 11.8, 13.0, 13.8, 15.6, 16.8, 17.6]
snr2= [-2.5, 0.0, 1.4, 3.0, 5.0, 7.0, 10.6, 14, 15.4, 15.8, 17, 17.8, 19.6, 20.8, 21.6]

rate = np.array([(0.15*2),(0.23*2),(0.38*2),(0.60*2),(0.88*2),(1.18*2),(1.48*4),(1.91*4),(4*2.41),(6*2.73),(6*3.32),(6*3.90),(6*4.52),(6*5.12),(6*5.55)])
rate = 1.1 * rate/100


fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
plt.plot(snr_db_e,value_kbits_e,label = 'Approx eMBB')
#plt.plot(snr_db_e,value_kbits_e, 'b*')
plt.plot(snr_db_u,value_kbits_u,label = 'Approx URLLC')
#plt.plot(snr_db,value_kbits_u, 'x')
plt.step(snr1,rate,label = 'AMC eMBB')
#plt.plot(snr1,rate, 'b*')
plt.step(snr2,rate,label = 'AMC URLLC')
#plt.plot(snr2,rate, 'x')
plt.legend()
#plt.title('')
plt.xlabel('SNR (in dB)')
plt.ylabel('Rate [Kbps]')
plt.grid()
plt.show()
fig.savefig('embb.svg', format='svg', dpi=300)
