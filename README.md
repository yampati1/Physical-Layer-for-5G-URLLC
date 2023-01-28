# Library Requirements

- To install required libraries, use the command in python pip install requirements.txt



# Design-of-Physical-Layer-for-5G-URLLC
- URLLC service represents that it requires high reliability and low latency communications. URLLC offers use cases that call for extremely low latency of just 1 ms for data transmission and high network reliability of more than 99.999\%.  
- The physical layer (PHY) in present 4G wireless network is not viable for this kind of communications because it is not designed specifically for achieving high reliability and latency requirements. Hence, in Release 15, the first complete set of 5G physical layer design for URLLC, was introduced by 3GPP. Release 16, the second 5G release, was completed in December 2019 and enables for increased latency and reliability measurements to accommodate new URLLC use cases. To address new industrial use cases, 3GPP Release 17 expands URLLC functionalities to unlicensed spectrum. 
- Based on physical layer enhancements proposed in standards and research, this focuses on resource management and allocation in both uplink and downlink for a network configured with URLLC users and the presence of eMBB users.

# Data Sum-rate Comparisons in downlink


![embb_urllc__dl_ecdf](https://user-images.githubusercontent.com/51235418/211160434-42df2af6-6f31-403b-bab0-42c3f328d800.svg)

- It is observed that when the network is configured with both eMBB and
URLLC users, the RR scheduler is failed to achieve the certain throughput requirement for
eMBB users. It can be observed from the simulation that the RR scheduler provides lower sum
throughput for eMBB users. This is because it assigns the resources without depending on channel
feedback but assigning resources to all users equally irrespective of QoS. The RR scheduler assigns
resources to both eMBB users and URLLC users but failed to achieve the minimum requirements
of eMBB users if the sum rate is higher. The EDS is also failed to provide the data sum rate less
than 30 Kilo bits (Kbits) (shown as dotted straight line) to 50% of eMBB users in the network.
The total number of available resources are divided equally for both eMBB and URLLC users and
allocated. As half of the resources are allocated to URLLC, the remaining are allocated to eMBB.
These resources are not sufficient to provide the minimum throughput to some of the eMBB users.
The PF scheduler is simulated with various TPF intervals. The simulation shows with increasing in
TPF increases the throughput of PF scheduler. It provides lower sum rate than EDS scheduler for
certain values of time interval TPF . And, it also fails for some of the users in the network if the
minimum sum rate is 30 Kbits. However, by respecting the isolation between service slices, the
optimization-based maximum rate scheduling algorithm, provides users with resources and satis-
fies the minimum rate constraint for eMBB users. In parallel, it also provides the URLLC users
with strict reliable and latency requirements. The results show that, the base-line scheduling algo-
rithms does not meet the isolation and minimum rate requirements when compared with discussed
optimization technique.
# Impact on eMBB data rate with URLLC Service in downlink
![embb_urllc_final](https://user-images.githubusercontent.com/51235418/211160441-dfd0e812-7bb0-4cb4-ab98-8fe9b243389b.svg)
- The results demonstrate that, as anticipated, the sum-rate of both eMBB and URLLC users increases with SNR, as indicated above. This is so that the UEs can select higher order MCS that increase the user's sum-rate as the SNR improves. The sum-rate of users serviced is seen to be higher when all of the cell's active users are connected to the eMBB services than when the cell is using solely URLLC. Users pick higher order MCS because there is less of a reliability constraint for eMBB, which leads to large data rates. The sum-rate is lower than eMBB service when all of the active users in the cell are URLLC, as demonstrated above. The customers prefer lower MCS compared to eMBB since the URLLC reliability limitation must be satisfied. The overall sum-rate of the users is higher than the total sum-rate of all URLLC users and lower than the total sum-rate of all eMBB users when the active users in the cell are linked to both types of services (both eMBB and URLLC). This is due to the fact that URLLC users are scheduled with lower MCS in relation to SNR while eMBB users are scheduled with higher MCS. As a result, the data rate is lower when compared to eMBB traffic but higher when compared to URLLC traffic.With varying reliability targets and a 1 ms URLLC latency, it is therefore noticed that the optimization method assigns the resources to both eMBB and URLLC users. The impact of increasing the number of resources allocated to URLLC users on eMBB data rate is being further investigated.

# Resource Consumption Comparisons in uplink

![reserved2](https://user-images.githubusercontent.com/51235418/211160637-c0f13f6a-65ec-4504-bea0-d0eeb21467a3.svg)
- It is depicted above how these schemes compare to the network's fluctuating number of URLLC users. Compared to various configurations suggested by the 3GPP, the implementation of an optimal scheme reduces the quantity of resources required for the same number of URLLC users in the network. According to the traditional 3GPP Release 15 scheme, the UE must wait until the following period if it cannot complete the specified number of repetitions by the gNB in order to meet the URLLC dependability requirement. So, in the worst-case situation with repK = 4, 4 transmission occasions happen each period, and the UE has to wait for 3 of them, which corresponds to 3 slots or 0.75 ms with SCS 60kHz. The four repeats that were planned for the upcoming period cannot be completed by the URLLC UEs because of how close this latency is to the 1 ms requirement of the URLLC. Because of this, the reliability cannot be guaranteed, which increases latency. While accomplishing the configured number of repeats via gNB in the target latency of 1 ms and meeting the reliability condition, the recommended ideal technique, on the other hand, enables the UE to start transmission immediately. Because less resources are set aside for URLLC users, the best resource allocation system for URLLC is taken into account for the research of eMBB and URLLC multiplexing.
# Collision probability analysis

![col_rbs](https://user-images.githubusercontent.com/51235418/211160548-0639d0db-207a-4189-bcaf-a09623e0348e.svg)
- The figure above depicts the collision probability for K = 4 when it comes to the magnitude of the reserved resources during all transmission occasions. When URLLC users are fixed in the network, it is possible to compute the number of reserved resources needed to achieve the desired collision probability. Based on the placement of reserved resource locations in the URLLC's optimal resource allocation, it is seen that the number of resource blocks lowers. It is clear that the collision probability lowers as the number of URLLC resources rises to service the same number of network users.

# License and Citation
- If you use this codes, please cite it as:

\cite{@software{Yampati_Physical_Layer_Design,
author = {Yampati, Venkat},
title = {{Physical Layer Design for 5G URLLC}}
}}


