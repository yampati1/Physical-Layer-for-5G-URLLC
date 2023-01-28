# Design-of-Physical-Layer-for-5G-URLLC
- URLLC service represents that it requires high reliability and low latency communications. URLLC offers use cases that call for extremely low latency of just 1 ms for data transmission and high network reliability of more than 99.999\%.  
- The physical layer (PHY) in present 4G wireless network is not viable for this kind of communications because it is not designed specifically for achieving high reliability and latency requirements. Hence, in Release 15, the first complete set of 5G physical layer design for URLLC, was introduced by 3GPP. Release 16, the second 5G release, was completed in December 2019 and enables for increased latency and reliability measurements to accommodate new URLLC use cases. To address new industrial use cases, 3GPP Release 17 expands URLLC functionalities to unlicensed spectrum. 
- Based on physical layer enhancements proposed in standards and research, this focuses on resource management and allocation in both uplink and downlink for a network configured with URLLC users and the presence of eMBB users.

# Data Sum-rate Comparisons in downlink


![embb_urllc__dl_ecdf](https://user-images.githubusercontent.com/51235418/211160434-42df2af6-6f31-403b-bab0-42c3f328d800.svg)
It is observed that when the network is configured with both eMBB and
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

# Resource Consumption Comparisons in uplink

![reserved2](https://user-images.githubusercontent.com/51235418/211160637-c0f13f6a-65ec-4504-bea0-d0eeb21467a3.svg)

# Collision probability analysis

![col_rbs](https://user-images.githubusercontent.com/51235418/211160548-0639d0db-207a-4189-bcaf-a09623e0348e.svg)



