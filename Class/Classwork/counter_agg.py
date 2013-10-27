# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 10:45:15 2013

@author: alexandersedgwick
"""

# Import required libraries

import sys

# Start a counter and store the textfile in memory

ages=[]

clicks_m_si = []
clicks_m_so = []
clicks_f_si = []
clicks_f_so = []

impressions_m_si =[]
impressions_m_so =[]
impressions_f_si =[]
impressions_f_so =[]



file = open("output_agg.txt", "w")


lines = sys.stdin.readlines()

#Remember pop removes the element from lines
lines.pop(0)

#For each line find the sum of the index 2 in the list.

for line in lines:
    if int(line.strip().split(',')[0]) != 0:
        ages.append(int(line.strip().split(',')[0]))
        


for counter in range(max(ages)):
    for line in lines:
        if int(line.strip().split(',')[0]) == counter:
            if int(line.strip().split(',')[1]) ==0:
                if int(line.strip().split(',')[4]) == 0:
                    impressions_f_so.append(int(line.strip().split(',')[2]))
                    clicks_f_so.append(int(line.strip().split(',')[3]))
                else:
                    impressions_f_si.append(int(line.strip().split(',')[2]))
                    clicks_f_si.append(int(line.strip().split(',')[3]))   
            else:
                if int(line.strip().split(',')[4]) == 0:
                    impressions_m_so.append(int(line.strip().split(',')[2]))
                    clicks_m_so.append(int(line.strip().split(',')[3]))
                else:
                    impressions_m_si.append(int(line.strip().split(',')[2]))
                    clicks_m_si.append(int(line.strip().split(',')[3])) 
    
    
   	file.write(str(counter) + ",")
    file.write(str(0) +",")
    file.write(str(0) +",")

    if len(clicks_f_so) > 0:
    	file.write(str(sum(clicks_f_so)/len(clicks_f_so))+",")
    else:
    	file.write(str(0)+",")

    if not clicks_f_so:
        file.write(str(0) +",")
    else:    
        file.write(str(max(clicks_f_so)) +",")

    if not impressions_f_so:
        file.write(str(0) +",\n")
    else:
        file.write(str(max(impressions_f_so)) +",\n")



    file.write(str(counter) + ",")
    file.write(str(0) +",")
    file.write(str(1) +",")

    if len(clicks_f_si) > 0:
    	file.write(str(sum(clicks_f_si)/len(clicks_f_si))+",")
    else:
    	file.write(str(0)+",")

    if not clicks_f_si:
        file.write(str(0) +",")
    else:    
        file.write(str(max(clicks_f_si)) +",")

    if not impressions_f_si:
        file.write(str(0) +",\n")
    else:
        file.write(str(max(impressions_f_so)) +",\n")


    file.write(str(counter) + ",")
    file.write(str(1) +",")
    file.write(str(0) +",")

    if len(clicks_m_so) > 0:
    	file.write(str(sum(clicks_m_so)/len(clicks_m_so))+",")
    else:
    	file.write(str(0)+",")

    if not clicks_m_so:
        file.write(str(0) +",")
    else:    
        file.write(str(max(clicks_m_so)) +",")

    if not impressions_m_so:
        file.write(str(0) +",\n")
    else:
        file.write(str(max(impressions_m_so)) +",\n")


    file.write(str(counter) + ",")
    file.write(str(1) +",")
    file.write(str(1) +",")

    if len(clicks_m_si) > 0:
    	file.write(str(sum(clicks_m_si)/len(clicks_m_si))+",")
    else:
    	file.write(str(0)+",")

    if not clicks_m_si:
        file.write(str(0) +",")
    else:    
        file.write(str(max(clicks_m_si)) +",")

    if not impressions_m_si:
        file.write(str(0) +",\n")
    else:
        file.write(str(max(impressions_m_si)) +",\n")

file.closed