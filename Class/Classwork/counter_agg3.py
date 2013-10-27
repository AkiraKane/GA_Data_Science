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

impressions_m_si = []
impressions_m_so = []
impressions_f_si = []
impressions_f_so = []


file = open("output_agg.txt", "w")


lines = sys.stdin.readlines()

#Remember pop removes the element from lines
lines.pop(0)

#For each line find the sum of the index 2 in the list.

for line in lines:
	ages.append(int(line.strip().split(',')[0]))
        
ages = set(ages)
ages = list(ages)

#for counter in range(0,max(ages)):
for counter in ages:
	del clicks_m_si[:]
	del clicks_m_so[:]
	del clicks_f_si[:]
	del clicks_f_so[:]
	del impressions_m_si[:]
	del impressions_m_so[:]
	del impressions_f_si[:]
	del impressions_f_so[:]
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
	file.write(str(sum(impressions_f_so)) + ",")
	file.write(str(sum(clicks_f_so)) + ",")

	file.write(str(sum(impressions_f_si)) +",")
	file.write(str(sum(clicks_f_si)) +",")

	file.write(str(sum(impressions_m_so)) +",")
	file.write(str(sum(clicks_m_so)) +",")

	file.write(str(sum(impressions_m_si)) +",")
	file.write(str(sum(clicks_m_si)) +"\n")
   

file.closed
