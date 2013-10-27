# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 10:45:15 2013

@author: alexandersedgwick
"""

# Import required libraries

import sys

# Start a counter and store the textfile in memory

ages=[]
genders=[]
signins=[]
clicks = []
impressions = []

lines = sys.stdin.readlines()

#Remember pop removes the element from lines
lines.pop(0)

#Open the file to write to
file = open("output_agg.txt", "w")

#For each line find the sum of the index 2 in the list.
for line in lines:
	ages.append(int(line.strip().split(',')[0]))
	genders.append(int(line.strip().split(',')[1]))
	signins.append(int(line.strip().split(',')[4]))    

ages = set(ages)
ages = list(ages)

genders = set(genders)
genders = list(genders)

signins = set(signins)
signins = list(signins)


#for counter in range(0,max(ages)):
for age in ages:
	for gender in genders:
		for signin in signins:
			
			del impressions[:]
			del clicks[:]

			for line in lines:

				if int(line.strip().split(',')[0]) == age:
					if int(line.strip().split(',')[1]) == gender:
						if int(line.strip().split(',')[4]) == signin:
							impressions.append(int(line.strip().split(',')[2]))
							clicks.append(int(line.strip().split(',')[3]))


			file.write(str(age) + ",")
			file.write(str(gender) + ",")
			file.write(str(signin) + ",")
			
			if len(clicks) == 0:
				file.write(str(float(0)) +",")
			else:
				file.write(str(float(sum(clicks))/(len(clicks))) +",")
			
			if len(impressions) == 0:
				file.write(str(float(0)) +",")
			else:
				file.write(str(float(sum(impressions))/(len(impressions))) +",")
			
			if len(clicks) == 0:
				file.write(str(float(0)) +",")
			else:	
				file.write(str(float(max(clicks))) +",")

			if len(impressions) == 0:
				file.write(str(float(0)) +",\n")
			else:
				file.write(str(float(max(impressions)))+",\n")

file.closed
