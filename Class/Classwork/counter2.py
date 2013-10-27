#!/usr/bin/python

# Import required libraries

import sys

# Start a counter and store the textfile in memory

ages=[]
clicks = []
impressions =[]

lines = sys.stdin.readlines()

#Remember pop removes the element from lines
lines.pop(0)

#For each line find the sum of the index 2 in the list.

for line in lines:
	if int(line.strip().split(',')[0]) != 0:
		ages.append(int(line.strip().split(',')[0]))
	impressions.append(int(line.strip().split(',')[2]))
	clicks.append(int(line.strip().split(',')[3]))


# Open a file
file = open("output.txt", "w")
file.write("Impressions Sum " + str(sum(impressions)) + "\n")
file.write('Average Age: ' + str(sum(ages)/len(ages))+ "\n")
file.write('Oldest Person: ' + str(max(ages))+ "\n")
file.write('Youngest Person: ' + str(min(ages))+ "\n")
file.write('Clicks per impression: '  + str(float(sum(clicks))/float(sum(impressions)))+ "\n")
file.closed

