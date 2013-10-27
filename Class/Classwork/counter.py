#!/usr/bin/python

# Import required libraries

import sys

# Start a counter and store the textfile in memory

count=0
lines = sys.stdin.readlines()

#Remember pop removes the element from lines
lines.pop(0)

#For each line find the sum of the index 2 in the list.

for line in lines:
	count = count + int(line.strip().split(',')[2])

print 'Impressions Sum: ', count
print 'Average Impresssions', float(count/len(lines))
