# Import required libraries
import sys
# Start a counter and store the textfile in memory
lines = sys.stdin.readlines()
lines.pop(0)
#Create a dictionary
dict = {}
for line in lines:
	if (int(line.strip().split(',')[0]),int(line.strip().split(',')[1]),int(line.strip().split(',')[4])) in dict.keys():
		
	else:
		dict[(int(line.strip().split(',')[0]),int(line.strip().split(',')[1]),int(line.strip().split(',')[4]))] = 1


for k,v in dict.items():
print k, v