# Import required libraries
import sys
# Start a counter and store the textfile in memory
lines = sys.stdin.readlines()
lines.pop(0)
#Create a dictionary
dict = {}
for line in lines:
	age = int(line.strip().split(',')[0])
	gender = int(line.strip().split(',')[1])
	sign_in = int(line.strip().split(',')[4])
	clicks = int(line.strip().split(',')[2])
	impressions = int(line.strip().split(',')[3])

	if (age, gender, sign_in) in dict.keys():	
		dict[age, gender, sign_in][0][0] = dict[age, gender, sign_in][0][0] + clicks
		dict[age, gender, sign_in][0][1] += 1
		dict[age, gender, sign_in][1][0] = dict[age, gender, sign_in][1][0] + impressions
		dict[age, gender, sign_in][1][1] += 1

		if clicks>dict[age, gender, sign_in][0][2]:
			dict[age, gender, sign_in][0][2] = clicks

		if clicks>dict[age, gender, sign_in][1][2]:
			dict[age, gender, sign_in][1][2] = impressions

	else:
		dict[age, gender, sign_in] = [[clicks,1,clicks],[impressions, 1, impressions]]
#Open the file to write to
file = open("output_agg.txt", "w")

for k,v in dict.items():
	file.write(str(k[0]) + ",") #Age
	file.write(str(k[1]) + ",")	#Gender
	file.write(str(k[2]) + ",") #Sign_in
	file.write(str(float(v[0][0])/float(v[0][1])) + ",") #clicks average
	file.write(str(float(v[1][0])/float(v[1][1])) + ",") #Impressions average
	file.write(str(v[0][2]) + ",") #max clicks
	file.write(str(v[1][2]) + ",\n") #max impressions



	#file.write(str(v[0][0]) + ",") #clicks sum
	#file.write(str(v[0][1]) + ",") #clicks count
	#file.write(str(v[1][0]) + ",") #Impressions sum
	#file.write(str(v[1][1]) + ",") #Impressions count


