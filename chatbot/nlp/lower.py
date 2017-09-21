from io import open
f2=open("models/original/all_features_lower-tree1.txt","w")
with open("models/original/all_features-tree1.txt",'r') as file:
	for line in file:
		f2.write(line.lower())