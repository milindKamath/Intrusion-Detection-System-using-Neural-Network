# This file contains code for pre-processing of data by normalizing and splitting the data into train and test.

__author__ = "Milind Kamath mk6715, Uddesh Karda uk8216"

# import libraries
import os
import numpy as np
import time
import re
import random

# attack buckets
attackDoS = ["Optimized_Smurf", "Optimized_Neptune", "Optimized_Back", "Optimized_TearDrop", "Optimized_Pod",
             "Optimized_Land"]
attackProbing = ["Optimized_Satan", "Optimized_IPSweep", "Optimized_PortSweep", "Optimized_NMap"]
attackU2R = ["Optimized_BufferOverflow", "Optimized_RootKit", "Optimized_LoadModule", "Optimized_Perl"]
attackR2L = ["Optimized_WarezClient", "Optimized_GuessPassword", "Optimized_WarezMaster", "Optimized_Imap",
             "Optimized_FTPWrite", "Optimized_MultiHop", "Optimized_PHF", "Optimized_Spy"]

# normalize and label
start = time.time()
print("Starting to parse the files from the folder\n")
output = open("dataset", 'a+')
for file in os.listdir("optimized_attacks_normal"):
    with open(os.getcwd() + "\\" + "optimized_attacks_normal" + "\\" + file) as f:
        for line in f.readlines():
            if re.search('\x00', line) is None:  # avoid NULL data
                content = np.array(list(map(float, line.split(","))))
                normalized = (content - np.mean(content)) / np.sqrt(np.sum(np.square(content - np.mean(content))))
                if file in attackDoS:
                    normalized = np.append(normalized, np.array(1))
                elif file in attackR2L:
                    normalized = np.append(normalized, np.array(2))
                elif file in attackU2R:
                    normalized = np.append(normalized, np.array(3))
                elif file in attackProbing:
                    normalized = np.append(normalized, np.array(4))
                elif file == "Optimized_Normal":
                    normalized = np.append(normalized, np.array(0))
                output.write(', '.join(list(map(str, list(normalized)))) + "\n")
        print(file + " parse completed")
output.close()

print("\nNew file containing normalized and labeled data created")

# split train and test
splitRatio = 0.75
train = open('train', 'a+')
test = open('test', 'a+')
with open('dataset', 'r') as d:
    data = d.readlines()
    random.shuffle(data)
    traindata = data[:int(splitRatio * len(data))]
    for row in traindata:
        train.write(row)
    testdata = data[int(splitRatio * len(data)):]
    for row in testdata:
        test.write(row)

print("\nData split into train and test data")

print("\nPre-processing Completed\n")
end = time.time()
print(end - start, "seconds")
