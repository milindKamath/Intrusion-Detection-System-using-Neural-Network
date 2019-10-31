# This file takes input from training data set and trains the neural network
__author__ = "Milind Kamath mk6715, Uddesh Karda uk8216"

import pickle
import numpy as np
import time
from tqdm import tqdm


# sigmoid activation
def sigmoid(val):
    return 1 / (1 + np.exp(-val))


# load the model
def readData():
    model = open("NNmodel.txt", 'rb')
    (weights, biases) = pickle.load(model)
    test = open("test", 'r')
    data = test.readlines()
    return weights, biases, data


# prediction
def predict(data, weights, biases):
    counter = 0
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    with tqdm(total=len(data), desc="Training") as prog:
        for line in data:
            content = np.array(list(map(float, line.split(","))))
            inputsequence = content[:len(content) - 2]
            outputsequence = np.array([0 for i in range(5)])
            outputsequence[int(content[-1])] = 1
            for j in range(1, len(weights)+1):
                res1 = np.matmul(weights[j], inputsequence)
                res2 = np.add(res1, biases[j])
                hidden = np.array(list(map(sigmoid, res2)))
                inputsequence = hidden
            x = list(inputsequence).index(max(list(inputsequence)))
            y = list(outputsequence).index(max(list(outputsequence)))
            if x == y and y > 0:
                truePositive += 1
            elif x == y and y == 0:
                trueNegative += 1
            elif x != y and y > 0:
                falsePositive += 1
            elif x != y and y == 0:
                falseNegative += 1
            counter += 1
            prog.update(1)

    precision = (truePositive / (truePositive + falsePositive))
    recall = (truePositive / (truePositive + falseNegative))

    print("Precision ", precision * 100, " percent")
    print("Recall: ", recall * 100, " percent")
    print("F-Measure: ", (2 / ((1 / precision) + (1 / recall))) * 100, " percent")
    print("Accuracy: ", ((truePositive + trueNegative) /
                         (truePositive + trueNegative + falsePositive + falseNegative)) * 100, " percent")


# start prediction
def start():
    weights, biases, data = readData()
    start = time.time()
    predict(data, weights, biases)
    end = time.time()
    print("Time taken: ", end - start, "seconds")


# main
if __name__ == '__main__':
    start()
