# This file takes input from training dataset and trains the neural network
__author__ = "Milind Kamath mk6715, Uddesh Karda uk8216"

# import libraries
import numpy as np
import time
from tqdm import tqdm
import pickle
import cProfile

# Dictionaries to save the weights, biases, errors, error derivatives and the output of every layer
Weights = {}
Bias = {}
Layers = {}
Errors = {}
DervOutput = {}
learningRate = 0.1


# generate random weights and biases
def createWeightsandBias(nodes, input, output):
    global Weights, Bias
    j = 1
    for i in range(len(nodes)):
        Weights[i+1] = np.random.random((nodes[i], input))
        Bias[i+1] = np.random.random((nodes[i]))
        input = nodes[i]
        j += 1
    Weights[j] = np.random.random((output, nodes[-1]))
    Bias[j] = np.random.random((output))


# read the training data
def readData():
    tr = open("train", 'r')
    data = tr.readlines()
    return data


# activation function sigmoid
def activation(val):
    return 1 / (1 + np.exp(- val))


# derivative of sigmoid
def derSigmoid(val):
    return val * (1 - val)


# matrix multiplication and activation
def matmul(input, weights, bias):
    res1 = np.matmul(weights, input)
    res = np.add(res1, bias)
    hidden = list(map(activation, res))
    return hidden


# user input for number of layers and nodes and epochs
def layers():
    hiddenLayers = int(input("Enter the number of hidden layers: "))
    layerNumber = 1
    nodes = []
    for i in range(hiddenLayers):
        nodes.append(int(input("Enter the number of nodes for hidden layer " + str(layerNumber) + ": ")))
        layerNumber += 1
    return nodes


# feed forward algorithm
def feedforward(inputsequence, nodes):
    global Weights, Bias, Layers
    layer = inputsequence
    counter = 1
    Layers[1] = inputsequence
    for i in range(1, len(nodes) + 1):
        hiddenLayer = matmul(layer, Weights[i], Bias[i])
        layer = hiddenLayer
        Layers[i+1] = np.array(hiddenLayer)
        counter += 1

    output = matmul(layer, Weights[counter], Bias[counter])
    Layers[counter+1] = np.array(output)
    return output


# backpropagation
def backpropagate(error, learningRate):
    global Weights, Errors, DervOutput, Layers
    Errors[len(Layers)] = error
    DervOutput[len(Layers)] = np.array(list(map(derSigmoid, Layers[len(Layers)])))
    for keys in sorted(Weights.keys(), reverse=True):
        Errors[keys] = np.matmul(Weights[keys].T, error)
        DervOutput[keys] = np.array(list(map(derSigmoid, Layers[keys])))
        error = Errors[keys]
    tempWeights = {}
    tempBias = {}
    tempLayer = []
    tempY = []
    for keys in sorted(Weights.keys(), reverse=True):
        x = np.multiply(Errors[keys+1], DervOutput[keys+1])
        y = np.multiply(learningRate, x)
        tempLayer.append(list(Layers[keys]))
        tempY.append(list(y))
        temp = np.array(tempLayer)
        newy = np.array(tempY)
        z = np.matmul(temp.T, newy)
        tempWeights[keys] = np.subtract(Weights[keys], z.T)
        tempBias[keys] = np.subtract(Bias[keys], y)
        tempLayer = []
        tempY = []
    Weights.update(tempWeights)
    Bias.update(tempBias)


# per epoch training
def epochTraining(data, nodes):
    global learningRate
    with tqdm(total=len(data), desc="Training") as prog:
        for line in data:
            content = np.array(list(map(float, line.split(","))))
            inputsequence = content[:len(content) - 2]
            outputsequence = np.array([0 for i in range(5)])
            outputsequence[int(content[-1])] = 1
            if not Weights and not Bias:
                createWeightsandBias(nodes, len(inputsequence), len(outputsequence))
                guess = feedforward(inputsequence, nodes)
            else:
                guess = feedforward(inputsequence, nodes)
            errorVector = np.subtract(guess, outputsequence)
            backpropagate(errorVector, learningRate)
            prog.update(1)
        learningRate /= 10


# train neural network
def neuralNet():
    global epoch
    data = readData()
    nodes = layers()
    epoch = int(input("Enter epoch: "))
    for i in range(epoch):
        print("\nEpoch ", i + 1)
        start = time.time()
        epochTraining(data, nodes)
        end = time.time()
        print("\nTime taken:", end - start)


# start
if __name__ == '__main__':
    start = time.time()
    neuralNet()
    #cProfile.run('neuralNet()')
    end = time.time()
    model = open("NNModel.txt", 'wb')
    pickle.dump((Weights, Bias), model)
    print("\nTotal Training Time:", end - start)
