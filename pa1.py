import pandas as pd
#import math
import numpy as np
from anytree import Node, RenderTree
from anytree.exporter import DotExporter

def entropy(data):
    num0 = 0
    num1 = 0

    for i in range(len(data.index)):
        if data.iloc[i][2] == 0:
            num0 += 1
        elif data.iloc[i][2] == 1:
            num1 += 1
    if data.size == 0:
        return 0
    tempEnt1 = float(num1/len(data.index)) * np.log2(float(num1/len(data.index)))
    tempEnt2 = float(num0/len(data.index)) * np.log2(float(num0/len(data.index)))
    return -1 * tempEnt1 - tempEnt2

def entropyHelper(num0, num1, length):
    tempEnt1 = float(num1/length) * np.log2(float(num1/length))
    tempEnt2 = float(num0/length) * np.log2(float(num0/length))
    return -1 * tempEnt1 - tempEnt2

# def informationGain(data):
#     parentEntropy = entropy(data)
#     leftNode = pd.DataFrame()
#     rightNode = pd.DataFrame()
#     for columns in data:
#         if data[columns][0] == 1:
#             break
#         else:
#             # sumOfFeature = data[columns].mean()
#             # avgOfFeature = float(sumOfFeature / len(data.index))
#             # for row in range(len(data.index)):
#             #     if data[columns][row] < avgOfFeature:
#             #         print("LESS + " + str(data[columns][row]))
#             #         leftNode.append(data[columns][row])
#             #     else:
#             #         print("GREATER")
#             #         #rightNode.append(data.iloc[row])
#             #         rightNode.append(data[columns][row])
#     # print("ENTROPY OF LEFT NODE: " + str(entropy(leftNode)))
#     # print("ENTROPY OF RIGHT NODE: " + str(entropy(rightNode)))

def informationGain(data, featureName):
    parentEntropy = entropy(data)
    leftNode = []
    rightNode = []

    split = pd.cut(data[featureName], 2, labels=[0,1])

    for i in split:
        if i == 0:
            leftNode.append(i)
        else:
            rightNode.append(i)
    
    infoGain = parentEntropy - entropyHelper(len(leftNode), len(rightNode), data.size)
    print(infoGain)
    return infoGain

columnLabels = ['feature1', 'feature2', 'label']
testData = pd.read_csv('synthetic-1.csv', header=None)
testData.columns = columnLabels

# find best attribute for first split
runningBest = 0
for column in columnLabels:
    if column != 'label':
        currInfoGain = informationGain(testData, column)
        if currInfoGain > runningBest:
            runningBest = currInfoGain
            runningBestLoc = column

def splitNode(node):
    

# set rootNode to be entire dataframe
rootNode = Node(testData)

# create left and right node for first split
leftNodeTemp = pd.DataFrame()
rightNodeTemp = pd.DataFrame()

# find average value for first attribute and split into two dataframes
avgOfFeature = testData[runningBestLoc].mean()

for row in range(len(testData.index)):
    if testData[runningBestLoc][row] < avgOfFeature:
        # print("LESS + " + str(testData[runningBestLoc][row]))
        leftNodeTemp.append(testData.iloc[row])
    else:
        # print("GREATER")
        rightNodeTemp.append(testData.iloc[row])
        #rightNode.append(data[columns][row])

leftNode = Node(leftNodeTemp, parent=rootNode)
rightNode = Node(rightNodeTemp, parent=rootNode)

print(RenderTree(rightNode))