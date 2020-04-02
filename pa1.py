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

def nodeSplit(rootNode):
    parentNode = Node(rootNode)
    # create left and right node for first split
    leftNodeFrame = pd.DataFrame(pd.np.empty((0, 3)))
    rightNodeFrame = pd.DataFrame(pd.np.empty((0, 3)))
    leftNodeFrame.columns = columnLabels
    rightNodeFrame.columns = columnLabels

    # find average value for first attribute and split into two dataframes
    avgOfFeature = rootNode[runningBestLoc].mean()

    #for row in range(len(rootNode.index)):
    for i, row in rootNode.iterrows():
        # if np.where(pd.isnull(rootNode[runningBestLoc][row])):
        #     print("HERE")
        if (rootNode[runningBestLoc][i] < avgOfFeature):
            # print("LESS + " + str(testData[runningBestLoc][row]))
            leftNodeFrame = leftNodeFrame.append(rootNode.loc[i])
        else:
            rightNodeFrame = rightNodeFrame.append(rootNode.loc[i])
            #rightNode.append(data[columns][row])
    #leftNode = Node(leftNodeFrame, parent=parentNode)
    #rightNode = Node(rightNodeFrame, parent=parentNode)

    return leftNodeFrame, rightNodeFrame

leftNodeFrame, rightNodeFrame = nodeSplit(testData)

# find best attribute for second split (left branch)
runningBest = 0
for column in columnLabels:
    if column != 'label':
        currInfoGain = informationGain(leftNodeFrame, column)
        if currInfoGain > runningBest:
            runningBest = currInfoGain
            runningBestLoc = column

leftLeftNodeFrame, leftRightNodeFrame = nodeSplit(leftNodeFrame)

# find best attribute for second split (right branch)
runningBest = 0
for column in columnLabels:
    if column != 'label':
        currInfoGain = informationGain(testData, column)
        if currInfoGain > runningBest:
            runningBest = currInfoGain
            runningBestLoc = column

rightLeftNodeFrame, rightRightNodeFrame = nodeSplit(rightNodeFrame)

# build tree using dataframes created
rootNode = Node(testData, parent=None)
leftNode = Node(leftNodeFrame, parent=rootNode)
rightNode = Node(rightNodeFrame, parent=rootNode)
leftLeftNode = Node(leftLeftNodeFrame, parent=leftNode)
leftRightNode = Node(leftRightNodeFrame, parent=leftNode)
rightLeftNode = Node(rightLeftNodeFrame, parent=rightNode)
rightRightNode = Node(rightRightNodeFrame, parent=rightNode)