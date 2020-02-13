import pandas as pd


def entropy(data):
    num0 = 0
    num1 = 0
    numrows = 0

    for index, row in data.iterrows():
        print(row['label'])
        if row['label'] == 0:
            num0 += 1
        elif row['label'] == 1:
            num1 += 1
        numrows += 1

    entTemp1 = float((num1/numrows)*)
    






columns = ['feature1', 'feature2', 'label']

testData = pd.read_csv('synthetic-1.csv')
testData.columns = columns
testEntropy = entropy(testData)
