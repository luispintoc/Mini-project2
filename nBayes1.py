import csv 
import random
import math

 
'''
**********LOAD DATA********
Load a csv file into a list of lists containing 
each data point Xj value

'''
def loadingCsv(filename): 
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)): 
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset 

#filename = "data.data.csv"
#dataset=loadingCsv(filename)
#print("loaded data file {0} with {1} rows").format(filename, len(dataset))
'''
*********SPLIT DATA INTO SETS********
Split data set (list of lists) into training set, test set 
'''
def splitDataset(dataset, splitRatio): 
    trainSize = int(len(dataset)*splitRatio)
    trainSet = [] 
    copy = list(dataset)
    while(len(trainSet)< trainSize): 
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

#dataset = [[1], [2], [3], [4], [5]]
#splitRatio = .67
#train, test = splitDataset(dataset, splitRatio)
#print('Split {0} rows into train with {1} and test {2}'). format(len(dataset), train, test)


'''
********SEPARATE IT BASED ON CLASS*****

Make a dictionary with the key as the class and value a
list of integer values associated with each data point Xj

Input: csv file 
Output: dictionary 
'''
def separateByClass(dataset): 
    separated = {} 
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated): 
            separated[vector[-1]] = [] #assume last val is class
        separated[vector[-1]].append(vector)
    return separated

#dataset = [[1,20,1], [2,21,0], [3,22,1]]
#separated = separateByClass(dataset)
#print ("separated the insances : {0}").format(separated)

#MEAN and STDEV
def mean(numbers): 
    return sum(numbers)/float(len(numbers))
def stdev(numbers): 
    avg = mean(numbers)
    variance = sum([pow(x-avg, 2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

#numbers = [1,2,3,4,5] 
#print("Summary of {0}: mean = {1}, stdev={2}").format(numbers, mean(numbers), stdev(numbers))

#MAKE MEAN AND COVARIANCE MATRIX 
def summarize(dataset): 
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries 

#dataset = [[1,20,0], [2,21,1], [3,22,0]]
#summary = summarize(dataset)
#print("Attribute summaries {0}").format(summary)

def summarizeByClass(dataset): 
    separated = separateByClass(dataset)
    #print(separated)
    summaries = {}
    for classValue, instances in separated.items(): 
        summaries[classValue] = summarize(instances)
    return summaries 

#dataset = [[1,20,1], [2,21,0],[3,22,1], [4,22,0]]
#summary = summarizeByClass(dataset)
#print("Summary by class value {0}").format(summary)


'''
****Calculate P(X|Y) single instance***** 
For each x, mean, and stdev plug into Gaussian probability
'''
def calculateProbability(x, mean, stdev): 
    exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev,2))))
    return(1/(math.sqrt(2*math.pi)*stdev))*exponent

#x = 71.5
#mean = 73
#stdev = 6.2

#probability = calculateProbability(x, mean, stdev)
#print('Probability of belonging to this class ((P(X|Y)) : {0}').format(probability)
'''
*******Calculate P(X|Y) for class*****
Extend the function above to calculate probability of an 
entire datapoint (Xj--> Xn)
rather than just a single instance Xj belonging to a class

Given a summary(mean, stdev) of each class, set all probabilities to one
then iterate through each class and calculate the posibility 
of Xj being in the class() then Xj+1, etc) multiplying as you go

'''
def calculateClassProbabilities(summaries, inputVector): 
    probabilities = {} 
    for classValue, classSummaries in summaries.items(): 
        probabilities[classValue] = 1
        for i in range(len(classSummaries)): 
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

#summaries = {0: [(1, 0.5)], 1:[(20,5.0)]}
#inputVector = [1.1]
#probabilities = calculateClassProbabilities(summaries, inputVector)

#print("probabilities for each class {0}").format(probabilities)

#PREDICTION FOR ONE FEATURE IN DATASET
def predict(summaries, inputVector): 
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb: 
            bestProb = probability
            bestLabel = classValue
    return bestLabel

#summaries={0:[(1,.5)], 1: [(20,5.0)]}
#inputVector = [1.1]
#result = predict(summaries, inputVector)
#print ("Prediction: {0}"). format(result)


def getPredictions(summaries, testSet): 
    predictions = [] 
    for i in range(len(testSet)): 
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions 

#summaries = {0:[(1,.5)], 1: [(20,5.0)]}
#testSet = [[1.1], [19.1]]
#predictions = getPredictions(summaries, testSet)
#print("Predictions {0}").format(predictions)

def getAccuracy(testSet, predictions): 
    correct = 0
    for i in range(len(testSet)): 
        if testSet[i][-1] == predictions[i]: 
            correct += 1
    return (correct/float(len(testSet)))* 100.00

def main(): 
    filename = "data.data.csv"
    splitRatio = 0.67
    dataset = loadingCsv(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print("SPLIT {0} ROWS into TRAIN = {1} and TEST = {2}"). format(len(dataset), len(trainingSet), len(testSet))
    summaries = summarizeByClass(trainingSet)
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print("ACCURACY IS {0}").format(accuracy)
main()