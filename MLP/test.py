import numpy
from pandas import read_csv

dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python',skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
#print(dataset.shape)
#print(dataset)
train_size = int(len(dataset)*0.67)
test_size = len(dataset) - train_size
#print(train_size)
#print(test_size)
train, test = dataset[0:train_size, :], dataset[train_size: len(dataset), :]


def create_dataset(train, look_back=1):
    dataX, datay = [], []
    for i in range(len(train) -look_back -1) :
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        datay.append(dataset[i+look_back-1, 0])
        return numpy.array(dataX), numpy.array(datay)


look_back=3
trainX, trainy = create_dataset(train, look_back)
testX, testy = create_dataset(test, look_back)
print(trainX.ndim)
print(trainX.shape[0])
print(trainX.shape[1])

print(testX.shape)
print(trainy.shape)
print(testy.shape)