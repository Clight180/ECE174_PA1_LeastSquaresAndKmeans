import numpy as np
import LSFun as fun
from mnist import MNIST


#K, number of classes
K = 10

#load MNIS data
mndata = MNIST('./ZippedDataset')
mndata.gz = True
trainImages, trainLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()

#One vs All:
Alpha = []
Beta = []
for c in range(K):
    X, Y = fun.XYPairing(trainImages, trainLabels, c, c, True)
    A, B = fun.getAB(X, Y)
    Alpha.append(A)
    Beta.append(B)

#now, make predictions and present confusion matrix
f_hat = fun.Classifier(Alpha, Beta, testImages, True)
accuracy = 1 - np.count_nonzero(testLabels-f_hat)/len(testLabels)
print('One vs. All accuracy:', format(accuracy*100, '.2f'), '%')
OneVsAll_ConfusionMatrix = fun.confusionMatrix(testLabels, f_hat)
fun.plotCM(OneVsAll_ConfusionMatrix)

#One vs One:
Alpha = []
Beta = []
for i in range(K-1):
    j = i+1
    while j < K:
        X, Y = fun.XYPairing(trainImages, trainLabels, i, j, False)
        A_ij, B_ij = fun.getAB(X, Y)
        Alpha.append(A_ij)
        Beta.append(B_ij)
        j += 1

#more predictions, another confusion matrix
f_hat = fun.Classifier(Alpha, Beta, testImages, False)
accuracy = 1 - np.count_nonzero(testLabels-f_hat)/len(testLabels)
print('One vs. One accuracy:', format(accuracy*100, '.2f'), '%')
OneVsOne_ConfusionMatrix = fun.confusionMatrix(testLabels, f_hat)
fun.plotCM(OneVsOne_ConfusionMatrix)

exit('Have a nice day.')
