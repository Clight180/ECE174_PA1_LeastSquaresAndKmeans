import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

K = 10

def XYPairing(Xlist, labels, i, j, binary):
    #Normalize. (list because it's faster than np.arrays)
    X = [[inner/255 for inner in outer] for outer in Xlist]

    #Convert Ytrain to appropriately conditioned TeacherVector
    TeacherVectorList = []
    index = 0
    for value in labels:
        if value == i:
            TeacherVectorList.append(1)
        elif binary:
            TeacherVectorList.append(-1)
        #For One vs One
        elif not binary and value == j:
            TeacherVectorList.append(-1)
        else:
            X.pop(index)
            index -= 1
        index += 1
    X = np.array(X)
    TeacherVector = np.array(TeacherVectorList)
    return X, TeacherVector

def getAB(X, Y):
    #Given feature matrix X and teacher signal Y, compute Alpha and Beta to minimum residual sum squares
    X = np.column_stack((np.ones((X.shape[0], 1)), X))
    Beta = np.linalg.pinv(X.T @ X) @ (X.T @ Y)
    Alpha = Beta[0]
    Beta = np.delete(Beta, 0)
    return Alpha, np.ndarray.tolist(Beta)

def Classifier(Alpha, Beta, X, binary):
    X = np.array(X)
    X = X/255
    f_hat = []

    for index in range(X.shape[0]):
        f_tilde = []
        if binary:
            for c in range(len(Alpha)):
                f_tilde.append(Beta[c] @ X[index] + Alpha[c])
            f_hat.append(np.argmax(f_tilde))
        else:
            f_tilde = [0] * 10
            ABndx = 0
            for i in range(K-1):
                j = i + 1
                while j < K:
                    confidence = Beta[ABndx] @ X[index] + Alpha[ABndx]
                    if confidence >= 0:
                        f_tilde[i] += 1
                    else:
                        f_tilde[j] += 1
                    j+=1
                    ABndx += 1
            f_hat.append(np.argmax(f_tilde))
    f_hat = np.array(f_hat)
    return f_hat

def confusionMatrix(Y, Ypredict):
    ConfusionMatrix = np.zeros((max(Y)+1, max(Y)+1), dtype=int)
    for index in range(len(Y)):
        ConfusionMatrix[Ypredict[index]][Y[index]] += 1
    return ConfusionMatrix

def plotCM(CM):
    plt.figure()
    ax = sns.heatmap(CM, annot=True, fmt="d", cbar=False, linewidths=.1, cmap='inferno')
    plt.xlabel('Truth', loc='left')
    plt.ylabel('Prediction', loc='top')
    plt.title('Confusion Matrix')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.show()
