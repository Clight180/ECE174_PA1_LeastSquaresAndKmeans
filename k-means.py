import numpy as np
import KMFun as fun
from mnist import MNIST
import matplotlib.pyplot as plt

noPlot = False
if noPlot:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt


'''--------------Step 1--------------'''


# K centroids
K = 10

#Create test points
points = np.random.multivariate_normal((0, 0), [[10, 0], [0, 10]], 10).T
for i in range(K - 1):
    points = np.hstack((points, np.random.multivariate_normal(
        (np.random.randint(-100, 100), np.random.randint(-100, 100)),
        [[np.random.randint(1, 100), 0], [0, np.random.randint(1, 100)]], 20).T))

N_groupAssignments, K_groupRepVectors, J_clust = fun.kmeans(points, K, True)
plt.plot(J_clust)
plt.show()


'''--------------Step 2--------------'''


K = 20

#load MNIS data
mndata = MNIST('./ZippedDataset')
mndata.gz = True
trainImages, trainLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()
MNIST_Labels = trainLabels
MNIST_Points = np.array([[inner/255 for inner in outer] for outer in trainImages]).T


N_groupAssignments_list, K_groupRepVectors_list, J_clust_list = fun.kmeans(MNIST_Points, K, False)
plt.figure()
plt.plot(J_clust_list)
plt.title('J_clust vs trials')
plt.show()

maxJclust = np.argmax(J_clust_list)
minJclust = np.argmin(J_clust_list)

print('Evaluating worst run. Occurrence on run: ', maxJclust)
fun.evaluateKmeans(N_groupAssignments_list[maxJclust], K_groupRepVectors_list[maxJclust], MNIST_Points, MNIST_Labels, K)
print('Evaluating best run. Occurrence on run: ', minJclust)
fun.evaluateKmeans(N_groupAssignments_list[minJclust], K_groupRepVectors_list[minJclust], MNIST_Points, MNIST_Labels, K)

exit('Have a nice day.')
