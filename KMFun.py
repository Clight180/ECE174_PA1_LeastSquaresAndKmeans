import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


randClusters = False
iterations = 30

def farthestNeighbors(points, k):
    points = np.ndarray.tolist(points.T)
    remaining_points = points[:]
    solution_set = []
    solution_set.append(remaining_points.pop(np.random.randint(0, len(remaining_points) - 1)))
    for K in range(k - 1):
        distances = [np.linalg.norm(np.subtract(point, solution_set[K])) for point in remaining_points]
        for index_rp, point_rp in enumerate(remaining_points):
            for index_ss, point_ss in enumerate(solution_set):
                distances[index_rp] = min(distances[index_rp], np.linalg.norm(np.subtract(point_rp, point_ss)))
        solution_set.append(remaining_points.pop(np.argmax(distances)))
    return solution_set

def kmeans(points, K, is2D):
    group_index = []
    centroids = []
    J_list = []
    if randClusters:
        # Pick clusters randomly from point set
        centroids_iter = [points[:, np.random.randint(0, points.shape[1])] for k in range(K)]
        group_index_iter = []
        for index_points in range(points.shape[1]):
            temp_distances = [np.linalg.norm(np.subtract(c, points[:, index_points])) for c in centroids_iter]
            group_index_iter.append(np.argmin(temp_distances))

        while set(group_index_iter) != set(range(K)):
            centroids_iter = [points[:, np.random.randint(0, points.shape[1])] for k in range(K)]
            group_index_iter = []
            for index_points in range(points.shape[1]):
                temp_distances = [np.linalg.norm(np.subtract(c, points[:, index_points])) for c in centroids_iter]
                group_index_iter.append(np.argmin(temp_distances))
    else:
        # Use farthest neighbors algorithm [https://flothesof.github.io/farthest-neighbors.html]
        centroids_iter = farthestNeighbors(points, K)

    # Present a nice graph if 2D
    if is2D:
        plt.figure()
        plt.subplot(121)
        plt.scatter(points[0], points[1])
        plt.title('Points generated. ' + str(K) + ' clusters chosen with multivariate_normal.')

        plt.subplot(122)
        plt.scatter(points[0], points[1])
        for i in range(len(centroids_iter)):
            plt.scatter(centroids_iter[i][0], centroids_iter[i][1], c='r')
        plt.title('First centroid picked.')
        if randClusters:
            supString = 'Point map generated and Centroids picked randomly'
        else:
            supString = 'Point map generated and Centroids picked by Farthest Neighbors algorithm'
        plt.suptitle(supString)
        plt.show()

    rgb_list = []
    colorSet = False
    for iter in range(iterations):
        # Now create N group assignment indices
        group_index_iter = []
        for index_points in range(points.shape[1]):
            temp_distances = [np.linalg.norm(np.subtract(c, points[:, index_points])) for c in centroids_iter]
            group_index_iter.append(np.argmin(temp_distances))

        if is2D:
            plt.figure()
            for c in range(K):
                clusterSet = np.array([points[:, i] for i in range(points.shape[1]) if c == group_index_iter[i]]).T
                if not colorSet:
                    rgb = (np.random.uniform(), np.random.uniform(), np.random.uniform())
                    rgb_list.append(rgb)
                plt.subplot(121)
                plt.scatter(clusterSet[0], clusterSet[1], c=[rgb_list[c]])
                plt.title('Clusters assigned to Centroids.')
                for i in range(len(centroids_iter)):
                    plt.scatter(centroids_iter[i][0], centroids_iter[i][1], c='r')
            colorSet = True

        #Save data before run improvements
        group_index.append(group_index_iter)
        centroids.append(centroids_iter)

        # Move centroid
        for c in range(K):
            centroids_iter[c] = np.mean(np.array([points[:, i] for i in range(points.shape[1]) if c == group_index_iter[i]]).T, axis=1)

        if is2D:
            for c in range(K):
                clusterSet = np.array([points[:, i] for i in range(points.shape[1]) if c == group_index_iter[i]]).T
                plt.subplot(122)
                plt.scatter(clusterSet[0], clusterSet[1], c=[rgb_list[c]])
                for i in range(len(centroids_iter)):
                    plt.scatter(centroids_iter[i][0], centroids_iter[i][1], c='r')
            plt.title('Centroids moved.')
            supTitleString = 'K-mean Clustering iteration: ' + str(iter+1)
            plt.suptitle(supTitleString)
            plt.show()

        # Find Jclust
        J = 0
        for i, value in enumerate(group_index_iter):
            z = centroids_iter[int(value)]
            J += np.linalg.norm(points.T[i] - z)**2 / len(points[0])
        J_list.append(J)
        print(J)
        if iter > 1 and J_list[iter] == J_list[iter - 1]:
            print('Convergence detected on iteration: ', iter+1)
            return group_index, centroids, J_list
    return group_index, centroids, J_list

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

def evaluateKmeans(N_groupAssignments, K_groupRepVectors, MNIST_Points, MNIST_Labels, K):
    # Group assignments are not related to their indicies!
    N_groupAssignments_list = np.array(N_groupAssignments)
    tempHash = [[0] * 10 for k in range(K)]
    for pointIndex in range(N_groupAssignments_list.shape[0]):
        tempHash[N_groupAssignments_list[pointIndex]][MNIST_Labels[pointIndex]] += 1
    hashmap = [-1] * K
    for c in range(K):
        hashmap[c] = np.argmax(tempHash[c])
    f_hat = [-1] * N_groupAssignments_list.shape[0]
    for pointIndex in range(N_groupAssignments_list.shape[0]):
        f_hat[pointIndex] = hashmap[N_groupAssignments_list[pointIndex]]
    f_hat = np.array(f_hat)

    accuracy = 1 - np.count_nonzero(MNIST_Labels - f_hat) / len(MNIST_Labels)
    print('Accuracy: ', accuracy*100)

    MNIST_ConfusionMatrix = confusionMatrix(MNIST_Labels, f_hat)
    plotCM(MNIST_ConfusionMatrix)

    dim = int(np.ceil((np.sqrt(K))))
    fig, axs = plt.subplots(dim, dim)
    for c in range(dim):
        for x in range(dim):
            if c * dim + x < K:
                image_matrix = np.reshape(K_groupRepVectors[c * dim + x], (28, 28))
                axs[c, x].imshow(image_matrix, cmap="gray")
                titleStr = 'Centroid label: ' + str(hashmap[c * dim + x])
                axs[c, x].set_title(titleStr)
    plt.suptitle('Centroid visualizations and their corresponding labels')
    plt.show()

    for c in range(K):
        nearestNeighbors = []
        remainingPoints = list(MNIST_Points.T)
        tempDists = []
        for i in range(len(remainingPoints)):
            tempDists.append(np.linalg.norm(np.subtract(K_groupRepVectors[c], remainingPoints[i])))
        for j in range(10):
            minDistIndex = int(np.argmin(tempDists))
            nearestNeighbors.append(remainingPoints.pop(minDistIndex))
            tempDists.pop(minDistIndex)

        fig, axs = plt.subplots(5, 2)
        for i in range(5):
            for j in range(2):
                image_matrix = np.reshape(nearestNeighbors[i * 2 + j], (28, 28))
                axs[i, j].imshow(image_matrix, cmap="gray")
                titleStr = 'Nearest Neighbor: ' + str(i * 2 + j + 1)
                axs[i, j].set_title(titleStr)
        plt.suptitle('Nearest neighbors to Centroid label: ' + str(hashmap[c]))
        plt.show()
