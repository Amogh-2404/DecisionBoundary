import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

PROTOTYPE = (0, 0)

def knn(X_train, y_train, X_test, k=3):
    predictions = []
    for test_point in X_test:
        # Compute distances from the test point to all training points
        distances = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))

        # Get indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:k]

        # Get the labels of the k nearest neighbors
        k_nearest_labels = y_train[k_indices]

        # Predict the label by majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        predictions.append(most_common[0][0])

    return np.array(predictions)


def generate_data_and_plot_knn(k_values=[1, 3, 5]):
    num_samples = 100
    class_1_mean = np.array([1.0, 1.0])
    class_2_mean = np.array([-2.0, -2.0])
    class_1_cov = np.array([[0.8, 0.4], [0.4, 0.8]])
    class_2_cov = np.array([[0.8, -0.6], [-0.6, 0.8]])

    X_class_1 = np.random.multivariate_normal(class_1_mean, class_1_cov, num_samples)
    X_class_2 = np.random.multivariate_normal(class_2_mean, class_2_cov, num_samples)

    X_train = np.vstack((X_class_1, X_class_2))
    y_train = np.hstack((np.zeros(num_samples), np.ones(num_samples)))

    # Generate grid for plotting decision boundaries
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    for k in k_values:
        Z = knn(X_train, y_train, np.c_[xx.ravel(), yy.ravel()], k=k)
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.5)
        plt.scatter(X_class_1[:, 0], X_class_1[:, 1], color='red', label='Class 1')
        plt.scatter(X_class_2[:, 0], X_class_2[:, 1], color='blue', label='Class 2')
        plt.title(f'K-Nearest Neighbours with k={k}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.savefig(f'Plots/K-Nearest Neighbours with k={k}.png')
        plt.show()




# Example usage
# generate_data_and_plot_knn([1, 3, 5])

def trainPrototype(X_train,y_train):
    num_samples = 100
    class0protoype = np.array([0.0, 0.0])
    class1protoype = np.array([0.0, 0.0])
    for point, group in zip(X_train, y_train):
        if group == 1:
            class0protoype += np.array(point)
        else:
            class1protoype += np.array(point)
    global PROTOTYPE
    PROTOTYPE = (class0protoype/num_samples, class1protoype/num_samples)

def prototypePrediction(X_train, y_train, test):
    num_samples = 100
    global PROTOTYPE
    class0prototype = np.array(PROTOTYPE[0])
    class1prototype = np.array(PROTOTYPE[1])

    predictions = []
    for point in test:
        dist1 = np.sqrt(np.sum((point - class1prototype)**2))
        dist0 = np.sqrt(np.sum((point - class0prototype)**2))

        if dist1<=dist0:
            predictions.append(1)
        else:
            predictions.append(0)
    return np.array(predictions)


def generate_data_and_plot_lwp():
    num_samples = 100
    class_1_mean = np.array([1.0, 1.0])
    class_2_mean = np.array([-2.0, -2.0])
    class_1_cov = np.array([[0.8, 0.4], [0.4, 0.8]])
    class_2_cov = np.array([[0.8, -0.6], [-0.6, 0.8]])

    X_class_1 = np.random.multivariate_normal(class_1_mean, class_1_cov, num_samples)
    X_class_2 = np.random.multivariate_normal(class_2_mean, class_2_cov, num_samples)

    X_train = np.vstack((X_class_1, X_class_2))
    y_train = np.hstack((np.zeros(num_samples), np.ones(num_samples)))

    # Generate grid for plotting decision boundaries
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    trainPrototype(X_train,y_train)

    Z = prototypePrediction(X_train, y_train, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.5)
    plt.scatter(X_class_1[:, 0], X_class_1[:, 1], color='red', label='Class 1')
    plt.scatter(X_class_2[:, 0], X_class_2[:, 1], color='blue', label='Class 2')
    plt.title(f'Learning with Prototype')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig('Plots/Learning with Prototype.png')
    plt.show()


# generate_data_and_plot_lwp()

def generate_data_and_plot_lwp_chi():
    k1 = 7
    k2 = 10
    num_samples = 100
    points_class_1 = np.random.chisquare(k1, (num_samples,2))
    points_class_2 = np.random.chisquare(k2, (num_samples, 2))


    X_train = np.vstack((points_class_1,points_class_2))
    y_train = np.hstack((np.zeros(num_samples), np.ones(num_samples)))

    # Generate grid for plotting decision boundaries
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    trainPrototype(X_train, y_train)

    Z = prototypePrediction(X_train, y_train, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.5)
    plt.scatter(points_class_1[:, 0], points_class_1[:, 1], color='red', label='Class 1')
    plt.scatter(points_class_2[:, 0], points_class_2[:, 1], color='blue', label='Class 2')
    plt.title(f'Learning with Prototype')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig('Plots/Learning with Prototype chi.png')
    plt.show()


def generate_data_and_plot_knn_chi(k_values):
    k1 = 7
    k2 = 10
    num_samples = 100
    points_class_1 = np.random.chisquare(k1, (num_samples,2))
    points_class_2 = np.random.chisquare(k2, (num_samples, 2))

    X_train = np.vstack((points_class_1,points_class_2))
    y_train = np.hstack((np.zeros(num_samples), np.ones(num_samples)))

    # Generate grid for plotting decision boundaries
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))


    for k in k_values:
        Z = knn(X_train, y_train, np.c_[xx.ravel(), yy.ravel()], k=k)
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.5)
        plt.scatter(points_class_1[:, 0], points_class_1[:, 1], color='red', label='Class 1')
        plt.scatter(points_class_2[:, 0], points_class_2[:, 1], color='blue', label='Class 2')
        plt.title(f'K-Nearest Neighbours with k={k}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.savefig(f'Plots/K-Nearest Neighbours chi with k={k}.png')
        plt.show()



generate_data_and_plot_knn(k_values = [1,3,5,10])
generate_data_and_plot_knn_chi(k_values = [1,3,5,10])
generate_data_and_plot_lwp()
generate_data_and_plot_lwp_chi()