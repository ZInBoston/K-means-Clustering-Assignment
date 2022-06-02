"""
Zachary Robinson
Class: CS 677
Date: April 25, 2022
"""
import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

# import data..
seeds = pd.read_csv('seeds_dataset.csv', delimiter='\t', header=None)
seeds.rename(columns={0: 'A', 1: 'P', 2: 'C', 3: 'Length_kernel', 4: 'Width', 5: 'Asymmetry', 6: 'Length_groove', 7: 'Class'}, inplace=True)
cols = ['A', 'P', 'C', 'Length_kernel', 'Width', 'Asymmetry', 'Length_groove', 'Class']

"""
Homework Problem 1
Implementing Kernel SVM methods on data
"""
# BUID last = 2, R = 0
seeds_1 = seeds[(seeds.Class == 1) | (seeds.Class == 2)]
seeds_train_1, seeds_test_1  = sklearn.model_selection.train_test_split(seeds_1, train_size=0.5, random_state=123)

def findings(prediction_data, test_data):
    tp, fp, tn, fn = 0, 0, 0, 0
    j = 0
    while j < len(test_data):
        if (prediction_data[j] == test_data[j]) & (test_data[j] == 2):
            tp +=1
            j += 1
        elif (prediction_data[j] != test_data[j]) & (test_data[j] == 2):
            fp += 1
            j += 1
        elif (prediction_data[j] == test_data[j]) & (test_data[j] == 1):
            tn += 1
            j += 1
        else:
            fn += 1
            j += 1
    tpr = round((tp/(tp + fn)) * 100, 2)
    tnr = round((tn/(tn + fp)) * 100, 2)
    return [tp, fp, tn, fn, tpr, tnr]

def svm_function(train_data, test_data, svm_type):
    X = train_data[cols].values
    scaler = StandardScaler()
    scaler.fit(X)
    Y = train_data['Class'].values

    svm_classifier = svm.SVC(kernel=svm_type)
    svm_classifier.fit(X, Y)
    Ypredict = svm_classifier.predict(test_data[cols].values)
    accuracy = round(svm_classifier.score(X, Y) * 100, 2)
    print("Accuracy",svm_type, ":", accuracy, '%')
    print("Rates", svm_type, findings(test_data['Class'].values, Ypredict))
    cf = confusion_matrix(test_data['Class'].values, Ypredict)
    sb.heatmap(cf.T, square = True, annot = True, fmt = 'd', cbar = True)
    plt.xlabel('True Class')
    plt.ylabel('Predicted Class')
    plt.title(svm_type + ' Confusion Matrix')
    plt.show()



svm_function(seeds_train_1, seeds_test_1, 'linear')
svm_function(seeds_train_1, seeds_test_1, 'rbf')
svm_function(seeds_train_1, seeds_test_1, 'poly')

"""
Homework Problem 2
Using kNN to predict class data
"""

def nearest_neighbor(k_value, train_data, test_data):
    X = train_data[cols].values
    Y = train_data['Class'].values
    knn_classifier = KNeighborsClassifier(n_neighbors=k_value)
    knn_classifier.fit(X, Y)
    predicted = knn_classifier.predict(test_data[cols].values)
    accuracy = round(accuracy_score(Y, predicted) * 100, 2)
    print("Accuracy kNN", accuracy, "%")
    print("Rates Knn:", findings(test_data['Class'].values, predicted))
    cf = confusion_matrix(test_data['Class'].values, predicted)
    sb.heatmap(cf.T, square = True, annot = True, fmt = 'd', cbar = True)
    plt.xlabel('True Class')
    plt.ylabel('Predicted Class')
    plt.title('kNN Confusion Matrix')
    plt.show()

nearest_neighbor(6, seeds_train_1, seeds_test_1)

"""
Homework Problem 3
Using k-means clustering to predict class data
"""
seeds_train, seeds_test  = sklearn.model_selection.train_test_split(seeds, train_size=0.5, random_state=123)
sse = []
for k in range(1, 9):
    kmeans_classifier = KMeans(n_clusters=k, init='random', random_state=123)
    y_means = kmeans_classifier.fit(seeds_train[cols])
    sse.append(kmeans_classifier.inertia_)
plt.plot(range(1, 9), sse, '-b')
plt.xlabel('k')
plt.ylabel('Intertia (SSE)')
plt.title('Knee Plot')
plt.show()
print("Best K Value:", sse.index(min(sse)) + 1)

random.seed(123)
rand = random.sample(cols, 2)
kmeans = KMeans(n_clusters=(sse.index(min(sse)) + 1), random_state=123)
X = seeds_train[rand]
kmeans.fit(X)
y_kmeans = kmeans.predict(seeds_test[rand])

plt.scatter(seeds_test[rand[0]], seeds_test[rand[1]], c=y_kmeans, s=50, cmap='viridis')
plt.xlabel(rand[0] + ' (Area)')
plt.ylabel(rand[1] + ' (Compactness)')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.8)
plt.show()

seeds_test['Kmeans'] = y_kmeans
kmeans_pivot = seeds_test.pivot_table(seeds_test, columns=["Kmeans", "Class"], aggfunc='size', )
for i in range(0, 8):
    print("Centroid for k = ", i+1, centers[i])
    print(kmeans_pivot.loc[i])

"""
Homework Problem 4
Computing euclidean distance with custom clusters
"""
x_arr = []
i = 0
while i < len(seeds):
    x_arr.append([seeds['A'].loc[i], seeds['C'].loc[i]])
    i += 1

seeds['Points'] = x_arr

def calc_euclidean(df, cluster_1, cluster_2, cluster_3):
    j = 0
    predict_class = []
    while j < len(df):
        a = np.sqrt(np.sum(np.square(df['Points'].loc[j] - cluster_1)))
        b = np.sqrt(np.sum(np.square(df['Points'].loc[j] - cluster_2)))
        c = np.sqrt(np.sum(np.square(df['Points'].loc[j] - cluster_3)))
        if (a > b) & (a > c):
            predict_class.append(2)
            j += 1
        elif (b > a) & (b > c):
            predict_class.append(3)
            j += 1
        else:
            predict_class.append(1)
            j += 1
    return predict_class

new_prediction = calc_euclidean(seeds, centers[0], centers[6], centers[7])
accuracy = round(accuracy_score(seeds['Class'], new_prediction) * 100, 2)
print("Accuracy of new three-cluster classifier:", accuracy, "%")

"""
Homework Problem 5
Comparing new classifier to SVM
"""
new_pred_drp, new_pred_test = sklearn.model_selection.train_test_split(seeds, train_size=0.5, random_state=123)
km_cl = []
for i in y_kmeans:
    if (y_kmeans[i] == 0) | (y_kmeans[i] == 3) | (y_kmeans[i] == 6):
        km_cl.append(3)
    elif (y_kmeans[i] == 1) | (y_kmeans[i] == 4) | (y_kmeans[i] == 7):
        km_cl.append(2)
    else:
        km_cl.append(1)
print(km_cl)
accuracy = round(accuracy_score(km_cl, new_pred_test['Class']) * 100, 2)
print("Accuracy of new three-cluster classifier when compared to SVM:", accuracy, "%")

cf = confusion_matrix(km_cl, new_pred_test['Class'])
sb.heatmap(cf.T, square=True, annot=True, fmt='d', cbar=True)
plt.xlabel('SVM Classifier')
plt.ylabel('New Custom Classifier')
plt.title('Confusion Matrix New Classifier vs. SVM')
plt.show()