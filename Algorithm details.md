1. Logistic Regression (LR)
Logistic Regression is a fundamental statistical model used for binary classification. It models the probability of a default class by fitting data to a logistic curve (sigmoid function). It is highly interpretable, fast to train, and works well when the relationship between features and the target is approximately linear.

Key Parameters:

penalty: Used to specify the norm used in the penalization (e.g., 'l1', 'l2', 'elasticnet').
C: Inverse of regularization strength; smaller values specify stronger regularization.
Code Example:

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0, penalty='l2')
model.fit(X_train_scaled, y_train)
preds = model.predict(X_test_scaled)
print(model.score(X_test_scaled, y_test))

2. K-Nearest Neighbors (KNN)
KNN is a non-parametric, instance-based learning algorithm. It classifies a data point based on how its neighbors are classified, typically using a majority vote of the 'k' nearest points in the feature space. It is simple to implement but can be computationally expensive as the dataset grows because it requires calculating distances to every training point.

Key Parameters:

n_neighbors: Number of neighbors to use (default is 5).
metric: The distance metric to use for the tree (e.g., 'minkowski', 'euclidean').
Code Example:

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(X_train_scaled, y_train)
labels = knn.predict(X_test_scaled)
print(knn.score(X_test_scaled, y_test))

3. Support Vector Machines (SVM)
SVMs find the hyperplane that best separates different classes by maximizing the margin between the closest points (support vectors) of the classes. Through the 'kernel trick,' SVMs can handle high-dimensional and non-linear data by projecting it into a higher-dimensional space where a linear separation is possible.

Key Parameters:

C: Regularization parameter that trades off margin maximization and classification error.
kernel: Specifies the kernel type used ('linear', 'poly', 'rbs', 'sigmoid').
Code Example:

from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train_scaled, y_train)
predictions = svm.predict(X_test_scaled)
print(svm.score(X_test_scaled, y_test))

4. Random Forest (RF)
Random Forest is an ensemble learning method that constructs a multitude of decision trees during training. For classification tasks, it outputs the mode of the classes of the individual trees. By training trees on different subsets of data and features (bagging), it significantly reduces the risk of overfitting compared to single decision trees.

Key Parameters:

n_estimators: The number of trees in the forest.
max_depth: The maximum depth of the individual trees.
Code Example:

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=None)
rf.fit(X_train_scaled, y_train)
results = rf.predict(X_test_scaled)
print(rf.score(X_test_scaled, y_test))

5. Gradient Boosting Machine (GBM)
Gradient Boosting builds an ensemble of trees sequentially, where each new tree attempts to correct the errors made by the previous trees. It uses gradient descent to minimize a loss function, making it a very powerful and flexible model that often achieves state-of-the-art results on tabular data, though it requires careful tuning.

Key Parameters:

learning_rate: Shrinks the contribution of each tree by its value.
n_estimators: The number of boosting stages to perform.
Code Example:

from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
gbm.fit(X_train_scaled, y_train)
out = gbm.predict(X_test_scaled)
print(gbm.score(X_test_scaled, y_test))


