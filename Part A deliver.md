Algorithm Cards (1-4)
=== Logistic Regression (LR) === When: Baseline model, need probability estimates, or linear relationships. Params: C, penalty, solver Pros: Fast, highly interpretable, no tuning required for simple cases. Cons: Assumes linearity, sensitive to outliers. Code: LogisticRegression(C=1.0).fit(X_train_scaled, y_train)
=== K-Nearest Neighbors (KNN) === When: Small/Medium datasets with non-linear boundaries. Params: n_neighbors, metric, weights Pros: Simple, makes no assumptions about data distribution. Cons: Computationally expensive at inference, sensitive to scaling. Code: KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)
=== Support Vector Machine (SVM) === When: High-dimensional data, clear margins of separation. Params: C, kernel, gamma Pros: Effective in high dimensions, versatile kernels (RBF/Poly). Cons: Not suitable for large datasets, difficult to interpret. Code: SVC(kernel='rbf', C=1.0).fit(X_train_scaled, y_train)
=== Decision Tree (DT) === When: Need visual interpretability, rule-based logic. Params: max_depth, min_samples_split, criterion Pros: Easy to visualize, requires little data prep (no scaling needed). Cons: High variance, prone to overfitting. Code: DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)

Algorithm Cards (5-8)
=== Random Forest (RF) === When: General purpose tabular data, need robustness. Params: n_estimators, max_depth, max_features Pros: Handles high dimensionality, reduces variance (hard to overfit). Cons: Can be slow to train many trees, large model size. Code: RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
=== Gradient Boosting (GBM) === When: Competitive predictive performance on tabular data. Params: n_estimators, learning_rate, max_depth Pros: High accuracy, handles complex non-linear patterns. Cons: Sensitive to noise/overfitting, requires careful tuning. Code: GradientBoostingClassifier(n_estimators=100).fit(X_train, y_train)
=== AdaBoost (Ada) === When: Boosting a weak learner (like stumps) for performance gains. Params: n_estimators, learning_rate, base_estimator Pros: Simple to implement, good for focusing on hard examples. Cons: Sensitive to noisy data and outliers. Code: AdaBoostClassifier(n_estimators=50).fit(X_train, y_train)
=== Gaussian Naive Bayes (NB) === When: Text classification or quick baseline for continuous data. Params: var_smoothing, priors Pros: Extremely fast, works well with small training data. Cons: Assumes independence between all features. Code: GaussianNB().fit(X_train_scaled, y_train)

### Model Ranking and Recommendation

Based on the cross-validation performance on the scaled dataset, here is the ranking:

1. **KNN (~94.5%)** - Top performer. Benefit: Non-parametric nature captures the complex decision boundaries of the 15 informative features effectively.
2. **SVM (~94.1%)** - Near-identical performance to KNN. Effective in high-dimensional spaces (20 features).
3. **RF (~91.2%)** - Strong performance through ensemble bagging, though slightly behind instance-based methods here.
4. **GBM (~89.2%)** - Solid performance, likely needs hyperparameter tuning to surpass RF.
5. **LR (~80.4%)** - Good baseline, but the 15.0% gap vs KNN suggests significant non-linear relationships.
6. **Ada (~80.0%)** - Performed similarly to LR, potentially limited by default weak learners.
7. **NB (~78.2%)** - Limited by the independence assumption in a dataset with 15 informative features.
8. **DT (~78.0%)** - Lowest performer due to high variance and lack of ensemble averaging.

**Recommendation:**
I recommend using **K-Nearest Neighbors (KNN)** or **Support Vector Machine (SVM)** for this specific dataset. Since the feature-to-sample ratio is relatively small (20:1000) and the features are scaled, these models are able to exploit the local structure and geometric boundaries of the data more effectively than the tree-based or linear counterparts.
