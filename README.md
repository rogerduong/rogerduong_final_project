# Identify Fraud from Enron Email

- Roger Duong
- 12 Dec 2017
- V1

This document is organized as follows:

1. Dataset and Question
2. Feature Selection/Engineering
3. Algorithm
4. Evaluation

## References

- `sklearn` documentation

# 1 Dataset and Question

## 1.1 Data Exploration

The purpose of this project is to identify the Persons of Interest (POI) in the Enron scandal, by means of a machine learning algorithm applied to the email dataset.

- Number of data points in dataset: 146
- Initial features per person: 21
- Number of POI: 18

The top 3 features with the lowest number of values are:

- `'director_fees'`: 17
- `'loan_advances'`: 4
- `'restricted_stock_deferred'`: 18

## 1.2 Outlier Investigation

A quick scan through the keys of the `data_dict` revealed the key `'THE TRAVEL AGENCY IN THE PARK'`, which looked like a good candidate for an outlier. However reading through the foodnotes of the `Insider pay` spreadsheet showed that it was a legitimate entry. Therefore this key was kept in the dictionary.

The entry `'TOTAL'` was removed because it did not correspond to any person.

# 2 Feature Selection/Engineering

The following features were added:

- Financial features:

  - the logarithm of total payments, to attenuate the variance between the values.
  - the logarithm of total stock value, for the same reason.

- Email features:

  - the proportion of email received by POI, to account for variations in the total number of emails received.
  - the proportion of email sent to POI, for the same reason.

## 2.1 Feature Scaling

Features are scaled using the `MinMaxScaler` function available in sklearn.

## 2.2 Feature Selection

Features are selected using the `SelectKBest` function, in the first step of the `GridSearchCV` step. Some of the features engineered previously are kept, some are not.

# 3 Algorithm

## 3.1 Algorithm Choice

The project uses the `GridSearchCV` and `pipeline` functions to sequentially apply and try a list of transforms and a final estimator, in order to select the best sequence of transforms and their associated parameters.

The pipeline steps are laid out as follows:

- `select_feat`: select the k-best features
- `reduce_dim`: reduce the dimensions, using PCA
- `clf`: classify the labels, using a variety of classifiers: Naive Bayes, Decision Tree, Random Forest, AdaBoost

## 3.2 Algorithm Validation

The following lists set out the various parameter values to try: `n_features`(number of features to select), `n_dimensions` (number of dimensions to reduce to, using PCA) and `param_grid` (parameters of the classifier).

The `GridSearchCV` is set to perform a 8-fold cross-validation.

# 4 Evaluation

## 4.1 Evaluation Metrics

The `GridSearchCV` is set to evaluate the classifier performance and recall score, as set in the dictionary `scoring`.

The precision measures the proportion of true positive POI over all the persons classified as POI by the algorithm (true positive + false positive). The recall measures the true positive POI over all the actual number of POI (true positive + false negative). The F1 score is the weighted average between the precision and recall. The best pipeline is chosen for the highest F1 score.

The step of evaluation and validation is important to assess the performance of the algorithm on test sets. This is done in order to verify the algorithm does not overfit, and that is generalizes satisfactorily to another dataset than the training dataset for which it was tuned.

## 4.2 Algorithm Performance

Using the `tester.py` script, the algorithm performance show precision and recall both above 0.3.
