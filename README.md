# Identify Fraud from Enron Email

- Roger Duong
- 27 Dec 2017
- V3

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
- Initial features per person: 19
- Number of POI: 18

With only 18 POI to detect in a population of 146, problems can arise in the classification algorithm due to class imbalance.

The top 3 features with the lowest number of values are:

- `'director_fees'`: 17
- `'loan_advances'`: 4
- `'restricted_stock_deferred'`: 18

## 1.2 Outlier Investigation

The entry `'TOTAL'` was removed because it did not correspond to any person.

Outliers were identified and removed by using the `remove_outliers_nan` function. It removes 23 the entries which have a number of NaN above a user-defined threshold, 15 by default.

After the removal of all outliers, the dataset becomes:

- Number of data points in dataset: 122
- Initial features per person: 19
- Number of POI: 18

# 2 Feature Selection/Engineering

Before the addition of engineered features, the algorithm returns the following precision and recall scores, which are lower than the objective of 0.3:

```python
Pipeline(memory=None,
     steps=[('select_feat', SelectKBest(k=7, score_func=<function f_classif at 0x1a0abe28c0>)), ('reduce_dim', PCA(copy=True, iterated_power='auto', n_components=2, random_state=42,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth...        min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best'))])
        Accuracy: 0.78823       Precision: 0.29415      Recall: 0.26900 F1: 0.28101     F2: 0.27368
        Total predictions: 13000        True positives:  538    False positives: 1291   False negatives: 1462   True negatives: 9709
```

In order to improve the representation of the dataset, the following features were added:

- Financial features:

  - the logarithm of total payments, in an attempt to attenuate the variance between the values. A quick visualization of the distribution of the dataset shows a highly skewed distribution.
  - the logarithm of total stock value, for the same reason.

- Email features:

  - the proportion of email received by POI, to reflect the number of emails relative to the total number of emails received, rather than an absolute amount.
  - the proportion of email sent to POI, for the same reason.

The function `find_classifier_POI` returns a list of all features and their ANOVA f-values (`f_classif`), indicating with a "[x]" the features that are eventually selected by the `SelectKBest` function.

```python
Selection of the 4 best features:
Features with [x] are selected
   [x] salary, (12.31)
   [ ] to_messages, (0.66)
   [ ] deferral_payments, (0.44)
   [ ] total_payments, (6.68)
   [ ] loan_advances, (5.95)
   [x] bonus, (15.21)
   [ ] restricted_stock_deferred, (0.09)
   [x] total_stock_value, (18.74)
   [ ] shared_receipt_with_poi, (5.37)
   [ ] long_term_incentive, (6.82)
   [x] exercised_stock_options, (19.52)
   [ ] from_messages, (0.33)
   [ ] other, (2.99)
   [ ] from_this_person_to_poi, (1.49)
   [ ] director_fees, (0.76)
   [ ] deferred_income, (8.80)
   [ ] expenses, (5.40)
   [ ] restricted_stock, (6.62)
   [ ] from_poi_to_this_person, (3.19)
   [ ] log_total_payments, (5.28)
   [ ] log_total_stock_value, (6.11)
   [ ] proportion_email_from_poi, (1.51)
   [ ] proportion_email_to_poi, (11.59)
```

In the chosen algorithm, the SelectKBest function selected 4 features out of the 23 available. None of the features engineered were included in the selection. The engineered features display scores significantly lower than the features selected.

The original dataset is stored in `labels_orig` and `features_orig`, while the enriched dataset with the engineered features is stored in `labels` and `features`.

After the removal of all outliers, the addition of features, the dataset becomes:

- Number of data points in dataset: 122
- Initial features per person: 23
- Number of POI: 18

The function `find_classifier_POI` chooses the best algorithm for both the original and enriched dataset. The kernel density estimates of precision and recall scores are plotted for each datasets, for each classifiers, and for all the parameters tested. The results show that the recall scores are generally better for the enriched dataset than for the original dataset.

## 2.1 Feature Scaling

Features are scaled using the `MinMaxScaler` function available in sklearn.

## 2.2 Feature Selection

Features are selected using the `SelectKBest` function, in the first step of the `GridSearchCV` step.

Feature selection is performed in two steps as part of the grid search:

1. SelectKBest features, using the following values to test for the number of features `k`: `range(4, 8)`.
2. Reduce dimensions using PCA using the following values for the number of components `n_components` : `range(2, 4)`.

# 3 Algorithm

## 3.1 Algorithm Choice

In this step, we choose the best algorithm and tune its parameters. This consists in:

- trying several classifiers
- defining a parameter space for the parameters we want to try
- setting a method for searching and sampling the parameter space
- determining a cross-validation scheme
- deciding a score function to evaluate the algorithm.

This step will use the `GridSearchCV` function to perform this tuning, in order to automatically pick out the combination of classifier parameters that will yield the best score. The advantage of `GridSearchCV` is that it tries out an exhaustive list of possibilities. Trying out a large parameter space reduces the likelihood of missing out a combination of parameters that would give the best performance, which could easily be the case with manual search.

The pipeline steps are laid out as follows:

- `select_feat`: select the k-best features
- `reduce_dim`: reduce the dimensions, using PCA
- `clf`: classify the labels, using a variety of classifiers: Decision Tree, Random Forest, AdaBoost

The Decision Tree classifier is tuned with the following parameters:

```
"clf__min_samples_split" : range(2, 5),
"clf__min_samples_leaf": range(1, 4),
"clf__random_state" : [42]
```

The Random Forest classifier is tuned with the following parameters:

```
"clf__n_estimators" : [5, 10, 15],
"clf__min_samples_split" : range(2, 5),
"clf__min_samples_leaf": range(1, 4),
"clf__random_state" : [42]
```

The AdaBoost classifier is tuned with the following parameters:

```
"clf__n_estimators" : range(8,12),
"clf__learning_rate" : [0.05, 0.1],
"clf__algorithm" : ["SAMME", "SAMME.R"],
"clf__random_state" : [42],
"clf__base_estimator__min_samples_split" : range(2, 4),
"clf__base_estimator__min_samples_leaf": range(1, 4),
"clf__base_estimator__random_state" : [42]
```

The kernel density of precision and recall scores for each classifiers are plotted by the function `plot_grid_search`, allowing for a quick visualization of the scores distribution, and the effect of each parameter tuning on the scores.

## 3.2 Algorithm Validation

Cross validation is performed inside `GridSearchCV` using a `StratifiedShuffleSplit` cross validator, applied to the whole dataset. This cross validator was chosen because of the dataset is imbalanced: 23 POI for 146 entries.

The `StratifiedShuffleSplit` allows to preserve the class structure of approximately 1 POI of 5 people in each of the folds returned.

# 4 Evaluation

## 4.1 Evaluation Metrics

The precision measures the proportion of true positive POI over all the persons classified as POI by the algorithm (true positive + false positive). The recall measures the true positive POI over all the actual number of POI (true positive + false negative). The F1 score is the weighted average between the precision and recall. The best pipeline is chosen for the highest F1 score.

The step of evaluation and validation is important to assess the performance of the algorithm on test sets. This is done in order to verify the algorithm does not overfit, and that is generalizes satisfactorily to another dataset than the training dataset for which it was tuned.

## 4.2 Algorithm Performance

Using the `tester.py` script, the algorithm performance shows the following results:

For the DecisionTreeClassifier, we have the following results:

```python
Pipeline(memory=None,
     steps=[('select_feat', SelectKBest(k=4, score_func=<function f_classif at 0x1a0abe28c0>)), ('reduce_dim', PCA(copy=True, iterated_power='auto', n_components=2, random_state=42,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth...        min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best'))])
        Accuracy: 0.80415       Precision: 0.37025      Recall: 0.38950 F1: 0.37963     F2: 0.38549
        Total predictions: 13000        True positives:  779    False positives: 1325   False negatives: 1221   True negatives: 9675
```

The Random Forest classifier yields a lower F1 and recall score:

```python
Pipeline(memory=None,
     steps=[('select_feat', SelectKBest(k=4, score_func=<function f_classif at 0x1a0abe28c0>)), ('reduce_dim', PCA(copy=True, iterated_power='auto', n_components=2, random_state=42,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='...estimators=5, n_jobs=1,
            oob_score=False, random_state=42, verbose=0, warm_start=False))])
        Accuracy: 0.83369       Precision: 0.43760      Recall: 0.28400 F1: 0.34445     F2: 0.30544
        Total predictions: 13000        True positives:  568    False positives:  730   False negatives: 1432   True negatives: 10270
```

The AdaBoost classifier yields a slightly lower precision, recall and F1 scores:

```python
Pipeline(memory=None,
     steps=[('select_feat', SelectKBest(k=4, score_func=<function f_classif at 0x1a0abe28c0>)), ('reduce_dim', PCA(copy=True, iterated_power='auto', n_components=2, random_state=42,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', AdaBoostClassifier(algorithm='SAMME',
          base_estimator=Decisi...=42,
            splitter='best'),
          learning_rate=0.05, n_estimators=12, random_state=42))])
        Accuracy: 0.80208       Precision: 0.36556      Recall: 0.38950 F1: 0.37715     F2: 0.38446
        Total predictions: 13000        True positives:  779    False positives: 1352   False negatives: 1221   True negatives: 9648
```

The comparison of the three classifiers scores result in the DecisionTreeClassifier being selected, thanks to higher scores in precision, recall and F1.

The AdaBoostClassifier ranks second, with scores marginally lower than the DecisionTreeClassifier. Both classifiers meet the objective of scoring higher than 0.3.

The RandomForestClassifier exhibits a recall score below the objective of 0.3, and therefore cannot be retained for our analysis.

# Conclusion

The DecisionTreeClassifier is selected as the best classifier for our POI analysis.
