#!/usr/bin/python

import sys
import pickle
import math
from time import time
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

t0 = time()

features_list = ['poi',
                 'salary',
                 'to_messages',
                 'deferral_payments',
                 'total_payments',
                 'loan_advances',
                 'bonus',
                 'restricted_stock_deferred',
                 'total_stock_value',
                 'shared_receipt_with_poi',
                 'long_term_incentive',
                 'exercised_stock_options',
                 'from_messages',
                 'other',
                 'from_this_person_to_poi',
                 'director_fees',
                  'deferred_income',
                 'expenses',
                 'restricted_stock',
                 'from_poi_to_this_person',
                 'log_total_payments',
                 'log_total_stock_value',
                 'proportion_email_from_poi',
                 'proportion_email_to_poi'
                 ]
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)
#data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
    
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Create a new feature adding the logs of total payments and total stock value
### Create a new feature: proportion of email received by POI
### Create a new feature: proportion of email sent to POI
for person in my_dataset:
    if (isinstance(my_dataset[person]['total_payments'], int) and
        my_dataset[person]['total_payments'] > 0 and
        isinstance(my_dataset[person]['total_stock_value'], int) and
        my_dataset[person]['total_stock_value'] >0):
        my_dataset[person]['log_total_payments'] = math.log(my_dataset[person]['total_payments'], 10)
        my_dataset[person]['log_total_stock_value'] = math.log(my_dataset[person]['total_stock_value'], 10)
    else:
        my_dataset[person]['log_total_payments'] = 0
        my_dataset[person]['log_total_stock_value'] =0
        
    if (isinstance(my_dataset[person]['from_messages'], int) and
        isinstance(my_dataset[person]['to_messages'], int) and
        isinstance(my_dataset[person]['from_poi_to_this_person'], int) and
        isinstance(my_dataset[person]['from_this_person_to_poi'], int)):
        my_dataset[person]['proportion_email_from_poi'] = 1.0 * my_dataset[person]['from_poi_to_this_person'] / my_dataset[person]['to_messages'] 
        my_dataset[person]['proportion_email_to_poi'] = 1.0 * my_dataset[person]['from_this_person_to_poi'] / my_dataset[person]['from_messages'] 
        
    else:
        my_dataset[person]['proportion_email_from_poi'] = 0
        my_dataset[person]['proportion_email_to_poi'] = 0

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Scale features
scaler = MinMaxScaler(copy=False)
features = scaler.fit_transform(features)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 4: Try a varity of classifiers
### Task 5: Tune your classifier to achieve better than .3 precision and recall 

### Define pipeline and parameters to pass to the gridsearch function

n_features = [8, 10, 12]
n_dimensions = [2, 4, 6]

pipe = Pipeline([
        ("select_feat", SelectKBest()),
        ("reduce_dim", None),
        ("clf", DecisionTreeClassifier())
        ])
    
param_grid = [
                {
                "select_feat" : [SelectKBest()],
                "select_feat__k" : n_features,
                "reduce_dim" : [PCA()],
                "reduce_dim__n_components" : n_dimensions,
                "clf" : [GaussianNB()]
                },
                {
                "select_feat" : [SelectKBest()],
                "select_feat__k" : n_features,
                "reduce_dim" : [PCA()],
                "reduce_dim__n_components" : n_dimensions,
                "clf" : [DecisionTreeClassifier()],
                "clf__min_samples_split" : [2 ,4 ,8],
                "clf__max_depth" : [2, 4, 8]
                },
                {
                "select_feat" : [SelectKBest()],
                "select_feat__k" : n_features,
                "reduce_dim" : [PCA()],
                "reduce_dim__n_components" : n_dimensions,
                "clf" : [RandomForestClassifier()],
                "clf__n_estimators" : [5, 10, 50, 100],
                "clf__max_depth" : [2, 4, 8]
                },
                {
                "select_feat" : [SelectKBest()],
                "select_feat__k" : n_features,
                "reduce_dim" : [PCA()],
                "reduce_dim__n_components" : n_dimensions,
                "clf" : [AdaBoostClassifier()],
                "clf__n_estimators" : [5, 10, 50, 100],
                },
                {
                "select_feat" : [SelectKBest()],
                "select_feat__k" : n_features,
                "reduce_dim" : [PCA()],
                "reduce_dim__n_components" : n_dimensions,
                "clf" : [SVC()],
                "clf__C" : [1, 10, 100, 1000],
                "clf__gamma" : [1, 10, 100, 1000]
                }
            ]

### Set scoring target
scoring = {"Precision" : "precision", "Recall" : "recall", "F1" : "f1"}

### Perform grid search
grid_search = GridSearchCV(pipe, cv=8, param_grid=param_grid, scoring=scoring, refit="F1",
                           n_jobs=4)

print "-----------------"
print "Pipeline:", [name for name, _ in pipe.steps]

### Select the best estimator
clf = grid_search.fit(features_train, labels_train).best_estimator_

print "-----------------"
print "Best Pipeline:"
print clf

### Display selected features
features_kbest =[]
features_kbest = clf.named_steps["select_feat"].fit(features_train, labels_train).get_support()
k_features = clf.named_steps["select_feat"].fit(features_train, labels_train).get_params()['k']
print "-----------------"
print "Selection of the {0} best features:".format(k_features)
for i in range(len(features_kbest)):
    if features_kbest[i]:
        print "   ", features_list[i+1]

### Evaluate scores
labels_pred = clf.predict(features_test)
best_index = grid_search.best_index_
print "-----------------"
#print "Actual POI: ", sum(labels_test)
#print confusion_matrix(labels_test, labels_pred)
print "Mean Test Precision: ",  grid_search.cv_results_['mean_test_Precision'][best_index]
print "Mean Test Recall: ", grid_search.cv_results_['mean_test_Recall'][best_index]
print "Mean Test F1: ", grid_search.cv_results_['mean_test_F1'][best_index]

### Task 6: Dump your classifier, dataset, and features_list so anyone can

dump_classifier_and_data(clf, my_dataset, features_list)

print "Total runtime: {:0.2f} s".format(time() - t0)