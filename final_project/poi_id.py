#!/usr/bin/python

import sys
import pickle
import math
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

t0 = time()

features_list = []

features_list_orig = ['poi',
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
                 ]

features_list = features_list_orig

n_features = range(4, 8)
n_dimensions = range(2, 4)

### Task 2: Remove outliers

def remove_outliers_nan(data_dict, threshold=15):
    #Remove entries with more than 15 unusable features

    persons_to_remove = []
    for person in data_dict:
        empty_entries = 0
        for key in data_dict[person].keys():
            if (data_dict[person][key] == 'NaN') :
                empty_entries += 1
        if (empty_entries > threshold):
            persons_to_remove.append(person)
            
    for person in persons_to_remove:
        data_dict.pop(person, 0)
    
    print "-----------------"
    print "The following {0} entries were removed:".format(len(persons_to_remove))
    print persons_to_remove
    print "Dataset size: {0} entries".format(len(data_dict))
    
    return data_dict
#
#def remove_outliers_iqr(features, labels):
#    # Remove outliers using IQR rule
#    # http://stamfordresearch.com/outlier-removal-in-python-using-iqr-rule/
#    q75, q25 = np.percentile(features, [75 ,25])
#    iqr = q75 - q25
#     
#    min_thr = q25 - (iqr*1.5)
#    max_thr = q75 + (iqr*1.5)
#    
#    select_index = []
#    for idx, feat in enumerate(features):
#        if not(any(feat < min_thr) or any(feat > max_thr)):
#            select_index.append(idx)
#    
#    features_cleaned = [features[i] for i in select_index]
#    labels_cleaned = [features[i] for i in select_index]
#    
#    return features_cleaned, labels_cleaned

### Task 3: Create new feature(s)
def add_features(my_dataset):

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
    
    features_list.extend(['log_total_payments',
                     'log_total_stock_value',
                     'proportion_email_from_poi',
                     'proportion_email_to_poi'])
    
    return my_dataset
    
def find_classifier_POI(features, labels):

    ### Scale features
    scaler = MinMaxScaler(copy=False)
    features = scaler.fit_transform(features)
 
    ### Cross Validation
    sss = StratifiedShuffleSplit(n_splits=20, random_state = 42)   
    cv = sss.split(features, labels)
    
    ### Define pipeline and parameters to pass to the gridsearch function
    
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
                    "reduce_dim__random_state" : [42],
                    "clf" : [DecisionTreeClassifier()],
                    "clf__min_samples_split" : range(2, 5),
                    "clf__min_samples_leaf": range(1, 4),
                    "clf__random_state" : [42]
                    },
#                    {
#                    "select_feat" : [SelectKBest()],
#                    "select_feat__k" : n_features,
#                    "reduce_dim" : [PCA()],
#                    "reduce_dim__n_components" : n_dimensions,
#                    "reduce_dim__random_state" : [42],
#                    "clf" : [RandomForestClassifier()],
#                    "clf__n_estimators" : [5, 10, 15],
#                    "clf__min_samples_split" : range(2, 5),
#                    "clf__min_samples_leaf": range(1, 4),
#                    "clf__random_state" : [42]
#                    },
#                    {
#                    "select_feat" : [SelectKBest()],
#                    "select_feat__k" : n_features,
#                    "reduce_dim" : [PCA()],
#                    "reduce_dim__random_state" : [42],
#                    "reduce_dim__n_components" : n_dimensions,
#                    "clf" : [AdaBoostClassifier(base_estimator=DecisionTreeClassifier())],
#                    "clf__n_estimators" : range(8,12),
#                    "clf__learning_rate" : [0.05, 0.1],
#                    "clf__algorithm" : ["SAMME", "SAMME.R"],
#                    "clf__random_state" : [42],
#                    "clf__base_estimator__min_samples_split" : range(2, 4),
#                    "clf__base_estimator__min_samples_leaf": range(1, 4),
#                    "clf__base_estimator__random_state" : [42]
#                    },
                ]
    
    ### Set scoring target
    scoring = {"Precision" : "precision", "Recall" : "recall", "F1" : "f1"}
    
    ### Perform grid search
    grid_search = GridSearchCV(pipe, cv=cv, param_grid=param_grid, scoring=scoring, refit="Recall",
                               n_jobs=4)
    
    print "-----------------"
    print "Pipeline:", [name for name, _ in pipe.steps]
    
    ### Select the best estimator
    clf = grid_search.fit(features, labels).best_estimator_
    
    print "-----------------"
    print "Best Pipeline:"
    print clf.get_params()
    
    ### Display selected features
    features_kbest =[]
    features_kbest = clf.named_steps["select_feat"].fit(features, labels).get_support()
    k_features = clf.named_steps["select_feat"].fit(features, labels).get_params()['k']
    print "-----------------"
    print "Selection of the {0} best features:".format(k_features)
    print "Features with [x] are selected"
    for i in range(len(features_kbest)):
        if features_kbest[i]:
            print "   [x] {feat}, ({feat_score:.2f})".format(feat=features_list[i+1], feat_score=clf.named_steps["select_feat"].scores_[i])
        else:
            print "   [ ] {feat}, ({feat_score:.2f})".format(feat=features_list[i+1], feat_score=clf.named_steps["select_feat"].scores_[i])
        
    ### Plot scores
    plot_grid_search(grid_search.cv_results_)
    
    return clf

def plot_grid_search(cv_results):
    # Rearrange test scores into data frame
    scores_clf = []
    # Get the classifier name
    for i in range(len(cv_results['params'])):
        scores_clf.append(re.split('\W+',str(type(cv_results['params'][i]['clf'])))[-2])
    
    scores_data = pd.DataFrame(
            {"classifier": scores_clf,
             "precision": cv_results['mean_test_Precision'],
             "recall": cv_results['mean_test_Recall']
                    }
            )
   
    # Plot Grid search scores
    sns.set()
    g = sns.FacetGrid(scores_data, col="classifier", aspect=1)
    g.map(sns.kdeplot, "recall", "precision")
    g.set(xlim=(0,0.6), ylim=(0,0.6), xticks=[0.1, 0.2, 0.3, 0.4, 0.5], yticks=[0.1, 0.2, 0.3, 0.4, 0.5])
    
def main():
    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
    
    ### Task 2: Remove outliers
    data_dict.pop("TOTAL", 0)
    data_dict = remove_outliers_nan(data_dict, threshold=15)
    
    ### Run the algorithm without the added features
#    my_dataset = data_dict
#    data_orig = featureFormat(my_dataset, features_list_orig, sort_keys = True)
#    labels_orig, features_orig = targetFeatureSplit(data_orig)   
#    clf = find_classifier_POI(features_orig, labels_orig)
    
    ### Run the algorithm with the added features
    my_dataset = add_features(data_dict)
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    clf = find_classifier_POI(features, labels)
    
    dump_classifier_and_data(clf, my_dataset, features_list)

if __name__ == '__main__':
    main()

print "Total runtime: {:0.2f} s".format(time() - t0)