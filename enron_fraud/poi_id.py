#!/usr/bin/python

### Identify Fraud in Enron Email
### file runs EstimatorSelectionHelper to test various Pipelines, identifying
### the best overall estimator; File also runs tester.py to test the
### precision and recall on the best estimator

import pickle

### import files
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
sorted_data_dict = sorted(data_dict)

# Exploring Dataset through Pandas DataFrame
import pandas as pd
df = pd.DataFrame.from_records(list(data_dict.values()))
df.set_index(pd.Series(list(data_dict.keys())), inplace=True)
print " "
print "Shape of Dataset, # of People & Features: {}".format(df.shape)
print " "

print df.groupby('poi')
print "# of POIs: True = POI; False = Not POI"
print df['poi'].value_counts()
print " "
# Remove Outliers
df.drop(['email_address',
         'deferred_income',
         'restricted_stock_deferred'],
         axis=1,inplace=True)

df.drop(['TOTAL',
         'THE TRAVEL AGENCY IN THE PARK'])

# Create my_dataset, train and test
# replace NaN by 0; change all to float
df.replace(to_replace= 'NaN', value= 0,inplace=True)
df=df.astype(float)

# Transform df to dictionary
data_dict= df.to_dict('index')

# Store to my_dataset for easy export below
my_dataset =data_dict
# get features_list
features_list = []
features_list = data_dict["PRENTICE JAMES"].keys()
features_list.insert(0, features_list.pop(features_list.index('poi')))

### create new feature: percent of total messages w/ poi
for k in my_dataset:
    to_poi = my_dataset[k]['from_this_person_to_poi']
    from_poi = my_dataset[k]['from_poi_to_this_person']
    shared_poi = my_dataset[k]['shared_receipt_with_poi']
    total_w_poi = to_poi + from_poi + shared_poi
    to_msgs = my_dataset[k]['to_messages']
    from_msgs = my_dataset[k]['from_messages']
    total_msgs = to_msgs + from_msgs
    if total_msgs > 0:
        my_dataset[k]['percent_msg_w_poi'] = (float(total_w_poi) / (total_msgs))
    else:
        my_dataset[k]['percent_msg_w_poi'] = 0

# add new feature to features_list
features_list.append('percent_msg_w_poi')
# print total number of features to be used
print " "
print "Total Number of Features: {}".format(len(features_list))
print " "

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# import statements; commented out methods are ones that were tried, but
# not used in final program
import sys
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.pipeline import Pipeline # FeatureUnion
from sklearn.preprocessing import MinMaxScaler, StandardScaler #RobustScaler
from sklearn.feature_selection import SelectKBest #RFE, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA # NMF, KernelPCA, SparsePCA
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# StratifiedShuffleSplit used for cross_validation
cv = StratifiedShuffleSplit(labels, 100, test_size=0.3, random_state = 42)

### imort ensemble classifiers
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier)

scaler = MinMaxScaler()
kbest = SelectKBest()
#kbest_chi2 = SelectKBest(chi2)
pca = PCA()
#kpca = KernelPCA()
#spca = SparsePCA()
#nmf = NMF()
#clf_ADB = AdaBoostClassifier()
#clf_GBC = GradientBoostingClassifier()
#rfe_ADB = RFE(clf_ADB)
#rfe_GBC = RFE(clf_GBC) # rfe returned decent results, but took way too long
clf_logistic = LogisticRegression(  C=10**20,
                                    tol=10**-10,
                                    )

### create pipelines
Pipe_DTC = Pipeline(steps=[
        ('scaling', MinMaxScaler()),
        ('selector', kbest),
        ('dim_red', pca),
        ("CLF", DecisionTreeClassifier())
    ]
)

Pipe_LR = Pipeline(steps=[
        ('scaling', StandardScaler()),
        ('selector', kbest),
        ('dim_red', pca),
        ("CLF", clf_logistic)
    ]
)

### models1 and parameters to feed 'fit' function from EstimatorSelectionHelper;
### many, many more combinations were tested; however, to minimize running time,
### most unsuccessful combinations have been removed
models = {
    'DecisionTreeClassifier': Pipe_DTC,
    'LogisticRegression': Pipe_LR,

}

params = {
    'DecisionTreeClassifier':  { 'dim_red__random_state': [42],
                                 'selector__k': [7,9,11],
                                 'CLF__random_state': [42],
 },
    'LogisticRegression': {  'selector__k': [14,16,18],
                             'dim_red__random_state': [42],
                             'dim_red__n_components': [10,12,14],
                             'dim_red__svd_solver': ['auto','full'],
                             'CLF__random_state': [42],
                             'CLF__class_weight': ['balanced'],

  },
}

### Hyperparameter Gridsearch across Multiple Classifiers
### by Panagiotis Katsaroumpas, taken from
### http://www.codiply.com/blog/hyperparameter-grid-search-across-multiple-models-in-scikit-learn/
### and modified to suit purposes of this project
class EstimatorSelectionHelper:
    '''takes models and parameters; provides fit() and score_summary() functions'''
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=cv, n_jobs=1, verbose=1, scoring='f1', refit=True):
        ''' takes features (X) and labels (y); runs GridSearchCV on Pipeline
            models and parameters received from __init__; returns
            'best_estimators' dictionary; updates 'grid_searches' dictionary
        '''
        best_estimators = {}
        for key in self.keys:
            # iterate through models and params dictionaries
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X,y)
            best_estimators[gs.best_estimator_] = gs.best_score_
            self.grid_searches[key] = gs
        return best_estimators

    def score_summary(self, sort_by='mean_score'):
        ''' returns dataframe of scores from gridsearches '''
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series(dict(params.items() + d.items()))

        rows = [row(k, gsc.cv_validation_scores, gsc.parameters)
                for k in self.keys
                for gsc in self.grid_searches[k].grid_scores_]
        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]

print " "
# create EstimatorSelectionHelper object
helper1 = EstimatorSelectionHelper(models, params)
best_estimators = helper1.fit(features, labels, cv=cv, scoring='f1', verbose=1)
# removed n_jobs=-1 b/c I don't have parallel processors
print " "
print helper1.score_summary(sort_by='mean_score')
print " "

# get overall best estimator and assign it to clf
clf = max(best_estimators, key=best_estimators.get)
print "clf: ", clf
print " "

# create a new list that contains the features selected by SelectKBest
# in the optimal model selected by GridSearchCV
weights = clf.named_steps['dim_red'].singular_values_
print "Singular Value Weights for PCA Components: "
print weights

features_selected=[features_list[i+1] for i in clf.named_steps['selector'].get_support(indices=True)]
importances = clf.named_steps['selector'].scores_
indices = np.argsort(importances)[::-1]
print 'Feature Ranking: '
for i in range(len(features_selected)):
    print "feature no. {}: {} ({})".format(i+1,features_selected[indices[i]],importances[indices[i]])


# import the test_classifier from tester.py
from tester import test_classifier
print ' '
# use test_classifier to evaluate the model
# selected by GridSearchCV in EstimatorSelectionHelper.fit()
print "Tester Classification report :"
test_classifier(clf, my_dataset, features_list)
print ' '

### Dump classifier, dataset, and features_list
dump_classifier_and_data(clf, my_dataset, features_list)
