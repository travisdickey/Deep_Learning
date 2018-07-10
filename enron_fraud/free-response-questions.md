Final Project: Investigate Fraud from Enron Email
=================================================

1.	Summarize goal of project and how machine learning is useful in accomplishing it.

	The goal of the project was to identify 'persons of interest, or POIs' in the Enron fraud case based on email and financial data. Machine learning allows us to employ a series of statistical models to identify patterns that otherwise would be much more difficult to find. Specifically, this dataset included 146 rows, or people, 18 of whom had previously been identified as persons of interest. For each person, there were 21 features concerning their financial information and email correspondence.

	Some outliers had to be removed from the dataset. For example, the row named `total` summed the data from each person, and one row was not a person at all, `The Travel Agency in the Park`. Also, each person's email address appeared as one of the features, so that had to be removed, and many datapoints appeared as 'NaNs'. Those had to be converted to 0. Finally, I removed `deferred_income` and `restricted_stock_deferred` because they contained negative values. I tested removing indivduals with no `total_payments` and no `total_stock_value`, because these people had a lot of missing values, but the results got worse after removing these.

	I removed feature outliers by converting the data to a Pandas dataframe and using the `.drop()` method. For the 'NaNs', I used the `.replace()` method.

	Machine learning allowed us to identify the most relevant features associated with the 'POIs' and then run those through a classifier algorithm to make predictions about which members were POI's. With sklearn metrics, we were then able to measure how effective our predictions were, based on precision and recall scores.

	Because there is significant class imbalance concerning POIs (18 POIs out of 146), this made preparing the dataset for training and testing a bit of a challenge, and it influenced the overall effectiveness of the learning model. Machine learning algorithms work best when there is relative equality between classes. Nevertheless, there are techniques that can be used to minimize problem.

2.	Features used in the POI identifier and the selection process used to pick them.

	I used `SelectKBest` in the Pipeline of my final algorithm, followed by PCA. I passed all the features to the algorithm, except the outliers previously mentioned. Using GridSearchCV, I tested a range of parameters for `K` in `SelectKBest` and for `n_components` in `PCA`. In the final algorithm, the GridSearchCV selected `K=18` and `n_components=10` as the optimal parameters. I also passed the parameters `auto` and `full` to PCA's `svd_solver` to GridSearchCV. It returned `auto` as the optimal value.

	I tried two different scalers, `MinMaxScaler` and `StandardScaler`. StandardScaler provided the best results. Scaling was necessary to normalize the data because we had two very different types of data: financial data and email correspondence. Also, the financial data varied significantly in amount from one feature to the next. Scaling allowed for an "apples to apples" comparison (so to speak) among the different types of data.

	My overall best estimator was a Pipeline using StandardScaler, PCA, and LogisticRegression. I engineered a new feature (`percent_msg_w_poi`) based on the percentage of each individual's messages involving and a POI, i.e., total of all messages from-person-to-poi, to-person-from-poi, and shared-poi divided by total-messages involving that person. The rationale for this was that presumably POIs would share a higher percentage of messages with one another compared to non-POIs. According to the rakings below, obtained through the `scores_` attribute of `SelectKBest`, the newly created feature had the highest ranking of all the features.

	```
	Feature Ranking:
	feature no. 1: percent_msg_w_poi (10.425279413)
	feature no. 2: shared_receipt_with_poi (8.90382155717)
	feature no. 3: from_poi_to_this_person (5.44668748333)
	feature no. 4: loan_advances (2.51826104452)
	feature no. 5: from_this_person_to_poi (2.47052122266)
	feature no. 6: to_messages (1.75169427903)
	feature no. 7: director_fees (0.54908420148)
	feature no. 8: total_payments (0.349627153043)
	feature no. 9: deferral_payments (0.238995889853)
	feature no. 10: exercised_stock_options (0.228267337291)
	feature no. 11: total_stock_value (0.16611912321)
	feature no. 12: from_messages (0.158770239213)
	feature no. 13: bonus (0.0779488557772)
	feature no. 14: other (0.0681945191596)
	feature no. 15: restricted_stock (0.0313332162976)
	feature no. 16: long_term_incentive (0.0222292708616)
	feature no. 17: expenses (0.0139784138218)
	feature no. 18: salary (0.000160054245696)


	```

3.	Algorithm used; others that I tried; how model performance differed.

	My overall best estimator was a Pipeline using `StandardScaler`, `SelectKBest`, `PCA`, and `LogisticRegression`:

	```
	clf:  Pipeline(memory=None,
	     steps=[('scaling', StandardScaler(copy=True, with_mean=True, with_std=True)),
	     ('selector', SelectKBest(k=18, score_func=<function f_classif at 0x0000000008F1F278>)),
	     ('dim_red', PCA(copy=True, iterated_power='auto', n_components=10, random_state=42,
	  svd_solver='auto', tol=0.0, whiten=False)), ('CL...,
	          random_state=42, solver='liblinear', tol=1e-10, verbose=0,
	          warm_start=False))])
	```

	I tried many, many combinations of classifiers and parameters before I finally found one that got a high enough recall score. I used `class EstimatorSelectionHelper` created by Panagiotis Katsaroumpas to help identify the best classifier. Katsaroumpas' code can be found [here](http://www.codiply.com/blog/hyperparameter-grid-search-across-multiple-models-in-scikit-learn/). I modified the code to suit my purposes. I added `StratifiedShuffleSplit` to the process, passed Pipelines to it, rather than just classifiers, and added a new dictionary (`best_estimators`) to keep the best performing parameters, along with `mean_score` for each Pipeline tested. Then I called the `max()` function on the `best_estimators` dictionary to get the overall best performing estimator and assigned that to `clf` to be used in `tester.py`.

	Below is an example of different models ranked by mean `f1` score. These are not by far all the models I tested. I tried about a thousand different combinations before I found one that achieved a high enough recall score. Other models included `DecisionTreeClassifier`, `AdaBoostClassifier`, `GradientBoostingClassifier`, and `SVM`. I also tried a variety of scaling, feature selection, and dimension reduction methods. Examples: `SelectKBest`, `RFE`, `PCA`, `KernelPCA`, and more.

	```
	  estimator             min_score mean_score max_score  std_score
	  0  LogisticRegression         0   0.337431  0.888889  0.165247
	  1  DecisionTreeClassifier     0   0.258988  0.833333  0.172876
	```

4.	What tuning parameters means; what can happen if it's not done well; how I tuned the parameters of my algorithm; what parameters I tuned.

	Tuning parameters involves trying different combinations of parameters to find the most effective combination. If parameter tuning is not done well, it can cause a few problems. First, if you do it manually, it can be a very tedious process: adjust, test, re-adjust, re-test, etc. If you do it in an unsupervised way, say by passing parameters to a Pipeline, it can throw an error if, for instance, you specify a number of features to test that does not fall within the range of components specified. Another problem can be that it may cause your program to run for a very long time if you specify too large a number of parameters to test.

	I used `GridsearchCV` inside of a function, which allowed me to iterate through various `Pipelines`. The `Pipelines` included a variety of selectors, dimension reduction methods, and classifiers along with their associated parameters. After trying out all the different combinations, the function returned a dictionary of the best performing estimator for each Pipeline. I then called the `max()` method on the dictionary to get the overall best performing estimator. Below are the parameters I passed to `GridsearchCV` for the `Pipeline` that ultimately was selected as the best estimator.

	```
	'LogisticRegression': { 'selector__k': [14,16,18],
	                        'dim_red__random_state': [42],
	                        'dim_red__n_components': [10,12,14],
	                        'dim_red__svd_solver': ['auto','full'],
	                        'CLF__random_state': [42],
	                        'CLF__class_weight': ['balanced'],
	    }


	```

	At first, I just radndomly threw numbers into the parameters for the Pipelines and hoped for the best. This did not prove to be particularly effective. My precision and recall scores would not go above about .222. Then I began to take a more systematic approach. I began trying to isolate the best values for key parameters. Then once those were identified, I would hold those values constant and try to identify the best values for other parameters one at a time.

	For example, initially the best performing classifier I found was `GradientBoostingClassifier`. To tune the parameters, I followed the process described in ["Complete Guide to Parameter Tuning in Gradient Boosting (GBM) in Python"](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/) by Aarshay Jain. First, I set initial estimates on a few parameters and then began tuning `max_depth` and `num_samples_split`. Once I found optimal values there, I held those constant and tuned `min_samples_leaf`. When I found the optimal value there, I tuned `max_features`. Finally, I moved to tuning `subsample` and then began tuning the `learning_rate` and making proportional inverse adjustments to the number of estimators.

	Unfortunately, after all this, the best `precision` and `recall` score I got was .29 for each. Frustrated, after having spent more time than I'd like to admit on this, I finally broke down and sought "Live Help" on the Udacity site. Thankfully, the mentor suggested a few classifiers that I had not tried. I was able to plug those classifiers into my functions and with just a little bit of tuning, was able to achieve precision and recall above 0.3, with final `precision` score of 0.406 and `recall` of 0.694.

	My takeaway message from this project is that to do this well, you have to both really understand the nature of the data and have a thorough knowledge of the myriad classifiers, selctors, and dimension reduction methods, so that you can choose the correct Pipeline to suit the data.

5.	What validation is; a classic mistake one can make if it's done wrong; how I validated my analysis.

	Validation involves dividing your dataset into training and testing splits so that a portion can be used to train the algorithm and a portion to test it. If you were to test your algorithm using the same subset of data that it was trained on, it would be impossible to attest to the validity of the score. In fact, this is a classic mistake that is often made, causing an overfit of the data. In doing so, you get a high accuracy score, but it cannot be trusted. It would return low precision and recall scores.

	In my algorithm, I used `StratifiedShuffleSplit`, which returns stratified, randomized folds. I used 100 folds in my tuning and testing and set the test size to include 30% of the data. Of course, `tester.py` used 1,000 folds in its test. `StratifiedShuffleSplit` was necessary becuase we had so little data. We didn't have enough to divide it into full trainig/testing splits. Instead, `StratifiedShuffleSplit` took the whole data over the specified number of "folds" and generated training and testing indices in each fold. The folds were passed to GridSearchCV, which then took the average best performing algorithm, including testing the various parameters on each fold. For example, from my final list of parameters, `GridSearchCV` in `tester.py` fit 1,000 folds for each of 40 candidates, totalling 40,000 fits. Below is the code for the `StratifiedShuffleSplit` that used for parameter tuning.

	```
	  cv = StratifiedShuffleSplit(y, 100, test_size=0.3, random_state = 42)
	```

6.	Two evaluation metrics and average performance for each; an interpretation of the metrics.

	The final precision, recall, and 'F1' scores for my algorithm are listed below. Taken together, precision and recall measure the overall effectiveness of the algorithm, where precision relates to the number of false-positive idnetifications and recall relates to the number of false-negative idnetifications.

	For this particular project, a high precision means the algorithm is identifying a good number of POIs correctly with few incorrect identifications (i.e., few false-positives). A high recall means that the algorithm is identifying a good number of POIs correctly without missing many actual POIs (i.e., low false-negatives). The 'F1' score is the harmonic mean of the two.

	```
	Accuracy: 0.82380   Precision: 0.40591  Recall: 0.69350
	F1: 0.51209 F2: 0.60743
	Total predictions: 15000    True positives: 1387    False positives: 2030   
	False negatives:  613   True negatives: 10970
	```

	My algorithm's recall score is slightly better than the precision score, which is explained by the fewer false-negatives than false_positives. In other words, the algorithm is identifying more people as POIs who are not POIs, than it is failing to identify ones who are.
