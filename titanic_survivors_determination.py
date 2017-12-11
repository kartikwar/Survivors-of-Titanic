import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
import re
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
import multiprocessing
from functools import partial
from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def label_encode_features(dataframe):
	new_dataframe = dataframe.copy()
	for col in new_dataframe.columns:
	    label_encoder = LabelEncoder()
	    # print col
	    try:
	        new_dataframe[col] = new_dataframe[col].fillna("")
	        new_dataframe[col] = label_encoder.fit_transform(new_dataframe[col])
	    except Exception as e :
	        print(col)
	        print(e)
	        pass
	# print dataframe.columns        
	return new_dataframe        

def find_corelations_for_survival(features_data, survival_flag):
	data  = features_data.copy()
	data['Survival'] = survival_flag
	# print data.columns
	correlation = data.corr()
	# print correlation.columns
	return correlation['Survival']

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""	

def get_name_length(name):
	name = str(name)
	length = len(name)
	return length

def train_model_in_folds(counter_kf_ele, x_train, y_train, x_test, oof_train, oof_test_skf, clf):
	x_tr = x_train.iloc[counter_kf_ele['indexes'][0]]
	y_tr = y_train.iloc[counter_kf_ele['indexes'][0]]
	x_te = x_train.iloc[counter_kf_ele['indexes'][1]]
	clf.fit(x_tr, y_tr)
	oof_train[counter_kf_ele['indexes'][1]] = clf.predict(x_te)
	oof_test_skf[counter_kf_ele['counter'], :] = clf.predict(x_test)

def get_oof(clf, x_train, y_train, x_test):
	# print (type(x_train))
	ntrain = x_train.shape[0]
	# print ntrain
	ntest = x_test.shape[0]
	oof_train = np.zeros((ntrain,))
	oof_test = np.zeros((ntest,))
	SEED = 0 # for reproducibility
	NFOLDS = 15 # set folds for out-of-fold prediction
	kf = KFold(len(x_train), n_folds= NFOLDS, random_state=SEED)
	kf = list(kf)
	oof_test_skf = np.empty((NFOLDS, ntest))
	
	counter = 0
	counter_kf_list = []
	for ele in kf:
		counter_kf = {'counter' : counter, 'indexes' : ele}
		counter_kf_list.append(counter_kf)

	with poolcontext(processes=3) as pool:
		results = pool.map(partial(train_model_in_folds, 
			x_train=x_train, y_train=y_train, oof_train=oof_train, oof_test_skf=oof_test_skf, clf=clf, x_test=x_test), counter_kf_list)
		print (oof_test_skf)
		oof_test[:] = oof_test_skf.mean(axis=0)	
		return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
	
	# kf = list(kf)  

	# for i, (train_index, test_index) in enumerate(kf):
	# 	x_tr = x_train.iloc[train_index]
	# 	y_tr = y_train.iloc[train_index]
	# 	x_te = x_train.iloc[test_index]
	# 	clf.fit(x_tr, y_tr)
	# 	oof_train[test_index] = clf.predict(x_te)
	# 	oof_test_skf[i, :] = clf.predict(x_test)

def feature_engineering(training_set, predict_set):
	full_set = [training_set, predict_set]
	for dataset in full_set:
		dataset['familySize'] = dataset['Parch'] + dataset['SibSp'] + 1
		dataset['hasCabin'] = dataset["Cabin"].apply(lambda x: 0 
		if type(x) == float else 1)
		bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

		bins = [0, 12, 17, 60, np.inf]
		labels = ['child', 'teenager', 'adult', 'elderly']
		age_groups = pd.cut(dataset.Age, bins, labels = labels)
		dataset['AgeGroup'] = age_groups

		dataset["Age"] = dataset["Age"].fillna(-0.5)		
		labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
		dataset['AgeGroup'] = pd.cut(dataset['AgeGroup'], bins, labels = labels)
		mr_age = dataset[dataset["Title"] == 1]["AgeGroup"].mode() #Young Adult
		miss_age = dataset[dataset["Title"] == 2]["AgeGroup"].mode() #Student
		mrs_age = dataset[dataset["Title"] == 3]["AgeGroup"].mode() #Adult
		master_age = dataset[dataset["Title"] == 4]["AgeGroup"].mode() #Baby
		royal_age = dataset[dataset["Title"] == 5]["AgeGroup"].mode() #Adult
		rare_age = dataset[dataset["Title"] == 6]["AgeGroup"].mode() #Adult

		for x in range(len(train["AgeGroup"])):
			if train["AgeGroup"][x] == "Unknown":
				train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]
		#map each Age value to a numerical value
		age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
		dataset['AgeGroup'] = dataset['AgeGroup'].map(age_mapping)		
		
		dataset['isAlone'] = np.where(dataset['familySize'] > 1, 0, 1)
		
		dataset['Title'] = dataset['Name'].apply(get_title)
		dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 
		'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')		
		dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
		dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
		dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
		dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
		
		dataset['NameLength'] = dataset['Name'].apply(get_name_length)
		dataset['Embarked'] = dataset['Embarked'].fillna('S') 
		drop_elements = ['PassengerId', 'Name', 'SibSp', 'Cabin', 'Ticket', 'Age']
		# Mapping Fare
		
		median_fare = dataset['Fare'].median()
		dataset['Fare'] = dataset['Fare'].fillna(median_fare)
		
		dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
		dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
		dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']  = 2
		dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
		# print dataset['Fare']
		# Mapping Age
		dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
		dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
		dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
		dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
		dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
 		# Sex
 		dataset.Sex.fillna('0', inplace=True)
		dataset.loc[dataset.Sex != 'male', 'Sex'] = 0
		dataset.loc[dataset.Sex == 'male', 'Sex'] = 1

		dataset = dataset.drop(drop_elements, axis = 1)	
	return (training_set, predict_set)	

def data_preprocessing():
	training_set = pd.read_csv('past_data_titanic.csv')
	predict_set = pd.read_csv('test_data_titanic.csv')

	training_set_X = training_set.drop('Survived', axis=1)
	training_set_Y = training_set['Survived']
	training_set_X['Age'] = training_set_X['Age'].apply(str)
	predict_set['Age'] = predict_set['Age'].apply(str)
	predict_set['Fare'] = predict_set['Fare'].apply(str)

	training_set_X, predict_set  = feature_engineering(training_set_X, predict_set)

	# X_train, X_test, y_train, y_test = train_test_split(training_set_X, training_set_Y, random_state = 0)
	X_train = training_set_X.copy()
	y_train = training_set_Y.copy()	
	
	corelations = find_corelations_for_survival(X_train, y_train)

	X_train = label_encode_features(X_train)
	
	return (X_train, y_train, predict_set)

def determine_best_params_random_forest(X_train, y_train):
	grid_values = {'n_estimators' : [1, 5, 25],
	'max_features': [1,  2 , 3, 4 , 5] 	 
	}
	clf = RandomForestClassifier(random_state = 0)
	grid_clf_accuracy = GridSearchCV(clf, param_grid=grid_values, 
		n_jobs=-1, scoring='accuracy')
	grid_clf_accuracy.fit(X_train, y_train)
	best_params =grid_clf_accuracy.best_params_
	return best_params	

def first_level_training(X_train, y_train, predict_set):
	rf = RandomForestClassifier(warm_start=True,  n_jobs=-1 , verbose=0,
		min_samples_leaf=2, n_estimators=500, max_features='sqrt',
		max_depth=6, random_state = 0)
	rf_predict_train, rf_predict_test = get_oof(rf, X_train, y_train, predict_set)

	et = ExtraTreesClassifier(n_estimators=500,  n_jobs=-1 , verbose=0,
		min_samples_leaf=2, max_depth=8, random_state=0)
	et_predict_train, et_predict_test = get_oof(et, X_train, y_train, predict_set)

	ada = AdaBoostClassifier(n_estimators= 500
		, learning_rate=0.75, random_state=0)
	ada_predict_train, ada_predict_test = get_oof(ada, X_train, y_train, predict_set)		

	gb = GradientBoostingClassifier(n_estimators=500, verbose= 0
		, max_depth= 5, min_samples_leaf=2 ,random_state=0)
	gb_predict_train, gb_predict_test = get_oof(gb, X_train, y_train, predict_set)	

	svc = SVC(kernel='linear', C=0.025, random_state=0)
	svc_predict_train, svc_predict_test = get_oof(svc, X_train, y_train, predict_set)

	knn = KNeighborsClassifier(n_neighbors=1)
	knn_predict_train, knn_predict_test = get_oof(knn, X_train, y_train, predict_set) 
	
	X_train = np.concatenate(( rf_predict_train, et_predict_train, ada_predict_train, 
		gb_predict_train, svc_predict_train, knn_predict_train), axis=1)
	predict_set = np.concatenate(( rf_predict_test, et_predict_test, 
		ada_predict_test, gb_predict_test, svc_predict_test, knn_predict_test), axis=1)

	return (X_train, predict_set)	

def second_level_training(X_train, y_train):
	gbm = xgb.XGBClassifier(n_estimators= 2000,max_depth= 4,min_child_weight= 2,
		gamma=0.9,subsample=0.8, objective='binary:logistic', 
		nthread= -1,scale_pos_weight=1).fit(X_train, y_train)
	return gbm

def build_classifier(X_train, y_train):
	best_params = determine_best_params_random_forest(X_train, y_train)
	# print best_params
	clf = RandomForestClassifier(random_state = 0
	, n_estimators = best_params['n_estimators'],
	max_features = best_params['max_features']).fit(X_train, y_train)	
	return clf

def find_accuracy_of_model(clf, X_train, X_test, y_train, y_test):
	training_accuracy = clf.score(X_train, y_train)
	test_accuracy = clf.score(X_test, y_test)
	return(training_accuracy, test_accuracy)

def save_to_csv(dataset, survival_predictions, file_name):
	dataset['Survived'] = survival_predictions
	dataset = dataset[['PassengerId', 'Survived']]
	dataset.to_csv(file_name, index=False)
	return None	

if __name__ == '__main__':
	X_train, y_train, predict_set =  data_preprocessing()
	predict_X = label_encode_features(predict_set)
	X_train, predict_X =  first_level_training(X_train, y_train, predict_X)
	clf = second_level_training(X_train, y_train)
	survival_predictions = clf.predict(predict_X)
	save_to_csv(predict_set, survival_predictions, 'predictions.csv')
	