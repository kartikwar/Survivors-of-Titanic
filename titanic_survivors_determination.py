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

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }
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

def feature_engineering(training_set, predict_set):
	full_set = [training_set, predict_set]
	for dataset in full_set:
		dataset['familySize'] = dataset['Parch'] + dataset['SibSp'] + 1
		dataset['hasCabin'] = dataset["Cabin"].apply(lambda x: 0 
		if type(x) == float else 1)
		dataset['isAlone'] = np.where(dataset['familySize'] > 1, 0, 1)
		dataset['Title'] = dataset['Name'].apply(get_title)
		dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 
		'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')		
		dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
		dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
		dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
		dataset['NameLength'] = dataset['Name'].apply(get_name_length)
		dataset['Embarked'] = dataset['Embarked'].fillna('S') 
		drop_elements = ['PassengerId', 'Name', 'SibSp', 'Cabin', 'Ticket']
		dataset = dataset.drop(drop_elements, axis = 1)	
	return (training_set, predict_set)	

def data_preprocessing():
	training_set = pd.read_csv('past_data_titanic.csv')
	predict_set = pd.read_csv('test_data_titanic.csv')
	# print((list(predict_set['PassengerId'])))

	training_set_X = training_set.drop('Survived', axis=1)
	training_set_Y = training_set['Survived']
	training_set_X['Age'] = training_set_X['Age'].apply(str)
	predict_set['Age'] = predict_set['Age'].apply(str)
	predict_set['Fare'] = predict_set['Fare'].apply(str)

	training_set_X, predict_set  = feature_engineering(training_set_X, predict_set)

	X_train, X_test, y_train, y_test = train_test_split(training_set_X, training_set_Y, random_state = 0)	
	
	corelations = find_corelations_for_survival(X_train, y_train)
	# print corelations

	# selected_features = feature_engineering(corelations)
	# X_train = X_train[selected_features]
	# X_test = X_test[selected_features]

	X_train = label_encode_features(X_train)
	# print(X_train.columns)
	X_test = label_encode_features(X_test)
	# print(X_test.columns)
	# X_train = pca_feature_engineering(X_train)
	# X_test = pca_feature_engineering(X_test)

	return (X_train, X_test, y_train, y_test, predict_set)

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

def first_level_training(X_train, y_train, X_test, y_test, predict_set):
	training_sets_list = [{'train' : X_train, 
	'test' : y_train}]

	rf = RandomForestClassifier(warm_start=True,  n_jobs=-1 , verbose=0,
		min_samples_leaf=2, n_estimators=500, max_features='sqrt',
		max_depth=6, random_state = 0).fit(X_train, y_train)
	rf_predict_train = rf.predict(X_train)
	rf_predict_test = rf.predict(X_test)
	rf_predict_set = rf.predict(predict_set)
	rf_predict_set = rf_predict_set.reshape(418,1)
	rf_predict_train = rf_predict_train.reshape(668, 1)
	rf_predict_test = rf_predict_test.reshape(223,1)


	et = ExtraTreesClassifier(n_estimators=500,  n_jobs=-1 , verbose=0,
		min_samples_leaf=2, max_depth=8, random_state=0).fit(X_train, y_train)
	et_predict_train = et.predict(X_train)
	et_predict_test = et.predict(X_test)
	et_predict_train = et_predict_train.reshape(668, 1)
	et_predict_test = et_predict_test.reshape(223,1)
	et_predict_set = et.predict(predict_set)
	et_predict_set = et_predict_set.reshape(418,1)

	ada = AdaBoostClassifier(n_estimators= 500
		, learning_rate=0.75, random_state=0).fit(X_train, y_train)	
	ada_predict_train = ada.predict(X_train)
	ada_predict_test = ada.predict(X_test)
	ada_predict_train = ada_predict_train.reshape(668, 1)
	ada_predict_test = ada_predict_test.reshape(223,1)
	ada_predict_set = ada.predict(predict_set)
	ada_predict_set = ada_predict_set.reshape(418,1)

	gb = GradientBoostingClassifier(n_estimators=500, verbose= 0
		, max_depth= 5, min_samples_leaf=2 ,random_state=0).fit(X_train, y_train)
	gb_predict_train = gb.predict(X_train)
	gb_predict_test = gb.predict(X_test)
	gb_predict_train = gb_predict_train.reshape(668, 1)
	gb_predict_test = gb_predict_test.reshape(223,1)
	gb_predict_set = gb.predict(predict_set)
	gb_predict_set = gb_predict_set.reshape(418,1)

	svc = SVC(kernel='linear', C=0.025, random_state=0).fit(X_train, y_train)
	svc_predict_train = svc.predict(X_train)
	svc_predict_test = svc.predict(X_test)
	svc_predict_train = svc_predict_train.reshape(668, 1)
	svc_predict_test = svc_predict_test.reshape(223,1)
	svc_predict_set = svc.predict(predict_set)
	svc_predict_set = svc_predict_set.reshape(418,1)

	
	X_train = np.concatenate(( rf_predict_train, et_predict_train, ada_predict_train, 
		gb_predict_train, svc_predict_train), axis=1)
	X_test = np.concatenate(( rf_predict_test, et_predict_test, 
		ada_predict_test, gb_predict_test, svc_predict_test), axis=1)
	predict_set = np.concatenate(( rf_predict_set, et_predict_set, 
		ada_predict_set, gb_predict_set, svc_predict_set ), axis=1)		
	return (X_train, X_test, predict_set)	

def second_level_training(X_train, y_train):
	# print(X_train.shape)
	# print(y_train.shape)
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
	X_train, X_test, y_train, y_test, predict_set =  data_preprocessing()
	predict_X = label_encode_features(predict_set)
	# print(predict_set.shape)
	X_train, X_test, predict_X =  first_level_training(X_train, y_train, X_test, 
		y_test, predict_X)
	clf = second_level_training(X_train, y_train)
	training_accuracy, test_accuracy = find_accuracy_of_model(clf, 
		X_train, X_test, y_train, y_test)
	print(training_accuracy, test_accuracy)
	survival_predictions = clf.predict(predict_X)
	save_to_csv(predict_set, survival_predictions, 'predictions.csv')
	# # print(survival_predictions)
	# print(training_accuracy, test_accuracy)
	