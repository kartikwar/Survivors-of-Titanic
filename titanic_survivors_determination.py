import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc


def label_encode_features(dataframe):
	for col in dataframe.columns:
	    label_encoder = LabelEncoder()
	    try:
	        dataframe[col] = dataframe[col].fillna("")
	        dataframe[col] = label_encoder.fit_transform(dataframe[col])
	    except Exception as e :
	        print(col)
	        print(e)
	        pass
	return dataframe        

# def pca_feature_engineering(dataset):
# 	pca = PCA(n_components=2).fit(dataset)
# 	dataset = pca.transform(dataset)
# 	return dataset

def find_corelations_for_survival(features_data, survival_flag):
	data  = features_data.copy()
	data['Survival'] = survival_flag
	correlation = data.corr(method='pearson')
	return correlation['Survival']

def feature_engineering(correlation):
	feature_list = list(correlation[abs(correlation) > 0.05].index)
	feature_list.remove('Survival')
	print feature_list
	return feature_list


def get_datasets():
	training_set = pd.read_csv('past_data_titanic.csv')
	predict_set = pd.read_csv('test_data_titanic.csv')

	training_set_X = training_set.drop('Survived', axis=1)
	training_set_Y = training_set['Survived']
	training_set_X['Age'] = training_set_X['Age'].apply(str)
	predict_set['Age'] = predict_set['Age'].apply(str)
	predict_set['Fare'] = predict_set['Fare'].apply(str)

	X_train, X_test, y_train, y_test = train_test_split(training_set_X, training_set_Y, random_state = 0)
	
	corelations = find_corelations_for_survival(X_train, y_train)
	print corelations

	# selected_features = feature_engineering(corelations)
	# X_train = X_train[selected_features]
	# X_test = X_test[selected_features]

	X_train = label_encode_features(X_train)
	X_test = label_encode_features(X_test)

	# X_train = pca_feature_engineering(X_train)
	# X_test = pca_feature_engineering(X_test)

	return (X_train, X_test, y_train, y_test, predict_set)

def build_classifier(X_train, y_train):
	clf = RandomForestClassifier().fit(X_train, y_train)
	return clf

def find_accuracy_of_model(clf, X_train, X_test, y_train, y_test):
	training_accuracy = clf.score(X_train, y_train)
	test_accuracy = clf.score(X_test, y_test)
	return(training_accuracy, test_accuracy)

if __name__ == '__main__':
	X_train, X_test, y_train, y_test, predict_set =  get_datasets()
	predict_X = label_encode_features(predict_set)
	clf = build_classifier(X_train,  y_train)
	training_accuracy, test_accuracy = find_accuracy_of_model(clf, X_train, X_test, y_train, y_test)
	print(training_accuracy, test_accuracy)
	