from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from ui_tools import regression_plot_with_feature_weights
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# INCOME CLASSIFICATION
######################
# income_data = pd.read_csv("incomedata.csv")
#
# income_train, income_test = train_test_split(income_data, test_size=0.001)
#
# training_x = income_train[["age", "some_number", "another_number", "who_knows"]]
# training_y = income_train["income"]
#
# test_x = income_test.loc[:, ["age", "some_number", "another_number", "who_knows"]]
# test_y = income_test["income"]
#
# model = LinearSVC()
# model.fit(training_x, training_y)
#
# predictions = model.predict(test_x)
#
# test_y = [-1 if test_x == ' <=50K' else 1 for test_x in test_y.tolist()]
# predictions = [-1 if test_x == ' <=50K' else 1 for test_x in predictions.tolist()]
#
# regression_plot_with_feature_weights(test_x, test_y, model.coef_.tolist()[0],
#                                      predictions,
#                                      model.intercept_[0], 'income_model')

# CAR EVALUATION CLASSIFICATION
######################
car_evaluation_data = pd.read_csv("carsafetydata.csv")
car_evaluation_data["eval"] = car_evaluation_data["eval"].replace(['acc', 'good', 'vgood'], 'acc')

car_evaluation_data["buying"] = 'buying_' + car_evaluation_data["buying"]
car_evaluation_data["maintenance"] = 'maintenance_' + car_evaluation_data["maintenance"]
car_evaluation_data["lug_boot"] = 'lug_boot_' + car_evaluation_data["lug_boot"]
car_evaluation_data["safety"] = 'safety_' + car_evaluation_data["safety"]

one_hot_buying = pd.get_dummies(car_evaluation_data["buying"])
one_hot_maintenance = pd.get_dummies(car_evaluation_data["maintenance"])
one_hot_lug_boot = pd.get_dummies(car_evaluation_data["lug_boot"])
one_hot_safety = pd.get_dummies(car_evaluation_data["safety"])

car_evaluation_data["doors"] = car_evaluation_data["doors"].replace(['5more'], '5')
car_evaluation_data["persons"] = car_evaluation_data["persons"].replace(['more'], '5')

car_evaluation_data = pd.concat([car_evaluation_data["doors"], car_evaluation_data["persons"], one_hot_buying, one_hot_lug_boot, one_hot_maintenance, one_hot_safety, car_evaluation_data["eval"]], axis=1)

car_evaluation_train, car_evaluation_test = train_test_split(car_evaluation_data, test_size=0.01)

training_y = car_evaluation_train["eval"]
car_evaluation_train = car_evaluation_train.drop(['eval'], axis=1)
training_x = car_evaluation_train

test_y = car_evaluation_test["eval"]
car_evaluation_test = car_evaluation_test.drop(['eval'], axis=1)
test_x = car_evaluation_test

model = LinearSVC()

model.fit(training_x, training_y)

predictions = model.predict(test_x)

test_y = [-1 if test_x == 'unacc' else 1 for test_x in test_y.tolist()]
predictions = [-1 if test_x == 'unacc' else 1 for test_x in predictions.tolist()]

regression_plot_with_feature_weights(test_x, test_y, model.coef_.tolist()[0],
                                     predictions,
                                     model.intercept_[0], 'car_evaluation_model')
