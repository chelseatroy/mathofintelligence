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
income_data = pd.read_csv("incomedata.csv")

income_train, income_test = train_test_split(income_data, test_size=0.001)

training_x = income_train[["age", "some_number", "another_number", "who_knows"]]
training_y = income_train["income"]

test_x = income_test.loc[:, ["age", "some_number", "another_number", "who_knows"]]
test_y = income_test["income"]

model = LinearSVC()
model.fit(training_x, training_y)

predictions = model.predict(test_x)

test_y = [-1 if test_x == ' <=50K' else 1 for test_x in test_y.tolist()]
predictions = [-1 if test_x == ' <=50K' else 1 for test_x in predictions.tolist()]

regression_plot_with_feature_weights(test_x, test_y, model.coef_.tolist()[0],
                                     predictions,
                                     model.intercept_[0], 'income_model')
