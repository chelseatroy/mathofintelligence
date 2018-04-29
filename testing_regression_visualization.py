from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from ui_tools import regression_plot_with_feature_weights
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CARS
####################
#
# car_columns = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year"]
#
# car_data = pd.read_csv("cardata.csv")
# car_train, car_test = train_test_split(car_data, test_size=0.1)
# car_y_train = car_train['mpg']
# car_y_test = car_test['mpg']
# car_x_train = car_train[car_columns]
# car_x_test = car_test[car_columns]
#
# car_regressor = LinearRegression(normalize=True)
# car_regressor.fit(car_x_train, car_y_train)
#
# print(car_regressor.score(car_x_test, car_y_test))
# print(car_regressor.coef_)
# regression_plot_with_feature_weights(car_x_test, car_y_test, car_regressor.coef_.tolist(), car_regressor.predict(car_x_test),
#                                      car_regressor.intercept_, 'car_model')

# FAST FOOD
######################

# food_columns = ["serving size", "total fat", "saturated fat", "trans fat", "sodium", "carbs", "sugars", "protein"]
#
# food_data = pd.read_csv("nutritiondata.csv")
# food_train, food_test = train_test_split(food_data, test_size=0.1)
# food_y_train = food_train["calories"]
# food_y_test = food_test["calories"]
# food_x_train = food_train[food_columns]
# food_x_test = food_test[food_columns]
#
# food_regressor = LinearRegression(normalize=True)
# food_regressor.fit(food_x_train, food_y_train)
#
# regression_plot_with_feature_weights(food_x_test, food_y_test, food_regressor.coef_.tolist(), food_regressor.predict(food_x_test),
#                                      food_regressor.intercept_, 'nutrition_model')

# HOME PRICES
######################
# home_price_data = pd.read_csv("homepricedata.csv")
# one_hot_locations = pd.get_dummies(home_price_data["location"])
#
# home_price_data = result = pd.concat([home_price_data, one_hot_locations], axis=1)
#
# home_price_train, home_price_test = train_test_split(home_price_data, test_size=0.1)
# home_price_y_train = home_price_train["price"]
# home_price_y_test = home_price_test["price"]
#
# home_price_columns = home_price_data.columns.values.tolist()
# home_price_columns.remove("location")
# home_price_columns.remove("price")
# home_price_columns.remove("price per sqft")
# home_price_x_train = home_price_train[home_price_columns]
# home_price_x_test = home_price_test[home_price_columns]
#
# home_price_regressor = LinearRegression(normalize=True)
#
# home_price_regressor.fit(home_price_x_train, home_price_y_train)
#
# print(home_price_regressor.score(home_price_x_test, home_price_y_test))
# print(home_price_regressor.coef_)
# predictions = home_price_regressor.predict(home_price_x_test)
# print(predictions)
#
# regression_plot_with_feature_weights(home_price_x_test, home_price_y_test, home_price_regressor.coef_.tolist(),
#                                      predictions,
#                                      home_price_regressor.intercept_, 'home_price_model')