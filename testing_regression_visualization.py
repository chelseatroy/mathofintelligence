from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from ui_tools import regression_plot_with_feature_weights
import pandas as pd

columns = ["cylinders","displacement","horsepower","weight","acceleration","model year"]

data = pd.read_csv("cardata.csv")
train, test = train_test_split(data, test_size=0.1)
y_train = train['mpg']
y_test = test['mpg']
x_train = train[columns]
x_test = test[columns]

regressor = LinearRegression(normalize=True)
regressor.fit(x_train, y_train)

print(regressor.score(x_test, y_test))
print(regressor.coef_)
regression_plot_with_feature_weights(x_test, y_test, regressor.coef_.tolist(), regressor.predict(x_test),
                                     regressor.intercept_, 'car_model')