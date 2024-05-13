
'''
 Background and intro
 Triggered by Zillow's shares price decrease after the company exits home-buying business,
 we write this project aims to predict levels of potential sale price (fair value) of real
 estate assets,when building models, we regard 'age' as the most important independent variable,
 and we also see others four variables: distance_to_MTR,number_of_stores,latitude,longitude
 as independent variables.
'''

# _______________________________________________________________________
'''
 preparation
'''

# CODE START HERE
# Import the necessary modules and libraries
import pandas as pd
import numpy as np

#read data
df = pd.read_csv('/Users/zhangdonglei/Desktop/23T1/3648/final/T1_2022_RE.csv')
print("data shape:", data.shape) #data shape: (414, 8)

# set the variables
df.columns # to find the valuse name
x_name = df[['age', 'distance_to_MTR', 'number_of_stores', 'latitude', 'longitude']]
y_name = df[['house_price_per_unit']]
x_val = df[['age', 'distance_to_MTR', 'number_of_stores', 'latitude', 'longitude']].values
y_val = df['house_price_per_unit'].values

#creat train and test dataset
import sklearn
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size=0.1)

# CODE END HERE



# _____________________________________________________________________

''' 
 Model 1: Polynomial
 When the linear regression has limited explanatory power, we consider the nonlinear model
 Step 1  set the model:y = beta0 +beta1 * age + delta1 * age^2 + .. + (coefficient_n) * age^n 
                           + ... 
                           + beta5 * longitude + delta5 * longitude + (coefficient_n) * longitude^n 
 Step 2  use train data to to fit the model
 Step 3  make prediction, test the accuracy and efficiency

'''

# CODE START HERE

# scatter plot: age vs price
import matplotlib.pyplot as plt
plt.scatter(df['age'], df['house_price_per_unit'])
plt.xlabel('age')
plt.ylabel('house_price_per_unit')
plt.title('age vs price')
# The scatter plot shows that the relationship between age and price is not described by a simple linear model


# tools
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# make predict & test
# degree = 2 ( eg: age^2 )
poly = Pipeline([('Poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=False))])
poly.fit(x_train, y_train)
# predict
y_pre_poly2 = poly.predict(x_test)
poly_mse2 = mean_squared_error(y_test, y_pre_poly2)
poly_r_square2 = r2_score(y_test, y_pre_poly2)
# print results
print(f'Poly Reg MSE:{poly_mse2}') #Poly Reg MSE:37.88484866427983
print(f'Poly R square:{poly_r_square2}') #Poly R square):0.8036972827586099

# degree = 3 (eg: age^3)
poly = Pipeline([('Poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
poly.fit(x_train, y_train)
# predict
y_pre_poly3 = poly.predict(x_test)
poly_mse3 = mean_squared_error(y_test, y_pre_poly3)
poly_r_square3 = r2_score(y_test, y_pre_poly3)
# print results
print(f'Poly Reg MSE:{poly_mse3}') #Poly Reg MSE:31.471783801238324
print(f'Poly R square:{poly_r_square3}') #Poly R square):Poly R square:0.8369269801929644

# degree = 4 (eg: age^4)
poly = Pipeline([('Poly', PolynomialFeatures(degree=4)), ('linear', LinearRegression(fit_intercept=False))])
poly.fit(x_train, y_train)
y_pre_poly4 = poly.predict(x_test)
poly_mse4 = mean_squared_error(y_test, y_pre_poly4)
poly_r_square4 = r2_score(y_test, y_pre_poly4)
print(f'Poly Reg MSE:{poly_mse4}') #Poly Reg MSE:1171.6137301151914
print(f'Poly R square:{poly_r_square4}') #Poly R square:-5.0707899566135115

# So we think let degree = 3 is the optimal choice

# Coefficients
poly.named_steps['linear'].coef_

# CODE END HERE

'''
 Explain results ：Polynomial
 When degree = 3, which means the model is :
    y = beta0 +beta1 * age + delta1 * age^2 + Omega * age^3 
              + ...
              + beta5 * longitude + delta5 * longitude + Omega5 * longitude^3
  
 As the R-square = 0.94 (based on the test set), the regression fits seems good.
 
 It is worth noting that the coefficients of some variables are very
    small, which may be due to the lack of normalization of the data.
'''

# ______________________________________________________________________


'''<
 Model 2:Decision Tree 
 The decision trees is used to fit a sine curve with addition noisy observation.
 As a result, it learns local linear regressions approximating the sine curve. 
 Step 1  use train data to to fit the model
 Step 2  make prediction, test the accuracy and efficiency
 >'''

# CODE START HERE

from sklearn.tree import DecisionTreeRegressor

# Fit regression model(depth = 2)
tree = DecisionTreeRegressor(max_depth=2)
tree.fit(x_train, y_train)

# predict
y_pre_tree2 = tree.predict(x_test)

# Accuracy & Efficiency
tree_mse2 = mean_squared_error(y_test, y_pre_tree2)
tree_r_square2 = r2_score(y_test, y_pre_tree2)

# print results
print(f'DecisionTree Reg MSE:{tree_mse2}')  # DecisionTree Reg MSE:60.01817593607378
print(f'DecisionTree Reg R-square:{tree_r_square2}')  # DecisionTree Reg R-square):0.6890120606121983

# Fit regression model(depth = 3)
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(x_train, y_train)

# predict
y_pre_tree3 = tree.predict(x_test)

# Accuracy & Efficiency
tree_mse3 = mean_squared_error(y_test, y_pre_tree3)
tree_r_square3 = r2_score(y_test, y_pre_tree3)

# print results
print(f'DecisionTree Reg MSE:{tree_mse3}')  # DecisionTree Reg MSE:170.90682293535934
print(f'DecisionTree Reg R-square:{tree_r_square3}')  # DecisionTree Reg R-square:0.11443558783602348

# try different depth but using for
tree_mse_list = []
tree_r_list = []
for depth in [2, 3, 4, 5, 6, 7]:
    tree = DecisionTreeRegressor(max_depth=depth)
    tree.fit(x_train, y_train)
    y_pre_tree = tree.predict(x_test)
    tree_mse = mean_squared_error(y_test, y_pre_tree)
    tree_mse_list.append(tree_mse)
    tree_r_square = r2_score(y_test, y_pre_tree)
    tree_r_list.append(tree_r_square)

# Plot the results
plt.plot([2, 3, 4, 5, 6, 7], tree_mse_list)
plt.xlabel('dmax_depth')
plt.ylabel('mse')
plt.show()

plt.plot([2, 3, 4, 5, 6, 7], tree_r_list)
plt.show()



# CODE END HERE

'''
 Explain results ：Decision Tree
 It can be seen that in the plot that, when max depth =2 , 
 the mse is the min(MSE:60.01817593607378) the r-square is the 
 max(R-square:0.11443558783602348). So max_depth = 2 is the optimal choice.
 We get a small decision tree model.
 
'''

# _________________________________________________________________

'''
 Explain results ：Compare results
 
 Comparing the MSE and R-square first,
 polynomial(degree=3):
 Poly Reg MSE:37.88484866427983
 Poly R square:0.8036972827586099
 Decision Tree(max depth=2)
 DecisionTree Reg MSE:60.01817593607378
 DecisionTree Reg R-square):0.6890120606121983
 
 MES=(y - y-estimated)^2/n, the less the better
 R^2 range in [0,1], the larger the better.
 So the Polynomial Model, with smaller MSE and larger R-square,
 is better than Decision Tree Model.
 
 It should be noted that a large r-square does not mean that 
 the model is absolutely good, because many factors such as the 
 number of independent variables are positively correlated 
 with the r-square, so if the condition is sufficient, you can compare the adjusted-r2
'''
