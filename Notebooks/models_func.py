import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF , ExpSineSquared , WhiteKernel ,Matern,Sum,Product
from sklearn.model_selection import train_test_split ,GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin



import math

#receives a trained model

def metrics_f(model , X_train ,X_test, y_train,y_test):
    # Make predictions on the training and testing sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate evaluation metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
  
    # Print the evaluation metrics
    print("Train MSE:", train_mse)
    print("Test MSE:", test_mse)
    print("Train RMSE:", train_rmse)
    print("Test RMSE:", test_rmse)
    print("Train MAE:", train_mae)
    print("Test MAE:", test_mae)
    print("Train R2:", train_r2)
    print("Test R2:", test_r2)


# recieves a dataframe , test size and random state and do stratify split by the Grade column 
def strat_split(df, test_size, random_state):
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    df['Grade_cat'] = pd.cut(df['Grade'], bins=[5,6,7, 8, 9,10], labels=['very low', 'low','medium', 'high', 'very high'])
    for train_index, test_index in split.split(df, df['Grade_cat']):
        strat_train_set = df.iloc[train_index]
        strat_test_set = df.iloc[test_index]
    for set_ in [df,strat_train_set, strat_test_set]:
        set_.drop('Grade_cat', axis= 1, inplace= True)
    y_train = strat_train_set['Curr Price'].copy()
    y_test = strat_test_set['Curr Price'].copy()
    for set_ in [strat_train_set, strat_test_set]:
        set_.drop('Curr Price', axis= 1, inplace= True)
    return strat_train_set, y_train, strat_test_set , y_test

def Checkin_day(df):
    df_copy = df.copy()
    df_copy['Checkin Day of week'] = df_copy.apply(lambda x: 2 if x['Monday'] == 1 else 4 if x['Wednesday'] == 1 else 6 ,axis=1)
    df_copy['Checkin Day of week'] = df_copy.apply(lambda x:(x['Checkin Day of week'] + x['TTT']) % 7,axis=1)
    df_copy['Checkin Day of week'].replace(0,7, inplace=True)
    return df_copy

def residual_plot(model , X_train ,X_test, y_train,y_test):    
     y_train_pred = model.predict(X_train)    
     y_test_pred = model.predict(X_test)       
     # Create a residual plot     
     train_residuals = y_train_pred - y_train    
     test_residuals = y_test_pred - y_test     
     plt.scatter(y_train_pred, train_residuals, c='blue', marker='o', label='Training data')     
     plt.scatter(y_test_pred, test_residuals, c='green', marker='s', label='Testing data', alpha=0.1)     
     plt.xlabel('Predicted values')     
     plt.ylabel('Residuals')     
     plt.legend(loc='upper left')     
     plt.hlines(y=0, xmin=0,xmax=700, lw=2, color='red')     
     plt.show()      
     # Create a residual histogram using seaborn for the training set     
      
     sns.histplot(train_residuals, kde=True, color='blue', edgecolor='black')    
     plt.axvline(x=np.mean(train_residuals), color='red', linestyle='--', label='Mean')    
     plt.xlabel("Residuals")     
     plt.ylabel("Frequency")     
     plt.title("Residual Histogram (Training Set)")     
     plt.legend()     
     plt.show()      
     
     # Create a residual histogram using seaborn for the test set     
     sns.histplot(test_residuals, kde=True, color='green', edgecolor='black')    
     plt.axvline(x=np.mean(test_residuals), color='red', linestyle='--', label='Mean')     
     plt.xlabel("Residuals")     
     plt.ylabel("Frequency")    
     plt.title("Residual Histogram (Test Set)")    
     plt.legend()    
     plt.show()
     
     
def feature_importance_with_drop(model, X_train, y_train, X_test, y_test):
    feature_importances_dict = {}
    for col in X_train.columns:
        new_model = model
        new_model.fit(X_train.drop(col , axis = 1), y_train)
        
        y_test_pred = new_model.predict(X_test.drop(col , axis = 1))
        test_r2 = r2_score(y_test, y_test_pred)
        feature_importances_dict[col] = 1-test_r2
    sorted_feature_importances_dict = sorted(feature_importances_dict.items(), key=lambda x:x[1], reverse=True)
    for i in sorted_feature_importances_dict:
        print(i)

'''--------------------------------------Pipelines----------------------------------------------'''

standartize = Pipeline([('std_scaler', StandardScaler())])


'''--------------------------------------linear regression----------------------------------------------'''
     
#receives a trained model
def feature_importances_LinearRegression(model,X_train):
    # Print feature coefficients
    for feature, coef in zip(X_train.columns, model.coef_):
        print("{} coefficient: {:.3f}".format(feature, coef))
        

'''--------------------------------------Decision Tree----------------------------------------------'''
def Tree_hyperparameters(X_train, y_train):
    # Define parameter grid to search
    param_grid={"splitter":["best","random"],
                "max_depth" : [2,3,4,5,6],
                "min_samples_leaf":[1,10,20,30,40],
                "max_leaf_nodes":[2,10,20,30,40,] }


    Tree = DecisionTreeRegressor()
    
    # fit gridsearchcv
    grid_search = GridSearchCV(Tree, param_grid, scoring='r2', cv = 5, verbose=5)
    grid_search.fit(X_train, y_train)
    
    # Print best parameters and best score
    print('Best Parameters:', grid_search.best_params_)
    print('Best Score:', grid_search.best_score_)

    # create Tree using best parameters
    Tree_best = DecisionTreeRegressor(**grid_search.best_params_)
    return Tree_best
    

 
 
'''--------------------------------------Gaussian Process Regression----------------------------------------------'''
  
def gp_hyperparameters(X_train, y_train):
    # Define the kernels to test
    kernels = [Sum(ConstantKernel(1.0, (1e-3, 1e3))*RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
                ExpSineSquared(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))),
               Sum(ConstantKernel(1.0, (1e-3, 1e3))*RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
                WhiteKernel(noise_level=1e-2))]

    # Define the hyperparameters for each kernel
    hyperparameters = [{'kernel__k1__k1__constant_value': [0.1, 1, 10],
                        'kernel__k1__k2__length_scale': [0.1, 1, 10],
                        'kernel__k2__length_scale': [0.1, 1, 5],
                        'kernel__k2__periodicity':[0.1, 1, 10],
                        "alpha": [0.01, 0.1, 1.0]},
                       {"kernel__k1__k1__constant_value": [1.0, 5.0, 10.0],
                        "kernel__k1__k2__length_scale": [1.0, 5.0, 10.0],
                        "kernel__k2__noise_level": [1e-4, 1e-3, 1e-2],
                        "alpha": [0.01, 0.1, 1.0]}]

    # Perform cross-validation to tune the hyperparameters
    best_score = -np.inf
    for i, kernel in enumerate(kernels):
        gp = GaussianProcessRegressor(kernel=kernel)
        grid_search  = GridSearchCV(gp, hyperparameters[i],scoring='r2',verbose=5, cv=4)
        grid_search.fit(X_train, y_train)
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_kernel = grid_search.best_estimator_.kernel_

    # Fit the Gaussian process to the data with the best kernel and hyperparameters
    gp = GaussianProcessRegressor(kernel=best_kernel)
    return gp
    
'''--------------------------------------KNN----------------------------------------------'''

def knn_hyperparameter(X_train, y_train):
    n_list = list(range(3, 30, 2))
    # Define parameter grid to search
    param_grid = {'n_neighbors': n_list,
                  'weights': ['uniform', 'distance'], # equal weight or weight to each point proportional to its inverse distance
                  'p': [1, 2], #for minkowski :  Manhattan distance or  Euclidean distance 
                  'leaf_size': [10, 20, 30],#for faster nearest neighbor search.
                  'metric': ['euclidean', 'manhattan', 'minkowski']}#compute the distance between two points in the dataset
                 

   
    knn = KNeighborsRegressor()

    
    grid_search = GridSearchCV(knn, param_grid,scoring='r2', verbose=4, cv=5)
 
    # fit gridsearchcv
    grid_search.fit(X_train, y_train)

    # Print best parameters and best score
    print('Best Parameters:', grid_search.best_params_)
    print('Best Score:', grid_search.best_score_)

    # create knn_best using best parameters
    knn_best = KNeighborsRegressor(**grid_search.best_params_)
    return knn_best

'''--------------------------------------XGBoost----------------------------------------------'''

def xgb_hyperparameter(X_train, y_train):
    hyperparameter = {
    'n_estimators': [100,500,900,1100,1500],
    'max_depth': [2,3,5,10,15],
    'eta': [0.05,0.1,0.15,0.2] , #learning rate
    'min_child_weight': [1,2,3,4],
    'booster': ['gbtree','gblinear'],
    'base_score': [0.025,0.5,0.75,1]    
    }
    
    xgb = XGBRegressor()

    random_search  = RandomizedSearchCV(xgb , hyperparameter , cv =5 ,scoring = 'r2' ,verbose=4 ,n_iter = 150)
    random_search.fit(X_train, y_train)

    # Print best parameters and best score
    print('Best Parameters:', random_search.best_params_)
    print('Best Score:', random_search.best_score_)

    # create xgb_best using best parameters
    xgb_best = XGBRegressor(**random_search.best_params_)
    return xgb_best

'''--------------------------------------???----------------------------------------------'''
