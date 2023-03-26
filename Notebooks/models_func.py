import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF , ExpSineSquared , WhiteKernel ,Matern,Sum,Product, DotProduct
from sklearn.model_selection import train_test_split ,GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score ,accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier


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
    for train_index, test_index in split.split(df, df['LOS']):
        strat_train_set = df.iloc[train_index]
        strat_test_set = df.iloc[test_index]
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

def residual_plot(model , X_train ,X_test, y_train,y_test , Min ,Max):    
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
     plt.hlines(y=0, xmin=Min,xmax=Max, lw=2, color='red')     
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


#get x_train and y_train for liner model
def hyper_feature( model_with_params , X , y):
    best_r2 = -float('inf')
    best_feature = None 
    

    for col in X.columns: 
        # squared the col
        col_squared = X[col] ** 2
        # new df 
        df_new = pd.concat([X.drop(col,axis=1), col_squared.rename(col + '_squared')], axis=1)
        #split the data
        X_train, X_test, y_train, y_test = train_test_split(df_new,y, test_size=0.3, random_state=42)

        model = model_with_params

        model.fit(X_train, y_train)

        r2 = r2_score( y_test, model.predict(X_test))
        #save the best result
        if r2 > best_r2:
            best_r2 = r2
            best_feature = col
            

        # Print the R-squared value of the model for the current feature
        print(f"Feature_squared: {col}, R-squared: {r2}")

    # Print the best feature squared and best R-squared value
    print(f"Best feature_squared: {best_feature}, Best R-squared: {best_r2}")
    return best_feature


def feature_squared_score( model_with_params ,col, X_train ,X_test, y_train ,y_test):
    col_squared_train = X_train[col] ** 2
    X_train_new = pd.concat([X_train.drop(col,axis=1), col_squared_train.rename(col + '_squared')], axis=1)
    col_squared_test = X_test[col] ** 2
    X_test_new = pd.concat([X_test.drop(col,axis=1), col_squared_test.rename(col + '_squared')], axis=1)
    model = model_with_params
    model.fit(X_train_new,y_train)
    print(r2_score(y_test, model.predict(X_test_new)))
    

'''--------------------------------------Pipelines----------------------------------------------'''

standartize = Pipeline([('std_scaler', StandardScaler())])


'''--------------------------------------linear regression----------------------------------------------'''
     
#receives a trained model
def feature_importances_LinearRegression(model,X_train):
    # Print feature coefficients
    feature_coef_dict = {}
    for feature, coef in zip(X_train.columns, model.coef_):
        feature_coef_dict[feature] = coef
        #print("{} coefficient: {:.3f}".format(feature, coef))
    sorted_feature_importances_dict = sorted(feature_coef_dict.items(), key=lambda x:x[1], reverse=True)
    for i in sorted_feature_importances_dict:
        print(i)

    

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
    kernels = [Product(DotProduct() + WhiteKernel(), RBF()),
              Product(DotProduct() + WhiteKernel(), Matern())]
   
    hyperparameters = [{'kernel__k1__k2__noise_level': [0.2, 0.5, 0.7], 
                       'kernel__k2__length_scale': [0.8, 1.0, 1.2]},
                       {'kernel__k1__k2__noise_level': [0.2, 0.5, 0.7], 
                       'kernel__k2__length_scale': [0.8, 1.0, 1.2],
                       'kernel__k2__nu': [1.5, 2.5]}]
    best_score = -np.inf
    for i, kernel in enumerate(kernels):
        gp = GaussianProcessRegressor(kernel=kernel)
        grid  = GridSearchCV(gp, hyperparameters[i],scoring='r2',verbose=5, cv=4)
        grid.fit(X_train, y_train)
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_kernel = grid.best_estimator_.kernel_

    # Fit the Gaussian process to the data with the best kernel and hyperparameters
    gp = GaussianProcessRegressor(kernel=best_kernel)
    return gp, best_kernel
    
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
    'max_depth': [2,3,5],
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

'''--------------------------------------ElasticNet----------------------------------------------'''

def  ElasticNet_hyperparameter(X_train, y_train):
    # Set up the grid search parameters
    param_grid = {'alpha': [0.1, 0.5 , 1.0, 2.0, 4.0, 6.0, 8.0 ,10.0],
                  'l1_ratio': [0.1,0.3 , 0.5,0.7, 0.9]}

    enet = ElasticNet()

    grid_search = GridSearchCV(enet, param_grid,scoring='r2', verbose=5, cv=5)
 
    # fit gridsearchcv
    grid_search.fit(X_train, y_train)

    # Print best parameters and best score
    print('Best Parameters:', grid_search.best_params_)
    print('Best Score:', grid_search.best_score_)

    # create ElasticNet_best using best parameters
    enet_best = ElasticNet(**grid_search.best_params_)
    return enet_best



'''--------------------------------------RandomForestR----------------------------------------------'''

def rf_hyperparameter(X_train, y_train):
    rfc = RandomForestClassifier(random_state=42)

    # Define the hyperparameter space
    hyperparameter = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 7,9],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
        }
    grid_search  = GridSearchCV(rfc , hyperparameter , cv =5 ,scoring = 'r2' ,verbose=4 )
    grid_search.fit(X_train, y_train)

    # Print best parameters and best score
    print('Best Parameters:', grid_search.best_params_)
    print('Best Score:', grid_search.best_score_)

    # create rfc_best using best parameters
    rfc_best = RandomForestRegressor(**grid_search.best_params_)
    return rfc_best

'''--------------------------------------RandomForestC----------------------------------------------'''

def rf_c_hyperparameter(X_train, y_train):
    hyperparameter = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4, 8],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestRegressor()

    grid_search  = GridSearchCV(rf , hyperparameter , cv =5 ,scoring = 'r2' ,verbose=4 )
    grid_search.fit(X_train, y_train)

    # Print best parameters and best score
    print('Best Parameters:', grid_search.best_params_)
    print('Best Score:', grid_search.best_score_)

    # create rf_best using best parameters
    rf_best = RandomForestRegressor(**grid_search.best_params_)
    return rf_best
