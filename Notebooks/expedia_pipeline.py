from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from forex_python.converter import CurrencyRates


class IndexTransform (BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['Index'] = X['Index'] + 1
        max_index = X['Index'].max()
        if max_index > 100:
            X = X.loc[X['Index'] < 100].reset_index(drop=True)
        return X
    
class TypeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['Snapshot'] = pd.to_datetime(X['Snapshot']).dt.date
        for col in ['Curr Price', 'Original Price', 'Num of Reviews']:
            X[col] = X[col].apply(lambda x: int(re.sub("[^0-9]", "", x)) if not pd.isnull(x) else x)
        X[['TTT', 'LOS']] = X[['TTT', 'LOS']].astype('int64')
        return X
    
class FillNa(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['Num of Reviews'] = X['Num of Reviews'].fillna(0)
        X['Grade'] = X['Grade'].fillna(round(random.uniform(5, 6),1))
        X['Original Price'] = X.apply(lambda x: x['Curr Price']  if pd.isnull(x['Original Price']) else x['Original Price'], axis = 1)
        return X

class PriceTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        c = CurrencyRates()
        x = X['Snapshot'].unique()
        curr_dict = {}
        for date in x:
            curr_dict[date] = c.get_rate('EUR','USD', date) 
        for col in ['Curr Price', 'Original Price']:
            X[col] = X.apply(lambda x: (x[col]*curr_dict[x['Snapshot']])/x['LOS']  , axis = 1)
        X['Percent of discount'] = X.apply(lambda x: round(1 - (x['Curr Price']/x['Original Price']) , 2) if x['Original Price'] > x['Curr Price'] else 0, axis = 1)
        return X
    
    class ExtrasTransform(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            X['Extras included'] = X['Extras included'].apply(lambda x: 1 if not pd.isnull(x) else 0)
            return X
    

