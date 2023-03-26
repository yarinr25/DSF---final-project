import random
from xgboost import XGBClassifier
import pandas as pd
import joblib
model = joblib.load('XGBC_Expedia.pkl')

class Hotel:
    def __init__(self, row):
        self.row = row
        
        # Add any other attributes you need

    def compare(self, other):
        if self.row['Hotel Name'] == other.row['Hotel Name']:
            return 1
        else:
            return 0
        
def expedia_compare(h1, h2):
   
   h1_df = pd.DataFrame(h1.row).T
   h2_df = pd.DataFrame(h2.row).T
   
   columns_to_drop = ['Hotel Name', 'Monday', 'Wednesday', 'Friday', 'Original Price']
   for df in [h1_df, h2_df]:
      df.drop(columns_to_drop, axis=1, inplace=True)
      
   compare_row = h1_df.merge(h2_df, how='left', on=['Snapshot', 'TTT', 'LOS'], suffixes=('_1', '_2'))
   
   numeric_cols = ['Grade', 'Num of Reviews', 'Curr Price']
   
   for col in numeric_cols:
      compare_row[col+'_diff'] = float(compare_row[col+'_1'] - compare_row[col+'_2'])
      compare_row.drop([col+'_1', col+'_2'], axis=1, inplace=True)
   
   model_cols = ['Grade_diff', 'Num of Reviews_diff', 'Curr Price_diff',
       'Is refundable_both', 'Is refundable_left', 'Is refundable_none',
       'Is refundable_right', 'Late payment_both', 'Late payment_left',
       'Late payment_none', 'Late payment_right', 'Extras included_both',
       'Extras included_left', 'Extras included_none', 'Extras included_right',
       'Option Member_both', 'Option Member_left', 'Option Member_none',
       'Option Member_right', 'Discount_both', 'Discount_left',
       'Discount_none', 'Discount_right']
   
   origin_boolean_cols = ['Is refundable', 'Late payment', 'Extras included', 'Option Member','Discount']
   
   for col in origin_boolean_cols:
     compare_row[col] = compare_row.apply(lambda x: col+'_both' if x[col+'_1'] == 1 and x[col+'_2'] == 1 else col+'_none' if x[col+'_1'] == 0 and x[col+'_2'] == 0 else col+'_left' if x[col+'_1'] == 1 and x[col+'_2'] == 0 else col+'_right',axis=1)
     compare_row.drop([col+'_1', col+'_2'], axis=1, inplace=True)
      
   columns_to_drop = ['Snapshot', 'TTT', 'LOS', 'Index_1', 'Index_2']
   compare_row.drop(columns_to_drop, axis=1, inplace=True)
   
   compare_row = pd.get_dummies(compare_row, prefix='', prefix_sep='', columns = compare_row.columns[3:])
   
   diff_cols = list(set(model_cols) - set(compare_row.columns))
   for col in diff_cols:
      compare_row[col] = 0
   compare_row = compare_row[model_cols]
   return model.predict(compare_row)[0]

def booking_compare(h1, h2):
   
   h1_df = pd.DataFrame(h1.row).T
   h2_df = pd.DataFrame(h2.row).T
      
   columns_to_drop = ['Hotel Name', 'Monday', 'Wednesday', 'Friday', 'Original Price', 'Type of room']
   for df in [h1_df, h2_df]:
      df.drop(columns_to_drop, axis=1, inplace=True)
      
   compare_row = h1_df.merge(h2_df, how='left', on=['Snapshot', 'TTT', 'LOS'], suffixes=('_1', '_2'))
   
   numeric_cols = ['Grade', 'Num of Reviews', 'Curr Price','Distance from center']
   
   for col in numeric_cols:
      compare_row[col+'_diff'] = float(compare_row[col+'_1'] - compare_row[col+'_2'])
      compare_row.drop([col+'_1', col+'_2'], axis=1, inplace=True)
   
   model_cols = ['Grade_diff', 'Num of Reviews_diff', 'Curr Price_diff',
       'Distance from center_diff', 'Location grade_both',
       'Location grade_left', 'Location grade_none', 'Location grade_right',
       'Free Cancellation_both', 'Free Cancellation_left',
       'Free Cancellation_none', 'Free Cancellation_right',
       'No prepayment needed_both', 'No prepayment needed_left',
       'No prepayment needed_none', 'No prepayment needed_right',
       'Breakfast included_both', 'Breakfast included_left',
       'Breakfast included_none', 'Breakfast included_right',
       'Cancel Later_both', 'Cancel Later_left', 'Cancel Later_none',
       'Cancel Later_right', 'Discount_both', 'Discount_left', 'Discount_none',
       'Discount_right']
   
   origin_boolean_cols = ['Location grade', 'Free Cancellation', 'No prepayment needed', 'Breakfast included','Cancel Later', 'Discount']
   
   for col in origin_boolean_cols:
     compare_row[col] = compare_row.apply(lambda x: col+'_both' if x[col+'_1'] == 1 and x[col+'_2'] == 1 else col+'_none' if x[col+'_1'] == 0 and x[col+'_2'] == 0 else col+'_left' if x[col+'_1'] == 1 and x[col+'_2'] == 0 else col+'_right',axis=1)
     compare_row.drop([col+'_1', col+'_2'], axis=1, inplace=True)
      
   columns_to_drop = ['Snapshot', 'TTT', 'LOS', 'Index_1', 'Index_2']
   compare_row.drop(columns_to_drop, axis=1, inplace=True)
   
   compare_row = pd.get_dummies(compare_row, prefix='', prefix_sep='', columns = compare_row.columns[3:])
   
   diff_cols = list(set(model_cols) - set(compare_row.columns))
   for col in diff_cols:
      compare_row[col] = 0
   compare_row = compare_row[model_cols]
   return model.predict(compare_row)[0]
    
def choose_comparison(h1, h2, site):
    match site:
        case 'Booking':
            return booking_compare(h1,h2)
        case 'Expedia':
            return expedia_compare(h1,h2)


class HotelsSearch:
    def __init__(self, df ,shuffledhotels=0):
        self.hotels = [Hotel(row) for _, row in df.iterrows()]
        shuffled_hotels = self.hotels[:]
        random.shuffle(shuffled_hotels)
        self.shuffled_hotels = shuffled_hotels


    def sort_by_model(self, site):
        n = len(self.shuffled_hotels)

        for i in range(n):
            for j in range(0, n-i-1):
                if not choose_comparison(self.shuffled_hotels[j],self.shuffled_hotels[j+1],site):
                    self.shuffled_hotels[j], self.shuffled_hotels[j+1] = self.shuffled_hotels[j+1], self.shuffled_hotels[j]


    def print_hotels(self):
        for h in self.hotels:
            print(h.row['Hotel Name'] )
            
    def print_hotels_compare(self):
        for i , h in enumerate(self.hotels):
            print(h.row['Hotel Name'] ,"   ",self.shuffled_hotels[i].row['Hotel Name'] )

    def print_shuffled_hotels(self):
        for h in self.shuffled_hotels:
            print(h.row['Hotel Name'])
    
    def get_sort_accuracy(self):

        #Calculate the percentage of successful sorts.

        correct_sorts = 0
        for i, hotel in enumerate(self.shuffled_hotels):
            if hotel.compare(self.hotels[i]):
                correct_sorts = correct_sorts+1
        return correct_sorts / len(self.shuffled_hotels)
    
    def get_sort_rmse(self):

        sum_range = 0
        for i, h_p in enumerate(self.shuffled_hotels):
            for j , h_o in enumerate(self.hotels):
                if h_p.compare(h_o):
                    sum_range = sum_range+ abs(i-j) 
                    break


        return sum_range / len(self.shuffled_hotels)



    