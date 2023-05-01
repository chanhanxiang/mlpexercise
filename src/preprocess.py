import pandas as pd

class preeda_clean(object):
    def __init__(self, df, df_num, df_new):
        self.df = df
        self.df_num = df_num
        self.df_new = df_new

    def datesplit(df):
        '''Split the date column into 3 seperate columns of year, month and date as the first step,
        and then converting the datatypes from object to integer as the second step'''
        df[["year", "month", "date"]] = df['Date'].str.split('-', expand=True)
        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)
        df['date'] = df['date'].astype(int)

    def rainpred(df):
        '''Create new column, "rainpred" by classifying Rainfall values that is >1.0mm as 1 (positive),
        and Rainfall values that are <= 1.0mm as 0 (negative).
        To align with fishing company’s definition, that a day is considered as rainy
        when there is more than 1.0 mm of rain in the day'''
        df['rainpred'] = [0 if x <= 1.0 else 1 for x in df['Rainfall']]

    def absolute(df):
        '''Negative values are observed in Sunshine column, most likely errorneously entered. To convert them as positive value'''
        df['Sunshine'] = df['Sunshine'].abs()

    def camelcase(df):
        '''To fix camelcase characters observed in Pressure9am and Pressure3pm columns'''
        df['Pressure9am'] = df['Pressure9am'].str.lower()
        df['Pressure3pm'] = df['Pressure3pm'].str.lower()

    def replacena(df):
        '''Around 3000 entries are filled as "None" instead of No or Yes. 
        In the context of weather assessment, None can be taken to imply as No rain'''
        df['RainToday'] = df['RainToday'].fillna(value='No')

    def dropna(df):
        '''Small numbers of NA (less than 500) have been observed out of a total of 12000 samples. 
        For reasons of consistency, all other columns with NA shall be dropped'''
        df.dropna(axis ='rows', inplace = True) 

    def nullcheck(df):
        '''Verify that all NaN have been procesed or dropped'''
        if df.isnull().any().any():
            print("There are null values in the DataFrame")
        else:
            print("There are no null values in the DataFrame")

class posteda_clean(object):
    def objmap(df_new):
        '''Replace No with 0 and Yes with 1'''
        df_new['RainToday'] = df_new['RainToday'].replace({'No': 0, 'Yes': 1})
        df_new['RainTomorrow'] = df_new['RainTomorrow'].replace({'No': 0, 'Yes': 1})

    def onehotencode_drop(df_new):
        '''Step 1: Apply one-hot encoding to multi-class categorical variables, 
        and re-concatenating them after performing one-hot encoding. 
        For the expanded categories, the prefix of the initial class is retained, 
        and the name of each attribute is appended onto the class. 
        Step 2: Drop the processed columns from the re-constituted main dataframe.
        At the final step, apply pickle so that dataframe can be loaded properly onto ipynb.'''
        Location_dummies = pd.get_dummies(df_new["Location"], drop_first=True, prefix="Location", dtype=int)
        Pressure9am_dummies = pd.get_dummies(df_new["Pressure9am"], drop_first=True, prefix="P9am", dtype=int)
        Pressure3pm_dummies = pd.get_dummies(df_new["Pressure3pm"], drop_first=True, prefix="P3pm", dtype=int)
        WindGustDir_dummies = pd.get_dummies(df_new["WindGustDir"], drop_first=True, prefix="WindGustDir", dtype=int)
        WindDir9am_dummies = pd.get_dummies(df_new["WindDir9am"], drop_first=True, prefix="WD9am", dtype=int)
        WindDir3pm_dummies = pd.get_dummies(df_new["WindDir3pm"], drop_first=True, prefix="WD3pm", dtype=int)
        df_new = pd.concat([df_new, Location_dummies,  Pressure9am_dummies, Pressure3pm_dummies, WindDir9am_dummies, \
                            WindDir3pm_dummies, WindGustDir_dummies], axis=1)
        df_new = df_new.drop(['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', \
                'ColourOfBoats', 'Pressure9am', 'Pressure3pm'], axis=1)
        df_new.to_pickle('df_new.pkl')

