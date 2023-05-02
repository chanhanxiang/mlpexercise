
import pandas as pd


# Steps for preprocessing df:

def df_preprocess(df):
    '''
    1. Split the date column into 3 seperate columns of year, month and date as the first step,
    and then converting the datatypes from object to integer as the second step
    2. Create new column, "rainpred" by classifying Rainfall values that is >1.0mm as 1 (positive),
    and Rainfall values that are <= 1.0mm as 0 (negative).
    To align with fishing company’s definition, that a day is considered as rainy
    when there is more than 1.0 mm of rain in the day
    3. Negative values are observed in Sunshine column, most likely errorneously entered. To convert them as positive value
    4. To fix camelcase characters observed in Pressure9am and Pressure3pm columns
    5. Around 3000 entries are filled as "None" instead of No or Yes. 
    In the context of weather assessment, None can be taken to imply as No rain.
    6. Small numbers of NA (less than 500) have been observed out of a total of 12000 samples. 
    For reasons of consistency, all other columns with NA shall be dropped.
    '''
    df[["year", "month", "date"]] = df['Date'].str.split('-', expand=True)
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['date'] = df['date'].astype(int)
    df['rainpred'] = [0 if x <= 1.0 else 1 for x in df['Rainfall']]
    df['Sunshine'] = df['Sunshine'].abs()
    df['Pressure9am'] = df['Pressure9am'].str.lower()
    df['Pressure3pm'] = df['Pressure3pm'].str.lower()
    df['RainToday'] = df['RainToday'].fillna(value='No')
    df.dropna(axis ='rows', inplace = True) 
    return df

#Split dataframe for EDA:

def dfcat(df, df_cat):
    '''Generate sub-dataframe for categorical variables (dtype = object)'''
    df_cat = df.copy().select_dtypes(include=["object"])
    return df_cat

def dfnum(df, df_num):
    '''Generate sub-dataframe for numerical variables (dtype = float or integer)'''
    df_num = df.copy().select_dtypes(include=["float64", "int32", "int64"])
    df_num = df_num.drop(["Rainfall", "rainpred"], axis=1).copy()
    return df_num

def IQR(df_num):
    '''Return rows that fall within Q3 and Q1, and drop outliers'''
    Q1 = df_num.quantile(0.25)
    Q3 = df_num.quantile(0.75)
    IQR = Q3 - Q1
    df_num = df_num[~((df_num < (Q1 - 1.5 * IQR)) |(df_num > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df_num

# Steps for preprocessing df (2nd round):

def rejoin(df, df_cat, df_num, df_new):
    '''Perform double inner join to reconstitute df'''
    df_new = df_num.join(df_cat, how='inner')
    df_new = df_new.join(df["rainpred"], how='inner')
    return df_new 

def objmap(df_new):
    '''Replace No with 0 and Yes with 1'''
    df_new['RainToday'] = df_new['RainToday'].replace({'No': 0, 'Yes': 1})
    df_new['RainTomorrow'] = df_new['RainTomorrow'].replace({'No': 0, 'Yes': 1})
    return df_new

def onehotencode_drop(df_new):
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
    return df_new