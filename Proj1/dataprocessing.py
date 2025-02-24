import pandas as pd
from sklearn import preprocessing


def clean_data_set(file_name):
    data_set_name = file_name
    df = pd.read_csv(data_set_name)
    le = preprocessing.LabelEncoder()
    del df['Date']
    # convert Time values to ints
    df['Time'] = le.fit_transform(df['Time'])
    for column in df:
        if column != 'Time':
            _remove_errors(df, column)
            if (column != 'Persons') and (column != 'PIR1') and (column != 'PIR2'):
                _remove_outliers(df, column)
        _normalize(df, column)
    df.to_csv('FinalDataSet.csv')
    return df


# Remove Errors
def _remove_errors(df, column):
    _mean = df[column].mean()
    df[column] = df[column].fillna(_mean)


# Remove Outliers
def _remove_outliers(df, column):
    _first_quartile = df[column].quantile(0.25)
    _third_quartile = df[column].quantile(0.75)
    _mean = df[column].mean()
    for value in df[column].values:
        if (value > (20*_third_quartile)) or (value < (_first_quartile/1.5)):
            df[column] = df[column].replace([value], _mean)


# Data Normalization
def _normalize(df, column):
    if (column != 'PIR1') and (column != 'PIR2') and (column != 'Persons'):
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
