import pandas as pd
import numpy as np

def load_data(filepath):
    
    return pd.read_csv(filepath)

def clean_data(df):
    df.dropna(subset = ['y'], inplace = True)
    df.drop(['contact', 'poutcome', 'duration'], axis = 1, inplace=True) # these features have too many non values 
    df.dropna(inplace=True) # others
    for column in df.columns:

        if pd.api.types.is_numeric_dtype(df[column]):
        
            if column not in ['previous', 'pdays']:

                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5*IQR
                upper = Q3 + 1.5*IQR
                upper_array = np.where(df[column] >= upper)[0]
                lower_array = np.where(df[column] <= lower)[0]
                upper_array = [idx for idx in upper_array if idx in df.index]
                lower_array = [idx for idx in lower_array if idx in df.index]
                df = df[(df[column] >= lower) & (df[column] <= upper)]
    return df

def encode_data(df):

    df_encoded = df.replace(['yes', 'no'], [1, 0])
    
    df_encoded.replace(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], 
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace=True)
    
    df_encoded = one_hot_encode(df_encoded, 'job')

    df_encoded = one_hot_encode(df_encoded, 'education')

    df_encoded = one_hot_encode(df_encoded, 'marital')

    return df_encoded

def one_hot_encode(df, column_name):
    if column_name in df.columns:

        df_encoded = pd.get_dummies(df[column_name], prefix=column_name, dtype=int)

        df.drop(column_name, axis=1, inplace=True)

        df = pd.concat([df, df_encoded], axis=1)

        print("One-hot encoding applied to column ", column_name, " successfully.")

        return df

    else:

        print(f"Column '{column_name}' not found in the DataFrame.")
    
    return df
