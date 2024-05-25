import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def scale_features(df):

    # apply normalization 
    std_scaler = StandardScaler()

    df_scaled = std_scaler.fit_transform(df.drop('y', axis=1).values)

    df_scaled = pd.DataFrame(df_scaled, columns=df.drop('y', axis=1).columns)

    df_scaled['y'] = df['y']

    df_scaled.index = df.index

    return df_scaled

def reduce_dimensions(df):

    # Select samples, transform and reduce features
    df_reduced = pd.concat([df[df['y'] == 1].head(4000).reset_index(drop=True),
                                df[df['y'] == 0].tail(4000).reset_index(drop=True)],
                                ignore_index=False).sort_index(kind='merge')

    dim_red = PCA()

    x_pca = dim_red.fit_transform(df_reduced.drop('y', axis=1))

    training_features = ['pca0', 'pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'pca8', 'pca9', 'pca10', 'pca11', 'pca12', 'pca13']

    df_pca = pd.DataFrame(x_pca, columns=dim_red.get_feature_names_out())

    df_pca_working_set = df_pca[training_features]

    df_pca_working_set = df_pca_working_set.reset_index(drop=True)

    df_pca_working_set['y'] = df_reduced.reset_index(drop=True)['y']

    return df_pca_working_set


