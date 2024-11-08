import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# import numpy as np

class DatasetHandler:
    """
    Loads in and handles the preprocessing and manipulation of
    the data to avoid unnecessarily cluttering the main program spaces. 
    """
    def __init__(self):
        df = pd.read_csv("data.csv", header=0)

        # Access mode needs to be one-hot-encoded, meaning that the dataframe needs to be transformed.
        self.features = ['technology_node','cache_size','cache_level_L2','cache_level_L3','associativity','ports.exclusive_read_port','ports.exclusive_write_port','uca_bank_count',
                    'access_mode_fast', 'access_mode_normal', 'access_mode_sequential']
        df = pd.get_dummies(df, columns=['access_mode', 'cache_level'], dtype=int)
        self.df = df[['technology_node','cache_size','cache_level_L2','cache_level_L3','associativity','ports.exclusive_read_port','ports.exclusive_write_port','uca_bank_count', 
                'access_mode_fast', 'access_mode_normal', 'access_mode_sequential', 'Access time (ns)', 'Cycle time (ns)',
                'Total dynamic read energy per access (nJ)', 'Total dynamic write energy per access (nJ)', 'Total leakage power of a bank (mW)', 'elapsed_time (s)']]
    
    def make_training_data(self, targets: list[str], random_seed: int, return_dataframes: bool):
        X = self.df[self.features].values
        y = self.df[targets].values

        # Split the X and y set into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

        # preprocess the data
        sc = StandardScaler()

        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        if return_dataframes:
            X_train = pd.DataFrame(X_train, columns=self.features)
            X_test = pd.DataFrame(X_test, columns=self.features)
            y_train = pd.DataFrame(y_train, columns=self.features)
            y_test = pd.DataFrame(y_test, columns=self.features)

        return X_train, X_test, y_train, y_test
