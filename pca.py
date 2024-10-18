import pandas as pd

df = pd.read_csv("data.csv", header=0)

# Access mode needs to be one-hot-encoded, meaning that the dataframe needs to be transformed.
features = ['technology_node','cache_size','cache_level_L2','cache_level_L3','associativity','ports.exclusive_read_port','ports.exclusive_write_port','uca_bank_count',
            'access_mode_fast', 'access_mode_normal', 'access_mode_sequential']
target = ['Access time (ns)']
df = pd.get_dummies(df, columns=['access_mode', 'cache_level'], dtype=int)
df = df[['technology_node','cache_size','cache_level_L2','cache_level_L3','associativity','ports.exclusive_read_port','ports.exclusive_write_port','uca_bank_count', 
         'access_mode_fast', 'access_mode_normal', 'access_mode_sequential', 'Access time (ns)', 'Cycle time (ns)',
         'Total dynamic read energy per access (nJ)', 'Total dynamic write energy per access (nJ)', 'Total leakage power of a bank (mW)', 'elapsed_time (s)']]

X = df[features].values
y = df[target].values

# Split the X and y set into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# preprocess the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)