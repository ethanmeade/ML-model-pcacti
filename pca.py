import pandas as pd

df = pd.read_csv("data.csv", header=0)

# Access mode needs to be one-hot-encoded, meaning that the dataframe needs to be transformed.
features = ['technology_node','cache_size','cache_level','associativity','ports.exclusive_read_port','ports.exclusive_write_port','uca_bank_count','access_mode']