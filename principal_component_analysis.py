# Try to use PCA to analyze data, look at shape. 
import math
from os import path
from os import mkdir
import numpy as np
from collect_data import inputs, input_names, output_names, get_dataframes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('data.csv')