import math
import numpy as np
from collect_data import inputs, input_names, output_names, get_dataframes
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from typing import Union, List, Tuple
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV
from tqdm import tqdm

settings_file = open("settings.cfg",'r')
while settings_file.__next__()!="Setup\n":
    pass

def parse_next_config(f,name):
    split_line = f.__next__().split(":")
    assert split_line[0].strip() == name
    split_line = split_line[1].split("#")[0]
    split_line = split_line.split(",")
    return [cfgval.strip() for cfgval in split_line]

config_random_seed = int(parse_next_config(settings_file, "Random State")[0])
config_split_method, config_split_argument2 = parse_next_config(settings_file, "Train Test Split Method")
given_output_name = parse_next_config(settings_file, "Output Select")[0]
config_output_idx = -1
if not given_output_name == "All":
    config_output_idx = output_names.index(given_output_name)
config_method = parse_next_config(settings_file, "Method")[0]
config_paramsearch = parse_next_config(settings_file, "Param Search")[0]
bayes_search_niter = 5

def input_transforms(name:str, value:str) -> Union[float, List[float]]:
    if name == "technology_node":
        return 1000 * float(value)
    if name == "cache_size" or name == "associativity":
        return math.log(int(value), 2)
    if name == "ports.exclusive_read_port" or name == "ports.exclusive_write_port":
        return float(value)
    if name == "uca_bank_count":
        return math.log(int(value), 2)
    if name == "access_mode":
        d = {"normal":[1,0,0], "sequential":[0,1,0], "fast":[0,0,1]}
        return d[value]
    if name == "cache_level": # take into account if L2 or L3
        d = {"L2":[1,0], "L3":[0,1]}
        return d[value]

def transform_frames(frames: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = list(), list()
    for frame in frames:
        X_row, Y_row = list(), list()
        for i,name in enumerate(input_names):
            transformed = input_transforms(name, frame[i])
            if isinstance(transformed, List):
                X_row.extend(transformed)
            else:
                X_row.append(transformed)
        for i,_ in enumerate(output_names):
            Y_row.append(float(frame[i+len(input_names)]))
        X.append(X_row); Y.append(Y_row)
    return np.array(X), np.array(Y)

def split_train_test(X, Y):
    "Splits data according to settings.cfg and shuffles"
    if config_split_method == "Random Split":
        test_ratio = float(config_split_argument2)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=config_random_seed)
    else:
        # manually shuffle data
        np.random.seed(config_random_seed)
        permute = np.random.permutation(len(X))
        X, Y = X[permute], Y[permute]
        target_node = input_transforms("technology_node", config_split_argument2)
        test_indices = (np.abs(X[:,0] - target_node) < 1e-6)
        X_train, Y_train = X[~test_indices], Y[~test_indices]
        X_test, Y_test = X[test_indices], Y[test_indices]
    return X_train, X_test, Y_train, Y_test

def search_for_best(X_train, y_train, X_test, y_test, bayes_search_niter):
    # model_name = ["KNN", "MLP", "SVR", "FOREST"]
    model_name = ["KNN", "SVR", "FOREST"]
    # model_name = ["KNN", "FOREST"]
    # model_name = ["SVR"]
    with open('best_results_all.txt', 'a') as f:
        print("Predicting "+output_names[config_output_idx]+"\n")
        f.write("\nPredicting "+output_names[config_output_idx]+"\n")
        y_variance = ((y_test - y_test.mean())**2).sum() / len(y_test)
        y_std = np.sqrt(y_variance)
        print("Variance and Standard Deviation of Ground Truth: {:.4g}, {:.4g}".format(y_variance, y_std))
        f.write("Variance and Standard Deviation of Ground Truth: {:.4g}, {:.4g}\n".format(y_variance, y_std))
    for i in model_name:
        print("Training a "+ i)
        if i == "MLP":
            # # Consider tuning the hidden_layer_sizes, solver and max_iter.
            # #regr = MLPRegressor(hidden_layer_sizes=(20, 20), solver='lbfgs', max_iter=5000, random_state=config_random_seed).fit(X_train, y_train)
            # print("Ignoring MLP since we're using a more robust neural network...")
            continue
        elif i == "KNN":
            param_dist = {
                'n_neighbors': (1, 10)
            }
            grid_search = BayesSearchCV(KNeighborsRegressor(), param_dist, n_iter=bayes_search_niter, cv=5)
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            regr = KNeighborsRegressor(**best_params).fit(X_train, y_train)
        elif i == "SVR":
            param_grid  = {
                'C': (1e-6, 1e+6),
                'gamma': (1e-6, 1e+1),
                'epsilon': (1e-6, 1e+1)
            }
            grid_search = BayesSearchCV(SVR(kernel='rbf'), param_grid, n_iter=bayes_search_niter, cv=5, random_state=config_random_seed)
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            regr = SVR(kernel='rbf',**best_params).fit(X_train, y_train)
        elif i =="FOREST":
            param_dist = {
                'n_estimators': (100, 1000),
                'max_depth': (5, 50),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 20),
            }
            grid_search = BayesSearchCV(RandomForestRegressor(random_state=config_random_seed), param_dist, n_iter=bayes_search_niter, cv=5, random_state=config_random_seed)
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            regr = RandomForestRegressor(**best_params, random_state=config_random_seed).fit(X_train, y_train)    
        else:
            print("Method not supported, exiting"); exit(0)
        with open('best_results_all.txt', 'a') as f:
            f.write(f"Best Params for {i}: {best_params}\n")
            y_pred = regr.predict(X_test)
            print("Predicting "+output_names[config_output_idx])
            mse = mean_squared_error(y_pred, y_test)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            print("MSE (Mean Squared Error): {:.4g}, Root MSE: {:.4g}".format(mse,rmse))
            f.write("MSE (Mean Squared Error): {:.4g}, Root MSE: {:.4g}\n".format(mse,rmse))
            f.write((f"R-squared: {r2}\n"))
            print(f"R-squared: {r2}")
            coeff_of_determination = regr.score(X_test, y_test)
            print("Coefficient of determination: {:.4g}".format(coeff_of_determination))
            f.write("Coefficient of determination: {:.4g}\n\n".format(coeff_of_determination))
            assert(np.abs(coeff_of_determination - (1-mse/y_variance)) < 1e-4)
            print()

if __name__ == "__main__":
    # preprocess data
    frames = get_dataframes()
    X_init, Y_init = transform_frames(frames)
    y = Y_init[:,config_output_idx]
    print("X[0]: ", X_init[0], " y[0]: ", y[0])
    print("X.shape: ", X_init.shape, " y.shape: ", y.shape)

    # for param2 in tqdm(["0.014", "0.016", "0.022", "0.032", "0.045", "0.065", "0.090"]):
    for param2 in tqdm(["0.014"]):
        config_split_argument2 = param2
        X_train, X_test, y_train, y_test = split_train_test(X_init, y)

        search_for_best(X_train, y_train, X_test, y_test, bayes_search_niter)

    # for ind in tqdm(["Cycle time (ns)", "Total dynamic read energy per access (nJ)", "Total dynamic write energy per access (nJ)", "Total leakage power of a bank (mW)"])
        
    #     for param2 in tqdm(["0.014", "0.016", "0.022", "0.032", "0.045", "0.065", "0.090"], leave=False):
    #         config_split_argument2 = param2
    #         X_train, X_test, y_train, y_test = split_train_test(X_init, y)

    #         search_for_best(X_train, y_train, X_test, y_test, bayes_search_niter)
        
    # train model and predict
    
    
    # if(config_paramsearch == "False"):
    #     print("Training a "+ config_method)
    #     if config_method == "MLP":
    #         regr = MLPRegressor(hidden_layer_sizes=(20, 20), solver='lbfgs', max_iter=5000, random_state=config_random_seed).fit(X_train, y_train)
    #     elif config_method == "KNN":
    #         regr = KNeighborsRegressor(n_neighbors=1).fit(X_train, y_train)
    #     elif config_method == "SVR":
    #         regr = SVR(kernel='rbf', C= 100, gamma=0.001,epsilon=0.001, degree=5).fit(X_train, y_train)
    #     elif config_method =="FOREST":
    #         regr = RandomForestRegressor(n_estimators=100, max_depth=30, random_state=config_random_seed).fit(X_train, y_train)    
    #     else:
    #         print("Method not supported, exiting"); exit(0)
    #     y_pred = regr.predict(X_test)
    #     # evaluate results
    #     print("Predicting "+output_names[config_output_idx])
    #     mse = mean_squared_error(y_pred, y_test)
    #     rmse = np.sqrt(mse)
    #     r2 = r2_score(y_test, y_pred)
    #     print("MSE (Mean Squared Error): {:.4g}, Root MSE: {:.4g}".format(mse,rmse))
    #     print(f"R-squared: {r2}")
    #     y_variance = ((y_test - y_test.mean())**2).sum() / len(y_test)
    #     y_std = np.sqrt(y_variance)
    #     print("Variance and Standard Deviation of Ground Truth: {:.4g}, {:.4g}".format(y_variance, y_std))
    #     coeff_of_determination = regr.score(X_test, y_test)
    #     assert(np.abs(coeff_of_determination - (1-mse/y_variance)) < 1e-4)
    #     print("Coefficient of determination: {:.4g}".format(coeff_of_determination))
    # else:
    #     # model_name = ["KNN", "MLP", "SVR", "FOREST"]
    #     model_name = ["KNN", "SVR", "FOREST"]
    #     for i in model_name:
    #         print("Training a "+ i)
    #         if i == "MLP":
    #             # Consider tuning the hidden_layer_sizes, solver and max_iter.
    #             #regr = MLPRegressor(hidden_layer_sizes=(20, 20), solver='lbfgs', max_iter=5000, random_state=config_random_seed).fit(X_train, y_train)
    #             print("Ignoring MLP since we're using a more robust neural network...")
    #             continue
    #         elif i == "KNN":
    #             param_dist = {
    #                 'n_neighbors': (1, 10)
    #             }
    #             grid_search = BayesSearchCV(KNeighborsRegressor(), param_dist, n_iter=bayes_search_niter, cv=5)
    #             grid_search.fit(X_train, y_train)
    #             best_params = grid_search.best_params_
    #             regr = KNeighborsRegressor(**best_params).fit(X_train, y_train)
    #         elif i == "SVR":
    #             param_grid  = {
    #                 'C': (1e-6, 1e+6),
    #                 'gamma': (1e-6, 1e+1),
    #                 'epsilon': (1e-6, 1e+1)
    #             }
    #             grid_search = BayesSearchCV(SVR(kernel='rbf'), param_grid, n_iter=bayes_search_niter, cv=5, random_state=config_random_seed)
    #             grid_search.fit(X_train, y_train)
    #             best_params = grid_search.best_params_
    #             regr = SVR(kernel='rbf',**best_params).fit(X_train, y_train)
    #         elif i =="FOREST":
    #             param_dist = {
    #                 'n_estimators': (100, 1000),
    #                 'max_depth': (5, 50),
    #                 'min_samples_split': (2, 20),
    #                 'min_samples_leaf': (1, 20),
    #             }
    #             grid_search = BayesSearchCV(RandomForestRegressor(random_state=config_random_seed), param_dist, n_iter=bayes_search_niter, cv=5, random_state=config_random_seed)
    #             grid_search.fit(X_train, y_train)
    #             best_params = grid_search.best_params_
    #             regr = RandomForestRegressor(**best_params, random_state=config_random_seed).fit(X_train, y_train)    
    #         else:
    #             print("Method not supported, exiting"); exit(0)
    #         print(best_params, "this is the best params for "+i)
    #         with open('best_params.txt', 'w') as f:
    #             f.write(i+" "+str(best_params)+'\n')
    #         y_pred = regr.predict(X_test)
    #         print("Predicting "+output_names[config_output_idx])
    #         mse = mean_squared_error(y_pred, y_test)
    #         rmse = np.sqrt(mse)
    #         r2 = r2_score(y_test, y_pred)
    #         print("MSE (Mean Squared Error): {:.4g}, Root MSE: {:.4g}".format(mse,rmse))
    #         print(f"R-squared: {r2}")
    #         y_variance = ((y_test - y_test.mean())**2).sum() / len(y_test)
    #         y_std = np.sqrt(y_variance)
    #         print("Variance and Standard Deviation of Ground Truth: {:.4g}, {:.4g}".format(y_variance, y_std))
    #         coeff_of_determination = regr.score(X_test, y_test)
    #         print(coeff_of_determination)
    #         assert(np.abs(coeff_of_determination - (1-mse/y_variance)) < 1e-4)
    #         print("Coefficient of determination: {:.4g}".format(coeff_of_determination))
    #         print()
