import math
from os import path
from os import mkdir
import numpy as np
from collect_data import inputs, input_names, output_names, get_dataframes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Union, List, Tuple
import torch
from torch import nn
from cactimod import CactiDataset
from cactimod import CactiNet
from tqdm import tqdm

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Get an argument parser to help control the automation?
# TODO: Add ability to iterate over all previous checkpoints in a file and use test set on them; how do they square up?
#   As in, does it begin to show worse performance at 225 epochs compared to 200? That sort of thing.
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(prog='Train MLP for all Params',
                                    description='Pass special instructions for what to do; if nothing passed then just read everything from settings.cfg and go from there.')
    parser.add_argument('--num_runs', type=int)
    parser.add_argument('-t', '--only_test', action='store_true')
    # The file input here SHOULD default to None (the Python type, not a string "None")
    parser.add_argument('-f', '--infile', type=str)
    parser.add_argument('--save_charts', action='store_true')
    parser.add_argument('-m', '--mode', default='All', choices=['All', 'Four', 'Single'])

    args = parser.parse_args()
    return args

settings_file = open("settings.cfg",'r')
while settings_file.__next__()!="Setup\n":
    pass

BATCH_SIZE = 64
EPOCHS = 1#150
LEARNING_RATE = 1e-4
SAVE_MODEL = True

def parse_next_config(f,name):
    split_line = f.__next__().split(":")
    assert split_line[0].strip() == name
    split_line = split_line[1].split("#")[0]
    split_line = split_line.split(",")
    return [cfgval.strip() for cfgval in split_line]

config_random_seed = int(parse_next_config(settings_file, "Random State")[0])
config_split_method, config_split_argument2, config_split_argument3 = "", "", ""
tmp = parse_next_config(settings_file, "Train Test Split Method")
if len(tmp) == 3:
    config_split_method, config_split_argument2, config_split_argument3 = tmp
else:
    config_split_method, config_split_argument2 = tmp
config_output_idx = output_names.index(parse_next_config(settings_file, "Output Select")[0])
# print(config_output_idx)
config_method = parse_next_config(settings_file, "Method")[0]
config_paramsearch = parse_next_config(settings_file, "Param Search")[0]
bayes_search_niter = 5

regression_mode = "Four"

out_dir_sub = ""
multi_out_str = "Multi/"

def technode_to_dir_name(technode):
    if technode == "0.014":
        # if config_split_argument3 == "0.016":
        #     return multi_out_str + "TNode14nm_16nm_All"
        return "14nm"
    elif technode == "0.016":
        return "16nm"
    elif technode == "0.022":
        return "22nm"
    elif technode == "0.032":
        return "32nm"
    elif technode == "0.045":
        return "45nm"
    elif technode == "0.065":
        return "65nm"
    elif technode == "0.090":
        return "90nm"
    else:
        raise ValueError(f"Unsupported tech node: {config_split_argument2}; check settings.cfg!")
        return ""

def get_out_dir_sub():
    #TODO: FIX THIS TO BEHAVE BETTER INSTEAD OF JUST A BIG ENUMERATION
    # Also take into account doing singles, as well.
    if config_split_method == "Random Split":
        return multi_out_str + "RandomSplit_All"
    elif config_split_method == "Tech Node":
        result = multi_out_str
        result += "TNode" + technode_to_dir_name(config_split_argument2)
        if not config_split_argument3 == "":
            result += "_" + technode_to_dir_name(config_split_argument3)
        # Now check for secondary tech node, and if its an "all 5" or "all (4)" deal.
        if regression_mode == 'All':
            result += "_All"
        elif regression_mode == 'Four':
            result += "_First4"
        else:
            result += "_Single"
        return result
    else:
        raise ValueError(f"Unsupported train-test split method: {config_split_method}; check settings.cfg!")
        return ""

if SAVE_MODEL:
    out_dir_sub = get_out_dir_sub()

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
    # Adding the option to choose a second node...
    "Splits data according to settings.cfg and shuffles"
    if config_split_method == "Random Split":
        test_ratio = float(config_split_argument2)
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=config_random_seed)
    else:
        # manually shuffle data
        np.random.seed(config_random_seed)
        permute = np.random.permutation(len(X))
        X, Y = X[permute], Y[permute]
        target_node = input_transforms("technology_node", config_split_argument2)
        test_indices = (np.abs(X[:,0] - target_node) < 1e-6)
        if not config_split_argument3 == "":
            target_node = input_transforms("technology_node", config_split_argument3)
            test_indices += (np.abs(X[:,0] - target_node) < 1e-6)
        # test_indices = (np.concatenate(test_indices, (np.abs(X[:,0] - target_node) < 1e-6)))
        X_train_val, Y_train_val = X[~test_indices], Y[~test_indices]
        X_test, Y_test = X[test_indices], Y[test_indices]
    return X_train_val, X_test, Y_train_val, Y_test

def split_train_val(X_train_val, Y_train_val):
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, Y_train_val, test_size=0.1, random_state=42)
    return X_train, X_val, y_train, y_val

def normalize_outputs(y_tensor, y_mean, y_std):
    return (y_tensor - y_mean) / y_std

def unnormalize_outputs(y_predicted, y_mean, y_std):
    return (y_predicted * y_std) + y_mean

def compute_multioutput_loss_all(y_predict, y_target, loss_func):
    acc_loss = loss_func(y_predict[:,0], y_target[:,0])
    cyc_loss = loss_func(y_predict[:,1], y_target[:,1])
    read_loss = loss_func(y_predict[:, 2], y_target[:,2])
    write_loss = loss_func(y_predict[:, 3], y_target[:,3])
    power_loss = loss_func(y_predict[:, 4], y_target[:,4])
    # Just sum up the MSE of all the individual losses...
    return (acc_loss + cyc_loss + read_loss + write_loss + power_loss, acc_loss, cyc_loss, read_loss, write_loss, power_loss)

def compute_multioutput_loss_four(y_predict, y_target, loss_func):
    acc_loss = loss_func(y_predict[:,0], y_target[:,0])
    cyc_loss = loss_func(y_predict[:,1], y_target[:,1])
    read_loss = loss_func(y_predict[:, 2], y_target[:,2])
    write_loss = loss_func(y_predict[:, 3], y_target[:,3])
    # Just sum up the MSE of all the individual losses...
    return (acc_loss + cyc_loss + read_loss + write_loss, acc_loss, cyc_loss, read_loss, write_loss)

def get_artifacts_filepath(use_case, relative_out_path, batch_size, curr_epochs, TOTAL_EPOCHS):
    """
    Takes in use_case (either graph, end, or checkpt) and some other values;
    checks to make sure that the directories to save to exists (creating them if they don't),
    then outputs the filepath to use. 

    Returns the string ERROR if the usecase doesn't match any of the valid ones. 
    """
    # First ensure that you can save to... wherever it is you're trying to save to. 
    file_prefix = path.join(".", "Saved_Models", str(relative_out_path))
    if not path.exists(file_prefix):
        if not path.exists(path.join(".", "Saved_Models")):
            mkdir(path.join(".", "Saved_Models"))
        mkdir(file_prefix)
    output_path = f'./Saved_Models/{relative_out_path}/cnet_epochs{TOTAL_EPOCHS}_batchsz{batch_size}_TechN{config_split_argument2}'
    if not config_split_argument3 == "":
        output_path += f'_TechNTwo_{config_split_argument3}'
    if use_case == 'graph':
        # Return for (first fragment of) model train-validation loss graphs save location
        pass
    elif use_case == 'end':
        # Return for end-checkpoint (done training)
        output_path += f'_final_checkpoint.tar'
    elif use_case == 'checkpt':
        # The proper return string for mid-training checkpoints
        output_path += f'_curr_epochs{curr_epochs}_checkpoint.tar'
    else:
        return f'ERROR'
    
    return output_path

def load_model_checkpoint(filepath_in, torch_device, network_in, optimizer_in):
    checkpoint = torch.load(argum.infile, map_location=device)
    network_in.load_state_dict(checkpoint['model_state_dict'])
    optimizer_in.load_state_dict(checkpoint['optimizer_state_dict'])
    starting_epoch = checkpoint['epoch']
    loss_stats = checkpoint['loss']
    return checkpoint, starting_epoch, loss_stats

def train_model_all(cnet, loss_stats, starting_epoch, batch_size, EPOCHS, train_loader, val_loader):
    for epoch in tqdm(range(starting_epoch, EPOCHS)):
            
            # Print epoch
            #print(f'Starting epoch {epoch+1}')

            # TRAINING SECTION

            cnet.train()
            
            # Set current loss value
            train_epoch_loss_total = 0.0
            train_epoch_loss_access = 0.0
            train_epoch_loss_cycle = 0.0
            train_epoch_loss_read = 0.0
            train_epoch_loss_write = 0.0
            train_epoch_loss_power = 0.0
            
            # Iterate over the DataLoader for training data
            # for X_train_batch, y_train_batch in tqdm(train_loader, leave=False):
            for X_train_batch, y_train_batch in tqdm(train_loader):
                
                # Send the training stuff to the device in use
                X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
                optimizer.zero_grad()

                y_train_pred = cnet(X_train_batch)

                # train_loss = loss_function(y_train_pred, y_train_batch.unsqueeze(1))
                train_loss_struct = compute_multioutput_loss_all(y_train_pred, y_train_batch, loss_function)
                train_loss, train_l_access, train_l_cycle, train_l_read, train_l_write, train_l_power = train_loss_struct
            
                train_loss.backward()
                optimizer.step()
                
                train_epoch_loss_total += train_loss.item()
                train_epoch_loss_access += train_l_access.item()
                train_epoch_loss_cycle += train_l_cycle.item()
                train_epoch_loss_read += train_l_read.item()
                train_epoch_loss_write += train_l_write.item()
                train_epoch_loss_power += train_l_power.item()

                # VALIDATION
                with torch.no_grad():   
                    val_epoch_loss_total = 0
                    val_epoch_loss_access = 0
                    val_epoch_loss_cycle = 0
                    val_epoch_loss_read = 0
                    val_epoch_loss_write = 0
                    val_epoch_loss_power = 0

                    cnet.eval()
                    for X_val_batch, y_val_batch in val_loader:
                        X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                        y_val_pred = cnet(X_val_batch)
                            
                        # val_loss = loss_function(y_val_pred, y_val_batch.unsqueeze(1))
                        # Just staring at the validation loss raws to make sure it's low like promised...
                        # print(f"Predicted validation: \n{y_val_pred[0]}")
                        # print(f"Actual validation val: \n{y_val_batch[0]}")
                        val_loss_struct = compute_multioutput_loss_all(y_val_pred, y_val_batch, loss_function)
                        val_loss, val_l_access, val_l_cycle, val_l_read, val_l_write, val_l_power = val_loss_struct
                
                val_epoch_loss_total += val_loss.item()
                val_epoch_loss_access += val_l_access.item()
                val_epoch_loss_cycle += val_l_cycle.item()
                val_epoch_loss_read += val_l_read.item()
                val_epoch_loss_write += val_l_write.item()
                val_epoch_loss_power += val_l_power.item()
            loss_stats['total']['train'].append(train_epoch_loss_total/len(train_loader))
            loss_stats['total']['val'].append(val_epoch_loss_total/len(val_loader))
            loss_stats['access']['train'].append(train_epoch_loss_access/len(train_loader))
            loss_stats['access']['val'].append(val_epoch_loss_access/len(val_loader))
            loss_stats['cycle']['train'].append(train_epoch_loss_cycle/len(train_loader))
            loss_stats['cycle']['val'].append(val_epoch_loss_cycle/len(val_loader))
            loss_stats['read']['train'].append(train_epoch_loss_read/len(train_loader))
            loss_stats['read']['val'].append(val_epoch_loss_read/len(val_loader))
            loss_stats['write']['train'].append(train_epoch_loss_write/len(train_loader))
            loss_stats['write']['val'].append(val_epoch_loss_write/len(val_loader))
            loss_stats['power']['train'].append(train_epoch_loss_power/len(train_loader))
            loss_stats['power']['val'].append(val_epoch_loss_power/len(val_loader))

            print(f'Epoch {epoch+0:03}: | Train Loss: {train_epoch_loss_total/len(train_loader):.5f} | Val Loss: {val_epoch_loss_total/len(val_loader):.5f}')

            if SAVE_MODEL and epoch % 15 == 0 and epoch > 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': cnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_stats
                }, get_artifacts_filepath('checkpt', out_dir_sub, batch_size, epoch, EPOCHS)
                )

def train_model_four(cnet, loss_stats, starting_epoch, batch_size, EPOCHS, train_loader, val_loader):
    for epoch in tqdm(range(starting_epoch, EPOCHS)):
            
            # Print epoch
            #print(f'Starting epoch {epoch+1}')

            # TRAINING SECTION

            cnet.train()
            
            # Set current loss value
            train_epoch_loss_total = 0.0
            train_epoch_loss_access = 0.0
            train_epoch_loss_cycle = 0.0
            train_epoch_loss_read = 0.0
            train_epoch_loss_write = 0.0
            
            # Iterate over the DataLoader for training data
            # for X_train_batch, y_train_batch in tqdm(train_loader, leave=False):
            for X_train_batch, y_train_batch in tqdm(train_loader):
                
                # Send the training stuff to the device in use
                X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
                optimizer.zero_grad()

                y_train_pred = cnet(X_train_batch)

                # train_loss = loss_function(y_train_pred, y_train_batch.unsqueeze(1))
                train_loss_struct = compute_multioutput_loss_four(y_train_pred, y_train_batch, loss_function)
                train_loss, train_l_access, train_l_cycle, train_l_read, train_l_write = train_loss_struct
            
                train_loss.backward()
                optimizer.step()
                
                train_epoch_loss_total += train_loss.item()
                train_epoch_loss_access += train_l_access.item()
                train_epoch_loss_cycle += train_l_cycle.item()
                train_epoch_loss_read += train_l_read.item()
                train_epoch_loss_write += train_l_write.item()

                # VALIDATION
                with torch.no_grad():   
                    val_epoch_loss_total = 0
                    val_epoch_loss_access = 0
                    val_epoch_loss_cycle = 0
                    val_epoch_loss_read = 0
                    val_epoch_loss_write = 0

                    cnet.eval()
                    for X_val_batch, y_val_batch in val_loader:
                        X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                        y_val_pred = cnet(X_val_batch)
                            
                        # val_loss = loss_function(y_val_pred, y_val_batch.unsqueeze(1))
                        # Just staring at the validation loss raws to make sure it's low like promised...
                        # print(f"Predicted validation: \n{y_val_pred[0]}")
                        # print(f"Actual validation val: \n{y_val_batch[0]}")
                        val_loss_struct = compute_multioutput_loss_four(y_val_pred, y_val_batch, loss_function)
                        val_loss, val_l_access, val_l_cycle, val_l_read, val_l_write = val_loss_struct
                
                val_epoch_loss_total += val_loss.item()
                val_epoch_loss_access += val_l_access.item()
                val_epoch_loss_cycle += val_l_cycle.item()
                val_epoch_loss_read += val_l_read.item()
                val_epoch_loss_write += val_l_write.item()
            loss_stats['total']['train'].append(train_epoch_loss_total/len(train_loader))
            loss_stats['total']['val'].append(val_epoch_loss_total/len(val_loader))
            loss_stats['access']['train'].append(train_epoch_loss_access/len(train_loader))
            loss_stats['access']['val'].append(val_epoch_loss_access/len(val_loader))
            loss_stats['cycle']['train'].append(train_epoch_loss_cycle/len(train_loader))
            loss_stats['cycle']['val'].append(val_epoch_loss_cycle/len(val_loader))
            loss_stats['read']['train'].append(train_epoch_loss_read/len(train_loader))
            loss_stats['read']['val'].append(val_epoch_loss_read/len(val_loader))
            loss_stats['write']['train'].append(train_epoch_loss_write/len(train_loader))
            loss_stats['write']['val'].append(val_epoch_loss_write/len(val_loader))

            print(f'Epoch {epoch+0:03}: | Train Loss: {train_epoch_loss_total/len(train_loader):.5f} | Val Loss: {val_epoch_loss_total/len(val_loader):.5f}')

            if SAVE_MODEL and epoch % 15 == 0 and epoch > 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': cnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_stats
                }, get_artifacts_filepath('checkpt', out_dir_sub, batch_size, epoch, EPOCHS)
                )
            
def generate_graphs_all(loss_stats, out_dir_sub, BATCH_SIZE, EPOCHS):
        train_val_loss_df = pd.DataFrame.from_dict(loss_stats['total']).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        plt.figure(figsize=(15,8))
        sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Total-Loss/Epoch')
        # plt.show()
        # TODO: Make more compact with a for loop or something of the sort
        plt.savefig(get_artifacts_filepath('graph', out_dir_sub, BATCH_SIZE, '', EPOCHS) + '_trainvalTotalLoss.png')

        train_val_loss_acc_df = pd.DataFrame.from_dict(loss_stats['access']).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        plt.figure(figsize=(15,8))
        sns.lineplot(data=train_val_loss_acc_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Access-Loss/Epoch')
        plt.savefig(get_artifacts_filepath('graph', out_dir_sub, BATCH_SIZE, '', EPOCHS) + '_trainvalAccessLoss.png')

        train_val_loss_cyc_df = pd.DataFrame.from_dict(loss_stats['cycle']).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        plt.figure(figsize=(15,8))
        sns.lineplot(data=train_val_loss_cyc_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Cycle-Loss/Epoch')
        plt.savefig(get_artifacts_filepath('graph', out_dir_sub, BATCH_SIZE, '', EPOCHS) + '_trainvalCycleLoss.png')

        train_val_loss_read_df = pd.DataFrame.from_dict(loss_stats['read']).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        plt.figure(figsize=(15,8))
        sns.lineplot(data=train_val_loss_read_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Read-Loss/Epoch')
        plt.savefig(get_artifacts_filepath('graph', out_dir_sub, BATCH_SIZE, '', EPOCHS) + '_trainvalReadLoss.png')

        train_val_loss_write_df = pd.DataFrame.from_dict(loss_stats['write']).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        plt.figure(figsize=(15,8))
        sns.lineplot(data=train_val_loss_write_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Write-Loss/Epoch')
        plt.savefig(get_artifacts_filepath('graph', out_dir_sub, BATCH_SIZE, '', EPOCHS) + '_trainvalWriteLoss.png')

        train_val_loss_power_df = pd.DataFrame.from_dict(loss_stats['power']).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        plt.figure(figsize=(15,8))
        sns.lineplot(data=train_val_loss_power_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Power-Loss/Epoch')
        plt.savefig(get_artifacts_filepath('graph', out_dir_sub, BATCH_SIZE, '', EPOCHS) + '_trainvalPowerLoss.png')

def generate_graphs_four(loss_stats, out_dir_sub, BATCH_SIZE, EPOCHS):
        train_val_loss_df = pd.DataFrame.from_dict(loss_stats['total']).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        plt.figure(figsize=(15,8))
        sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Total-Loss/Epoch')
        # plt.show()
        # TODO: Make more compact with a for loop or something of the sort
        plt.savefig(get_artifacts_filepath('graph', out_dir_sub, BATCH_SIZE, '', EPOCHS) + '_trainvalTotalLoss.png')

        train_val_loss_acc_df = pd.DataFrame.from_dict(loss_stats['access']).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        plt.figure(figsize=(15,8))
        sns.lineplot(data=train_val_loss_acc_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Access-Loss/Epoch')
        plt.savefig(get_artifacts_filepath('graph', out_dir_sub, BATCH_SIZE, '', EPOCHS) + '_trainvalAccessLoss.png')

        train_val_loss_cyc_df = pd.DataFrame.from_dict(loss_stats['cycle']).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        plt.figure(figsize=(15,8))
        sns.lineplot(data=train_val_loss_cyc_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Cycle-Loss/Epoch')
        plt.savefig(get_artifacts_filepath('graph', out_dir_sub, BATCH_SIZE, '', EPOCHS) + '_trainvalCycleLoss.png')

        train_val_loss_read_df = pd.DataFrame.from_dict(loss_stats['read']).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        plt.figure(figsize=(15,8))
        sns.lineplot(data=train_val_loss_read_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Read-Loss/Epoch')
        plt.savefig(get_artifacts_filepath('graph', out_dir_sub, BATCH_SIZE, '', EPOCHS) + '_trainvalReadLoss.png')

        train_val_loss_write_df = pd.DataFrame.from_dict(loss_stats['write']).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        plt.figure(figsize=(15,8))
        sns.lineplot(data=train_val_loss_write_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Write-Loss/Epoch')
        plt.savefig(get_artifacts_filepath('graph', out_dir_sub, BATCH_SIZE, '', EPOCHS) + '_trainvalWriteLoss.png')

if __name__ == "__main__":
    argum = parse_arguments()
    regression_mode = argum.mode
    # preprocess data
    frames = get_dataframes()
    X, Y = transform_frames(frames)
    y = Y
    if regression_mode == "Four":
        # Include all EXCEPT for power leakage. Can calculate that on it's own.
        y = Y[:,0:-1]
    print("X[0]: ", X[0], " y[0]: ", y[0])
    print("X.shape: ", X.shape, " y.shape: ", y.shape)

    # for param2 in tqdm(["0.014", "0.016", "0.032"]):
    # for param2 in tqdm(["0.045", "0.065", "0.090"]):
    # for param2 in tqdm(["0.045", "0.065"]):
    # TODO: FIX THIS TO BE LESS HARD CODED
    # for param2 in tqdm(["0.032"]):

    # config_split_argument2 = param2
    out_dir_sub = get_out_dir_sub()
    print(f"Currently operating on: {config_split_argument2} {config_split_argument3}")

    X_train_val, X_test, y_train_val, y_test = split_train_test(X, y)
    
    # Normalize the y_train_val outputs so they aren't so crazy
    y_mean = np.mean(y_train_val, axis=0)
    y_std = np.std(y_train_val, axis=0)

    y_train_val = normalize_outputs(y_train_val, y_mean, y_std)

    #TODO: Fix the rest of this to work with multiple output s***

    # Just checks to see which tech nodes are being used in training...
    unique_tech_node = []
    for line in X_train_val:
        if line[0] not in unique_tech_node:
            unique_tech_node.append(line[0])
    print(f"Unique Tech Nodes: {unique_tech_node}")

    X_train, X_val, y_train, y_val = split_train_val(X_train_val, y_train_val)

    # print(f"X_Train: {X_train[0:3]}\n\nX_Test: {X_test[0:3]}")
    # train model and predict

    # Check metal GPU is around
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # If CUDA is available instead, jump for them:
    if torch.cuda.is_available():
        device = torch.device("cuda")

    print(f"DEVICE: {device}")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train, y_train = np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32)
    X_val, y_val = np.array(X_val, dtype=np.float32), np.array(y_val, dtype=np.float32)
    X_test, y_test = np.array(X_test, dtype=np.float32), np.array(y_test, dtype=np.float32)

    train_dataset = CactiDataset(X_train, y_train)
    val_dataset = CactiDataset(X_val, y_val)
    test_dataset = CactiDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1)

    # Initialize the MLP
    cnet = CactiNet(outputs=regression_mode)
    cnet.to(device)
    
    # Define the loss function and optimizer
    #loss_function = nn.L1Loss()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(cnet.parameters(), lr=LEARNING_RATE)

    starting_epoch = 0
    loss_stats = {}
    if regression_mode == "All":
        loss_stats = {
            'total': {'train': [], 'val': []},
            'access': {'train': [], 'val': []},
            'cycle': {'train': [], 'val': []},
            'read': {'train': [], 'val': []},
            'write': {'train': [], 'val': []},
            'power': {'train': [], 'val': []}
        }
    elif regression_mode == "Four":
        loss_stats = {
            'total': {'train': [], 'val': []},
            'access': {'train': [], 'val': []},
            'cycle': {'train': [], 'val': []},
            'read': {'train': [], 'val': []},
            'write': {'train': [], 'val': []}
        }

    if argum.infile:
        checkpoint, starting_epoch, loss_stats = load_model_checkpoint(argum.infile, device, cnet, optimizer)
    
    # Do training only if not told not to OR not given any previous data to work with.
    if not argum.only_test or not argum.infile:
        # Run the training loop
        # for epoch in tqdm(range(starting_epoch, EPOCHS), leave=False):
        if regression_mode == "All":
            train_model_all(cnet, loss_stats, starting_epoch, BATCH_SIZE, EPOCHS, train_loader, val_loader)
        else:
            # Do four grade
            train_model_four(cnet, loss_stats, starting_epoch, BATCH_SIZE, EPOCHS, train_loader, val_loader)


        # Training is complete.
        print('Training process has finished.')

        if SAVE_MODEL and argum.save_charts:
            if regression_mode == "All":
                generate_graphs_all(loss_stats, out_dir_sub, BATCH_SIZE, EPOCHS)
            else:
                generate_graphs_four(loss_stats, out_dir_sub, BATCH_SIZE, EPOCHS)


    else:
        epoch = -1

    y_pred_list = []
    with torch.no_grad():
        cnet.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = cnet(X_batch)
            y_pred_list.append(y_test_pred.cpu().numpy())
    y_pred_list = np.array([a.squeeze().tolist() for a in y_pred_list])

    # Apply the reverse normalization
    y_pred_list = unnormalize_outputs(y_pred_list, y_mean, y_std)

    # Column names for pandas dataframe
    column_names = ["Access Time (ns)", "Cycle Time (ns)", "Total dynamic read energy per access (nJ)", "Total dynamic write energy per access (nJ)", "Total leakage power of a bank (mW)"]

    # Index can be done with range()
    df_index = range(y_test.shape[0])

    # mse = mean_squared_error(y_test, y_pred_list)
    # r_square = r2_score(y_test, y_pred_list)
    # print("Mean Squared Error :",mse)
    # print("R^2 :",r_square)

    mse_acc = mean_squared_error(y_test[:,0], y_pred_list[:,0])
    print(f"Mean Squared Error for Access Time (ns): {mse_acc}")
    mse_cyc = mean_squared_error(y_test[:,1], y_pred_list[:,1])
    print(f"Mean Squared Error for Cycle Time (ns): {mse_cyc}")
    mse_read = mean_squared_error(y_test[:,2], y_pred_list[:,2])
    print(f"Mean Squared Error for Dynam. Read Energy (nJ): {mse_read}")
    mse_write = mean_squared_error(y_test[:,3], y_pred_list[:,3])
    print(f"Mean Squared Error for Dynam. Write Energy (nJ): {mse_write}")
    mse_total_avg = 0
    if regression_mode == 'All':
        mse_power = mean_squared_error(y_test[:,4], y_pred_list[:,4])
        print(f"Mean Squared Error for Leakage Power (mW): {mse_power}")
        mse_total_avg = (mse_acc + mse_cyc + mse_read + mse_write + mse_power)/5
    else:
        mse_total_avg = (mse_acc + mse_cyc + mse_read + mse_write)/4
    print(f"Average Mean Squared Error across all outputs: {mse_total_avg}")

    r2_acc = r2_score(y_test[:,0], y_pred_list[:,0])
    print(f"R^2 Score for Access Time (ns): {r2_acc}")
    r2_cyc = r2_score(y_test[:,1], y_pred_list[:,1])
    print(f"R^2 Score for Cycle Time (ns): {r2_cyc}")
    r2_read = r2_score(y_test[:,2], y_pred_list[:,2])
    print(f"R^2 Score for Dynam. Read Energy (nJ): {r2_read}")
    r2_write = r2_score(y_test[:,3], y_pred_list[:,3])
    print(f"R^2 Score for Dynam. Write Energy (nJ): {r2_write}")
    r2_total_avg = 0
    if regression_mode == 'All':
        r2_power = r2_score(y_test[:,4], y_pred_list[:,4])
        print(f"R^2 Score for Leakage Power (mW): {r2_power}")
        r2_total_avg = (r2_acc + r2_cyc + r2_read + r2_write + r2_power)/5
    else:
        r2_total_avg = (r2_acc + r2_cyc + r2_read + r2_write)/4
    print(f"Average R^2 Score across all outputs: {r2_total_avg}")

    # test_loss = compute_multioutput_loss(torch.tensor(y_pred_list), torch.tensor(y_test), loss_function)

    if SAVE_MODEL and not argum.only_test:

        # if epoch > EPOCHS:
        #     EPOCHS = epoch

        torch.save({
                'epoch': EPOCHS,
                'model_state_dict': cnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_stats
            }, get_artifacts_filepath('end', out_dir_sub, BATCH_SIZE, '', EPOCHS)
            )

