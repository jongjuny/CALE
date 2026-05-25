############################################################################################
## Arguments
## - op: operator name (Train & Test for a single operator)
## - ata: 6-digit ATA number (Train & Test for a single ATA)
## - epochs: default as 100
## Date: 2026. 2. 26.
############################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

import os
import sys
import argparse

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

from models.mix_model import *

from tqdm.notebook import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

import random
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def save_pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    
#############################################################################
## Global Variables

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## Directory set-up
base_dir = "/home/mhi/Data/dataset"
# model_dir_name = 'train_model'
# predict_dir_name = 'test_results_all'
scaler_dir_name = 'scalers_logit'
config_dir_name = 'configs_logit'
model_dir_name = 'train_model_logit'
predict_dir_name = 'test_result_logit'

## Group (regional) of operators
europe_ops=['ANE', 'BCY', 'CLH']
asia_ops = ['HXA', 'IBX']
na_grp = ['EDV', 'PSY', 'SKW']
ac_model = 'CRJ700'

# all_operators = ['SKW'] #, 'PSY', 'EDV', 'ANE', 'BCY', 'CLH','HXA', 'IBX']
# all_operators2 = ['SKW', 'PSY', 'EDV', 'ANE', 'BCY', 'CLH','HXA', 'IBX']
all_operators = ['EDV', 'CLH','HXA', 'IBX', 'ANE', 'JZA', 'SKW', 'PSY']
target_atas_all= [243201, 324301, 344401, 215206]

target_atas_all2 = [243203, 253203, 313301, 324101, 324201, 324101, 335141, 722110]
## For temporal use
m_cols = ['PART_NO', 'PART_SN', 'INSTALL_DATE', 'REMOVAL_DATE', 'REMOVAL_TYPE_CODE', 'ATA_NUMBER', 
          'AC_SN', 'OPERATOR_CODE', 'FLIGHT_HOURS', 'FLIGHT_CYCLES']

## Minimum flight hours for use
MIN_FH = 10

## Important datetime split
considered_date = datetime.datetime(2015,1,1)
latest_date = datetime.datetime(2025,10,1)
test_date = datetime.datetime(2023,3,1)
# margin_duration = 365*2

## Parameter setting & default parameters
parser = argparse.ArgumentParser()
parser.add_argument('-group_op', help='  : select group of ops (ex. asia, europe, na, ...)', default='None')
parser.add_argument('-op', help='   : select a single ATA (ex. HXA)', default='SKW')
parser.add_argument('-ata6', help = '  : select ATA6', default='all')
parser.add_argument('-epochs', help= '   : set epochs for training', default=100)
parser.add_argument('-type', help='     : set training set for removal type code (all, U, S)', default='U')
parser.add_argument('-scaler', help='     : set scaler between standard, minmax, robust', default='standard')
parser.add_argument('-seq_len_ata', help='      : sequence length for train/test inputs', default=3)
parser.add_argument('-seq_len_sn', help='      : sequence length for train/test inputs', default=3)
parser.add_argument('-deviation', help='        : cum or inc for PN prediction', default='inc')
parser.add_argument('-lr', help='        : learning rate', default=0.0005)
parser.add_argument('-set_config', help='       : for Large model', default=2)
parser.add_argument('-ata_model', help='        : model name for ATA4', default='TCN_AttenMulti')
parser.add_argument('-batch', help='        : batch size for all models', default=16)
parser.add_argument('-min_records', help='        : minimum records for selected ATA-OP', default=50)
parser.add_argument('-min_acs', help='        : minimum unique ACs for selected ATA-OP', default=10)
args = parser.parse_args()

def param_passer(args):
    if args.group_op is None or args.group_op == 'None':
        selected_operators = [[args.op]]
        print('Single OP: ', selected_operators[0])
    else:
        sel_ops = args.group_op
        print('OPs: ', sel_ops)
        if args.group_op == 'asia':
            selected_operators = [asia_ops]
        elif args.group_op == 'europe':
            selected_operators = [europe_ops]
        elif args.group_op == 'na':
            selected_operators = [na_grp]
        elif args.group_op == 'all':
            selected_operators = [all_operators]
        elif args.group_op == 'region':
            selected_operators = [na_grp, europe_ops, asia_ops]
        elif args.group_op == 'not_na':
            selected_operators = [europe_ops, asia_ops]
        elif args.group_op == 'each':
            selected_operators = [[op] for op in all_operators]
            # selected_operators = [[op] for op in europe_ops]
        elif args.group_op == 'test_all':
            selected_operators = []
        else:
           print(f'No group of the name {args.group_op}')
           return None, None, None, None

    if args.ata6 == 'all':
        target_atas = target_atas_all
    elif args.ata6 == 'all2':
        target_atas = target_atas_all2
    else:
        target_atas = int(args.ata6)

    epochs = int(args.epochs)
    sel_type = args.type

    scaler_type = args.scaler
    seq_len_ata  = int(args.seq_len_ata)
    seq_len_sn  = int(args.seq_len_sn)
    deviation = args.deviation
    learning_rate = float(args.lr)
    set_config = int(args.set_config)
    model_name = args.ata_model
    batch = int(args.batch)
    min_records = int(args.min_records)
    min_acs = int(args.min_acs)

    return selected_operators, target_atas, epochs, sel_type, scaler_type, seq_len_ata, seq_len_sn, deviation, learning_rate, set_config, model_name, batch, min_records, min_acs

## main function
def main(argv, args):
    print('\n')
    print('argv: ', argv)
    print('args: ', args)

    selected_operators, target_atas, epochs, sel_type, scaler_type, seq_len_ata, seq_len_sn, deviation, learning_rate, set_config, model_name, batch, min_records, min_acs = param_passer(args)

    
    # target_atas = 243201
    
    ## For relatively small-size model
    if set_config==1:
        embed_dim = 64  ## <-- out_channel increase (both CNN and Attention)
        num_heads = 8
        OUT_LAYER = 1
        STRIDE =1
        KERNEL_SIZE = 2 ## <-- kernel size 
        DROPOUT = 0.1

        ## for TCN data
        sel_ones = 'all'
        embedding_dim =3    ## for categorical embedding
        revise=True

        ## Parameters for Transformer
        input_dim = 2
        d_model = 64
        nhead = 8
        num_encoder_layers = 2 
        dim_feedforward = 128  

    ## For relatively large-size model
    elif set_config==2:
        embed_dim = 128  ## <-- out_channel increase (both CNN and Attention)
        num_heads = 8
        OUT_LAYER = 1
        STRIDE =1
        KERNEL_SIZE = 2 ## <-- kernel size 
        DROPOUT = 0.1

        ## for TCN data
        sel_ones = 'all'
        embedding_dim =8    ## for categorical embedding
        revise=True

        ## Parameters for Transformer
        input_dim = 2
        d_model = 128    ## <--- increase
        nhead = 8
        num_encoder_layers = 4  ## <-- increase 
        dim_feedforward = 256   ## <-- increase

    if selected_operators is None:
        print('Error: No valid arguments')
        return None
    else:
        print('OPs:', selected_operators)
    
    ## Load base data
    df_rep, df_util_diff = load_data(base_dir, MIN_FH, latest_date, ac_model)

    if selected_operators == [] and args.group_op == 'test_all':
        pair_counts = (df_rep.groupby(['ATA_NUMBER', 'OPERATOR_CODE']).agg(record_count=('ATA_NUMBER', 'size'),  unique_ac_sn=('AC_SN', 'nunique')).reset_index())
        valid_pairs = pair_counts[(pair_counts['record_count'] > min_records) & (pair_counts['unique_ac_sn'] > min_acs)]
        valid_pairs_ata_lists = valid_pairs['ATA_NUMBER'].unique().tolist()
    else:
        op_list = all_operators

    ## op_grp: all or regional or independent operators
    for op_grp in selected_operators:

        print(f"GRP: {op_grp}, ATA: {target_atas}")
        df_process_data = get_selected_data(df_rep, df_util_diff, op_grp, target_atas)

        ## For TCN (ATA4) networks
        if model_name in ['TCN_AttenMulti', 'TSTMulti', 'TST']:
            X_train_nums, X_train_ones, y_train_list, u_train_list, X_test_nums, X_test_ones, y_test_list, u_test_list, df_test, df_train, pn_list = get_data_tcn(
                df_process_data, df_util_diff, int(target_atas), op_list, considered_date=considered_date, latest_date=latest_date, test_date=test_date, seq_len=seq_len_ata, embedding_dim=embedding_dim)
            
            print("[CHK] length of train and test ==> ", len(df_train), len(df_test))
            if len(df_train) < 10:
                print(f'Not enough training set for {op_grp}. AC: {len(df_train)}')
                continue
        
        elif model_name == 'Transformer':
        ## Transformer (TST) uses the almost same inputs as TCN, but with different sequence and tensor formatting.
            X_U, y_U, X_S, y_S, X_US, y_US, X_test_sn, y_test_sn, y_test_type_sn, df_test_sn, df_train_sn = get_selected_data_pn(
                df_process_data, target_atas, op_grp, deviation, latest_date, test_date, seq_len=seq_len_sn)
            
            print("[CHK SN] length of train and test ==> ", len(df_train_sn), len(df_test_sn))
            if len(df_train_sn) < 10:
                print(f'Not enough training set for {op_grp}. SN: {len(df_train_sn)}')
                continue

        
        for col in m_cols:
            if col in ['INSTALL_DATE', 'REMOVAL_DATE']:
                if model_name in ['TCN_AttenMulti', 'TSTMulti', 'TST']:
                    df_train[col] = pd.to_datetime(df_train[col])   ## from TCN
                    df_test[col] = pd.to_datetime(df_test[col])
                elif model_name == 'Transformer':
                    df_train_sn[col] = pd.to_datetime(df_train_sn[col]) ## from Transformer
                    try:
                        df_test_sn[col] = pd.to_datetime(df_test_sn[col])
                    except:
                        continue
            else:
                if model_name in ['TCN_AttenMulti', 'TSTMulti', 'TST']:
                    df_train[col] = df_train[col].astype(str)
                    df_test[col] = df_test[col].astype(str)
                elif model_name == 'Transformer':
                    df_train_sn[col] = df_train_sn[col].astype(str)
                    try:
                        df_test_sn[col] = df_test_sn[col].astype(str)
                    except:
                        continue

        m_cols_rev = [m for m in m_cols if m!= 'FLIGHT_HOURS' ]
        if model_name in ['TCN_AttenMulti', 'TSTMulti', 'TST']:
            df_train = df_train.rename(columns={'FLIGHT_HOURS': 'FLIGHT_HOURS_tcn'})
            df_test = df_test.rename(columns={'FLIGHT_HOURS': 'FLIGHT_HOURS_tcn'})

            u_weight = len(df_train[df_train['isU']==0]) / len(df_train[df_train['isU']==1])

            ## Pre-training TCN (ATA4) Encoder
            X_train_t, y_train_t, u_train_t, y_test_t, u_test_t, X_test_t, scalerX, scalerY =tensor_TCN(
                X_train_nums, X_train_ones, y_train_list, u_train_list,
                X_test_nums, X_test_ones, y_test_list, u_test_list, sel_ones, scaler_type=scaler_type, device=device)
            if model_name == 'TCN_AttenMulti':
                print('TCN: ', torch.isnan(X_train_t).any(), torch.isinf(X_train_t).any())
                model_ata4, feat_ata4 = train_TCNMulti(X_train_t, y_train_t, u_train_t, epochs, 
                                                                      embed_dim, num_heads, OUT_LAYER, STRIDE, KERNEL_SIZE, DROPOUT, learning_rate, device, model_name=model_name, batch=batch, u_weight=u_weight)
                fname = f"Multi_TCN_ops_{'_'.join(op_grp)}__ata_{'_'.join([str(target_atas)])}_CONFIG_{set_config}_model_{model_name}_seqata_{seq_len_ata}_seqsn_{seq_len_sn}_b_{batch}"
            elif model_name == 'TSTMulti':
                print('TST: ', torch.isnan(X_train_t).any(), torch.isinf(X_train_t).any())
                model_ata4, feat_ata4 = train_TSTMulti(X_train_t, y_train_t, u_train_t, epochs, 
                                                                      d_model, nhead, num_encoder_layers, dim_feedforward, learning_rate, device, model_name=model_name, batch=batch, u_weight=u_weight)
                fname = f"Multi_TST_ops_{'_'.join(op_grp)}__ata_{'_'.join([str(target_atas)])}_CONFIG_{set_config}_model_{model_name}_seqata_{seq_len_ata}_seqsn_{seq_len_sn}_b_{batch}"
            elif model_name == 'TST':
                print('TST: ', torch.isnan(X_train_t).any(), torch.isinf(X_train_t).any())
                model_ata4, feat_ata4 = train_TST(X_train_t, y_train_t, X_test_t, y_test_t, epochs, 
                                                                      d_model, nhead, num_encoder_layers, dim_feedforward, learning_rate, device, model_name=model_name, batch=batch)
                fname = f"TST_ops_{'_'.join(op_grp)}__ata_{'_'.join([str(target_atas)])}_CONFIG_{set_config}_model_{model_name}_seqata_{seq_len_ata}_seqsn_{seq_len_sn}_b_{batch}"

            torch.save(model_ata4.state_dict(), f'./{model_dir_name}/ata4_{fname}.pth')

            save_pickle(scalerX, f'./{scaler_dir_name}/scalerX_AC_{fname}.pkl')
            save_pickle(scalerY, f'./{scaler_dir_name}/scalerY_AC_{fname}.pkl')

            model_ata4, df_test = test_TCNMulti(
                model_ata4, X_test_t, y_test_list, scalerY, df_test, device, model_name)

            df_test.to_csv(f'./{predict_dir_name}/ata4_{fname}.csv')


        elif model_name == 'Transformer':
            df_train_sn = df_train_sn.rename(columns={'FLIGHT_HOURS': 'FLIGHT_HOURS_sn'})        
            df_test_sn = df_test_sn.rename(columns={'FLIGHT_HOURS': 'FLIGHT_HOURS_sn'})

            ## Pre-training Transformer (PartSN) Encoder
            X_train_t_sn, y_train_t_sn, X_test_t_sn, y_test_t_sn, scalerX_sn, scalerY_sn = tensor_SN(
                X_U, y_U, X_S, y_S, X_US, y_US, X_test_sn, y_test_sn, y_test_type_sn, 
                sel_type=sel_type, scaler_type=scaler_type, device=device)
            print('Transformer: ', torch.isnan(X_train_t_sn).any(), torch.isinf(X_train_t_sn).any())
            model_sn, feat_sn = pretrain_SN(X_train_t_sn, y_train_t_sn, X_test_t_sn, y_test_t_sn, epochs, 
                                            input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, learning_rate, device=device, batch=batch)
            torch.save(model_sn.state_dict(), f'./{model_dir_name}/sn_{fname}.pth')

            save_pickle(scalerX_sn, f'./{scaler_dir_name}/scalerX_SN_{fname}.pkl')
            save_pickle(scalerY_sn, f'./{scaler_dir_name}/scalerY_SN_{fname}.pkl')

            model_sn, df_test_sn = pretrain_test_SN(
                model_sn, X_test_t_sn, y_test_sn, scalerY_sn, df_test_sn, device)

            df_test_sn.to_csv(f'./{predict_dir_name}/sn_{fname}.csv')



if __name__ == '__main__':
    argv = sys.argv
    main(argv, args)