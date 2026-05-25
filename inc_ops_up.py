############################################################################################
## Arguments
## - op: airline name (Train & Test for a single operator)
## - ata: 6-digit ATA number (Train & Test for a single ATA)
## - epochs: default as 100
## Date: 2026. 5.25.
## Description:
## - CALE test with incrementing training set (w/ adding airlines)
## - In here, we fixed the test set as target operator, but increasing training set only
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
scaler_dir_name = 'scalers_up'
config_dir_name = 'configs_up'
model_dir_name = 'train_model_up'
predict_dir_name = 'test_result_up'

all_operators = ['E', 'C','H', 'I', 'A', 'J', 'S', 'P']
target_atas_all= [243201, 324301,  243203, 313301, 344401, 324201]
ata_names = ['Main \nBattery', 'Break \nAssembly', 'APU \nBattery', 'Quick \nAccess \nRecorder', 
             'Radio \nTransceiver', 'Nose \nWheel &\nTire \nAssembly']
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
parser.add_argument('-group_op', help='  : select group of ops (ex. asia, europe, na, ...)', default='all')
parser.add_argument('-op', help='   : select a single ATA (ex. HXA)', default='S')
parser.add_argument('-ata6', help = '  : select ATA6', default='243203')
parser.add_argument('-epochs', help= '   : set epochs for training', default=100)
parser.add_argument('-type', help='     : set training set for removal type code (all, U, S)', default='U')
parser.add_argument('-scaler', help='     : set scaler between standard, minmax, robust', default='standard')
parser.add_argument('-seq_len_ata', help='      : sequence length for train/test inputs', default=3)
parser.add_argument('-seq_len_sn', help='      : sequence length for train/test inputs', default=3)
parser.add_argument('-deviation', help='        : cum or inc for PN prediction', default='inc')
parser.add_argument('-lr', help='        : learning rate', default=0.0005)
parser.add_argument('-set_config', help='       : for Large model', default=2)
parser.add_argument('-ata_model', help='        : model name for ATA4', default='TCNAtten')
parser.add_argument('-batch', help='        : batch size for all models', default=16)
parser.add_argument('-min_records', help='        : minimum records for selected ATA-OP', default=50)
parser.add_argument('-min_acs', help='        : minimum unique ACs for selected ATA-OP', default=10)
args = parser.parse_args()

### Controlling input data w/ adding airlines.
def get_incremental_operator_groups(start_op, operators):
    ordered_ops = [start_op]
    ordered_ops += [op for op in operators if op != start_op]
    return [ordered_ops[:idx + 1] for idx in range(len(ordered_ops))]

def filter_test_by_operator(test_items, df_test, test_op):
    if df_test is None or len(df_test) == 0:
        return (*test_items, df_test)

    test_mask = (df_test['OPERATOR_CODE'].astype(str) == str(test_op)).to_numpy()
    filtered_items = []
    for item in test_items:
        if item is None:
            filtered_items.append(item)
        elif torch.is_tensor(item):
            torch_mask = torch.as_tensor(test_mask, dtype=torch.bool, device=item.device)
            filtered_items.append(item[torch_mask])
        else:
            filtered_items.append([value for value, keep in zip(item, test_mask) if keep])

    df_filtered = df_test.loc[test_mask].reset_index(drop=True)
    return (*filtered_items, df_filtered)

def param_passer(args):
    if args.group_op is None or args.group_op == 'None':
        selected_operators = [[args.op]]
        print('Single OP: ', selected_operators[0])
    else:
        sel_ops = args.group_op
        print('OPs: ', sel_ops)
        if args.group_op == 'all':
            selected_operators = [all_operators]
        elif args.group_op == 'each':
            selected_operators = [[op] for op in all_operators]
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

#############################################################################
## main function
def main(argv, args):
    print('\n')
    print('argv: ', argv)
    print('args: ', args)

    selected_operators, target_atas, epochs, sel_type, scaler_type, seq_len_ata, seq_len_sn, deviation, learning_rate, set_config, model_name, batch, min_records, min_acs = param_passer(args)
    
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
        X_train_nums, X_train_ones, y_train_list, u_train_list, X_test_nums, X_test_ones, y_test_list, u_test_list, df_test, df_train, pn_list = get_data_tcn(
            df_process_data, df_util_diff, int(target_atas), op_list, considered_date=considered_date, latest_date=latest_date, test_date=test_date, seq_len=seq_len_ata, embedding_dim=embedding_dim)
        X_test_nums, X_test_ones, y_test_list, u_test_list, df_test = filter_test_by_operator(
            [X_test_nums, X_test_ones, y_test_list, u_test_list], df_test, args.op)
        
        ## For Transformer-encoder (PartSN) networks
        X_U, y_U, X_S, y_S, X_US, y_US, X_test_sn, y_test_sn, y_test_type_sn, df_test_sn, df_train_sn = get_selected_data_pn(
            df_process_data, target_atas, op_grp, deviation, latest_date, test_date, seq_len=seq_len_sn)
        X_test_sn, y_test_sn, y_test_type_sn, df_test_sn = filter_test_by_operator(
            [X_test_sn, y_test_sn, y_test_type_sn], df_test_sn, args.op)


        ## For duplicated datasets for two encoders (train & test)
        print("[CHK] length of train and test ==> ", len(df_train), len(df_test), len(df_train_sn), len(df_test_sn))
        if len(df_train) < 10 or len(df_train_sn) < 10:
            print(f'Not enough training set for {op_grp}. AC: {len(df_train)}, SN: {len(df_train_sn)}')
            continue
        
        for col in m_cols:
            if col in ['INSTALL_DATE', 'REMOVAL_DATE']:
                df_train[col] = pd.to_datetime(df_train[col])   ## from TCN
                df_train_sn[col] = pd.to_datetime(df_train_sn[col]) ## from Transformer
                df_test[col] = pd.to_datetime(df_test[col])
                try:
                    df_test_sn[col] = pd.to_datetime(df_test_sn[col])
                except:
                    continue
            else:
                df_train[col] = df_train[col].astype(str)
                df_train_sn[col] = df_train_sn[col].astype(str)
                df_test[col] = df_test[col].astype(str)
                try:
                    df_test_sn[col] = df_test_sn[col].astype(str)
                except:
                    continue

        # m_cols_rev= m_cols - ['FLIGHT_HOURS']
        m_cols_rev = [m for m in m_cols if m!= 'FLIGHT_HOURS' ]
        df_train = df_train.rename(columns={'FLIGHT_HOURS': 'FLIGHT_HOURS_tcn'})
        df_train_sn = df_train_sn.rename(columns={'FLIGHT_HOURS': 'FLIGHT_HOURS_sn'})
        df_test = df_test.rename(columns={'FLIGHT_HOURS': 'FLIGHT_HOURS_tcn'})
        df_test_sn = df_test_sn.rename(columns={'FLIGHT_HOURS': 'FLIGHT_HOURS_sn'})
        
        ## For decoder & fine-tune
        df_train_dup = df_train.merge(df_train_sn, how='inner', on=m_cols_rev)
        try:
            df_test_dup = df_test.merge(df_test_sn, how='inner', on =m_cols_rev)
        except:
            print('No test for SN')

        ## Pre-training TCN (ATA4) Encoder
        X_train_t, y_train_t, u_train_t, X_test_t, y_test_t, u_test_t, scalerX, scalerY =tensor_TCN(
            X_train_nums, X_train_ones, y_train_list, u_train_list,
            X_test_nums, X_test_ones, y_test_list, u_test_list, sel_ones, scaler_type=scaler_type, device=device)
        
        print('TCN: ', torch.isnan(X_train_t).any(), torch.isinf(X_train_t).any())
        model_ata4, X_test_t_serial, feat_ata4 = pretrain_TCN(X_train_t, y_train_t, X_test_t, y_test_t, epochs, 
                                                              embed_dim, num_heads, OUT_LAYER, STRIDE, KERNEL_SIZE, DROPOUT, learning_rate, device, model_name=model_name, batch=batch)
        fname = f"CIKM_TCN_ops_{'_'.join(op_grp)}__ata_{'_'.join([str(target_atas)])}_CONFIG_{set_config}_model_{model_name}_seqata_{seq_len_ata}_seqsn_{seq_len_sn}_b_{batch}"
        torch.save(model_ata4.state_dict(), f'./{model_dir_name}/ata4_{fname}.pth')

        save_pickle(scalerX, f'./{scaler_dir_name}/scalerX_AC_{fname}.pkl')
        save_pickle(scalerY, f'./{scaler_dir_name}/scalerY_AC_{fname}.pkl')

        model_ata4, df_test = pretrain_test_TCN(
            model_ata4, X_test_t_serial, y_test_list, scalerY, df_test, device, model_name)
        
        df_test.to_csv(f'./{predict_dir_name}/ata4_{fname}.csv')

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

        ## Config for two-encoders
        if model_name == 'MultiheadAttenCNN':
            input_len_fl = X_train_t.size(1) * X_train_t.size(2)
        elif model_name == 'TCNAtten':
            input_len_fl = X_train_t.size(2)
        config_ata4 = CONFIG_ATA4(in_channel=input_len_fl, out_channel=embed_dim, num_heads=num_heads,
                                  output_dim=OUT_LAYER, stride=STRIDE, kernel_size=KERNEL_SIZE, 
                                  dropout_ratio=DROPOUT, pn_list=pn_list, feat_ata4=feat_ata4, num_train=len(y_train_t))
        
        config_sn = CONFIG_SN(input_dim=input_dim, d_model=d_model, nhead=nhead,
                              num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
                              X_train_t_sn=X_train_t_sn, feat_sn=feat_sn, num_train_sn=len(y_train_t_sn))
        
        save_pickle(config_ata4, f'./{config_dir_name}/config_ata4_{fname}.pkl')
        save_pickle(config_sn, f'./{config_dir_name}/config_sn_{fname}.pkl')
        ## Decoder
        u_weight = len(df_train[df_train['isU']==0]) / len(df_train[df_train['isU']==1])
        loss_gaussian = nn.GaussianNLLLoss(reduction='none').to(device)
        loss_logit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([u_weight]).to(device))
        multi_task_loss = MultiTaskLoss().to(device)

        X_tr_t_d, y_tr_t_d, X_te_t_d, X_tr_t_sn_d, y_tr_t_sn_d, X_te_t_sn_d, isU_t, pns_t, reinstall_t, fh_last_t, has_history, isU_te, pns_te, reinstall_te, fh_last_te, has_history_te = tensor_decoder(
            df_train_dup, df_test_dup, scalerX, scalerY, pn_list, y_train_t, y_train_t_sn, scalerX_sn, scalerY_sn, device=device)
        
        model_UP = UnscheduledPredictor(config_ata4, config_sn, model_name = model_name)
        model_UP.aircraft_encoder.load_state_dict(torch.load(f'./{model_dir_name}/ata4_{fname}.pth'))
        model_UP.part_encoder.load_state_dict(torch.load(f'./{model_dir_name}/sn_{fname}.pth'))
        model_UP.to(device)


        ## Training decoder (2nd-phase)
        for param in model_UP.aircraft_encoder.parameters():
            param.requires_grad = False
        for param in model_UP.part_encoder.parameters():
            param.requires_grad = False


        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_UP.parameters()), lr=1e-4)
        
        ## Set y as next flight hours --> y_tr = y_tr_t_d
        model_UP_tr, loss_UP = train_UP(model_UP, X_tr_t_d, X_tr_t_sn_d, y_tr_t_d, y_tr_t_sn_d, 
                                        isU_t, pns_t, reinstall_t, fh_last_t, has_history, 
                                        optimizer, loss_gaussian, loss_logit, multi_task_loss, epochs=50, lr = 0.0001, device=device)
        
        torch.save(model_UP_tr.state_dict(), f'./{model_dir_name}/UP_2nd_{fname}.pth')
        ## Secondary test
        mu_ps, sigma_ps, alpha_ps, logit_ps, y_dup_ac, y_dup_pn = test_UP(
            model_UP, X_te_t_d, X_te_t_sn_d, isU_te, pns_te, reinstall_te, fh_last_te, has_history, device=device)
        
        yp_en1 = scalerY.inverse_transform(np.array(y_dup_ac).reshape(-1,1))
        yp_en2 = scalerY_sn.inverse_transform(np.array(y_dup_pn).reshape(-1,1))
        yp_mu = scalerY.inverse_transform(np.array(mu_ps).reshape(-1,1))
        yp_sigma = np.array(sigma_ps).reshape(-1,1)*scalerY.scale_
        df_test_dup['mu'] = yp_mu
        df_test_dup['sigma'] = yp_sigma
        df_test_dup['alpha'] = alpha_ps
        df_test_dup['logit'] = logit_ps
        df_test_dup['y_ac'] = yp_en1
        df_test_dup['y_pn'] = yp_en2

        ## Fine Tuning
        for param in model_UP.parameters():
            param.requires_grad = True


        optimizer = torch.optim.Adam(model_UP.parameters(), lr=1e-6)
        model_FL_tr_f, loss_FL_f = train_UP(
            model_UP, X_tr_t_d, X_tr_t_sn_d, y_tr_t_d, y_tr_t_sn_d, isU_t, pns_t, reinstall_t, fh_last_t, 
            has_history, optimizer, loss_gaussian, loss_logit, multi_task_loss, epochs=50, lr = 0.0001, device=device)
        torch.save(model_FL_tr_f.state_dict(), f'./{model_dir_name}/UP_tune_{fname}.pth')


        mu_ps_f, sigma_ps_f, alpha_ps_f, logit_ps_f, y_dup_ac_f, y_dup_pn_f = test_UP(
            model_UP, X_te_t_d, X_te_t_sn_d, isU_te, pns_te, reinstall_te, fh_last_te, has_history, device=device)
        
        yp_en1_f = scalerY.inverse_transform(np.array(y_dup_ac_f).reshape(-1,1))
        yp_en2_f = scalerY_sn.inverse_transform(np.array(y_dup_pn_f).reshape(-1,1))
        yp_mu_f = scalerY.inverse_transform(np.array(mu_ps_f).reshape(-1,1))
        yp_sigma_f = np.array(sigma_ps_f).reshape(-1,1)*scalerY.scale_
        df_test_dup['mu_f'] = yp_mu_f
        df_test_dup['sigma_f'] = yp_sigma_f
        df_test_dup['alpha_f'] = alpha_ps_f
        df_test_dup['logit_f'] = logit_ps_f
        df_test_dup['y_ac_f'] = yp_en1_f                
        df_test_dup['y_pn_f'] = yp_en2_f

        df_test_dup.to_csv(f'./{predict_dir_name}/UP_{fname}.csv')


if __name__ == '__main__':
    argv = sys.argv
    main(argv, args)
