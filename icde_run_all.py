import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# 
from models.preprocessing import convert_to_int, related_at_install_series
from models.model_ata import *
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import argparse

import random
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
base_dir = "./dataset"

all_operators = ['E', 'C','H', 'I', 'A', 'J', 'S', 'P']
target_atas_all= [243201, 324301, 344401, 215206]
sel_columns= ['PART_NO', 'PART_SN','INSTALL_DATE', 'AC_SN', 'AC_MODEL','OPERATOR_CODE', 'ATA_NUMBER',
       'REMOVAL_DATE', 'REMOVAL_TYPE_CODE', 'QPA', 'FLIGHT_HOURS', 'FLIGHT_CYCLES',
       'ATA4','AU_DATE', 'ac_sn', 'MONTH' ]

## 
OUT_LAYER = 1
learning_rate = 0.001

## Parameters for Transformer
# input_dim = 2
d_model = 64
nhead = 8
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 128
sequence_length = 5

embedding_dim =3

## Parameter for LSTMAtten
hidden_dim = 64
num_layer = 1

## Parameters for MLP-Atten & CNN-Atten
HIDDEN_LAYER1, HIDDEN_LAYER2 = 100, 100     ### Temp.
embed_dim, num_heads = 64, 8
STRIDE, KERNEL_SIZE = 1, 2

parser = argparse.ArgumentParser()
parser.add_argument('-group_op', help='  : select group of ops (ex. asia, europe, na, ...)', default=None)
parser.add_argument('-op', help='   : select a single ATA (ex. HXA)', default='HXA')
parser.add_argument('-ata6', help = '  : select ATA6', default='all')
parser.add_argument('-epochs', help= '   : set epochs for training', default=100)
parser.add_argument('-type', help='     : set training set for removal type code (all, U, S)', default='all')
parser.add_argument('-scaler', help='     : set scaler between standard, minmax, robust', default='standard')
parser.add_argument('-seq_len', help='      : sequence length for train/test inputs', default=5)
parser.add_argument('-embed', help='      : set embedding for categorical variables (True, False)', default='True')
parser.add_argument('-sel_ones', help='     : selection of features (all, part, month)', default='all')
parser.add_argument('-revise', help= '      : filter out anomalies or not (True / False)', default='False')
args = parser.parse_args()

sel_columns= ['PART_NO', 'PART_SN','INSTALL_DATE', 'AC_SN', 'AC_MODEL','OPERATOR_CODE', 'ATA_NUMBER',
       'ATA_CHAPTER', 'ATA_SECTION', 'ATA_COMPONENT','REMOVAL_DATE', 'REMOVAL_TYPE_CODE',
       'QPA', 'FLIGHT_HOURS', 'FLIGHT_CYCLES', 'CUML_HOURS', 'CUML_CYCLES',
       'ATA4','AU_DATE', 'ac_sn', 'MONTH', 'diff_hours', 'diff_cycle',
       'hour_per_cycle', 'CUMULATIVE_FLIGHT_HOURS', 'CUMULATIVE_CYCLES','days']

considered_date = datetime.datetime(2015,1,1)
latest_date = datetime.datetime(2025,3,1)
test_date = datetime.datetime(2023,3,1)

def param_passer(args):
    if args.group_op is None:
        selected_operators = [args.op]
        print('Single OP: ', selected_operators[0])
    else:
        sel_ops = args.group_op
        print('OPs: ', sel_ops)
        if args.group_op == 'all':
            selected_operators = all_operators
        else:
           print(f'No group of the name {args.group_op}')
           return None, None, None, None

    if args.ata6 == 'all':
        target_atas = target_atas_all
    else:
        target_atas = [int(args.ata6)]

    epochs = int(args.epochs)
    sel_type = args.type

    scaler_type = args.scaler
    seq_len  = int(args.seq_len)
    embed = True if args.embed == 'True' else False
    sel_ones = args.sel_ones
    revise_valid = True if args.revise == 'True' else False

    return selected_operators, target_atas, epochs, sel_type, scaler_type, seq_len, embed, sel_ones, revise_valid

    

def main(argv, args):
    print('\n')
    print('argv: ', argv)
    print('args: ', args)

    selected_operators, target_atas, epochs, sel_type, scaler_type, seq_len, embed, sel_ones, revise_valid= param_passer(args)
    if selected_operators is None:
        print('Error: No valid arguments')
        return None

    df_util_diff = pd.read_csv(f'{base_dir}/util_rev.csv', parse_dates=['AU_DATE'], index_col=0)
    df_util_diff = df_util_diff.rename(columns={'Aircraft':'ac_sn'})
    df_util_diff = df_util_diff.dropna(axis=0)

    df_rep = pd.read_csv(f'{base_dir}/CT_Export.csv', index_col=None, thousands=',', parse_dates=['INSTALL_DATE', 'REMOVAL_DATE', 'DATE'], date_format='mixed')
    df_rep['PART_NO'] = df_rep['PART_NO'].apply(convert_to_int)
    df_rep = df_rep.dropna(subset=['ATA_NUMBER'], how='any', axis=0) 
    df_rep = df_rep[df_rep['REMOVAL_DATE'] <= datetime.datetime(2025,3,26)]
    df_rep = df_rep[df_rep['FLIGHT_HOURS'] >= 10]
    df_rep = df_rep.groupby('AC_MODEL').get_group(ac_model)
    df_rep['lifetime'] = (df_rep['REMOVAL_DATE']-df_rep['INSTALL_DATE']).dt.days
    df_rep['avg_hours'] = df_rep['FLIGHT_HOURS']/df_rep['FLIGHT_CYCLES']
    df_rep['avg_dates'] = df_rep['lifetime']/df_rep['FLIGHT_CYCLES']

    df_rep['ATA4'] = df_rep['ATA_NUMBER']/100
    df_rep['ATA4'] = df_rep['ATA4'].astype(int)

    for ii, sel_op in enumerate(selected_operators):
        df_op = df_rep.groupby('OPERATOR_CODE').get_group(sel_op)
        print('OPERATOR: ', sel_op, ' # of records: ', len(df_op))
        for target_ata in target_atas:
            df_train_data = pd.DataFrame()
            print('TARGET ATA: ', target_ata)
            ata4 = int(target_ata/100)
            ata_chapter_target = int(ata4/100)
            ata_section_target = ata4-ata_chapter_target*100
            ## selected ATA6
            df_rep_rev = df_op.groupby('ATA_CHAPTER').get_group(ata_chapter_target)
            df_rep_rev = df_rep_rev.groupby('ATA_SECTION').get_group(ata_section_target)

            df_rep_rev_util = pd.merge(left=df_rep_rev, right=df_util_diff, how='left', left_on=['AC_SN', 'REMOVAL_DATE'], right_on=['ac_sn','AU_DATE'], sort=False)
            df_rep_rev_util.reset_index(inplace=True)
            df_train_data = pd.concat([df_train_data, df_rep_rev_util])

            chk_ata = int(target_ata)
            df_train_data = df_train_data[sel_columns]
            df_train_data = df_train_data.sort_values(by='REMOVAL_DATE', ascending=True)

            pn_df = df_train_data.groupby('ATA_NUMBER').get_group(chk_ata)
            pn_list = list(pn_df.groupby('PART_NO').count().index)
            print('PNs:', pn_list)
            ata_list = df_train_data.groupby('ATA_NUMBER').count().index
            all_tests, all_chks = related_at_install_series(df_train_data, df_util_diff, chk_ata, ata_list, considered_date, latest_date)
            # all_tests2 = add_before_fh(all_tests)
        
            # X_train_nums, X_train_ones, y_train_list, X_test_nums, X_test_ones, y_test_list, df_test = train_test_transformer(all_tests2, all_chks, seq_len, pn_list, test_date, embed=embed)
            X_train_nums, X_train_ones, y_train_list, X_test_nums, X_test_ones, y_test_list, df_test = train_test_ata4_series(all_tests, all_chks, seq_len, pn_list, test_date, sel_ones = sel_ones, embed=embed, embedding_dim = embedding_dim, use_type =False)

            if scaler_type == 'standard':
                scalerX, scalerY = StandardScaler(), StandardScaler()
            elif scaler_type == 'minmax':
                scalerX, scalerY = MinMaxScaler(), MinMaxScaler()
            elif scaler_type == 'robust':
                scalerX, scalerY = RobustScaler(), RobustScaler()

            X_train_nums, X_test_nums = np.array(X_train_nums), np.array(X_test_nums)

            y_tr = np.array(y_train_list)
            scalerY.fit(np.array(y_tr).reshape(-1, 1))
            y_train_n = scalerY.transform(np.array(y_tr).reshape(-1,1))
            y_train_t = makeTensor(y_train_n, device)
            y_train_t = y_train_t.unsqueeze(-1)

            y_test_n = scalerY.transform(np.array(y_test_list).reshape(-1,1))
            y_test_t = makeTensor(y_test_n, device)
            y_test_t = y_test_t.unsqueeze(-1)

            X_tr_2d = X_train_nums.reshape(-1, X_train_nums.shape[-1])
            scalerX.fit(X_tr_2d)

            X_tr_2d_sc = scalerX.transform(X_tr_2d)
            X_tr_sc = X_tr_2d_sc.reshape(X_train_nums.shape[0], X_train_nums.shape[1], X_train_nums.shape[2])
            X_train_n = np.concatenate([X_tr_sc, X_train_ones], axis=2)
            X_train_t = makeTensor(X_train_n, device)

            X_te_2d = X_test_nums.reshape(-1, X_test_nums.shape[-1])
            X_te_2d_sc = scalerX.transform(X_te_2d)
            X_te_sc = X_te_2d_sc.reshape(X_test_nums.shape[0], X_test_nums.shape[1], X_test_nums.shape[2])
            X_test_n = np.concatenate([X_te_sc, X_test_ones], axis=2)
            X_test_t = makeTensor(X_test_n, device)

            input_len = len(X_train_n[0])
            print('X Len:', len(X_train_n[0]))

            model_name = 'TransformerRegression'
            print(f'{model_name} Start')
            model = TimeSeriesTransformer(X_train_t.shape[-1], d_model, nhead, num_encoder_layers, dim_feedforward, X_train_t.shape[0]).to(device)
            filter_out = True
            epochs_valid = None
            model, losses, test_losses_val, out_inds_tr, X_tr_out, y_tr_out = train_model(model, model_name, X_train_t, y_train_t, X_test_t, y_test_t, X_valid=None, y_valid=None, revise_valid = False, filter_out=filter_out, learning_rate=0.001, epochs=epochs, epochs_valid=epochs_valid, device=device)

            yp = test_model(model, model_name, X_test_t, device=device)
            yp_inv = scalerY.inverse_transform(np.array(yp).reshape(-1,1))

            df_test['FH_org'] = y_test_list
            df_test['FH_TR'] = yp_inv

            print(f'{model_name} End')

            ## LSTM
            model_name = 'MultiheadAttenLSTM'
            print(f'{model_name} Start')
            model = MultiheadAttenLSTM(X_train_t.shape[-1], hidden_dim, num_layer, False, nhead, OUT_LAYER).to(device)
            filter_out = True
            epochs_valid = None
            model, losses, test_losses_val, out_inds_tr, X_tr_out, y_tr_out = train_model(model, model_name, X_train_t, y_train_t, X_test_t, y_test_t, X_valid=None, y_valid=None, revise_valid = False, filter_out=filter_out, learning_rate=0.001, epochs=epochs, epochs_valid=epochs_valid, device=device)

            yp = test_model(model, model_name, X_test_t, device=device)
            yp_inv = scalerY.inverse_transform(np.array(yp).reshape(-1,1))

            df_test['FH_MH_LSTM'] = yp_inv
            print(f'{model_name} End')

            ## MLP
            input_len_mlp = X_train_t.size(1) * X_train_t.size(2)
            X_train_t_serial = X_train_t.view(X_train_t.size(0), X_train_t.size(1)*X_train_t.size(2))
            X_test_t_serial = X_test_t.view(X_test_t.size(0), X_test_t.size(1)*X_test_t.size(2))
            model_name = 'MultiHeadAttentionRegression_MLP'
            print(f'{model_name} Start')
            model = MultiHeadAttentionRegression_MLP(input_len_mlp, HIDDEN_LAYER1, HIDDEN_LAYER2, embed_dim, num_heads, OUT_LAYER).to(device)

            filter_out = True
            epochs_valid = None
            model, losses, test_losses_val, out_inds_tr, X_tr_out, y_tr_out = train_model(model, model_name, X_train_t_serial, y_train_t, X_test_t_serial, y_test_t, X_valid=None, y_valid=None, revise_valid = False, filter_out=filter_out, learning_rate=0.001, epochs=epochs, epochs_valid=epochs_valid, device=device)

            yp = test_model(model, model_name, X_test_t_serial, device=device)
            yp_inv = scalerY.inverse_transform(np.array(yp).reshape(-1,1))

            df_test['FH_MH_MLP'] = yp_inv
            print(f'{model_name} End')

            ## CNN
            model_name = 'MultiheadAttenCNN'
            print(f'{model_name} Start')
            model = MultiheadAttenCNN(input_len_mlp, embed_dim, num_heads, OUT_LAYER, STRIDE, KERNEL_SIZE, OUT_LAYER).to(device)

            filter_out = True
            epochs_valid = None
            model, losses, test_losses_val, out_inds_tr, X_tr_out, y_tr_out = train_model(model, model_name, X_train_t_serial, y_train_t, X_test_t_serial, y_test_t, X_valid=None, y_valid=None, revise_valid = False, filter_out=filter_out, learning_rate=0.001, epochs=epochs, epochs_valid=epochs_valid, device=device)

            yp = test_model(model, model_name, X_test_t_serial, device=device)
            yp_inv = scalerY.inverse_transform(np.array(yp).reshape(-1,1))

            # result = pd.DataFrame(columns=view_cols + ['yp_MLP', 'yp_CNN'])
            df_test['FH_MH_CNN'] = yp_inv
            print(f'{model_name} End')

            df_test.to_csv(f'./data_ata4/icde_test_ata_{sel_op}_{target_ata}_len_{seq_len}_embed_{embed}_sel_{sel_ones}.csv')

    

if __name__ == '__main__':
    argv = sys.argv
    main(argv, args)



