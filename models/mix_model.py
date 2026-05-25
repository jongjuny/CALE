import pandas as pd
import numpy as np
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from models.preprocessing import convert_to_int, build_chains_by_qpa, related_at_install_series
from models.model_ata import MultiheadAttenCNN, TCN_Atten, TCN_AttenMulti, TimeSeriesTransformer, TimeSeriesTransformerMulti,train_model_rev, train_model_multi, train_test_ata4_series, train_test_part_SN_serial_rev, test_model_rev, test_model_multi, makeTensor

from dataclasses import dataclass
from typing import Sequence

@dataclass
class CONFIG_ATA4:
    in_channel: int
    out_channel: int
    num_heads: int
    output_dim: int
    stride: int
    kernel_size: int
    dropout_ratio: float
    pn_list: Sequence
    feat_ata4: np.ndarray
    num_train: int

    pn_size: int = None
    feat_dim: int = None

    def __post_init__(self):
        self.pn_size = len(self.pn_list)
        self.feat_dim = self.feat_ata4.shape[1]

@dataclass
class CONFIG_SN:
    input_dim: int
    d_model: int
    nhead: int
    num_encoder_layers: int
    dim_feedforward: int
    X_train_t_sn: np.ndarray
    feat_sn: np.ndarray
    num_train_sn: int

    max_len: int = None
    feat_dim: int = None

    def __post_init__(self):
        self.max_len = self.X_train_t_sn.shape[0]
        self.feat_dim = self.feat_sn.shape[1]

def load_data(base_dir, min_fh, latest_date, ac_model):
    df_util_diff = pd.read_csv(f'{base_dir}/util_2025.csv', parse_dates=['AU_DATE'], index_col=0)
    df_util_diff = df_util_diff.rename(columns={'Aircraft':'ac_sn'})
    df_util_diff = df_util_diff.dropna(axis=0)

    df_rep = pd.read_csv(f'{base_dir}/CT_Export_2025.csv', index_col=None, thousands=',', 
                         parse_dates=['INSTALL_DATE', 'REMOVAL_DATE', 'DATE'], date_format='mixed')

    df_rep = (
        df_rep.assign(PART_NO=lambda x: x['PART_NO'].map(convert_to_int),isU=lambda x: (x['REMOVAL_TYPE_CODE'] == 'U').astype('int8'))
        .dropna(subset=['ATA_NUMBER', 'ATA_COMPONENT'])
        .loc[(df_rep['REMOVAL_DATE'] <= latest_date) &(df_rep['FLIGHT_HOURS'] >= min_fh) &(df_rep['AC_MODEL'] == ac_model)]
    )
    df_rep['ATA4'] = (df_rep['ATA_NUMBER'] // 100).astype('int16')
    
    df_rep['INSTALL_DATE'] = pd.to_datetime(df_rep['INSTALL_DATE'])
    df_rep['REMOVAL_DATE'] = pd.to_datetime(df_rep['REMOVAL_DATE'])
    
    df_rep = df_rep.astype({'ATA_NUMBER': 'int32','ATA_CHAPTER': 'int16','ATA_SECTION': 'int16','ATA_COMPONENT': 'int16'})    
    df_rep = (df_rep.sort_values(['PART_SN', 'REMOVAL_DATE']).assign(reinstall_cnt=lambda x: x.groupby(['ATA_NUMBER', 'PART_SN']).cumcount()))

    return df_rep, df_util_diff

#############################################################################################
## target_ata: 6-digit ATA
## ops: multiple operators (ex. ['EDV', 'SKW', 'PSY])
def get_selected_data(df_rep, df_util, ops, target_ata):
    ata4 = int(target_ata/100)

    ## Subset of ATA4 
    df_ata_chk = df_rep[df_rep['ATA_NUMBER'] // 100 == ata4] 

    df_op = pd.DataFrame()
    for sel_op in ops:
        ac_lists_t = df_ata_chk.loc[df_ata_chk['OPERATOR_CODE'] == sel_op,'AC_SN'].unique().tolist()

        ## Get data based on Aircraft Serial Number (in Op)
        df_filtered = df_ata_chk[df_ata_chk['AC_SN'].isin(ac_lists_t)]
        df_t_rev = df_filtered[df_filtered['ATA_NUMBER'] // 100 == ata4]
        df_op = pd.concat([df_op, df_t_rev])
    
    df_op = df_op.drop_duplicates()
    df_process_data = pd.merge(left=df_op, right=df_util, how='left', left_on=['AC_SN', 'REMOVAL_DATE'], 
                           right_on=['ac_sn','AU_DATE'], sort=False)
    df_process_data.reset_index(inplace=True)
    df_process_data = df_process_data.sort_values(by='REMOVAL_DATE', ascending=True)

    return df_process_data

def get_data_tcn(df_tcn_data, df_util, chk_ata, op_list=None, 
                 considered_date=datetime.datetime(2015,1,1), latest_date=datetime.datetime(2025,10,1), 
                 test_date = datetime.datetime(2023,3,1), seq_len=3, embedding_dim =8):

    # pn_list = df_tcn_data.loc[df_tcn_data['ATA_NUMBER'] == chk_ata, 'PART_NO'].unique().tolist()
    # pn_list = sorted(pn_list, key=lambda x:str(x))
    pn_series = df_tcn_data.loc[df_tcn_data['ATA_NUMBER'] == chk_ata, 'PART_NO']
    pn_list = (pn_series.dropna().astype(str).str.strip().unique().tolist())
    pn_list = sorted(pn_list)

    ata_list = df_tcn_data.groupby('ATA_NUMBER').count().index

    print('PNs:', pn_list, ata_list)
    all_tests, all_chks = related_at_install_series(df_tcn_data, df_util, chk_ata, ata_list, 
                                                    considered_date, latest_date)
    
    print('TEST and CHKS: ', len(all_tests), len(all_chks))
    if len(all_tests) == 0:
        return None, None, None, None, None, None, None, None, None, None, None

    X_train_nums, X_train_ones, y_train_list, u_train_list, X_test_nums, X_test_ones, y_test_list, u_test_list, df_test, df_train = train_test_ata4_series(
        all_tests, all_chks, seq_len, pn_list, test_date, 
        sel_ones = 'all', embed=True, embedding_dim = embedding_dim, use_type =False, op_list=op_list)
    
    if X_train_nums is None:
        return None, None, None, None, None, None, None, None, None, None, None
    return X_train_nums, X_train_ones, y_train_list, u_train_list, X_test_nums, X_test_ones, y_test_list, u_test_list, df_test, df_train, pn_list


def get_selected_data_pn(df_data, target_ata, selected_operators, deviation, 
                         latest_date, test_date, seq_len=3):
    df_ata = df_data.groupby('ATA_NUMBER').get_group(target_ata)
    df_ata = df_ata.dropna(subset=['OPERATOR_CODE'])

    sn_lists = df_ata.groupby('PART_SN').count()
    sn_lists = sn_lists[['PART_NO']]
    print(f'All: {len(sn_lists)}')
    print(f'More than 1: {len(sn_lists[sn_lists.PART_NO >1])}')
    if len(df_ata[df_ata['REMOVAL_TYPE_CODE']=='U']) >0:
        print(len(df_ata[df_ata['REMOVAL_TYPE_CODE']=='U']))
    else:
        return None, None, None, None, None, None, None, None, None, None, None

    X_U, y_U, X_S, y_S, X_US, y_US, X_test_sn, y_test_sn, y_test_type_sn, df_test_sn, df_train_sn = train_test_part_SN_serial_rev(
        df_ata, selected_operators, latest_date, test_date, seq_len=seq_len, deviation=deviation, filter_single=False)


    return X_U, y_U, X_S, y_S, X_US, y_US, X_test_sn, y_test_sn, y_test_type_sn, df_test_sn, df_train_sn


## Converting numerical and categorical data into tensor types (for our model only)
def tensor_TCN(X_train_nums, X_train_ones, y_train_list, u_train_list,
                  X_test_nums, X_test_ones, y_test_list, u_test_list, sel_ones, scaler_type='standard', device=torch.device('cpu')):
    
    if scaler_type == 'standard':
        scalerX, scalerY = StandardScaler(), StandardScaler()
    elif scaler_type == 'minmax':
        scalerX, scalerY = MinMaxScaler(), MinMaxScaler()
    elif scaler_type == 'robust':
        scalerX, scalerY = RobustScaler(), RobustScaler()

    ## Convert to tensor for numerical and categorical datasets (train)
    X_train_nums, X_test_nums = np.array(X_train_nums), np.array(X_test_nums)

    y_tr = np.array(y_train_list)
    scalerY.fit(np.array(y_tr).reshape(-1, 1))
    y_train_n = scalerY.transform(np.array(y_tr).reshape(-1,1))

    y_train_t = makeTensor(y_train_n, device)
    y_train_t = y_train_t.unsqueeze(-1)

    u_tr = np.array(u_train_list)
    u_train_t = makeTensor(u_tr, device)
    u_train_t = u_train_t.unsqueeze(-1)

    if len(y_test_list) >0:
        y_test_n = scalerY.transform(np.array(y_test_list).reshape(-1,1))
        y_test_t = makeTensor(y_test_n, device)
        y_test_t = y_test_t.unsqueeze(-1)

        u_test_t = makeTensor(np.array(u_test_list), device)
        u_test_t = u_test_t.unsqueeze(-1)   
    else:
        y_test_t = y_test_list
        u_test_t = u_test_list

    X_tr_2d = X_train_nums.reshape(-1, X_train_nums.shape[-1])
    scalerX.fit(X_tr_2d)
    X_tr_2d_sc = scalerX.transform(X_tr_2d)

    X_tr_sc_t = torch.tensor(
        X_tr_2d_sc.reshape(X_train_nums.shape[0], X_train_nums.shape[1], X_train_nums.shape[2]), 
        dtype=X_train_ones.dtype, device=X_train_ones.device)
    if sel_ones is None:
        X_train_t = X_tr_sc_t
    else:
        X_train_t = torch.cat([X_tr_sc_t, X_train_ones], dim=2)

    ## Conver to tensor for numerical and categorical datasets (test)
    if len(X_test_nums) > 0:
        X_te_2d = X_test_nums.reshape(-1, X_test_nums.shape[-1])
        X_te_2d_sc = scalerX.transform(X_te_2d) 
    
        X_te_sc_t = torch.tensor(
            X_te_2d_sc.reshape(X_test_nums.shape[0], X_test_nums.shape[1], X_test_nums.shape[2]),
            dtype=X_test_ones.dtype, device=X_test_ones.device)
        if sel_ones is None:
            X_test_t = X_te_sc_t
        else:
            X_test_t = torch.cat([X_te_sc_t, X_test_ones], dim=2)
    else:
        X_test_t = []
     
    return X_train_t, y_train_t, u_train_t, X_test_t, y_test_t, u_test_t, scalerX, scalerY

## Pre-training the TCN_Atten encoder model
## It also can work independently
def pretrain_TCN(X_train_t, y_train_t, X_test_t, y_test_t, epochs, 
                 embed_dim, num_heads, OUT_LAYER, STRIDE, KERNEL_SIZE, DROPOUT, learning_rate, device, model_name, batch=1, u_weight=None):

    input_len = len(X_train_t[0])
    print('X Len:', len(X_train_t[0]))

    ## For comparison, fixed now
    filter_out = True
    epochs_valid = None

    print(f'{model_name} Start')
    ## Support two model only here.
    ## If necessary, we can extend MLPAtten
    if model_name == 'MultiheadAttenCNN':
        input_len_mlp = X_train_t.size(1) * X_train_t.size(2)
        X_train_t_serial = X_train_t.view(X_train_t.size(0), X_train_t.size(1)*X_train_t.size(2))
        if len(X_test_t) >0:
            X_test_t_serial = X_test_t.view(X_test_t.size(0), X_test_t.size(1)*X_test_t.size(2))
        else:
            X_test_t_serial = X_test_t

        model_ata4 = MultiheadAttenCNN(input_len_mlp, embed_dim, num_heads, 
                                       OUT_LAYER, STRIDE, KERNEL_SIZE, DROPOUT).to(device)
        
        model_ata4, _, _, _, _, _, feat_ata4 = train_model_rev(
            model_ata4, model_name, X_train_t_serial, y_train_t, X_test_t_serial, y_test_t, 
            X_valid=None, y_valid=None, revise_valid = False, filter_out=filter_out, 
            learning_rate=learning_rate, epochs=epochs, epochs_valid=epochs_valid, batch=batch, device=device) 

        return model_ata4, X_test_t_serial, feat_ata4
  
    ## This is the current model
    elif model_name == 'TCNAtten':
        input_len = X_train_t.size(2)
        model_ata4 = TCN_Atten(input_len, embed_dim, num_heads, 
                              OUT_LAYER, STRIDE, KERNEL_SIZE, DROPOUT).to(device)
        model_ata4, _, _, _, _, _, feat_ata4 = train_model_rev(
            model_ata4, model_name, X_train_t, y_train_t, X_test_t, y_test_t, 
            X_valid=None, y_valid=None, revise_valid = False, filter_out=filter_out, 
            learning_rate=learning_rate, epochs=epochs, epochs_valid=epochs_valid, batch=batch, device=device)   

        if len(X_test_t) > 0:
            print('TCN: ', X_test_t.shape)
        else: 
            print('No TEST SET')
        return model_ata4, X_test_t, feat_ata4

def train_TCNMulti(X_train_t, y_train_t, u_train_t, epochs, 
                 embed_dim, num_heads, OUT_LAYER, STRIDE, KERNEL_SIZE, DROPOUT, learning_rate, device, model_name, batch=1, u_weight=9.0):

    input_len = len(X_train_t[0])
    print('X Len:', len(X_train_t[0]))

    ## For comparison, fixed now
    filter_out = True
    epochs_valid = None

    print(f'{model_name} Start')
    ## Support two model only here.
  
    ## This is the current model
    if model_name == 'TCN_AttenMulti':
        input_len = X_train_t.size(2)
        model_ata4 = TCN_AttenMulti(input_len, embed_dim, num_heads, 
                              OUT_LAYER, STRIDE, KERNEL_SIZE, DROPOUT).to(device)
        model_ata4, _, _, _, _, _, feat_ata4 = train_model_multi(
            model_ata4, model_name, X_train_t, y_train_t, u_train_t, filter_out=filter_out, 
            learning_rate=learning_rate, epochs=epochs, epochs_valid=epochs_valid, batch=batch, pos_weight = u_weight, device=device)   

        return model_ata4, feat_ata4
    else:
        print('Error: Unsupported model name')
        return None, None

def train_TSTMulti(X_train_t, y_train_t, u_train_t, epochs, 
                 d_model, nhead, num_encoder_layers, dim_feedforward, learning_rate, device, model_name, batch=1, u_weight=9.0):

    input_len = len(X_train_t[0])
    print('X Len:', len(X_train_t[0]))

    ## For comparison, fixed now
    filter_out = True
    epochs_valid = None

    print(f'{model_name} Start')
    ## Support two model only here.
  
    ## This is the current model
    if model_name == 'TSTMulti':
        input_len = X_train_t.size(2)
        model_ata4 = TimeSeriesTransformerMulti(input_len, d_model, nhead, 
                              num_encoder_layers, dim_feedforward, X_train_t.shape[0]).to(device)
        model_ata4, _, _, _, _, _, feat_ata4 = train_model_multi(
            model_ata4, model_name, X_train_t, y_train_t, u_train_t, filter_out=filter_out, 
            learning_rate=learning_rate, epochs=epochs, epochs_valid=epochs_valid, batch=batch, pos_weight = u_weight, device=device)   


        return model_ata4, feat_ata4
    else:
        print('Error: Unsupported model name')
        return None, None

def train_TST(X_train_t, y_train_t, X_test_t, y_test_t, epochs, 
                 d_model, nhead, num_encoder_layers, dim_feedforward, learning_rate, device, model_name, batch=1):

    input_len = len(X_train_t[0])
    print('X Len:', len(X_train_t[0]))

    ## For comparison, fixed now
    filter_out = True
    epochs_valid = None

    print(f'{model_name} Start')
    ## Support two model only here.
  
    ## This is the current model
    if model_name == 'TST':
        input_len = X_train_t.size(2)
        model_ata4 = TimeSeriesTransformer(input_len, d_model, nhead, 
                              num_encoder_layers, dim_feedforward, X_train_t.shape[0]).to(device)

        model_ata4, _, _, _, _, _, feat_ata4 = train_model_rev(
            model_ata4, model_name, X_train_t, y_train_t, X_test_t, y_test_t, X_valid=None, y_valid=None, revise_valid=False, filter_out=filter_out, 
            learning_rate=learning_rate, epochs=epochs, epochs_valid=epochs_valid, batch=batch, device=device)   

        return model_ata4, feat_ata4
    else:
        print('Error: Unsupported model name')
        return None, None
    
## This is only for temporal comparison.
## Testing TCN model right away
def pretrain_test_TCN(model_ata4, X_test_t_serial, y_test_list, scalerY, df_test, device, model_name):
    
    yp = test_model_rev(model_ata4, model_name, X_test_t_serial, device=device)
    
    yp_inv = scalerY.inverse_transform(np.array(yp).reshape(-1,1))
    df_test['FH_org'] = y_test_list
    df_test['FH_MH_CNN'] = yp_inv

    print(f'TCN End')
    return model_ata4, df_test

def test_TCNMulti(model_ata4, X_test_t_serial, y_test_list, scalerY, df_test, device, model_name):
    
    yp, u_pred = test_model_multi(model_ata4, model_name, X_test_t_serial, device=device)

    yp_inv = scalerY.inverse_transform(np.array(yp).reshape(-1,1))
    df_test['FH_org'] = y_test_list
    if model_name == 'TCN_AttenMulti':
        df_test['FH_MH_TCN'] = yp_inv
        df_test['u_pred'] = u_pred
    elif model_name == 'TSTMulti':
        df_test['FH_MH_TST'] = yp_inv
        df_test['u_pred'] = u_pred
    elif model_name == 'TST':
        df_test['FH_TST'] = yp_inv
    

    print(f'{model_name} End')
    return model_ata4, df_test

## Converting PART_SN data into tensor type, for our Transformer encoder model
def tensor_SN(X_U, y_U, X_S, y_S, X_US, y_US, X_test_sn, y_test_sn, y_test_type_sn, 
              sel_type='U', scaler_type='standard', device=torch.device('cpu')):
    
    if scaler_type == 'standard':
        scalerX_sn, scalerY_sn = StandardScaler(), StandardScaler()
    elif scaler_type == 'minmax':
        scalerX_sn, scalerY_sn = MinMaxScaler(), MinMaxScaler()
    elif scaler_type == 'robust':
        scalerX_sn, scalerY_sn = RobustScaler(), RobustScaler()

    ## We only use sel_type == 'U' now
    if sel_type == 'U':
        y_tr_sn = np.array(y_U)
        X_tr_sn = np.array(X_U)
    elif sel_type == 'S':
        y_tr_sn = np.array(y_U + y_S)
        X_tr_sn = np.array(X_U + X_S)
    else:
        y_tr_sn = np.array(y_U + y_S + y_US)
        X_tr_sn = np.array(X_U + X_S + X_US)

    scalerY_sn.fit(np.array(y_tr_sn).reshape(-1,1))
    y_train_n_sn = scalerY_sn.transform(np.array(y_tr_sn).reshape(-1,1))
    
    y_train_t_sn = makeTensor(y_train_n_sn, device)
    y_train_t_sn = y_train_t_sn.unsqueeze(-1)

    ## Scale only the 1st feature (cum_FH)
    X_v = X_tr_sn[...,0].reshape(-1, 1)
    X_s = X_tr_sn[...,1].reshape(-1, 1)
    scalerX_sn.fit(X_v)
    X_vs = scalerX_sn.transform(X_v)

    ## restore shape
    X_vs = X_vs.reshape(X_tr_sn.shape[0], X_tr_sn.shape[1], 1)
    X_s = X_s.reshape(X_tr_sn.shape[0], X_tr_sn.shape[1], 1)
    X_train_n_sn = np.concatenate((X_vs, X_s), axis=-1)
    X_train_t_sn = makeTensor(X_train_n_sn, device)
    # X_train_t = X_train_t.unsqueeze(-1)

    if len(X_test_sn) >0:
        ## Test set
        y_test_n_sn = scalerY_sn.transform(np.array(y_test_sn).reshape(-1,1))
        y_test_t_sn = makeTensor(y_test_n_sn, device)
        y_test_t_sn = y_test_t_sn.unsqueeze(-1)

        X_test_sn = np.array(X_test_sn)
        X_tv = X_test_sn[...,0].reshape(-1, 1)
        X_ts = X_test_sn[...,1].reshape(-1, 1)
        X_tvs = scalerX_sn.transform(X_tv)

        X_tvs = X_tvs.reshape(X_test_sn.shape[0], X_test_sn.shape[1], 1)
        X_ts = X_ts.reshape(X_test_sn.shape[0], X_test_sn.shape[1], 1)
        X_test_n_sn = np.concatenate((X_tvs, X_ts), axis=-1)
        X_test_t_sn = makeTensor(X_test_n_sn, device)
        # X_test_t = X_test_t.unsqueeze(-1)
    else:
        X_test_t_sn, y_test_t_sn = [], []

    return X_train_t_sn, y_train_t_sn, X_test_t_sn, y_test_t_sn, scalerX_sn, scalerY_sn

## Pretrain the Transformer encoder model with PART_SN data
def pretrain_SN(X_train_t_sn, y_train_t_sn, X_test_t_sn, y_test_t_sn, epochs, 
                input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, learning_rate, batch=1, device=torch.device('cpu')):
    
    # input_len = len(X_train_t_sn[0])
    print('Transformer X Len:', len(X_train_t_sn[0]))

    model_name = 'TransformerRegression'
    model_sn = TimeSeriesTransformer(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, 
                                     X_train_t_sn.shape[0]).to(device)
    filter_out = True
    epochs_valid = None
    model_sn, _, _, _, _, _, feat_sn = train_model_rev(
        model_sn, model_name, X_train_t_sn, y_train_t_sn, X_test_t_sn, y_test_t_sn, 
        X_valid=None, y_valid=None, revise_valid = False, filter_out=filter_out, 
        learning_rate=learning_rate, epochs=epochs, epochs_valid=epochs_valid, batch=batch, device=device)
    
    return model_sn, feat_sn

## For independent test only
def pretrain_test_SN(model_sn, X_test_t_sn, y_test_sn, scalerY_sn, df_test_sn, device):
    model_name='TransformerRegression'
    yp = test_model_rev(model_sn, model_name, X_test_t_sn, device=device)
    yp_inv = scalerY_sn.inverse_transform(np.array(yp).reshape(-1,1))   

    df_test_sn['FH_org'] = y_test_sn
    df_test_sn['FH_TR'] = yp_inv
    print(f'Transformer End')

    return model_sn, df_test_sn

convert_cols = ['FLIGHT_HOURS_tcn', 'X_numeric', 'X_hist', 'Y']

def tensor_decoder(df_train_dup, df_test_dup, scalerX, scalerY, pn_list, y_train_t, y_train_t_sn, 
                   scalerX_sn, scalerY_sn, device=torch.device('cpu')):
    ## ATA4 AC encoder
    ## Target y
    df_train_dup[convert_cols] = df_train_dup[convert_cols].apply(np.array)
    y_tr_d = df_train_dup['FLIGHT_HOURS_tcn'].to_numpy()
    

    ## Feature X Nums (ATA FHs), Train
    X_tr_nums_d = df_train_dup['X_numeric'].to_numpy()
    X_tr_nums_d = np.stack(X_tr_nums_d)
    X_tr_2d_d = X_tr_nums_d.reshape(-1, X_tr_nums_d.shape[-1])

    ## Feature X Category, Train
    X_tr_ones_d = list(df_train_dup['X_category'])
    X_tr_ones_d = [x[0] for x in X_tr_ones_d]
    X_tr_ones_d = torch.stack(X_tr_ones_d, dim=0)

    ## Test X, Numeric
    if len(df_test_dup) > 0:
        df_test_dup[convert_cols] = df_test_dup[convert_cols].apply(np.array)
        X_te_nums_d = df_test_dup['X_numeric'].to_numpy()
        X_te_nums_d = np.stack(X_te_nums_d)
        X_te_2d_d = X_te_nums_d.reshape(-1, X_te_nums_d.shape[-1])
        X_te_2d_sc_d = scalerX.transform(X_te_2d_d)
        X_te_sc_d = X_te_2d_sc_d.reshape(X_te_nums_d.shape[0], X_te_nums_d.shape[1], X_te_nums_d.shape[2])

        ## Test X, Category
        X_te_ones_d = list(df_test_dup['X_category'])
        X_te_ones_d = [x[0] for x in X_te_ones_d]
        X_te_ones_d = torch.stack(X_te_ones_d, dim=0)

        X_te_sc_d = torch.tensor(X_te_sc_d, dtype=X_te_ones_d.dtype, device=X_te_ones_d.device)
        X_te_t_d = torch.cat([X_te_sc_d, X_te_ones_d], dim=2)

        y_te_d = df_test_dup['FLIGHT_HOURS_tcn'].to_numpy()
        y_te_n_d = scalerY.transform(np.array(y_te_d).reshape(-1,1))
        y_te_t_d = makeTensor(y_te_n_d, device)
        y_te_t_d = y_te_t_d.unsqueeze(-1)

        ## SN encoder
        y_te_sn_d = df_test_dup['Y'].to_numpy()
        y_te_n_sn_d = scalerY_sn.transform(y_te_sn_d.reshape(-1,1))
        y_te_t_sn_d = makeTensor(y_te_n_sn_d, device)
        y_te_t_sn_d = y_te_t_sn_d.unsqueeze(-1)

        ## Test set
        X_te_sn_d = df_test_dup['X_hist'].to_numpy()
        X_te_sn_d = np.stack(X_te_sn_d)
        X_tv_d = X_te_sn_d[...,0].reshape(-1, 1)
        X_ts_d = X_te_sn_d[...,1].reshape(-1, 1)
        X_tvs_d = scalerX_sn.transform(X_tv_d)

        X_tvs_d = X_tvs_d.reshape(X_te_sn_d.shape[0], X_te_sn_d.shape[1], 1)
        X_ts_d = X_ts_d.reshape(X_te_sn_d.shape[0], X_te_sn_d.shape[1], 1)
        X_te_n_sn_d = np.concatenate((X_tvs_d, X_ts_d), axis=-1)
        X_te_t_sn_d = makeTensor(X_te_n_sn_d, device)

        ## Test for gating
        isU_td = df_test_dup['isU_y'].to_numpy().astype(int)
        isU_te = makeTensor(isU_td.reshape(-1,1), device)
        isU_te = isU_te.unsqueeze(-1)

        ## Part_Number: embedding based on whole training set from AC encoder
        pn_list = np.array(pn_list).astype(str)
        part_to_idx = {pn:idx for idx, pn in enumerate(pn_list)}

        part_idx_te = df_test_dup['PART_NO'].map(part_to_idx).to_numpy()
        pns_te = torch.tensor(part_idx_te, dtype=torch.long)

        # reinstall
        reinstall_td = df_test_dup['reinstall_cnt_y'].to_numpy().astype(int)
        reinstall_te = makeTensor(reinstall_td.reshape(-1,1), device)

        # cum_FH
        fh_last_td = df_test_dup['X_hist'].apply(lambda x:x[-1,0]).to_numpy().reshape(-1,1)
        fh_last_te = makeTensor(fh_last_td,device)

    else:
        X_te_t_d, X_te_t_sn_d, isU_te, pns_te, reinstall_te, fh_last_te, has_history_te = [], [], [], [], [], [], []


    y_tr_n_d = scalerY.transform(np.array(y_tr_d).reshape(-1,1))
    X_tr_2d_sc = scalerX.transform(X_tr_2d_d)

    y_tr_t_d = makeTensor(y_tr_n_d, device)
    y_tr_t_d = y_tr_t_d.unsqueeze(-1)    

    X_tr_sc_d = X_tr_2d_sc.reshape(X_tr_nums_d.shape[0], X_tr_nums_d.shape[1], X_tr_nums_d.shape[2])
    X_tr_sc_d = torch.tensor(X_tr_sc_d, dtype=X_tr_ones_d.dtype, device=X_tr_ones_d.device)
    X_tr_t_d = torch.cat([X_tr_sc_d, X_tr_ones_d], dim=2)

    input_len_ac = len(X_tr_t_d[0])

    ## SN encoder
    y_tr_n_sn_d = df_train_dup['Y'].to_numpy()
    y_tr_n_sn_d = scalerY_sn.transform(y_tr_n_sn_d.reshape(-1,1))
    y_tr_t_sn_d = makeTensor(y_tr_n_sn_d, device)
    y_tr_t_sn_d = y_tr_t_sn_d.unsqueeze(-1)

    ## scale only numeric 1st feature (cumFH)
    X_tr_sn_d = df_train_dup['X_hist'].to_numpy()
    X_tr_sn_d = np.stack(X_tr_sn_d)

    X_v_d = X_tr_sn_d[...,0].reshape(-1,1)
    X_s_d = X_tr_sn_d[...,1].reshape(-1,1)
    X_vs_d = scalerX_sn.transform(X_v_d)

    ## Restore shape
    X_vs_d = X_vs_d.reshape(X_tr_sn_d.shape[0], X_tr_sn_d.shape[1], 1)
    X_s_d = X_s_d.reshape(X_tr_sn_d.shape[0], X_tr_sn_d.shape[1],1)
    X_tr_n_sn_d = np.concatenate((X_vs_d, X_s_d), axis=-1)
    X_tr_t_sn_d = makeTensor(X_tr_n_sn_d, device)

    ## Inputs for gating
    isU_d = df_train_dup['isU_y'].to_numpy().astype(int)
    isU_t = makeTensor(isU_d.reshape(-1,1), device)
    isU_t = isU_t.unsqueeze(-1)

    ## Part_Number: embedding based on whole training set from AC encoder
    pn_list = np.array(pn_list).astype(str)
    part_to_idx = {pn:idx for idx, pn in enumerate(pn_list)}

    part_idx = df_train_dup['PART_NO'].map(part_to_idx).to_numpy()
    pns_t = torch.tensor(part_idx, dtype=torch.long)

    # reinstall
    reinstall = df_train_dup['reinstall_cnt_y'].to_numpy().astype(int)
    reinstall_t = makeTensor(reinstall.reshape(-1,1), device)

    # cum_FH
    fh_last = df_train_dup['X_hist'].apply(lambda x:x[-1,0]).to_numpy().reshape(-1,1)
    fh_last_t = makeTensor(fh_last,device)

    has_history = torch.tensor(len(y_train_t_sn)/ len(y_train_t)).to(device)

    has_history_te = torch.tensor(len(y_train_t_sn)/ len(y_train_t)).to(device)

    return X_tr_t_d, y_tr_t_d, X_te_t_d, X_tr_t_sn_d, y_tr_t_sn_d, X_te_t_sn_d, isU_t, pns_t, reinstall_t, fh_last_t, has_history, isU_te, pns_te, reinstall_te, fh_last_te, has_history_te

## Gating model
class MetaAttentionGating(nn.Module):
    def __init__(self, ata4_dim, sn_dim, meta_dim, hidden_dim):
        super().__init__()
        input_dim = ata4_dim + sn_dim + meta_dim
        # print('METADIM:', input_dim, ata4_dim, sn_dim, meta_dim)
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )

    def forward(self, feat_ac, feat_part, meta_data, has_history):
        # print('DIM Gate:', feat_ac.shape, feat_part.shape, meta_data.shape)
        # feat_ac = feat_ac[:,-1,:]
        combined_query = torch.cat([feat_ac, feat_part, meta_data], dim=-1)
        
        alpha = self.gate_mlp(combined_query) # [B, 1]
        # alpha = alpha * has_history
        return alpha

class FullIntegratedModel(nn.Module):
    def __init__(self, config_ata4, config_sn, common_dim = 64, ata_embed = 16, model_name='TCNAtten'):
        super().__init__()
        if model_name == 'MultiheadAttenCNN':
            self.aircraft_encoder = MultiheadAttenCNN(
                config_ata4.in_channel, config_ata4.out_channel, config_ata4.num_heads, 
                config_ata4.output_dim, config_ata4.stride, config_ata4.kernel_size, config_ata4.dropout_ratio)

        elif model_name == 'TCNAtten':
            self.aircraft_encoder = TCN_Atten(
                config_ata4.in_channel, config_ata4.out_channel, config_ata4.num_heads, 
                config_ata4.output_dim, config_ata4.stride, config_ata4.kernel_size, config_ata4.dropout_ratio)

        self.part_encoder = TimeSeriesTransformer(
            config_sn.input_dim, config_sn.d_model, config_sn.nhead, 
            config_sn.num_encoder_layers, config_sn.dim_feedforward, config_sn.max_len)
        
        self.ata_emb = nn.Embedding(config_ata4.pn_size, ata_embed)
        
        # meta_dim = 16(ata) + 8(op) + 1(repaired_count) + 1(cum_fh) = 26
        # meta_dim = 16(pn) + 1(repaired_count) + 1(cum_fh) = 18
        self.gating_unit = MetaAttentionGating(
            ata4_dim=config_ata4.feat_dim, sn_dim=config_sn.feat_dim, meta_dim=ata_embed+2, hidden_dim=64)
        
        # hidden1 = config_ata4.feat_dim + config_sn.feat_dim
        self.ac_proj = nn.Linear(config_ata4.feat_dim, common_dim)
        self.part_proj = nn.Linear(config_sn.feat_dim, common_dim)

        self.fc_fusion = nn.Linear(common_dim, common_dim)
        self.reg_mu = nn.Linear(common_dim, 1)
        self.reg_sigma = nn.Sequential(
            nn.Linear(common_dim, 1), 
            nn.Softplus()
        )
        self.classifier = nn.Linear(common_dim, 1)

    def forward(self, x_ac, x_part, pn, repair, cum_fh, has_history):
        x_ac = x_ac.unsqueeze(0)
        out_ac, f_ac = self.aircraft_encoder(x_ac)
        out_part, f_part = self.part_encoder(x_part)
        
        ata_f = self.ata_emb(pn)
        # print('CHK Dim:', ata_f.shape, pn, repair, cum_fh)
        meta_feat = torch.cat([ata_f, repair, cum_fh], dim=-1)
        meta_feat = meta_feat.reshape(1, len(meta_feat))
        
        alpha = self.gating_unit(f_ac, f_part, meta_feat, has_history)
        
        f_ac = self.ac_proj(f_ac)
        f_part = self.part_proj(f_part)

        fused = (1 - alpha) * f_ac + alpha * f_part
        fused = F.relu(self.fc_fusion(fused))

        mu = self.reg_mu(fused)
        mu = mu[-1]
        sigma = self.reg_sigma(fused) + 1e-6
        sigma = sigma[-1]

        logit = self.classifier(fused)
        logit = logit[-1]
        
        return mu.squeeze(), sigma.squeeze(), alpha.squeeze(), logit, out_ac.squeeze(), out_part.squeeze()
        # return mu.squeeze(-1), sigma.squeeze(-1), alpha.squeeze(-1), logit, out_ac.squeeze(-1), out_part.squeeze(-1)


def train_FL(model, X_train_ac, X_train_sn, y_train_ac, y_train_sn, isUs, pns, repairs, cum_fhs, 
             has_history, optimizer, loss_gaussian, loss_logit, 
             epochs=50, lr = 0.0001, device=torch.device('cpu')):
    for epoch in range(epochs):
        # epoch_loss = 0
        # train_loss_vec = []
        model.train()
        # idx = 0
        has_history = has_history.to(device)

        for x_ac, x_sn, y_ac, y_sn, isu, pn, repair, cum_fh in zip(X_train_ac, X_train_sn, y_train_ac, y_train_sn, isUs, pns, repairs, cum_fhs):
        # for i in range(0, len(X_train_ac), batch):
            # print(x_ac.shape)
            x_ac, x_sn = x_ac.to(device), x_sn.to(device)
            y_ac, y_sn = y_ac.to(device), y_sn.to(device)
            isu = isu.to(device)
            pn, repair, cum_fh = pn.to(device), repair.to(device), cum_fh.to(device)
            has_history = has_history.to(device)
            
            mu, sigma, alpha, logit, out_ac, out_part = model(
                x_ac, x_sn, pn, repair, cum_fh, has_history
            )
            loss_reg = loss_gaussian(mu, y_ac, sigma)
            # print('LOGIT:', logit, logit.squeeze(), isu, isu.squeeze())
            loss_cls = loss_logit(logit.flatten(), isu.flatten())

            total_loss = loss_reg + loss_cls

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epochs {epoch}: logit={logit.item()}, alpha={alpha.item():.4f}, loss_reg={loss_reg.item()}, loss_cls={loss_cls.item()}")
    
    return model, total_loss.item()


def test_FL(model, X_test_ac, X_test_sn, isUs, pns, repairs, cum_fhs, has_history, device=torch.device('cpu')):
    mu_ps, sigma_ps, alpha_ps, logit_ps, out_acs, out_parts = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for x_ac, x_sn, isu, pn, repair, cum_fh in zip(X_test_ac, X_test_sn, isUs, pns, repairs, cum_fhs):
            x_ac, x_sn = x_ac.to(device), x_sn.to(device)
            # y_ac, y_sn = y_ac.to(device), y_sn.to(device)
            isu = isu.to(device)
            pn, repair, cum_fh = pn.to(device), repair.to(device), cum_fh.to(device)
            has_history = has_history.to(device)

            mu, sigma, alpha, logit, out_ac, out_part = model(
                x_ac, x_sn, pn, repair, cum_fh, has_history
            )
            mu_ps.append(mu.item())
            sigma_ps.append(sigma.item())
            alpha_ps.append(alpha.item())
            logit_ps.append(logit.item())
            out_acs.append(out_ac.item())
            out_parts.append(out_part.item())


    return mu_ps, sigma_ps, alpha_ps, logit_ps, out_acs, out_parts



class UnscheduledPredictor(nn.Module):
    def __init__(self, config_ata4, config_sn, common_dim = 64, ata_embed = 16, model_name='TCNAtten', dropout=0.1):
        super().__init__()
        self.dropout = dropout
        if model_name == 'MultiheadAttenCNN':
            self.aircraft_encoder = MultiheadAttenCNN(
                config_ata4.in_channel, config_ata4.out_channel, config_ata4.num_heads, 
                config_ata4.output_dim, config_ata4.stride, config_ata4.kernel_size, config_ata4.dropout_ratio)


        elif model_name == 'TCNAtten':
            self.aircraft_encoder = TCN_Atten(
                config_ata4.in_channel, config_ata4.out_channel, config_ata4.num_heads, 
                config_ata4.output_dim, config_ata4.stride, config_ata4.kernel_size, config_ata4.dropout_ratio)


        self.part_encoder = TimeSeriesTransformer(
            config_sn.input_dim, config_sn.d_model, config_sn.nhead, 
            config_sn.num_encoder_layers, config_sn.dim_feedforward, config_sn.max_len)
        
        self.ata_emb = nn.Embedding(config_ata4.pn_size, ata_embed)
        
        # meta_dim = 16(ata) + 8(op) + 1(repaired_count) + 1(cum_fh) = 26
        # meta_dim = 16(pn) + 1(repaired_count) + 1(cum_fh) = 18
        self.gating_unit = MetaAttentionGating(
            ata4_dim=config_ata4.feat_dim, sn_dim=config_sn.feat_dim, meta_dim=ata_embed+2, hidden_dim=64)
        
        # hidden1 = config_ata4.feat_dim + config_sn.feat_dim
        self.ac_proj = nn.Linear(config_ata4.feat_dim, common_dim)
        self.part_proj = nn.Linear(config_sn.feat_dim, common_dim)

        self.fc_fusion = nn.Linear(common_dim, common_dim)
        self.reg_mu = nn.Linear(common_dim, 1)
        self.reg_sigma = nn.Sequential(
            nn.Linear(common_dim, 1), 
            nn.Softplus()
        )

        self.log_tau = nn.Parameter(torch.zeros(1))

        self.classifier = nn.Sequential(
            nn.Linear(common_dim + 5, common_dim //2), ## delta, abs(delta), p1, p2, agreement
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(common_dim //2 , 1)   ## BCEWithLogitLoss
        )
        # self.classifier = nn.Linear(common_dim, 1)

    @property
    def tau(self):
        return self.log_tau.exp() + 1e-6

    def safe_cat(self, tensors, batch_size=1):
        reshaped = [t.reshape(batch_size, -1) for t in tensors]
        return torch.cat(reshaped, dim=1)


    def forward(self, x_ac, x_part, pn, repair, cum_fh, has_history):
        x_ac = x_ac.unsqueeze(0)
        out_ac, f_ac = self.aircraft_encoder(x_ac)
        out_part, f_part = self.part_encoder(x_part)

        ## For classification
        delta = out_part - out_ac
        delta_abs = delta.abs()
        agreement = torch.exp(-delta_abs / self.tau)
        
        ata_f = self.ata_emb(pn)
        # print('CHK Dim:', ata_f.shape, pn, repair, cum_fh)
        meta_feat = torch.cat([ata_f, repair, cum_fh], dim=-1)
        meta_feat = meta_feat.reshape(1, len(meta_feat))
        
        alpha = self.gating_unit(f_ac, f_part, meta_feat, has_history)
        
        f_ac = self.ac_proj(f_ac)
        f_part = self.part_proj(f_part)


        fused = (1 - alpha) * f_ac + alpha * f_part
        fused = F.relu(self.fc_fusion(fused))


        mu = self.reg_mu(fused)
        mu = mu[-1]
        sigma = self.reg_sigma(fused) + 1e-6
        sigma = sigma[-1]

        # for name, t in [('fused', fused), ('delta', delta), ('delta_abs', delta_abs),
                # ('out_part', out_part), ('out_ac', out_ac), ('agreement', agreement)]:
            # print(f"{name}: {t.shape}")
        # print(f'dim. fused: {fused.shape}, delta: {delta.shape}, abs: {delta_abs.shape}, part: {out_part.shape}, ac: {out_ac.shape}, agreement: {agreement.shape}')
        inputs = self.safe_cat([fused, delta, delta_abs, out_part, out_ac, agreement])
        logit = self.classifier(inputs)
        # logit = self.classifier(torch.cat([
            # fused.squeeze(0), 
            # delta.squeeze(), 
            # delta_abs.squeeze(), 
            # out_part.unsqueeze(0) if out_part.dim() ==1 else out_part.squeeze(0), 
            # out_ac.squeeze(), 
            # agreement.squeeze()
            # ], dim=0).unsqueeze(0))
        logit = logit[-1]
        
        return mu.squeeze(), sigma.squeeze(), alpha.squeeze(), logit, out_ac.squeeze(), out_part.squeeze()
        # return mu.squeeze(-1), sigma.squeeze(-1), alpha.squeeze(-1), logit, out_ac.squeeze(-1), out_part.squeeze(-1)

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # log(σ²) 형태로 학습 — 음수도 가능하게
        self.log_var_reg = nn.Parameter(torch.zeros(1))
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_con = nn.Parameter(torch.zeros(1))

    # def forward(self, L_reg, L_cls, L_consist):
        # L / σ² + log(σ²)
        # loss = (L_reg * torch.exp(-self.log_var_reg) + self.log_var_reg +
                # L_cls * torch.exp(-self.log_var_cls) + self.log_var_cls +
                # L_consist * torch.exp(-self.log_var_con) + self.log_var_con)
        # return loss
    def forward(self, L_reg, L_cls, L_consist, y_unsch):
        # y_unsch=1인 샘플만 regression loss에 반영
        # L_reg는 이미 sample-wise로 계산되어 있어야 함 (reduction='none')
        w_reg = y_unsch  # [B] — unscheduled만 regression 학습
        L_reg_masked = (w_reg * L_reg).sum() / (w_reg.sum() + 1e-8)

        loss = (L_reg_masked * torch.exp(-self.log_var_reg) + self.log_var_reg +
                L_cls     * torch.exp(-self.log_var_cls) + self.log_var_cls +
                L_consist * torch.exp(-self.log_var_con) + self.log_var_con)
        return loss
    
# def consist_loss(pred1, pred2, y_cls, epsilon=0.7):
def consist_loss(pred1, pred2, y_cls, margin=1.0):
    """
    pred1: 모델1 예측 (scheduled 무관 removal hour)
    pred2: 모델2 예측 (next unscheduled removal hour)
    y_cls: 실제 unscheduled 여부 (1 or 0)
    epsilon: scheduled 샘플의 최소 마진
    """
    delta = pred2 - pred1

    unsch_loss = y_cls * delta ** 2

    # unscheduled=0 → pred2가 pred1보다 epsilon 이상 커야 함
    # sch_loss = (1 - y_cls) * torch.clamp(epsilon - delta, min=0)
    sch_loss = (1 - y_cls) * torch.clamp(margin - delta, min=0) ** 2

    return (unsch_loss + sch_loss).mean()

def train_UP(model, X_train_ac, X_train_sn, y_train_ac, y_train_sn, isUs, pns, repairs, cum_fhs, 
             has_history, optimizer, loss_gaussian, loss_logit, multi_task_loss, 
             epochs=50, lr = 0.0001, l1=1, l2=0.3, margin=1.0, device=torch.device('cpu')):
    

    for epoch in range(epochs):
        # epoch_loss = 0
        # train_loss_vec = []
        model.train()
        # idx = 0
        has_history = has_history.to(device)


        for x_ac, x_sn, y_ac, y_sn, isu, pn, repair, cum_fh in zip(X_train_ac, X_train_sn, y_train_ac, y_train_sn, isUs, pns, repairs, cum_fhs):
        # for i in range(0, len(X_train_ac), batch):
            # print(x_ac.shape)
            optimizer.zero_grad()

            x_ac, x_sn = x_ac.to(device), x_sn.to(device)
            y_ac, y_sn = y_ac.to(device), y_sn.to(device)
            isu = isu.to(device)
            pn, repair, cum_fh = pn.to(device), repair.to(device), cum_fh.to(device)
            has_history = has_history.to(device)
            
            mu, sigma, alpha, logit, out_ac, out_part = model(
                x_ac, x_sn, pn, repair, cum_fh, has_history
            )

            loss_reg = loss_gaussian(mu, y_ac, sigma**2)
            loss_reg = loss_reg.mean(dim=-1) if loss_reg.dim() > 1 else loss_reg
            loss_cls = loss_logit(logit.flatten(), isu.flatten())

            # loss_consist = isu.flatten()*(out_ac-out_part)**2 + (1-isu.flatten())*max(0, epsilon-(out_part-out_ac))
            loss_consist = consist_loss(out_ac, out_part, isu.flatten(), margin=margin)

            # total_loss = loss_reg + l1*loss_cls + l2*loss_consist
            total_loss = multi_task_loss(loss_reg, loss_cls, loss_consist, isu.flatten())

            total_loss.backward()
            optimizer.step()


        with torch.no_grad():
            w_reg = torch.exp(-multi_task_loss.log_var_reg).item()
            w_cls = torch.exp(-multi_task_loss.log_var_cls).item()
            w_con = torch.exp(-multi_task_loss.log_var_con).item()
        if epoch % 10 == 0:
            print(f"Epochs {epoch}: logit={logit.item():.4f}, loss_reg={loss_reg.item():.4f}, loss_cls={loss_cls.item():.4f}, loss_consist={loss_consist.item():.4f}")
            print(f"        : w_reg={w_reg:.3f}, w_cls={w_cls:.3f}, w_con={w_con:.3f}, loss_total={total_loss.item():.4f}")
    
    return model, total_loss.item()




def test_UP(model, X_test_ac, X_test_sn, isUs, pns, repairs, cum_fhs, has_history, device=torch.device('cpu')):
    mu_ps, sigma_ps, alpha_ps, logit_ps, out_acs, out_parts = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for x_ac, x_sn, isu, pn, repair, cum_fh in zip(X_test_ac, X_test_sn, isUs, pns, repairs, cum_fhs):
            x_ac, x_sn = x_ac.to(device), x_sn.to(device)
            # y_ac, y_sn = y_ac.to(device), y_sn.to(device)
            isu = isu.to(device)
            pn, repair, cum_fh = pn.to(device), repair.to(device), cum_fh.to(device)
            has_history = has_history.to(device)


            mu, sigma, alpha, logit, out_ac, out_part = model(
                x_ac, x_sn, pn, repair, cum_fh, has_history
            )
            mu_ps.append(mu.item())
            sigma_ps.append(sigma.item())
            alpha_ps.append(alpha.item())
            logit_ps.append(logit.item())
            out_acs.append(out_ac.item())
            out_parts.append(out_part.item())




    return mu_ps, sigma_ps, alpha_ps, logit_ps, out_acs, out_parts