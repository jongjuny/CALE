
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
from torch import device, optim
from torch.nn import functional as F
import torch.nn.utils.weight_norm as weight_norm
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import warnings

# from models.mix_model import MultiTaskLoss
warnings.filterwarnings('ignore')

######################################################################
## After aligning target ATA6 removals into ATA4-grps in the same AC, 
## - Make numerical feature sets (ATA4-groups' estimated flight hours)
## - Make categorical feature sets (Operator code, PART_NO, Removal Months)
## - Divide Train / Test datasets for ATA4-group model (Ex. TCN_Atten)

def train_test_ata4_series(
        all_tests, all_chks, seq_len, pn_list, split_date, sel_ones='all', 
        embed=True, embedding_dim = 3, use_type =True, op_list=None):

    save_columns = ['PART_NO', 'PART_SN', 'AC_SN', 'INSTALL_DATE',
       'OPERATOR_CODE', 'ATA_NUMBER', 'ATA_CHAPTER', 'ATA_SECTION',
       'ATA_COMPONENT', 'REMOVAL_DATE', 'REMOVAL_TYPE_CODE',
       'QPA', 'FLIGHT_HOURS', 'FLIGHT_CYCLES',
       'CUML_CYCLES', 'ATA4', 'isU', 'reinstall_cnt',
       'CUMULATIVE_FLIGHT_HOURS', 'CUMULATIVE_CYCLES',
       'days', 'prev_fh', 'X_numeric', 'X_category']

    temp_columns = [col for col in save_columns if col not in ['X_numeric', 'X_category']]
    
    df_test = pd.DataFrame(columns=save_columns)
    df_train = pd.DataFrame(columns=save_columns)

    pn_list = np.array(pn_list).astype(str)
    if op_list is not None:
        op_list = np.array(op_list).astype(str)
        UNK_TOKEN='UNK'
        op_list = np.append(op_list, UNK_TOKEN)
        op_to_idx = {op:idx for idx, op in enumerate(op_list)}
        unk_idx = op_to_idx[UNK_TOKEN]

    if embed:
        part_emb = nn.Embedding(num_embeddings= len(pn_list), embedding_dim= embedding_dim)
        month_emb = nn.Embedding(num_embeddings=12, embedding_dim=embedding_dim)
        if op_list is not None:
            op_emb = nn.Embedding(num_embeddings=len(op_list), embedding_dim= 16)
    else:
        part_encoder = OneHotEncoder(categories=[pn_list], sparse_output=False, handle_unknown='ignore')
        part_encoder.fit(np.array(pn_list).reshape(-1,1))

        month_encoder = OneHotEncoder(categories=[np.arange(1, 13)], sparse_output=False)
        month_encoder.fit(np.arange(1, 13).reshape(-1,1))

    ## Divide numeric and others
    X_train_nums, X_train_ones, X_train_list, y_train_list, u_train_list = [], [], [], [], []
    X_test_nums, X_test_ones, X_test_list, y_test_list, u_test_list = [], [], [], [], []
    # ata_dim = len(all_chks[0][0])  # assuming all_chks[i][j] have the same columns
    ata_dim = 0
    for chk_tmp in all_chks:
        if len(chk_tmp) > 0:
            for chk_df in chk_tmp:
                if len(chk_df) > 0:
                    ata_dim = len(chk_df)
                else: continue
        else: continue

        if ata_dim != 0: break

    print(ata_dim)

    part_to_idx = {pn:idx for idx, pn in enumerate(pn_list)}

    for test_df, chk_list in zip(all_tests, all_chks):
        n = len(test_df)

        ## filter out single series
        if n <2: continue

        y_train_list, y_test_list = list(y_train_list), list(y_test_list)
        u_train_list, u_test_list = list(u_train_list), list(u_test_list)

        # Convert categorical features
        test_df['PART_NO'] = test_df['PART_NO'].astype(str)

        if sel_ones is not None:
            if embed:
                if sel_ones == 'all' or sel_ones == 'part':
                    # part_idx = test_df['PART_NO'].map(part_to_idx).to_numpy()
                    test_df['PART_NO_STR'] = test_df['PART_NO'].astype(str).str.strip()
                    part_idx = test_df['PART_NO_STR'].map(part_to_idx).to_numpy()
                    
                    x_part = part_emb(torch.tensor(part_idx, dtype=torch.long)) 
                    if op_list is not None:
                        op_idx = (test_df['OPERATOR_CODE'].map(lambda x: op_to_idx.get(x, unk_idx)).to_numpy())
                        x_op = op_emb(torch.tensor(op_idx, dtype=torch.long)) 
                if sel_ones == 'all' or sel_ones == 'month':
                    month_idx = test_df['INSTALL_DATE'].dt.month.to_numpy() -1
                    x_month = month_emb(torch.tensor(month_idx, dtype=torch.long)) 
                    if op_list is not None:
                        op_idx = (test_df['OPERATOR_CODE'].map(lambda x: op_to_idx.get(x, unk_idx)).to_numpy())
                        x_op = op_emb(torch.tensor(op_idx, dtype=torch.long)) 
            else:
                if sel_ones == 'all' or sel_ones == 'part':
                    x_part = part_encoder.transform(test_df['PART_NO'].to_numpy().reshape(-1, 1))
                if sel_ones == 'all' or sel_ones == 'month':
                    x_month = month_encoder.transform(test_df['INSTALL_DATE'].dt.month.to_numpy().reshape(-1, 1))

        if use_type:
            test_df['isU'] = test_df['prev_type'].map(lambda x:1 if x == 'U' else 0)
            x_type = test_df['isU'].to_numpy().reshape(-1, 1)
        
        ## previous flight hours
        x_prev = test_df['prev_fh'].to_numpy().reshape(-1, 1)

        # Build x_other (same length as test_df): ATA4 flight hours
        x_other = []
        for chk_df in chk_list:
            num_df = chk_df.sort_values(by='ATA_NUMBER')['FH_CUM']
            if num_df.empty:
                arr = np.zeros(ata_dim)
            else:
                # Use FH_CUM or all numerical columns
                arr = num_df.to_numpy().flatten()
                if len(arr) < ata_dim:
                    arr = np.pad(arr, (0, ata_dim - len(arr)))
                elif len(arr) > ata_dim:
                    arr = arr[:ata_dim]
            x_other.append(arr)
        x_other = np.stack(x_other, axis=0)  # (n, other_dim)

        # Merge all features
        y_all = test_df['FLIGHT_HOURS'].to_numpy()
        u_all = test_df['isU'].to_numpy()   ## Only for comparing
        removal_dates = test_df['REMOVAL_DATE']

        X_numeric = np.concatenate([x_prev, x_other], axis=1).astype(float)
        features = []

        ### Select and manage categorical features (saved into X_ones)
        if sel_ones in ['all', 'part']: features.append(x_part.detach())
        if sel_ones in ['all', 'month']: features.append(x_month.detach())
        if op_list is not None: features.append(x_op.detach())
        if use_type: features.append(x_type.detach())

        X_ones = torch.cat(features, dim=1) if features else None

        # Build subsequences with zero padding
        for t in range(1, n):
            start = max(0, t - seq_len)
            seq_num = X_numeric[start:t]
            if sel_ones is not None:
                seq_ones = X_ones[start:t]
            pad_len = seq_len - len(seq_num)
            if pad_len > 0:
                seq_num = np.concatenate([np.zeros((pad_len, X_numeric.shape[1])), seq_num], axis=0)
                if sel_ones is not None:
                    # seq_ones = np.concatenate([np.zeros((pad_len, X_ones.shape[1])), seq_ones], axis=0)
                    pad = torch.zeros((pad_len, X_ones.shape[1]),device=X_ones.device,dtype=X_ones.dtype)
                    seq_ones = torch.cat([pad, seq_ones], dim=0)

            # decide train/test by REMOVAL_DATE[t]
            if removal_dates.iloc[t] < split_date:
                # X_train_list.append([seq_num, seq_ones])
                X_train_nums.append(seq_num)
                if sel_ones is not None: X_train_ones.append(seq_ones)
                else: X_train_ones.append(None)
                y_train_list.append(y_all[t])
                u_train_list.append(u_all[t])

                df_add = test_df[temp_columns].iloc[t].copy()

                df_add['X_numeric'] = seq_num.tolist()
                df_add['X_category'] = [seq_ones.detach().clone()] if sel_ones is not None else None
                df_add  = df_add.to_frame().T
                df_train = pd.concat([df_train, df_add], ignore_index=True)
            else:
                # X_test_list.append([seq_num, seq_ones])
                X_test_nums.append(seq_num)
                if sel_ones is not None:
                    X_test_ones.append(seq_ones)
                else:
                    X_test_ones.append(None)
                y_test_list.append(y_all[t])
                u_test_list.append(u_all[t])
                df_add = test_df[temp_columns].iloc[t].copy()
                df_add['X_numeric'] = seq_num.tolist()
                df_add['X_category'] = [seq_ones.detach().clone()] if sel_ones is not None else None
                df_add  = df_add.to_frame().T
                df_test = pd.concat([df_test, df_add], ignore_index=True)
    
    print('TRAIN: ', len(X_train_nums), len(X_train_ones))
    if len(X_train_nums) < 10:
        return None, None, None, None, None, None, None, None, None, None
    X_train_ones = torch.stack(X_train_ones, dim=0).to(seq_ones.device)
    if len(X_test_ones) >0:
        X_test_ones = torch.stack(X_test_ones, dim=0).to(seq_ones.device)

    return X_train_nums, X_train_ones, y_train_list, u_train_list, X_test_nums, X_test_ones, y_test_list, u_test_list, df_test, df_train

## For extract zero-padded subsequence withlength seq_len, from PART_SN series
## For both U-series (train, test) and S-series (test only)
def _get_series_data(st, seq_len, df_t, split_date=None):
    X_series, df_train = [], pd.DataFrame()

    s_idx, e_idx = st
    subseqs = []

    # Init.
    e = s_idx
    s = max(0, e - seq_len)
    if s != e: subseqs.append([s, e])

    while e < e_idx:
        e += 1
        s = max(0, e - seq_len)
        if s != e: subseqs.append([s, e])

    subseqs.sort(key=lambda x: x[0])

    for sub_st in subseqs:
        if split_date is not None:
            if df_t.iloc[sub_st[1]]['REMOVAL_DATE'] < split_date:
                continue
        df_x = df_t[['cum_FH', 'isU']].iloc[sub_st[0]:sub_st[1]]
        series_X = df_x.to_numpy()
        if len(series_X) < seq_len:
            series_X = np.pad(series_X, ((seq_len-len(series_X), 0), (0, 0)), 'constant', constant_values=0)
        X_series.append(series_X)
        df_add = df_t.iloc[sub_st[1]]
        df_add  = df_add.to_frame().T
        df_train = pd.concat([df_train, df_add], ignore_index=True)

    df_train['X_hist'] = X_series
    return X_series, df_train


## From the given subset (by PART_SN, i.e., df_t), extract X_train & X_test 
## Arguments work as pointer, thus they would be updated for each cases
def append_series(
    df_t, st, seq_len, split_date,
    X_train, y_train,
    X_test, y_test,
    df_test, df_train,
    deviation
):
    end_idx = st[1]

    # TRAIN
    if df_t.iloc[end_idx]['REMOVAL_DATE'] < split_date:
        X_series, df_tr = _get_series_data(st, seq_len, df_t)
        if X_series:
            X_train.extend(X_series)
            if deviation == 'cum':
                y_train.extend([df_t['cum_FH'].iloc[end_idx]] * len(X_series))
                df_tr['Y'] = [df_t['cum_FH'].iloc[end_idx]] * len(X_series)
            
            ## We used 'inc' option only 
            else:  # inc
                y_train.extend(df_t['cum_FH'].iloc[end_idx]- np.array([x[-1, 0] for x in X_series]))
                df_tr['Y'] = df_t['cum_FH'].iloc[end_idx]- np.array([x[-1, 0] for x in X_series])

            df_train = pd.concat([df_train, df_tr])

    # TEST
    else:
        X_series, df_tr = _get_series_data(st, seq_len, df_t, split_date=split_date)
        if X_series:
            X_test.extend(X_series)
            if deviation == 'cum':
                y_test.extend([df_t['cum_FH'].iloc[end_idx]] * len(X_series))
                df_tr['Y'] = [df_t['cum_FH'].iloc[end_idx]] * len(X_series)
            
            ## We used 'inc' option only
            else:
                y_test.extend(df_t['cum_FH'].iloc[end_idx]- np.array([x[-1, 0] for x in X_series]))
                df_tr['Y'] = df_t['cum_FH'].iloc[end_idx]- np.array([x[-1, 0] for x in X_series])

            df_test = pd.concat([df_test, df_tr])

    return df_test, df_train

def findU(lst):
    result = []
    start = 0
    for i, val in enumerate(lst):
        if val == 'U':
            result.append([start, i])
            start = i + 1

    return result

######################################################################
## Extract PART_SN trajectories, building train / test for Transformer-encoder 

def train_test_part_SN_serial_rev(
    df_ata, sel_ops,
    latest_date, split_date,
    seq_len=5, deviation='cum',
    filter_single=True
):
    save_columns = ['PART_NO', 'PART_SN', 'AC_SN', 'INSTALL_DATE',
        'OPERATOR_CODE', 'ATA_NUMBER', 'ATA_CHAPTER', 'ATA_SECTION',
        'ATA_COMPONENT', 'REMOVAL_DATE', 'REMOVAL_TYPE_CODE',
        'QPA', 'FLIGHT_HOURS', 'FLIGHT_CYCLES',
        'CUML_CYCLES', 'ATA4', 'isU', 'reinstall_cnt',
        'CUMULATIVE_FLIGHT_HOURS', 'CUMULATIVE_CYCLES',
        'days', 'prev_fh', 'X_hist', 'Y']
    
    temp_columns = [col for col in save_columns if col not in ['X_hist']]
    df_ata = df_ata.dropna(subset=['OPERATOR_CODE'])

    X_train_U, y_train_U = [], []
    X_train_S, y_train_S = [], []
    X_train_US, y_train_US = [], []

    X_test, y_test = [], []
    y_test_type = []
    df_test, df_train = pd.DataFrame(), pd.DataFrame()

    for sn in tqdm(df_ata['PART_SN'].unique()):
        df_t = df_ata[df_ata['PART_SN'] == sn]

        if not df_t['OPERATOR_CODE'].isin(sel_ops).any():continue

        ## Use cumulative flight hours for Transformer encoder
        df_t = df_t.sort_values('INSTALL_DATE')
        df_t['cum_FH'] = df_t['FLIGHT_HOURS'].cumsum()
        df_t['cum_FC'] = df_t['FLIGHT_CYCLES'].cumsum()
        df_t['isU'] = (df_t['REMOVAL_TYPE_CODE'] == 'U').astype(int)

        ## Filter out single removals (per PART_SN)
        if filter_single and len(df_t) == 1: continue

        # Case 1: U Series
        if 'U' in df_t['REMOVAL_TYPE_CODE'].values:
            st_list = findU(list(df_t['REMOVAL_TYPE_CODE']))

            for st in st_list:
                if df_t.iloc[st[1]]['REMOVAL_DATE'] >= latest_date: continue

                df_test, df_train = append_series(
                    df_t, st, seq_len, split_date,
                    X_train_U, y_train_U,
                    X_test, y_test,
                    df_test, df_train,
                    deviation
                )

            # ---------- US-series ----------
            ## only updating 'test' sets
            if (df_t.iloc[-1]['REMOVAL_TYPE_CODE'] != 'U' and df_t.iloc[-1]['REMOVAL_DATE'] < latest_date):
                st = [st_list[-1][1] + 1, len(df_t) - 1]

                df_test, df_train = append_series(
                    df_t, st, seq_len, split_date,
                    X_train_US, y_train_US,
                    X_test, y_test,
                    df_test, df_train,
                    deviation
                )

        # Case 2: S-series
        ## only updating 'test' sets
        else:
            if df_t.iloc[-1]['REMOVAL_DATE'] < latest_date:
                st = [0, len(df_t) - 1]

                df_test, df_train = append_series(
                    df_t, st, seq_len, split_date,
                    X_train_S, y_train_S,
                    X_test, y_test,
                    df_test, df_train,
                    deviation
                )

    return (
        X_train_U, y_train_U,
        X_train_S, y_train_S,
        X_train_US, y_train_US,
        X_test, y_test, y_test_type,
        df_test, df_train
    )


def makeTensor(array, device):
    """
    Transform numpy to tensor
    """
    return torch.from_numpy(np.array(array)).float().to(device)

def one_hot_encode(encode_list, sel):
    """
    One-hot encoding for categorical data including part number and weekdays
    """
    t_encoding = np.zeros(len(encode_list))
    t_encoding[encode_list.index(sel)] = 1
    return t_encoding

def add_before_fh(all_tests):
    for a_test in all_tests:
        chk_prev = []
        for i in range(len(a_test)):
            no_add = True
            if i ==0: 
                chk_prev.append(a_test.iloc[i]['FLIGHT_HOURS'])
                continue
            for j in range(i, -1, -1):
                if a_test.iloc[i]['INSTALL_DATE'] >= a_test.iloc[j]['REMOVAL_DATE']:
                    chk_prev.append(a_test.iloc[j]['FLIGHT_HOURS'])
                    no_add =False
                    break
            if no_add:
                chk_prev.append(a_test.iloc[i]['FLIGHT_HOURS'])
            
        # print(len(a_test), len(chk_prev))
        a_test['prev_fh'] = chk_prev

    return all_tests

def train_test_split_rev_fh(all_tests, all_chks, pn_list, op_list, split_date, valid=False, valid_date=None):
    """
    Split train and test sets from all_tests & all_chks (return from the related_at_install)
    Support CNN and MLP model in this file
    """
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    if len(all_tests) == 0: 
        print('No Test Set!')
        return [], [], [], []
    
    y_train_df, y_test_df = pd.DataFrame(columns=all_tests[0].columns), pd.DataFrame(columns=all_tests[0].columns)
    X_train, X_test, y_train, y_test = [], [], [], []

    all_tests2 = pd.DataFrame(columns = all_tests[0].columns)
    all_chks2, list_date = [], []
    for tests, chks in zip(all_tests, all_chks):
        all_tests2 = pd.concat([all_tests2, tests])
        all_chks2 += chks
    list_date = list(all_tests2['REMOVAL_DATE'])

    all_tests3 = all_tests2.sort_values(by='REMOVAL_DATE', ascending=True)
    date3 = np.array(list_date)
    date3_idx = np.argsort(list_date)
    all_chks3 = [all_chks2[i] for i in date3_idx]


    if valid:
        if valid_date is None:
            print('Error. Need to specify the valid date')
            return None
        y_valid_df = pd.DataFrame(columns=all_tests[0].columns)
        X_valid, y_valid = [], []

    all_tests3 = all_tests3.dropna()
    
    for id in range(len(all_tests3)):
        if all_tests3.iloc[id]['FLIGHT_HOURS'] ==0: continue
        if len(all_chks3[id]) == 0: continue
        if all_tests3.iloc[id]['OPERATOR_CODE'] not in op_list: continue
        if all_tests3.iloc[id]['REMOVAL_DATE'] < split_date:
            pn_en = one_hot_encode(pn_list, all_tests3.iloc[id]['PART_NO'])
            month_en = one_hot_encode(months, all_tests3.iloc[id]['INSTALL_DATE'].month)
            if len(op_list) == 1:
                op_en = []
            else:
                op_en = one_hot_encode(op_list, all_tests3.iloc[id]['OPERATOR_CODE'])
            if valid:
                if all_tests3.iloc[id]['REMOVAL_DATE'] >= valid_date:
                    y_valid.append(all_tests3.iloc[id]['FLIGHT_HOURS'])
                    X_valid.append(list(all_chks3[id].sort_values(by='ATA_NUMBER')['FH_CUM']) + [all_tests3.iloc[id]['prev_fh']]+ list(pn_en) + list(month_en) + list(op_en))
                    y_valid_df = pd.concat([y_valid_df, pd.DataFrame(all_tests3.iloc[id]).T])
                    continue
                
            y_train.append(all_tests3.iloc[id]['FLIGHT_HOURS'])
            X_train.append(list(all_chks3[id].sort_values(by='ATA_NUMBER')['FH_CUM'])+ [all_tests3.iloc[id]['prev_fh']]+ list(pn_en) + list(month_en) + list(op_en))
            y_train_df = pd.concat([y_train_df, pd.DataFrame(all_tests3.iloc[id]).T])

        else:
            y_test.append(all_tests3.iloc[id]['FLIGHT_HOURS'])
            X_test.append(list(all_chks3[id].sort_values(by='ATA_NUMBER')['FH_CUM']) + [all_tests3.iloc[id]['prev_fh']]+ list(pn_en) + list(month_en) + list(op_en))
            y_test_df = pd.concat([y_test_df, pd.DataFrame(all_tests3.iloc[id]).T])
    
    if valid:
        return X_train, X_test, X_valid, y_train, y_test, y_valid, y_train_df, y_test_df, y_valid_df
    else:
        return X_train, X_test, y_train, y_test, y_train_df, y_test_df
    

#######################################################################################
## Models
def scaled_dot_product_attention(Q, K, V, mask=None):
    # Compute the dot products between Q and K, then scale by the square root of the key dimension
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # Apply mask if provided (useful for masked self-attention in transformers)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax to normalize scores, producing attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Compute the final output as weighted values
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        # Define linear transformations for Q, K, V
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        # Generate Q, K, V matrices
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Calculate attention using our scaled dot-product function
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        return out

def cross_attention(Q, K, V, mask=None):
    # Compute the dot products between Q and K, then scale
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax to normalize scores and get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Weighted sum of values
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

class MultiHeadAttentionRegression_MLP(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, embed_dim, num_heads, output_dim, dropout_ratio=0.3):
        super(MultiHeadAttentionRegression_MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim1 = hidden1
        self.hidden_dim2 = hidden2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden1, bias=True),
            # nn.BatchNorm1d(in2),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden1, hidden2, bias=True),
            # nn.BatchNorm1d(in3),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden2, embed_dim, bias=True),
            nn.Sigmoid()
        )

        # self.input_proj = nn.Linear(input_dim, embed_dim)

        self.multihead_atten = nn.MultiheadAttention(embed_dim, num_heads)

        self.fc = nn.Linear(embed_dim, output_dim)
        

    def forward(self, x):
        # x = self.input_proj(x)
        x = self.mlp(x)

        atten_output, _ = self.multihead_atten(x, x, x)

        output = self.fc(atten_output)

        return output

class AttenCNN(nn.Module):
    def __init__(self, in_channel, out_channel, stride, kernel_size, dropout_ratio = 0.1):
        super(AttenCNN, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=1, padding_mode='zeros'), 
            nn.ReLU(),
            nn.Dropout1d(dropout_ratio),
            nn.MaxPool1d(kernel_size=2, stride=stride)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=1, padding_mode='zeros'), 
            nn.ReLU(),
            nn.Dropout1d(dropout_ratio),
            nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
        )
        ## Attention
        
        self.self_attention = SelfAttention(out_channel)
        # self.multi_attention = MultiHeadAttention(out_channel, 4)
        ## Fill in the size for Linear
        self.fc = nn.Linear(out_channel, 1, bias=True)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = torch.cat([x], dim=0)
        out = self.layer1(x.reshape(-1,1))
        ## revise the layers
        out = self.layer2(out)

        out = self.self_attention(out.view(-1, out.size(0)))
        # out = self.multi_attention(out.view(-1, out.size(0)))

        # out = out.view(-1, out.size(0)) ## Flatten
        # print('3', out.shape)
        
        out = self.fc(out)
        return out

class MultiheadAttenCNN(nn.Module):
    def __init__(self, in_channel, out_channel, num_heads, output_dim, stride, kernel_size, dropout_ratio = 0.1):
        super(MultiheadAttenCNN, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=1, padding_mode='zeros'), 
            nn.ReLU(),
            nn.Dropout1d(dropout_ratio),
            nn.MaxPool1d(kernel_size=2, stride=stride)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=1, padding_mode='zeros'), 
            nn.ReLU(),
            nn.Dropout1d(dropout_ratio),
            nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
        )   ## <-- increase layers
        ## Attention
        
        # self.self_attention = SelfAttention(out_channel)
        self.multi_attention = nn.MultiheadAttention(out_channel, num_heads)
        ## Fill in the size for Linear
        self.fc = nn.Linear(out_channel, output_dim, bias=True)
        ## <-- increase FFN ()
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = torch.cat([x], dim=0)
        out = self.layer1(x.reshape(-1,1))
        ## revise the layers
        out = self.layer2(out)
        out = out.reshape(-1,out.size(0))

        # out = self.self_attention(out.view(-1, out.size(0)))
        atten_out, _ = self.multi_attention(out, out, out)

        # out = out.view(-1, out.size(0)) ## Flatten
        
        out = self.fc(atten_out)
        return out, atten_out

class TCN_Atten(nn.Module):
    def __init__(self, in_channel, out_channel, num_heads, output_dim, stride, kernel_size, dropout_ratio = 0.1):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.layer1 = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=1, padding_mode='zeros')), 
            nn.ReLU(),
            nn.Dropout1d(dropout_ratio)
        )
        self.layer2 = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=1, padding_mode='zeros')), 
            nn.ReLU(),
            nn.Dropout1d(dropout_ratio),
        )   ## <-- increase layers
        ## Attention
        
        self.multi_attention = nn.MultiheadAttention(embed_dim=out_channel, num_heads=num_heads, batch_first=True)
        ## Fill in the size for Linear
        self.fc = nn.Linear(out_channel, output_dim, bias=True)
        ## <-- increase FFN ()
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.transpose(1,2)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.transpose(1, 2)

        atten_out, _ = self.multi_attention(out, out, out)        
        out = self.fc(atten_out[:, -1, :])
        return out, atten_out[:, -1, :]

class TCN_AttenMulti(nn.Module):
    def __init__(self, in_channel, out_channel, num_heads, output_dim, stride, kernel_size, dropout_ratio = 0.1):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.layer1 = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=1, padding_mode='zeros')), 
            nn.ReLU(),
            nn.Dropout1d(dropout_ratio)
        )
        self.layer2 = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=1, padding_mode='zeros')), 
            nn.ReLU(),
            nn.Dropout1d(dropout_ratio),
        )   ## <-- increase layers
        ## Attention
        
        self.multi_attention = nn.MultiheadAttention(embed_dim=out_channel, num_heads=num_heads, batch_first=True)
        ## Fill in the size for Linear
        self.fc = nn.Linear(out_channel, output_dim, bias=True)
        self.unsch = nn.Linear(out_channel, 1, bias=True)
        ## <-- increase FFN ()
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.transpose(1,2)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.transpose(1, 2)
        atten_out, _ = self.multi_attention(out, out, out)        
        out = self.fc(atten_out[:, -1, :])
        unsch = self.unsch(atten_out[:, -1, :])
        return out, atten_out[:, -1, :], unsch
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model) or (seq_len, d_model)
        """
        # If no batch, 
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, seq_len, d_model)
            squeeze_out = True
        else:
            squeeze_out = False

        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device).type_as(x)

        if squeeze_out:
            x = x.squeeze(0)
        return x
    
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=2, d_model=64, nhead=8, num_layers=2, dim_feedforward=128, max_len=5000):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True  # (batch, seq_len, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.predictor = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (batch, seq_len, feature_dim) or (seq_len, feature_dim)
        """
        # if no batch (single sequence)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, seq_len, feature_dim)
            squeeze_out = True
        else:
            squeeze_out = False

        # 입력 linear projection
        x = self.input_linear(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)   # (batch, seq_len, d_model)
        x = self.encoder(x)       # (batch, seq_len, d_model)

        x_last = x[:, -1, :]      # (batch, d_model)
        out = self.predictor(x_last)  # (batch, 1)

        if squeeze_out:
            out = out.squeeze(0)  # (1)

        return out, x_last

class TimeSeriesTransformerMulti(nn.Module):
    def __init__(self, input_dim=2, d_model=64, nhead=8, num_layers=2, dim_feedforward=128, max_len=5000):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True  # (batch, seq_len, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.predictor = nn.Linear(d_model, 1)
        self.unsch = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (batch, seq_len, feature_dim) or (seq_len, feature_dim)
        """
        # if no batch (single sequence)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, seq_len, feature_dim)
            squeeze_out = True
        else:
            squeeze_out = False

        # 입력 linear projection
        x = self.input_linear(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)   # (batch, seq_len, d_model)
        x = self.encoder(x)       # (batch, seq_len, d_model)

        x_last = x[:, -1, :]      # (batch, d_model)
        out = self.predictor(x_last)  # (batch, 1)
        unsch = self.unsch(x_last)    # (batch, 1)
        if squeeze_out:
            out = out.squeeze(0)  # (1)
            unsch = unsch.squeeze(0)  # (1)

        return out, x_last, unsch

###########################################################################################################
## Previous version
def train_model(
    model, model_name, X_train, y_train, X_test, y_test, X_valid=None, y_valid=None, revise_valid = False, 
    learning_rate=0.001, epochs=300, epochs_valid=300, out_per=None, device=torch.device('cpu')
    ):

    loss_fn = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses, re_losses, val_losses = [], [], []
    test_losses_val, test_losses = [], []

    for epoch in range(epochs):
        epoch_loss = 0
        train_loss_vec = []
        model.train()
        idx = 0

        for s1 in X_train:
            s1 = s1.to(device)
            optimizer.zero_grad()
            if model_name in ['MultiHeadAttentionRegression', 'MultiHeadAttentionRegression_MLP']:
                y_pred = model(s1.view(1, 1, s1.size(0)))
            else:
                y_pred = model(s1)
            loss = loss_fn(y_pred.reshape(-1), y_train[idx].to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_loss_vec.append(loss.item())
            idx +=1

        if revise_valid:
            val_loss = 0
            model.eval()
            for val_x, val_y in zip(X_valid, y_valid):
                val_x = val_x.to(device)
                if model_name in ['MultiHeadAttentionRegression', 'MultiHeadAttentionRegression_MLP']:
                    y_vp = model(val_x.view(1, 1, val_x.size(0)))
                else:
                    y_vp = model(val_x)
                loss = loss_fn(y_vp.reshape(-1), val_y.to(device))
                val_loss += loss.item()

        test_loss = 0
        model.eval()
        for test_x, test_y in zip(X_test, y_test):
            test_x = test_x.to(device)
            if model_name in ['MultiHeadAttentionRegression', 'MultiHeadAttentionRegression_MLP']:
                y_tp= model(test_x.view(1, 1, test_x.size(0)))
            else:
                y_tp = model(test_x)
            loss_test = loss_fn(y_tp.reshape(-1), test_y.to(device))
            test_loss += loss_test.item()

        if epoch % (epochs/10) == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, epochs, epoch_loss/len(X_train)))
            if revise_valid and len(X_valid) > 0:
                print('Validation: {:4d}/{} Cost: {:.6f}'.format(epoch, epochs, val_loss/len(X_valid)))
            print('Test: {:4d}/{} Cost: {:.6f}'.format(epoch, epochs, test_loss/len(X_test)))

        losses.append(epoch_loss/len(X_train))
        if revise_valid and len(X_valid) > 0: val_losses.append(val_loss/len(X_valid))
        else: val_losses.append(0)
        test_losses_val.append(test_loss/len(X_test))

        if epoch == int(epochs/2):
        # if epoch == int(epochs/2):
            if out_per is None:
                th = np.mean(train_loss_vec) + 3*np.std(train_loss_vec)
            else:
                nt = math.ceil(len(train_loss_vec)* out_per/100)
                th = sorted(train_loss_vec, reverse=True)[nt-1]

            out_inds_tr = [i for i, val in enumerate(train_loss_vec) if val > th]
            X_tr_out = [X_train[i] for i in out_inds_tr]
            y_tr_out = [y_train[i] for i in out_inds_tr]
            X_tr_re = [X_train[i] for i in range(len(X_train)) if i not in out_inds_tr]
            y_tr_re = [y_train[i] for i in range(len(y_train)) if i not in out_inds_tr]
            X_train, y_train = X_tr_re, y_tr_re


    if revise_valid:
        for epoch in range(epochs_valid):
            valid_loss_vec = []
            epoch_loss = 0
            idx = 0
            model.train()
            for seq in X_valid:
                seq.to(device)
                optimizer.zero_grad()
                if model_name in ['MultiHeadAttentionRegression', 'MultiHeadAttentionRegression_MLP']:
                    y_pred = model(seq.view(1, 1, seq.size(0)))
                else:
                    y_pred = model(seq)
                loss = loss_fn(y_pred.reshape(-1), y_valid[idx].to(device))
                loss.backward()
                optimizer.step()
                idx +=1
                epoch_loss += loss.item()
                valid_loss_vec.append(loss.item())

            test_loss = 0
            model.eval()
            for test_x, test_y in zip(X_test, y_test):
                test_x.to(device)
                if model_name in ['MultiHeadAttentionRegression', 'MultiHeadAttentionRegression_MLP']:
                    y_tp = model(test_x.view(1, 1, test_x.size(0)))
                else:
                    y_tp = model(test_x)
                loss_test = loss_fn(y_tp.reshape(-1), test_y.to(device))
                test_loss += loss_test.item()
            if epoch % (epochs_valid/10) == 0 and len(X_valid)>0:
                print('[RE] Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, epochs_valid, epoch_loss/len(X_valid)))
                print('Test: {:4d}/{} Cost: {:.6f}'.format(epoch, epochs, test_loss/len(X_test)))
            if len(X_valid)>0 : re_losses.append(epoch_loss/len(X_valid))
            else: re_losses.append(0)
            test_losses.append(test_loss/len(X_test))

            if epoch == int(epochs/2):
                out_inds_val = [i for i, val in enumerate(valid_loss_vec) if val > th]
                X_val_out = [X_valid[i] for i in out_inds_val]
                y_val_out = [y_valid[i] for i in out_inds_val]
                X_val_re = [X_valid[i] for i in range(len(X_valid)) if i not in out_inds_val]
                y_val_re = [y_valid[i] for i in range(len(y_valid)) if i not in out_inds_val]
                X_valid, y_valid = X_val_re, y_val_re

    return model

def test_model(model, model_name, X_test, device=torch.device('cpu')):
    y_ps = []
    model.eval()
    with torch.no_grad():
        for s1 in X_test:
            if model_name in ['MultiHeadAttentionRegression', 'MultiHeadAttentionRegression_MLP']:
                y_ps.append(model(s1.view(1, 1, s1.size(0))).item())
            else:
                y_ps.append(model(s1).item())
    return y_ps


###########################################################################################################
## Current version
def train_model_rev(
    model, model_name, X_train, y_train, X_test, y_test, X_valid=None, y_valid=None, revise_valid = False, filter_out=True, 
    learning_rate=0.001, epochs=300, epochs_valid=300, out_per=None, batch=1, device=torch.device('cpu')
    ):

    loss_fn = nn.MSELoss().to(device)
    if model_name == 'TransformerRegression':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses, re_losses, val_losses = [], [], []
    test_losses_val, test_losses = [], []

    for epoch in range(epochs):
        epoch_loss = 0
        train_loss_vec = []
        model.train()
        idx = 0

        # for s1, y1 in zip(X_train, y_train):
        for i in range(0, len(X_train), batch):
            
            x_b = X_train[i:i+batch].to(device)
            y_b = y_train[i:i+batch].to(device)

            # x_b = x_b.unsqueeze(1)

            optimizer.zero_grad()
            y_pred, feat_vec = model(x_b)

            loss = loss_fn(y_pred.view(-1), y_b)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_loss_vec.append(loss.item())
            idx +=1

        if revise_valid:
            val_loss = 0
            model.eval()
            for val_x, val_y in zip(X_valid, y_valid):
                val_x = val_x.to(device)
                if model_name in ['MultiHeadAttentionRegression', 'MultiHeadAttentionRegression_MLP', 'MLPAtten', 'MLP_VL']:
                    y_vp, feat_vec = model(val_x.view(1, 1, val_x.size(0)))
                elif model_name == 'TransformerRegression':
                    y_vp, feat_vec = model(val_x)
                elif model_name == 'TCNAtten':
                    val_x = val_x.unsqueeze(0)
                    # val_x = val_x.transpose(1,2)
                    y_vp, feat_vec = model(val_x)
                else:
                    y_vp, feat_vec = model(val_x)

                if model_name == 'TCNAtten':
                    loss = loss_fn(y_vp.view(-1), val_y.to(device))
                else:
                    loss = loss_fn(y_vp.reshape(-1), val_y.to(device))
                val_loss += loss.item()

        test_loss = 0

        if epoch % (epochs/10) == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, epochs, epoch_loss/len(X_train)))
            if revise_valid and len(X_valid) > 0:
                print('Validation: {:4d}/{} Cost: {:.6f}'.format(epoch, epochs, val_loss/len(X_valid)))

        losses.append(epoch_loss/len(X_train))
        if revise_valid and len(X_valid) > 0: val_losses.append(val_loss/len(X_valid))
        else: val_losses.append(0)
        if len(X_test) == 0:
            test_losses_val = 0
        else:
            test_losses_val.append(test_loss/len(X_test))

        if filter_out and epoch == int(epochs/2):
        # if epoch == int(epochs/2):
            if out_per is None:
                th = np.mean(train_loss_vec) + 3*np.std(train_loss_vec)
            else:
                nt = math.ceil(len(train_loss_vec)* out_per/100)
                th = sorted(train_loss_vec, reverse=True)[nt-1]

            out_inds_tr = [i for i, val in enumerate(train_loss_vec) if val > th]
            X_tr_out = [X_train[i] for i in out_inds_tr]
            y_tr_out = [y_train[i] for i in out_inds_tr]
            X_tr_re = [X_train[i] for i in range(len(X_train)) if i not in out_inds_tr]
            y_tr_re = [y_train[i] for i in range(len(y_train)) if i not in out_inds_tr]
            X_train, y_train = X_tr_re, y_tr_re
            X_train = torch.stack(X_train)
            y_train = torch.stack(y_train)

    if revise_valid:
        for epoch in range(epochs_valid):
            valid_loss_vec = []
            epoch_loss = 0
            idx = 0
            model.train()
            for seq, yv in zip(X_valid, y_valid):
                seq.to(device)
                optimizer.zero_grad()
                if model_name in ['MultiHeadAttentionRegression', 'MultiHeadAttentionRegression_MLP', 'MLPAtten', 'MLP_VL']:
                    y_pred,feat_vec = model(seq.view(1, 1, seq.size(0)))
                elif model_name == 'TransformerRegression':
                    y_pred,feat_vec = model(s1)
                elif model_name == 'TCNAtten':
                    s1 = s1.unsqueeze(0)
                    # s1 = s1.transpose(1,2)
                    y_pred, feat_vec = model(s1)
                else:
                    y_pred,feat_vec = model(seq)
                loss = loss_fn(y_pred.reshape(-1), yv.to(device))
                loss.backward()
                optimizer.step()
                idx +=1
                epoch_loss += loss.item()
                valid_loss_vec.append(loss.item())

            test_loss = 0
            if epoch % (epochs_valid/10) == 0 and len(X_valid)>0:
                print('[RE] Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, epochs_valid, epoch_loss/len(X_valid)))

            if len(X_valid)>0 : re_losses.append(epoch_loss/len(X_valid))
            else: re_losses.append(0)
            test_losses.append(test_loss/len(X_test))

            if filter_out and epoch == int(epochs/2):
                out_inds_val = [i for i, val in enumerate(valid_loss_vec) if val > th]
                X_val_out = [X_valid[i] for i in out_inds_val]
                y_val_out = [y_valid[i] for i in out_inds_val]
                X_val_re = [X_valid[i] for i in range(len(X_valid)) if i not in out_inds_val]
                y_val_re = [y_valid[i] for i in range(len(y_valid)) if i not in out_inds_val]
                X_valid, y_valid = X_val_re, y_val_re

    if filter_out:
        if revise_valid:
            return model, losses, val_losses, test_losses_val, re_losses, test_losses, out_inds_tr, X_tr_out, y_tr_out, out_inds_val, X_val_out, y_val_out, feat_vec
        else:
            return model, losses, test_losses_val, out_inds_tr, X_tr_out, y_tr_out, feat_vec
    else:
        if revise_valid:
            return model, losses, val_losses, test_losses_val, re_losses, test_losses, feat_vec
        else:
            return model, losses, test_losses_val, feat_vec

def test_model_rev(model, model_name, X_test, device=torch.device('cpu')):
    y_ps = []
    print('Shape of X_test:', X_test[0].shape, X_test[0].unsqueeze(0).shape)
    model.eval()
    with torch.no_grad():
        for s1 in X_test:
            s1 = s1.to(device)
            if model_name in ['MultiHeadAttentionRegression', 'MultiHeadAttentionRegression_MLP', 'MLPAtten', 'MLP_VL']:
                yp, feat_v = model(s1.view(1, 1, s1.size(0)))
                y_ps.append(yp.item())
            elif model_name == 'TCNAtten':
                s1 = s1.unsqueeze(0)
                # s1 = s1.transpose(1,2)
                yp, feat_v = model(s1)
                yp = yp[-1]
                y_ps.append(yp.item())
            else:
                yp, feat_v =model(s1)
                y_ps.append(yp.item())
    return y_ps

##########################################################################################################

class MultiTaskLossClass(nn.Module):
    """
    Kendall et al. (2018) uncertainty weighting.
    regression  : Gaussian NLL  → weight  exp(-log_var_reg)
    classification : BCE        → weight  exp(-log_var_cls)
    """
    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.log_var_reg = nn.Parameter(torch.zeros(1))   # learnable
        self.log_var_cls = nn.Parameter(torch.zeros(1))


        self.reg_loss_fn = nn.MSELoss()
        self.cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    def forward(
        self,
        reg_pred:  torch.Tensor,   # (seq, batch, output_dim)
        reg_target: torch.Tensor,
        cls_pred:  torch.Tensor,   # (seq, batch, 1)  or  (batch, 1)
        cls_target: torch.Tensor,  # float 0/1, same shape as cls_pred
    ) -> tuple[torch.Tensor, dict]:


        l_reg = self.reg_loss_fn(reg_pred, reg_target)
        l_cls = self.cls_loss_fn(cls_pred, cls_target)


        # uncertainty weighting
        # loss = (1/2) * exp(-s) * L + (1/2) * s   where s = log_var
        weighted_reg = 0.5 * torch.exp(-self.log_var_reg) * l_reg + 0.5 * self.log_var_reg
        weighted_cls = 0.5 * torch.exp(-self.log_var_cls) * l_cls + 0.5 * self.log_var_cls


        total = weighted_reg + weighted_cls


        info = {
            "loss_reg": l_reg.item(),
            "loss_cls": l_cls.item(),
            "log_var_reg": self.log_var_reg.item(),
            "log_var_cls": self.log_var_cls.item(),
        }
        return total, info
    

## For predicting multi-task (regression + classification)
def train_model_multi(
    model, model_name, X_train, y_train, u_train, filter_out=True, 
    learning_rate=0.001, epochs=300, epochs_valid=300, out_per=None, batch=1, pos_weight=9.0, device=torch.device('cpu')
    ):


    if model_name == 'TSTMulti':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    pos_weight = torch.tensor([pos_weight], device=device)
    multi_loss  = MultiTaskLossClass(pos_weight=pos_weight).to(device)


    for epoch in range(epochs):
        epoch_loss = 0
        train_loss_vec = []
        model.train()
        idx = 0

        # for s1, y1 in zip(X_train, y_train):
        for i in range(0, len(X_train), batch):            
            x_b = X_train[i:i+batch].to(device)
            y_b = y_train[i:i+batch].to(device)
            u_b = u_train[i:i+batch].to(device)

            optimizer.zero_grad()

            y_pred, feat_vec, unsch = model(x_b)

            loss, info = multi_loss(y_pred, y_b, unsch, u_b.float())

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_loss_vec.append(loss.item())
            idx +=1

        if epoch % (epochs/10) == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, epochs, epoch_loss/len(X_train)))


        losses.append(epoch_loss/len(X_train))


        if filter_out and epoch == int(epochs/2):
        # if epoch == int(epochs/2):
            if out_per is None:
                th = np.mean(train_loss_vec) + 3*np.std(train_loss_vec)
            else:
                nt = math.ceil(len(train_loss_vec)* out_per/100)
                th = sorted(train_loss_vec, reverse=True)[nt-1]


            out_inds_tr = [i for i, val in enumerate(train_loss_vec) if val > th]
            X_tr_out = [X_train[i] for i in out_inds_tr]
            y_tr_out = [y_train[i] for i in out_inds_tr]
            u_tr_out = [u_train[i] for i in out_inds_tr]
            X_tr_re = [X_train[i] for i in range(len(X_train)) if i not in out_inds_tr]
            y_tr_re = [y_train[i] for i in range(len(y_train)) if i not in out_inds_tr]
            u_tr_re = [u_train[i] for i in range(len(u_train)) if i not in out_inds_tr]
            X_train, y_train, u_train = X_tr_re, y_tr_re, u_tr_re
            X_train = torch.stack(X_train)
            y_train = torch.stack(y_train)
            u_train = torch.stack(u_train)


    if filter_out:
        return model, losses, out_inds_tr, X_tr_out, y_tr_out, u_tr_out, feat_vec
                
    else:
        return model, losses, feat_vec
    

def test_model_multi(model, model_name, X_test, device=torch.device('cpu')):
    y_ps, unsch_ps = [], []
    model.eval()
    print(model_name)
    with torch.no_grad():
        for s1 in X_test:
            s1 = s1.to(device)
            # if model_name in ['MultiHeadAttentionRegression', 'MultiHeadAttentionRegression_MLP', 'MLPAtten', 'MLP_VL']:
                # yp, feat_v = model(s1.view(1, 1, s1.size(0)))
                # y_ps.append(yp.item())
            if model_name in ['TCN_AttenMulti', 'TSTMulti']:
                s1 = s1.unsqueeze(0)
                # s1 = s1.transpose(1,2)
                yp, feat_v, unsch = model(s1)
                yp = yp[-1]
                unsch = unsch[-1]
                y_ps.append(yp.item())
                unsch_ps.append(unsch.item())
            else:
                yp, feat_v =model(s1)
                y_ps.append(yp.item())
    return y_ps, unsch_ps