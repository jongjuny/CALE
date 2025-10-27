
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
import datetime
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

## Numpy array to Tensor
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

## Save previous flight hours of removal (same AC case)
def add_before_fh(all_tests):
    for a_test in all_tests:
        chk_prev, type_prev = [], []
        for i in range(len(a_test)):
            no_add = True
            if i ==0: 
                chk_prev.append(a_test.iloc[i]['FLIGHT_HOURS'])
                type_prev.append(a_test.iloc[i]['REMOVAL_TYPE_CODE'])
                continue
            for j in range(i, -1, -1):
                if a_test.iloc[i]['INSTALL_DATE'] >= a_test.iloc[j]['REMOVAL_DATE']:
                    chk_prev.append(a_test.iloc[j]['FLIGHT_HOURS'])
                    type_prev.append(a_test.iloc[j]['REMOVAL_TYPE_CODE'])
                    no_add =False
                    break
            if no_add:
                chk_prev.append(a_test.iloc[i]['FLIGHT_HOURS'])
                type_prev.append(a_test.iloc[i]['REMOVAL_TYPE_CODE'])
            
        # print(len(a_test), len(chk_prev))
        a_test['prev_fh'] = chk_prev
        a_test['prev_type'] = type_prev

    return all_tests

## Divide train / test set based on the conditoin
## all_tests (list) : Containing removal history of a Aircraft (pandas dataframe
## all_chks (list) : Containing dataframe of correlated ATA4's elpased flight hours, for each record in the all_tests
## split_date: divide train / test based on the date (datetime), on Removal date
def train_test_ata4_series(all_tests, all_chks, seq_len, pn_list, split_date, sel_ones='all', embed=True, embedding_dim = 3, use_type =True):
    df_test = pd.DataFrame()
    pn_list = np.array(pn_list).astype(str)

    if embed:
        part_emb = nn.Embedding(num_embeddings= len(pn_list), embedding_dim= embedding_dim)
        month_emb = nn.Embedding(num_embeddings=12, embedding_dim=embedding_dim)
    else:
        part_encoder = OneHotEncoder(categories=[pn_list], sparse_output=False, handle_unknown='ignore')
        part_encoder.fit(np.array(pn_list).reshape(-1,1))

        month_encoder = OneHotEncoder(categories=[np.arange(1, 13)], sparse_output=False)
        month_encoder.fit(np.arange(1, 13).reshape(-1,1))

    ## Divide numeric and others
    X_train_nums, X_train_ones, y_train_list = [], [], []
    X_test_nums, X_test_ones, y_test_list = [], [], []
    ata_dim = len(all_chks[0][0])  # assuming all_chks[i][j] have the same columns
    part_to_idx = {pn:idx for idx, pn in enumerate(pn_list)}

    for test_df, chk_list in zip(all_tests, all_chks):
        n = len(test_df)

        if n <2:
            continue

        y_train_list, y_test_list = list(y_train_list), list(y_test_list)
        # Convert categorical features
        test_df['PART_NO'] = test_df['PART_NO'].astype(str)

        if sel_ones is not None:
            if embed:
                if sel_ones == 'all' or sel_ones == 'part':
                    part_idx = test_df['PART_NO'].map(part_to_idx).to_numpy()
                    x_part = part_emb(torch.tensor(part_idx, dtype=torch.long)).detach().numpy()
                if sel_ones == 'all' or sel_ones == 'month':
                    month_idx = test_df['INSTALL_DATE'].dt.month.to_numpy() -1
                    x_month = month_emb(torch.tensor(month_idx, dtype=torch.long)).detach().numpy()
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
        removal_dates = test_df['REMOVAL_DATE']

        X_numeric = np.concatenate([x_prev, x_other], axis=1).astype(float)
        if use_type:
            if sel_ones == 'all':
                X_ones = np.concatenate([x_part, x_month, x_type], axis=1).astype(float)
            elif sel_ones == 'part':
                X_ones = np.concatenate([x_part, x_type], axis=1).astype(float)
            elif sel_ones == 'month':
                X_ones = np.concatenate([x_month, x_type], axis=1).astype(float)
            elif sel_ones is None:
                X_ones = np.concatenate([x_type], axis=1).astype(float)
        else:
            if sel_ones == 'all':
                X_ones = np.concatenate([x_part, x_month], axis=1).astype(float)
            elif sel_ones == 'part':
                X_ones = np.concatenate([x_part], axis=1).astype(float)
            elif sel_ones == 'month':
                X_ones = np.concatenate([x_month], axis=1).astype(float)
            elif sel_ones is None:
                X_ones = None

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
                    seq_ones = np.concatenate([np.zeros((pad_len, X_ones.shape[1])), seq_ones], axis=0)

            # decide train/test by REMOVAL_DATE[t]
            if removal_dates.iloc[t] <= split_date:
                # X_train_list.append([seq_num, seq_ones])
                X_train_nums.append(seq_num)
                if sel_ones is not None:
                    X_train_ones.append(seq_ones)
                y_train_list.append(y_all[t])
            else:
                # X_test_list.append([seq_num, seq_ones])
                X_test_nums.append(seq_num)
                if sel_ones is not None:
                    X_test_ones.append(seq_ones)
                y_test_list.append(y_all[t])
                df_test = pd.concat([df_test, test_df.iloc[[t]]], ignore_index=True)
    
    return X_train_nums, X_train_ones, y_train_list, X_test_nums, X_test_ones, y_test_list, df_test

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
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden1, hidden2, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden2, embed_dim, bias=True),
            nn.Sigmoid()
        )

        self.multihead_atten = nn.MultiheadAttention(embed_dim, num_heads)

        self.fc = nn.Linear(embed_dim, output_dim)
        

    def forward(self, x):
        x = self.mlp(x)

        atten_output, _ = self.multihead_atten(x, x, x)

        output = self.fc(atten_output)

        return output


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
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, seq_len, feature_dim)
            squeeze_out = True
        else:
            squeeze_out = False

        x = self.input_linear(x) 
        x = self.pos_encoder(x)  
        x = self.encoder(x)      

        x_last = x[:, -1, :]      # (batch, d_model)
        out = self.predictor(x_last)  # (batch, 1)

        if squeeze_out:
            out = out.squeeze(0) 

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
        )
        ## Attention
        self.multi_attention = nn.MultiheadAttention(out_channel, num_heads)
        ## Fill in the size for Linear
        self.fc = nn.Linear(out_channel, output_dim, bias=True)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = torch.cat([x], dim=0)
        out = self.layer1(x.reshape(-1,1))
        ## revise the layers
        out = self.layer2(out)
        out = out.reshape(-1,out.size(0))

        atten_out, _ = self.multi_attention(out, out, out)
        
        out = self.fc(atten_out)
        return out

class MultiheadAttenLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional, num_heads, output_dim, dropout_ratio = 0.1):
        super(MultiheadAttenLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_ratio = dropout_ratio

        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout_ratio if num_layers >1 else 0,
            bidirectional = bidirectional
        )
        ## Attention
        enc_output_dim = hidden_dim *(2 if bidirectional else 1)
        self.multi_attention = nn.MultiheadAttention(enc_output_dim, num_heads)
        ## Fill in the size for Linear
        self.fc = nn.Sequential(
            nn.Linear(enc_output_dim, enc_output_dim),  ## 1st linear: reduced into half
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(enc_output_dim, output_dim)
        )

    def forward(self, x):
        if x.ndim ==2:
            x = x.unsqueeze(0)
        lstm_out, _= self.lstm(x)
        # lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        atten_out, _ = self.multi_attention(lstm_out, lstm_out, lstm_out)
        # atten_out = atten_out.transpose(0, 1)
        out = torch.mean(atten_out, dim=1)
        
        out = self.fc(out)
        return out.squeeze(0)

########################################################################################################################

def train_model(
    model, model_name, X_train, y_train, X_test, y_test, X_valid=None, y_valid=None, revise_valid = False, filter_out=True,
    learning_rate=0.001, epochs=300, epochs_valid=300, out_per=None, device=torch.device('cpu')
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

        for s1, y1 in zip(X_train, y_train):
            s1 = s1.to(device)
            optimizer.zero_grad()
            if model_name in ['MultiHeadAttentionRegression_MLP', 'MLPAtten']:
                y_pred = model(s1.view(1, 1, s1.size(0)))
            else:
                y_pred = model(s1)
            # loss = loss_fn(y_pred.reshape(-1), y_train[idx].to(device))
            loss = loss_fn(y_pred.reshape(-1), y1.to(device))
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
                if model_name in ['MultiHeadAttentionRegression_MLP', 'MLPAtten']:
                    y_vp = model(val_x.view(1, 1, val_x.size(0)))
                else:
                    y_vp = model(val_x)
                loss = loss_fn(y_vp.reshape(-1), val_y.to(device))
                val_loss += loss.item()

        test_loss = 0
        model.eval()
        for test_x, test_y in zip(X_test, y_test):
            test_x = test_x.to(device)
            if model_name in ['MultiHeadAttentionRegression_MLP', 'MLPAtten']:
                y_tp= model(test_x.view(1, 1, test_x.size(0)))
            elif model_name == 'TransformerRegression':
                y_tp = model(test_x)
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

        if filter_out and epoch == int(epochs/2):
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
            for seq, yv in zip(X_valid, y_valid):
                seq.to(device)
                optimizer.zero_grad()
                if model_name in ['MultiHeadAttentionRegression_MLP', 'MLPAtten']:
                    y_pred = model(seq.view(1, 1, seq.size(0)))
                elif model_name == 'TransformerRegression':
                    y_pred = model(s1)
                else:
                    y_pred = model(seq)
                loss = loss_fn(y_pred.reshape(-1), y_valid.to(device))
                loss.backward()
                optimizer.step()
                idx +=1
                epoch_loss += loss.item()
                valid_loss_vec.append(loss.item())

            test_loss = 0
            model.eval()
            for test_x, test_y in zip(X_test, y_test):
                test_x.to(device)
                if model_name in ['MultiHeadAttentionRegression_MLP', 'MLPAtten']:
                    y_tp = model(test_x.view(1, 1, test_x.size(0)))
                elif model_name == 'TransformerRegression':
                    y_tp = model(test_x)
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

            if filter_out and epoch == int(epochs/2):
                out_inds_val = [i for i, val in enumerate(valid_loss_vec) if val > th]
                X_val_out = [X_valid[i] for i in out_inds_val]
                y_val_out = [y_valid[i] for i in out_inds_val]
                X_val_re = [X_valid[i] for i in range(len(X_valid)) if i not in out_inds_val]
                y_val_re = [y_valid[i] for i in range(len(y_valid)) if i not in out_inds_val]
                X_valid, y_valid = X_val_re, y_val_re

    if filter_out:
        if revise_valid:
            return model, losses, val_losses, test_losses_val, re_losses, test_losses, out_inds_tr, X_tr_out, y_tr_out, out_inds_val, X_val_out, y_val_out
        else:
            return model, losses, test_losses_val, out_inds_tr, X_tr_out, y_tr_out
    else:
        if revise_valid:
            return model, losses, val_losses, test_losses_val, re_losses, test_losses
        else:
            return model, losses, test_losses_val

def test_model(model, model_name, X_test, device=torch.device('cpu')):
    y_ps = []
    model.eval()
    with torch.no_grad():
        for s1 in X_test:
            if model_name in ['MultiHeadAttentionRegression_MLP', 'MLPAtten']:
                y_ps.append(model(s1.view(1, 1, s1.size(0))).item())
            else:
                y_ps.append(model(s1).item())
    return y_ps



def findU(lst):
    result = []
    start = 0

    for i, val in enumerate(lst):
        if val == 'U':
            result.append([start, i])
            start = i + 1

    return result

def train_test_transformer(all_tests, all_chks, seq_len, pn_list, split_date, embed=True):

    df_test = pd.DataFrame()
    pn_list = np.array(pn_list).astype(str)

    if embed:
        part_emb = nn.Embedding(num_embeddings= len(pn_list), embedding_dim=3)
        month_emb = nn.Embedding(num_embeddings=12, embedding_dim=3)

    else:
        # 1. Collect all unique part numbers for consistent one-hot encoding
        part_encoder = OneHotEncoder(categories=[pn_list], sparse_output=False, handle_unknown='ignore')
        part_encoder.fit(np.array(pn_list).reshape(-1, 1))

        # 2. month encoder (1~12)
        month_encoder = OneHotEncoder(categories=[np.arange(1, 13)], sparse_output=False)
        month_encoder.fit(np.arange(1, 13).reshape(-1, 1))
    
    X_train_nums, X_train_ones, X_train_list, y_train_list = [], [], [], []
    X_test_nums, X_test_ones, X_test_list, y_test_list = [], [], [], []
    ata_dim = len(all_chks[0][0])  # assuming all_chks[i][j] have the same columns
    part_to_idx = {pn:idx for idx, pn in enumerate(pn_list)}

    for test_df, chk_list in zip(all_tests, all_chks):
        n = len(test_df)

        if n <2:
            continue

        y_train_list, y_test_list = list(y_train_list), list(y_test_list)
        # Convert categorical features
        test_df['PART_NO'] = test_df['PART_NO'].astype(str)

        if embed:    
            part_idx = test_df['PART_NO'].map(part_to_idx).to_numpy()
            x_part = part_emb(torch.tensor(part_idx, dtype=torch.long)).detach().numpy()
            month_idx = test_df['INSTALL_DATE'].dt.month.to_numpy() -1
            x_month = month_emb(torch.tensor(month_idx, dtype=torch.long)).detach().numpy()
        else:
            x_part = part_encoder.transform(test_df['PART_NO'].to_numpy().reshape(-1, 1))
            x_month = month_encoder.transform(test_df['INSTALL_DATE'].dt.month.to_numpy().reshape(-1, 1))
        test_df['isU'] = test_df['prev_type'].map(lambda x:1 if x == 'U' else 0)
        x_type = test_df['isU'].to_numpy().reshape(-1, 1)
        x_prev = test_df['prev_fh'].to_numpy().reshape(-1, 1)

        # Build x_other (same length as test_df)
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
        removal_dates = test_df['REMOVAL_DATE']

        X_numeric = np.concatenate([x_prev, x_other], axis=1).astype(float)
        X_ones = np.concatenate([x_part, x_month, x_type], axis=1).astype(float)

        # Build subsequences with zero padding
        for t in range(1, n):
            start = max(0, t - seq_len)
            seq_num = X_numeric[start:t]
            seq_ones = X_ones[start:t]
            pad_len = seq_len - len(seq_num)
            if pad_len > 0:
                seq_num = np.concatenate([np.zeros((pad_len, X_numeric.shape[1])), seq_num], axis=0)
                seq_ones = np.concatenate([np.zeros((pad_len, X_ones.shape[1])), seq_ones], axis=0)

            # decide train/test by REMOVAL_DATE[t]
            if removal_dates.iloc[t] <= split_date:
                # X_train_list.append([seq_num, seq_ones])
                X_train_nums.append(seq_num)
                X_train_ones.append(seq_ones)
                y_train_list.append(y_all[t])
            else:
                # X_test_list.append([seq_num, seq_ones])
                X_test_nums.append(seq_num)
                X_test_ones.append(seq_ones)
                y_test_list.append(y_all[t])
                df_test = pd.concat([df_test, test_df.iloc[[t]]], ignore_index=True)

    return X_train_nums, X_train_ones, y_train_list, X_test_nums, X_test_ones, y_test_list, df_test