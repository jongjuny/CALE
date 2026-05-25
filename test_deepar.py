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

from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.dataset.common import ListDataset
from gluonts.torch.distributions import StudentTOutput

from neuralprophet import NeuralProphet

if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float

#############################################################################
## Global Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
## Directory set-up
base_dir = "/home/mhi/Data/dataset"
# model_dir_name = 'train_model'
# predict_dir_name = 'test_results_all'
scaler_dir_name = 'scalers_up'
config_dir_name = 'configs_up'
model_dir_name = 'train_model_up'
predict_dir_name = 'test_result_up'

## Group (regional) of operators
europe_ops=['ANE', 'BCY', 'CLH']
asia_ops = ['HXA', 'IBX']
na_grp = ['EDV', 'PSY', 'SKW']
ac_model = 'CRJ700'

# selected_operators = ['SKW', 'PSY', 'EDV', 'ANE', 'BCY', 'CLH','HXA', 'IBX']
# selected_operators = ['SKW', 'PSY', 'EDV']
all_operators = ['EDV', 'CLH','HXA', 'IBX', 'ANE', 'JZA', 'SKW', 'PSY']
# all_operators = ['IBX', 'ANE', 'JZA', 'SKW', 'PSY']

# target_atas_all= [243201, 324301, 344401, 215206]
target_atas_all = [243201, 324301, 344401, 243203, 313301, 324201]
# target_atas_all = [313301, 324201]

view_cols = ['PART_NO', 'PART_SN', 'INSTALL_DATE', 'REMOVAL_DATE', 'REMOVAL_TYPE_CODE', 'AC_SN', 'OPERATOR_CODE', 'FLIGHT_HOURS', 'FLIGHT_CYCLES', 'CUML_HOURS']

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


parser = argparse.ArgumentParser()
parser.add_argument('-group_op', help='  : select group of ops (ex. asia, europe, na, ...)', default=None)
parser.add_argument('-op', help='   : select a single ATA (ex. HXA)', default='HXA')
parser.add_argument('-ata6', help = '  : select ATA6', default='all')
parser.add_argument('-epochs', help= '   : set epochs for training', default=100)
# parser.add_argument('-type', help='     : set training set for removal type code (all, U, S)', default='all')
# parser.add_argument('-scaler', help='     : set scaler between standard, minmax, robust', default='standard')
parser.add_argument('-seq_len', help='      : sequence length for train/test inputs', default=3)
# parser.add_argument('-embed', help='      : set embedding for categorical variables (True, False)', default='True')
# parser.add_argument('-sel_ones', help='     : selection of features (all, part, month)', default='all')
args = parser.parse_args()

prediction_length = 1

def create_listdataset(df, freq):
    dataset = []
    for pid, group in df.groupby('AC_SN'):
        start_date = group['INSTALL_DATE'].min()
        target = group['FLIGHT_HOURS'].values.astype(float)
        dataset.append({
            "start": pd.Timestamp(start_date),
            "target": target,
            "item_id": str(pid),
            "feat_static_cat": [group['PART_NO_cat'].iloc[0]]  # PART_NO categorical
        })
    return ListDataset(dataset, freq=freq)

def param_passer(args):
    if args.group_op is None:
        selected_operators = [args.op]
        print('Single OP: ', selected_operators[0])
    else:
        sel_ops = args.group_op
        print('OPs: ', sel_ops)
        if args.group_op == 'asia':
            selected_operators = asia_ops
        elif args.group_op == 'europe':
            selected_operators = europe_ops
        elif args.group_op == 'na':
            selected_operators = na_ops
        elif args.group_op == 'all':
            selected_operators = all_operators
        else:
           print(f'No group of the name {args.group_op}')
           return None, None, None, None

    if args.ata6 == 'all':
        target_atas = target_atas_all
    else:
        target_atas = [int(args.ata6)]

    epochs = int(args.epochs)
    # sel_type = args.type

    # scaler_type = args.scaler
    seq_len  = int(args.seq_len)
    # embed = True if args.embed.lower() == 'True' else False
    # sel_ones = args.sel_ones

    return selected_operators, target_atas, epochs, seq_len

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


def main(argv, args):
    print('\n')
    print('argv: ', argv)
    print('args: ', args)

    selected_operators, target_atas, epochs, seq_len= param_passer(args)
    if selected_operators is None:
        print('Error: No valid arguments')
        return None

    ## Load base data
    df_rep, df_util_diff = load_data(base_dir, MIN_FH, latest_date, ac_model)

    for ii, sel_op in enumerate(selected_operators):
        print(f"GRP: {sel_op}, ATA: {target_atas}")

        # df_op = df_rep.groupby('OPERATOR_CODE').get_group(sel_op)
        
        for target_ata in target_atas:
            df_rep_rev = get_selected_data(df_rep, df_util_diff, [sel_op], target_ata)
            print('OPERATOR: ', sel_op, 'ATA:', target_ata, ' # of records: ', len(df_rep_rev))

            df_train_data = pd.DataFrame()
            # print('TARGET ATA: ', target_ata)

            if len(df_rep_rev) == 0:
                continue

            df_train_data = df_rep_rev

            chk_ata = int(target_ata)
            # df_train_data = df_train_data[sel_columns]
            df_train_data = df_train_data.sort_values(by='REMOVAL_DATE', ascending=True)

            pn_df = df_train_data.groupby('ATA_NUMBER').get_group(chk_ata)
            pn_list = list(pn_df.groupby('PART_NO').count().index)
            print('PNs:', pn_list)
            ata_list = df_train_data.groupby('ATA_NUMBER').count().index
            all_tests, all_chks = related_at_install_series(df_train_data, df_util_diff, chk_ata, ata_list, considered_date, latest_date)
            all_tests2 = add_before_fh(all_tests)

            
            df_comparison = pd.DataFrame(columns=['AC_SN', 'INSTALL_DATE', 'FLIGHT_HOURS'])
            df_test = pd.DataFrame(columns=['AC_SN', 'INSTALL_DATE', 'FLIGHT_HOURS'])
            for df in all_tests2:
                df_tr = df[df['INSTALL_DATE'] <= test_date].copy()
                df_comparison = pd.concat([df_comparison, df_tr])
                df_te  = df[df['INSTALL_DATE'] > test_date].copy()
                df_test = pd.concat([df_test, df_te])

            if len(df_comparison) ==0: continue
            
            df_ar = df_comparison[['AC_SN', 'PART_NO', 'INSTALL_DATE', 'REMOVAL_DATE', 'FLIGHT_HOURS']].copy()
            df_ar['INSTALL_DATE'] = pd.to_datetime(df_ar['INSTALL_DATE'])
            df_ar = df_ar.sort_values(['AC_SN','INSTALL_DATE'])

            df_test = df_test[['AC_SN', 'PART_NO', 'INSTALL_DATE', 'REMOVAL_DATE', 'FLIGHT_HOURS']].copy()
            df_test['INSTALL_DATE'] = pd.to_datetime(df_test['INSTALL_DATE'])
            df_test = df_test.sort_values(['AC_SN', 'INSTALL_DATE'])

            ac_sn_map   = {v:i for i,v in enumerate(df_ar['AC_SN'].unique())}
            part_no_map = {v:i for i,v in enumerate(df_ar['PART_NO'].unique())}

            df_ar['AC_SN_cat']   = df_ar['AC_SN'].map(ac_sn_map)
            df_ar['PART_NO_cat'] = df_ar['PART_NO'].map(part_no_map)

            df_test['AC_SN_cat']   = df_test['AC_SN'].map(ac_sn_map)
            df_test['PART_NO_cat'] = df_test['PART_NO'].map(part_no_map)

            ### Deep AR
            print('DeepAR Start')
            train_df = df_ar
            test_df = df_test

            freq = "D"
            context_length = seq_len

            train_ds = create_listdataset(train_df, freq)

            estimator = DeepAREstimator(
                prediction_length=prediction_length,
                context_length=context_length,
                freq=freq,
                hidden_size=50,
                dropout_rate=0.1,
                distr_output=StudentTOutput(),
                cardinality=[df_ar['PART_NO'].nunique()],
                trainer_kwargs={
                    "max_epochs": epochs,
                    "logger": False,
                },
            )

            predictor = estimator.train(train_ds)

            predictions = []

            for pid, group in test_df.groupby('AC_SN'):
                # 초기 context: train_df 마지막 context_length
                subset = train_df[train_df['AC_SN'] == pid]['PART_NO_cat']
                if not subset.empty:
                    part_no_cat = subset.iloc[-1]
                else:
                    # 없는 경우 기본값 지정 (예: 0)
                    part_no_cat = 0
                
                context = train_df[train_df['AC_SN'] == pid]['FLIGHT_HOURS'].values[-context_length:].tolist()
                # part_no_cat = train_df[train_df['AC_SN'] == pid]['PART_NO_cat'].iloc[-1]  # 마지막 값 사용

                for fh_true in group['FLIGHT_HOURS'].values:
                    # context를 dict로 wrapping해서 ListDataset 생성
                    input_ds = ListDataset([{
                        "start": pd.Timestamp('2020-01-01'),  # 임의 start, 실제 날짜는 의미 없음
                        "target": np.array(context, dtype=float),
                        "item_id": str(pid),
                        "feat_static_cat": [part_no_cat]  
                    }], freq='D')

                    # predict 호출
                    forecast_it = predictor.predict(input_ds)
                    forecast = next(forecast_it)

                    # 평균 예측
                    fh_pred = forecast.mean[ -1]  # 마지막 step 예측
                    predictions.append(fh_pred)

                    # context 업데이트: 실제값 사용
                    context = context[1:] + [fh_true]
            
        

            df_test['FH_AR2'] = predictions

            df_test.to_csv(f'./data_ata4/deepar2_icde_test_ata_{sel_op}_{target_ata}_len_{seq_len}.csv')

    

if __name__ == '__main__':
    argv = sys.argv
    main(argv, args)
