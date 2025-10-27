import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

from models.preprocessing import convert_to_int, related_at_install_series
from models.model_exp import *

from tqdm.notebook import tqdm
from tqdm.contrib import tzip
import copy
import pickle
import warnings
warnings.filterwarnings('ignore')

from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.dataset.common import ListDataset
from gluonts.torch.distributions import StudentTOutput

if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float

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

considered_date = datetime.datetime(2015,1,1)
latest_date = datetime.datetime(2025,3,1)
test_date = datetime.datetime(2023,3,1)

parser = argparse.ArgumentParser()
parser.add_argument('-group_op', help='  : select group of ops (ex. asia, europe, na, ...)', default=None)
parser.add_argument('-op', help='   : select a single ATA (ex. HXA)', default='HXA')
parser.add_argument('-ata6', help = '  : select ATA6', default='all')
parser.add_argument('-epochs', help= '   : set epochs for training', default=100)
parser.add_argument('-seq_len', help='      : sequence length for train/test inputs', default=5)
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
            "item_id": str(pid)
        })
    return ListDataset(dataset, freq=freq)

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
    seq_len  = int(args.seq_len)

    return selected_operators, target_atas, epochs, seq_len


def main(argv, args):
    print('\n')
    print('argv: ', argv)
    print('args: ', args)

    selected_operators, target_atas, epochs, seq_len= param_passer(args)
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

            ## Extract ATA4 features within the aircraft, after combining utilization data
            ## Also, filter out and augment missing data while extracting
            all_tests, all_chks = related_at_install_series(df_train_data, df_util_diff, chk_ata, ata_list, considered_date, latest_date)
            all_tests2 = add_before_fh(all_tests)

            
            df_comparison = pd.DataFrame(columns=['AC_SN', 'INSTALL_DATE', 'FLIGHT_HOURS'])
            df_test = pd.DataFrame(columns=['AC_SN', 'INSTALL_DATE', 'FLIGHT_HOURS'])
            for df in all_tests2:
                df_tr = df[df['INSTALL_DATE'] <= test_date].copy()
                df_comparison = pd.concat([df_comparison, df_tr])
                df_te  = df[df['INSTALL_DATE'] > test_date].copy()
                df_test = pd.concat([df_test, df_te])

            df_ar = df_comparison[['AC_SN', 'INSTALL_DATE', 'REMOVAL_DATE', 'FLIGHT_HOURS']].copy()
            df_ar['INSTALL_DATE'] = pd.to_datetime(df_ar['INSTALL_DATE'])
            df_ar = df_ar.sort_values(['AC_SN','INSTALL_DATE'])

            df_test = df_test[['AC_SN', 'INSTALL_DATE', 'REMOVAL_DATE', 'FLIGHT_HOURS']].copy()
            df_test['INSTALL_DATE'] = pd.to_datetime(df_test['INSTALL_DATE'])
            df_test = df_test.sort_values(['AC_SN', 'INSTALL_DATE'])

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
                trainer_kwargs={
                    "max_epochs": epochs,
                },
            )

            predictor = estimator.train(train_ds)

            predictions = []

            for pid, group in test_df.groupby('AC_SN'):
                context = train_df[train_df['AC_SN'] == pid]['FLIGHT_HOURS'].values[-context_length:].tolist()

                for fh_true in group['FLIGHT_HOURS'].values:
                    input_ds = ListDataset([{
                        "start": pd.Timestamp('2015-01-01'),  # init.
                        "target": np.array(context, dtype=float),
                        "item_id": str(pid)
                    }], freq='D')

                    # predict
                    forecast_it = predictor.predict(input_ds)
                    forecast = next(forecast_it)

                    # Estimate the average
                    fh_pred = forecast.mean[ -1]  # Take the last step
                    predictions.append(fh_pred)

                    ## For incremental prediction
                    context = context[1:] + [fh_true]
            
        

            df_test['FH_AR'] = predictions

            df_test.to_csv(f'./data_ata4/deepar_icde_test_ata_{sel_op}_{target_ata}_len_{seq_len}.csv')

    

if __name__ == '__main__':
    argv = sys.argv
    main(argv, args)
