import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import copy
from tqdm.contrib import tzip
# from tqdm.notebook import tqdm
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# base_dir = "/home/mhi/Data/dataset"
base_dir = "./dataset"

###########################################################################################################
## Data Read
def convert_to_int(pn):
    if isinstance(pn, str) and pn.isdigit():
        return int(pn)
    return pn

def build_chains_by_qpa(df_input, qpa=2):
    MAX_DAY = datetime.timedelta(1000)
    # Step 1: FLIGHT_HOURS > 0 filter out
    df_filtered = df_input[df_input['FLIGHT_HOURS'] > 0].copy()
    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered['RowID'] = df_filtered.index  

    # Step 2: matching REMOVAL with INSTALL date 
    # install_lists = df_filtered[['RowID', 'INSTALL_DATE', 'REMOVAL_DATE']]
    chains = [pd.DataFrame(columns=df_filtered.columns) for _ in range(qpa)]

    for _, r in df_filtered.iterrows():
        min_day = MAX_DAY
        # print(f'{r.RowID} Install: {r.INSTALL_DATE}')
        for i, chain in enumerate(chains):
            if len(chain) == 0:# and min_day == MAX_DAY:
                # chains[i] = pd.concat(chains[i], pd.DataFrame(r).T)
                min_day, min_chain = -1, i
                break
            if chain['REMOVAL_DATE'].iloc[-1] > r['INSTALL_DATE']: 
                # print(f'NA: {chain.REMOVAL_DATE.iloc[-1]} for Chain {i}')
                continue
            else:
                d_day = r['INSTALL_DATE'] - chain['REMOVAL_DATE'].iloc[-1]
                if min_day > d_day:
                    min_day, min_chain = d_day, i
                # print(f'Day: {d_day} / {min_day}, for {i}')
    
        chains[min_chain] = pd.concat([chains[min_chain], pd.DataFrame(r).T])
    return chains

def revise_util_data(f):
    df_util = pd.read_excel(f, index_col=None, parse_dates=['MONTH'], date_format='mixed')  
    ac_list = list(df_util.groupby('Aircraft').count().index)

    df_util2 = pd.DataFrame(columns=['AU_DATE', 'Aircraft', 'MONTH', 'diff_hours', 'diff_cycle',
        'hour_per_cycle', 'CUMULATIVE_FLIGHT_HOURS', 'CUMULATIVE_CYCLES','days'])
    for ac in tqdm(ac_list):
        df_tmp = df_util.groupby('Aircraft').get_group(int(ac))
        # df_tmp = df_tmp.rename(columns={"Aircraft": "ac_sn", "MONTH": "AU_DATE"})

        all_days = pd.date_range(start=df_tmp['MONTH'].min(), end=df_tmp['MONTH'].max() + pd.DateOffset(months=1) - pd.DateOffset(days=1), freq='D')
        daily_df = pd.DataFrame({'AU_DATE': all_days})

        df_tmp['next_month'] = df_tmp['MONTH'].shift(-1, fill_value=df_tmp['MONTH'].iloc[-1] + pd.DateOffset(months=1))
        df_tmp['days_in_month'] = (df_tmp['next_month'] - df_tmp['MONTH']).dt.days
        df_tmp['fh_in_month'] = df_tmp['FH_Total'].shift(-1, fill_value=df_tmp['FH_Total'].iloc[-1])
        df_tmp['cycle_in_month'] = df_tmp['FC_Total'].shift(-1, fill_value=df_tmp['FC_Total'].iloc[-1])

        df_tmp['diff_hours'] = df_tmp['fh_in_month'] / df_tmp['days_in_month']
        df_tmp['diff_cycle'] = df_tmp['cycle_in_month'] / df_tmp['days_in_month']
        df_tmp['hour_per_cycle'] = df_tmp['diff_hours'] / df_tmp['diff_cycle']

        daily_df = daily_df.merge(df_tmp[['Aircraft', 'MONTH', 'diff_hours', 'diff_cycle', 'hour_per_cycle']], left_on=daily_df['AU_DATE'].dt.to_period('M'), right_on=df_tmp['MONTH'].dt.to_period('M'), how='left').drop(columns=['key_0'])
        daily_df['CUMULATIVE_FLIGHT_HOURS'] = daily_df['diff_hours'].cumsum()
        daily_df['CUMULATIVE_CYCLES'] = daily_df['diff_cycle'].cumsum()
        daily_df['days'] = daily_df['AU_DATE'].dt.dayofweek
        df_util2 = pd.concat([df_util2, daily_df])
        # print(ac)

    return df_util2
    
def revise_util_data_new(f):

    df_util = pd.read_csv(f, index_col=None, parse_dates=['AU_DATE'], date_format='mixed')  
    ac_list = list(df_util.groupby('AC_SN').count().index)


    df_util2 = pd.DataFrame(columns=['AU_DATE', 'AC_SN', 'FLIGHT_HOURS_MONTH', 'diff_hours', 'diff_cycle',
        'hour_per_cycle', 'CUMULATIVE_FLIGHT_HOURS', 'CUMULATIVE_CYCLES','days'])
    for ac in tqdm(ac_list):
        df_tmp = df_util.groupby('AC_SN').get_group(int(ac))

        all_days = pd.date_range(start=df_tmp['AU_DATE'].min(), end=df_tmp['AU_DATE'].max() + pd.DateOffset(months=1) - pd.DateOffset(days=1), freq='D')
        daily_df = pd.DataFrame({'AU_DATE': all_days})

        df_tmp['next_month'] = df_tmp['AU_DATE'].shift(-1, fill_value=df_tmp['AU_DATE'].iloc[-1] + pd.DateOffset(months=1))
        df_tmp['days_in_month'] = (df_tmp['next_month'] - df_tmp['AU_DATE']).dt.days
        df_tmp['fh_in_month'] = df_tmp['FLIGHT_HOURS_MONTH'].shift(-1, fill_value=df_tmp['FLIGHT_HOURS_MONTH'].iloc[-1])
        df_tmp['cycle_in_month'] = df_tmp['CYCLES_MONTH'].shift(-1, fill_value=df_tmp['CYCLES_MONTH'].iloc[-1])

        df_tmp['diff_hours'] = df_tmp['fh_in_month'] / df_tmp['days_in_month']
        df_tmp['diff_cycle'] = df_tmp['cycle_in_month'] / df_tmp['days_in_month']
        df_tmp['hour_per_cycle'] = df_tmp['diff_hours'] / df_tmp['diff_cycle']

        daily_df = daily_df.merge(df_tmp[['AC_SN', 'FLIGHT_HOURS_MONTH', 'diff_hours', 'diff_cycle', 'hour_per_cycle']], left_on=daily_df['AU_DATE'].dt.to_period('M'), right_on=df_tmp['AU_DATE'].dt.to_period('M'), how='left').drop(columns=['key_0'])
        daily_df['CUMULATIVE_FLIGHT_HOURS'] = daily_df['diff_hours'].cumsum()
        daily_df['CUMULATIVE_CYCLES'] = daily_df['diff_cycle'].cumsum()
        daily_df['days'] = daily_df['AU_DATE'].dt.dayofweek
        df_util2 = pd.concat([df_util2, daily_df])
        # print(ac)


    df_util2 = df_util2.rename(columns={'AC_SN':'ac_sn', 'FLIGHT_HOURS_MONTH':'MONTH'})
    df_util2['AU_DATE'] = pd.to_datetime(df_util2['AU_DATE'])
    df_util2['MONTH'] = df_util2['AU_DATE'].dt.to_period('M').dt.to_timestamp()
    df_util2['MONTH'] = df_util2['MONTH'].dt.date
    # df_util2 = df_util2.rename(columns={'AC_SN':'ac_sn'})
    return df_util2

##########################################################################################\
## For the last updates
def related_at_install_series(df_rep, df_util, set_ata, ata_list, chk_date, testing_date):
    all_tests = []
    all_chks = []
    
    acs = df_rep['AC_SN'].unique()
    grp_rep = df_rep.groupby('AC_SN', sort=False)
    grp_util = df_util.groupby('ac_sn', sort=False)

    for ac in tqdm(acs):
        if ac not in grp_rep.groups: continue

        df_ac = grp_rep.get_group(ac)
        mask = (
            (df_ac['INSTALL_DATE'] >= chk_date) &
            (df_ac['REMOVAL_DATE'] < testing_date) &
            (df_ac['FLIGHT_HOURS'] > 0) &
            (df_ac['ATA_NUMBER'] == set_ata)
        )
        df_target = df_ac.loc[mask]
        if df_target.empty: continue

        if ac not in grp_util.groups: continue
        df_util_ac = grp_util.get_group(ac)

        ## Re-align and get series of removal info. by QPA
        if df_target['QPA'].iloc[0] >1:
            chain_list = build_chains_by_qpa(df_target, qpa=df_target['QPA'].iloc[0])
        else:
            chain_list = [df_target]

        for df_test in chain_list:
            df_chks = []
            df_test = df_test.sort_values(by='INSTALL_DATE')
            
            if len(df_test) < 2: continue

            df_test['prev_fh'] = df_test['FLIGHT_HOURS'].shift(1).fillna(0)
            df_test['INSTALL_DATE'] = pd.to_datetime(df_test['INSTALL_DATE'], errors='coerce')
            df_m = pd.merge_asof(df_test[['INSTALL_DATE']], df_util_ac, left_on='INSTALL_DATE', right_on='AU_DATE', direction='backward')

            for i in range(len(df_test)):
                inst_d = df_test['INSTALL_DATE'].iloc[i]
                inst_cum_fh = df_m['CUMULATIVE_FLIGHT_HOURS'].iloc[i]
                df_chk2 = df_ac[(df_ac['INSTALL_DATE'] < inst_d) & (df_ac['REMOVAL_DATE'] > inst_d)]
                j = len(df_chk2)

                df_chk2 = pd.merge(left=df_chk2, right=df_util_ac, how='inner', left_on=['AC_SN', 'INSTALL_DATE'], right_on=['ac_sn', 'AU_DATE'])
                df_chk2['FH_CUM'] = inst_cum_fh - df_chk2['CUMULATIVE_FLIGHT_HOURS_y']

                if (df_chk2['FH_CUM'] < 0).any():
                    print(df_util_ac[df_util_ac['AU_DATE']== df_test['INSTALL_DATE'].iloc[i]]['CUMULATIVE_FLIGHT_HOURS'])
                    return df_chk2, df_test
                
                df_chk2['CUMULATIVE_FLIGHT_HOURS'] = df_chk2['CUMULATIVE_FLIGHT_HOURS_x']

                ## If there is no removal records for ATA6s, take the last removal_date as their installation
                idx_last = df_ac.groupby('ATA_NUMBER')['REMOVAL_DATE'].idxmax()
                df_last = df_ac.loc[idx_last]

                df_last = df_last[df_last['REMOVAL_DATE'] <= inst_d]
                for _, row in df_last.iterrows():
                    df_chk2.loc[j] = row
                    df_chk2.loc[j, 'FH_CUM'] = inst_cum_fh - row['CUMULATIVE_FLIGHT_HOURS']

                    if df_chk2.loc[j, 'FH_CUM'] < 0:
                        print('CHK2', inst_cum_fh, acs)
                        print(
                            df_util_ac[
                                df_util_ac['AU_DATE'] == df_test['INSTALL_DATE'].iloc[i]
                            ]['CUMULATIVE_FLIGHT_HOURS']
                        )
                        return df_chk2, df_test

                    j += 1

                ## For the same ATA6, select older one or younger one? 
                # df_chk2 = df_chk2[['AC_SN', 'PART_NO', 'ATA_NUMBER', 'INSTALL_DATE', 'FH_CUM']]
                df_chk2 = df_chk2[['AC_SN','ATA_NUMBER', 'INSTALL_DATE', 'FH_CUM']]

                df_chk2 = (
                    df_chk2.dropna(subset=['FH_CUM'])
                    .loc[lambda x: x.groupby('ATA_NUMBER')['FH_CUM'].idxmax()]
                    [['AC_SN','ATA_NUMBER','INSTALL_DATE','FH_CUM']]
                    .drop_duplicates()
                    .sort_values('ATA_NUMBER')
                    .reset_index(drop=True)
                )                

                if df_chk2.empty:
                    df_chks.append(df_chk2)
                    continue

                idx_add = len(df_chk2)

                for a_t in ata_list:
                    if a_t in list(df_chk2['ATA_NUMBER']):
                        if df_chk2[df_chk2['ATA_NUMBER'] == a_t]['FH_CUM'].isna().iloc[0] == True:
                            df_chk2['FH_CUM'] = df_chk2['FH_CUM'].fillna(0)
                    else:
                        if len(df_chks) ==0:
                            df_chk2.loc[idx_add]  = [df_chk2.iloc[0]['AC_SN']] + [a_t] + [df_chk2.iloc[0]['INSTALL_DATE']] +[0]
                        else:
                            ## If there is no record for this ATA6 (in ATA4-grp), set FH_CUM as 0 (with install date from the 1st record)
                            init_idx = next((ik for ik, df in enumerate(df_chks) if not df.empty), None)
                            # print(init_idx)
                            if init_idx is not None:
                                df_chk2.loc[idx_add]  = [df_chks[init_idx].iloc[0]['AC_SN']] + [a_t] + [df_chks[init_idx].iloc[0]['INSTALL_DATE']] +[0]
                        idx_add +=1

                df_chks.append(df_chk2)

            all_tests.append(df_test)
            all_chks.append(df_chks)

    return all_tests, all_chks

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

# @ chk_date: only save removal records when its install_date is greater than chk_date
def related_at_install(df_rep, df_util, set_ata, ata_list, chk_date, testing_date):
    all_tests = []
    all_chks = []
    acs = df_rep.groupby('AC_SN').count().index
    for ac in tqdm(acs):
        df_ac = df_rep.groupby('AC_SN').get_group(ac)
        df_ac = df_ac.sort_values(by='INSTALL_DATE')
        df_test = df_ac[df_ac['INSTALL_DATE'] >= chk_date]
        df_test = df_test[df_test['REMOVAL_DATE'] < testing_date]
        df_test = df_test[df_test['FLIGHT_HOURS'] >0]
        if len(df_test) ==0:
            continue
        try:
            df_test = df_test.groupby('ATA_NUMBER').get_group(set_ata)
            # print(ac, len(df_test))
        except:
            # print(ac, 'NO', set_ata)
            continue
        try:
            df_util_ac = df_util.groupby('ac_sn').get_group(ac)
        except:
            continue
        df_chks = []
        for i in range(len(df_test)):
            inst_d = df_test['INSTALL_DATE'].iloc[i]
            if len(df_util_ac[df_util_ac['AU_DATE']== df_test['INSTALL_DATE'].iloc[i]]['CUMULATIVE_FLIGHT_HOURS']) >0:
                inst_cum_fh = float(df_util_ac[df_util_ac['AU_DATE']== df_test['INSTALL_DATE'].iloc[i]]['CUMULATIVE_FLIGHT_HOURS'])
            else:
                inst_cum_fh = df_util_ac['CUMULATIVE_FLIGHT_HOURS'].iloc[0] - (df_util_ac['AU_DATE'].iloc[0]- df_test['INSTALL_DATE'].iloc[i]).days*df_util_ac['diff_hours'].iloc[0]

            df_chk = df_ac[df_ac['INSTALL_DATE']< inst_d]
            df_chk2 = df_chk[df_chk['REMOVAL_DATE']>  inst_d]
            j = len(df_chk2)
            
            df_chk2 = pd.merge(left=df_chk2, right=df_util_ac, how='inner', left_on=['AC_SN', 'INSTALL_DATE'], right_on=['ac_sn', 'AU_DATE'])
            df_chk2['FH_CUM'] = inst_cum_fh - df_chk2['CUMULATIVE_FLIGHT_HOURS_y']
            df_chk2['CUMULATIVE_FLIGHT_HOURS'] = df_chk2['CUMULATIVE_FLIGHT_HOURS_x']
            # df_chk2['FH_CUM'] = (inst_d-df_chk2['INSTALL_DATE'])

            ## If there is no removal records for ATA6s, take the last removal_date as their installation
            atas_ac = df_ac.groupby('ATA_NUMBER').count().index
            for ata_ac in atas_ac:
                df_t = df_ac.groupby('ATA_NUMBER').get_group(ata_ac)
                if np.max(df_t['REMOVAL_DATE']) <= inst_d:
                    df_chk2.loc[j] = df_t.iloc[np.argmax(df_t['REMOVAL_DATE'])]
                    df_chk2['FH_CUM'].loc[j] = inst_cum_fh - df_chk2['CUMULATIVE_FLIGHT_HOURS'].loc[j]
                    j +=1

            ## For the same ATA6, select older one or younger one? 
            # df_chk2 = df_chk2[['AC_SN', 'PART_NO', 'ATA_NUMBER', 'INSTALL_DATE', 'FH_CUM']]
            df_chk2 = df_chk2[['AC_SN','ATA_NUMBER', 'INSTALL_DATE', 'FH_CUM']]
            ata_t = df_chk2.groupby('ATA_NUMBER').count().index

            for a_t in ata_t:
                chk_ata = df_chk2.groupby('ATA_NUMBER').get_group(a_t)
                df_chk2 = df_chk2.drop(chk_ata[chk_ata['FH_CUM'] < np.max(chk_ata['FH_CUM'])].index)
                chk_ata = df_chk2.groupby('ATA_NUMBER').get_group(a_t)

            df_chk2.drop_duplicates(inplace=True)
            df_chk2 =df_chk2.sort_values(by=['ATA_NUMBER'])

            if len(df_chk2) ==0:
                df_chks.append(df_chk2)
                continue
            df_chk2.reset_index(drop=True, inplace=True)

            idx_add = len(df_chk2)
            
            for a_t in ata_list:
                if a_t in list(df_chk2['ATA_NUMBER']):
                    if df_chk2[df_chk2['ATA_NUMBER'] == a_t]['FH_CUM'].isna().iloc[0] == True:
                        # df_chk2[df_chk2['ATA_NUMBER'] == a_t]['FH_CUM'] = 0
                        df_chk2['FH_CUM'] = df_chk2['FH_CUM'].fillna(0)
                    # continue
                else:
                    # df_chk2.loc[idx_add] = [df_chk2.iloc[0]['AC_SN']] + ['NON'] + [a_t] + [df_chk2.iloc[0]['INSTALL_DATE']] +[0]
                    # print('FALSE:', a_t)
                    df_chk2.loc[idx_add] = [df_chk2.iloc[0]['AC_SN']] + [a_t] + [df_chk2.iloc[0]['INSTALL_DATE']] +[0]
                    idx_add +=1

            # if len(df_chk2) != 3:
                # display(df_chk2)
            df_chks.append(df_chk2)

        all_tests.append(df_test)
        all_chks.append(df_chks)
    return all_tests, all_chks
    # break

# @ chk_date: only save removal records when its removal_date is greater than chk_date
def related_at_install_test(df_rep, df_util, set_ata, ata_list, chk_date, testing_date):
    all_tests = []
    all_chks = []
    acs = df_rep.groupby('AC_SN').count().index
    for ac in tqdm(acs):
        df_ac = df_rep.groupby('AC_SN').get_group(ac)
        df_ac = df_ac.sort_values(by='INSTALL_DATE')
        df_test = df_ac[df_ac['REMOIVAL_DATE'] >= chk_date]
        df_test = df_test[df_test['REMOVAL_DATE'] < testing_date]
        df_test = df_test[df_test['FLIGHT_HOURS'] >0]
        if len(df_test) ==0:
            continue
        try:
            df_test = df_test.groupby('ATA_NUMBER').get_group(set_ata)
            # print(ac, len(df_test))
        except:
            # print(ac, 'NO', set_ata)
            continue
        try:
            df_util_ac = df_util.groupby('ac_sn').get_group(ac)
        except:
            continue
        df_chks = []
        for i in range(len(df_test)):
            inst_d = df_test['INSTALL_DATE'].iloc[i]
            if len(df_util_ac[df_util_ac['AU_DATE']== df_test['INSTALL_DATE'].iloc[i]]['CUMULATIVE_FLIGHT_HOURS']) >0:
                inst_cum_fh = float(df_util_ac[df_util_ac['AU_DATE']== df_test['INSTALL_DATE'].iloc[i]]['CUMULATIVE_FLIGHT_HOURS'])
            else:
                inst_cum_fh = df_util_ac['CUMULATIVE_FLIGHT_HOURS'].iloc[0] - (df_util_ac['AU_DATE'].iloc[0]- df_test['INSTALL_DATE'].iloc[i]).days*df_util_ac['diff_hours'].iloc[0]

            df_chk = df_ac[df_ac['INSTALL_DATE']< inst_d]
            df_chk2 = df_chk[df_chk['REMOVAL_DATE']>  inst_d]
            j = len(df_chk2)
            
            df_chk2 = pd.merge(left=df_chk2, right=df_util_ac, how='inner', left_on=['AC_SN', 'INSTALL_DATE'], right_on=['ac_sn', 'AU_DATE'])
            df_chk2['FH_CUM'] = inst_cum_fh - df_chk2['CUMULATIVE_FLIGHT_HOURS_y']
            df_chk2['CUMULATIVE_FLIGHT_HOURS'] = df_chk2['CUMULATIVE_FLIGHT_HOURS_x']
            # df_chk2['FH_CUM'] = (inst_d-df_chk2['INSTALL_DATE'])

            ## If there is no removal records for ATA6s, take the last removal_date as their installation
            atas_ac = df_ac.groupby('ATA_NUMBER').count().index
            for ata_ac in atas_ac:
                df_t = df_ac.groupby('ATA_NUMBER').get_group(ata_ac)
                if np.max(df_t['REMOVAL_DATE']) <= inst_d:
                    df_chk2.loc[j] = df_t.iloc[np.argmax(df_t['REMOVAL_DATE'])]
                    df_chk2['FH_CUM'].loc[j] = inst_cum_fh - df_chk2['CUMULATIVE_FLIGHT_HOURS'].loc[j]
                    j +=1

            ## For the same ATA6, select older one or younger one? 
            # df_chk2 = df_chk2[['AC_SN', 'PART_NO', 'ATA_NUMBER', 'INSTALL_DATE', 'FH_CUM']]
            df_chk2 = df_chk2[['AC_SN','ATA_NUMBER', 'INSTALL_DATE', 'FH_CUM']]
            ata_t = df_chk2.groupby('ATA_NUMBER').count().index

            for a_t in ata_t:
                chk_ata = df_chk2.groupby('ATA_NUMBER').get_group(a_t)
                df_chk2 = df_chk2.drop(chk_ata[chk_ata['FH_CUM'] < np.max(chk_ata['FH_CUM'])].index)
                chk_ata = df_chk2.groupby('ATA_NUMBER').get_group(a_t)

            df_chk2.drop_duplicates(inplace=True)
            df_chk2 =df_chk2.sort_values(by=['ATA_NUMBER'])

            if len(df_chk2) ==0:
                df_chks.append(df_chk2)
                continue
            df_chk2.reset_index(drop=True, inplace=True)

            idx_add = len(df_chk2)
            
            for a_t in ata_list:
                if a_t in list(df_chk2['ATA_NUMBER']):
                    if df_chk2[df_chk2['ATA_NUMBER'] == a_t]['FH_CUM'].isna().iloc[0] == True:
                        # df_chk2[df_chk2['ATA_NUMBER'] == a_t]['FH_CUM'] = 0
                        df_chk2['FH_CUM'] = df_chk2['FH_CUM'].fillna(0)
                    # continue
                else:
                    # df_chk2.loc[idx_add] = [df_chk2.iloc[0]['AC_SN']] + ['NON'] + [a_t] + [df_chk2.iloc[0]['INSTALL_DATE']] +[0]
                    # print('FALSE:', a_t)
                    df_chk2.loc[idx_add] = [df_chk2.iloc[0]['AC_SN']] + [a_t] + [df_chk2.iloc[0]['INSTALL_DATE']] +[0]
                    idx_add +=1

            # if len(df_chk2) != 3:
                # display(df_chk2)
            df_chks.append(df_chk2)

        all_tests.append(df_test)
        all_chks.append(df_chks)
    return all_tests, all_chks
    # break

def get_valid_ata_lists_op(df_rep, op, chk_date, validate_date, testing_date, end_date):
    i =0

    ## For distinct ATA4
    df_ata_chks = df_rep[['ATA4', 'ATA_NUMBER']]
    df_ata_chks.drop_duplicates(inplace=True)
    df_chks_cnt = df_ata_chks.groupby('ATA4').count()
    df_chks_cnt2 = df_chks_cnt[df_chks_cnt.index >= 1000]
    df_chks_cnt2 = df_chks_cnt2[df_chks_cnt2.index <10000]

    df_chks_cnt3 = df_chks_cnt2[df_chks_cnt2['ATA_NUMBER'] >=2]
    ata4_list = list(df_chks_cnt3.index)

    df_stats = pd.DataFrame(columns=['op', 'ATA4', 'ATA6', 'train', 'validate', 'test', 'all'])

    ind = 0
    for ata4 in ata4_list:
        try:
            df_rep_ata4 = df_rep.groupby('ATA4').get_group(ata4)
            ata6_list= list(df_rep_ata4.groupby('ATA_NUMBER').count().index)
            # print(ata6_list)
            for ata6 in ata6_list:
                df_rep_ata6 = df_rep_ata4.groupby('ATA_NUMBER').get_group(ata6)
                df_rep_ata6_rev = df_rep_ata6[df_rep_ata6['INSTALL_DATE'] >= chk_date]
                df_2023 = df_rep_ata6_rev[df_rep_ata6_rev['REMOVAL_DATE']>=validate_date]
                df_2023 = df_2023[df_2023['REMOVAL_DATE'] < testing_date]
                df_2024 = df_rep_ata6_rev[df_rep_ata6_rev['REMOVAL_DATE']>=testing_date]
                df_stats.loc[ind] = [op] + [ata4] + [ata6] + [len(df_rep_ata6_rev[df_rep_ata6_rev['REMOVAL_DATE'] < validate_date])] + [len(df_2023)] + [len(df_2024)] + [len(df_rep_ata6)]
                ind +=1
        except:
            continue

    df_stats['range'] = df_stats['train'] + df_stats['validate'] + df_stats['test']

    return df_stats
    
def get_ata4_lists(df_rep, selected_operators):
    ata_lists = []
    ata4_lists = []
    i = 0
    for op in selected_operators:
        df_rep_op = df_rep.groupby('OPERATOR_CODE').get_group(op)
        df_rep_op['ATA4'] = df_rep_op['ATA_NUMBER']/100
        df_rep_op['ATA4'] = df_rep_op['ATA4'].astype(int)
    
        ## Number of ATA6
        # ata_cnt = df_rep_op.groupby('ATA_NUMBER').count()
        # ata_cnt2 = ata_cnt[ata_cnt.index > 100000]
        # ata_cnt2 = ata_cnt2[ata_cnt2.index < 1000000]
        # ata_lists.append(list(ata_cnt2.index))
    
        ## For distinct ATA4
        df_ata_chks = df_rep_op[['ATA4', 'ATA_NUMBER']]
        df_ata_chks.drop_duplicates(inplace=True)
        df_chks_cnt = df_ata_chks.groupby('ATA4').count()
        df_chks_cnt2 = df_chks_cnt[df_chks_cnt.index >=1000]
        df_chks_cnt2 = df_chks_cnt2[df_chks_cnt2.index <10000]
    
        ## Checking ATA4 containing >=2 ATA6
        df_chks_cnt3 = df_chks_cnt2[df_chks_cnt2['ATA_NUMBER'] >=2]
        ata4_lists.append(list(df_chks_cnt3.index))
        i +=1

    ata4_together = []
    for ll in ata4_lists[0]:
        for j in range(1, len(selected_operators)):
            if ll in ata4_lists[j]: continue
            else: break
        # print(i)
        if j == len(selected_operators)-1: ata4_together.append(ll)
    return ata4_together, ata4_lists

def get_ata4_stats(df_rep, ata4_lists, selected_operators, chk_date, validate_date, testing_date, end_date):

    df_stats = pd.DataFrame(columns=['op', 'ATA4', 'ATA6', 'train', 'validate', 'test', 'all'])

    ind = 0
    for op in selected_operators:
        df_rep_op = df_rep.groupby('OPERATOR_CODE').get_group(op)
        df_rep_op['ATA4'] = df_rep_op['ATA_NUMBER']/100
        df_rep_op['ATA4'] = df_rep_op['ATA4'].astype(int)

        for ata4 in ata4_lists:
            try:
                df_rep_ata4 = df_rep_op.groupby('ATA4').get_group(int(ata4))
                ata6_lists = list(df_rep_ata4.groupby('ATA_NUMBER').count().index)
                for ata6 in ata6_lists:
                    df_rep_ata6 = df_rep_ata4.groupby('ATA_NUMBER').get_group(ata6)
                    df_rep_ata6_rev = df_rep_ata6[df_rep_ata6['INSTALL_DATE'] >= chk_date]
                    df_2023 = df_rep_ata6_rev[df_rep_ata6_rev['REMOVAL_DATE']>=validate_date]
                    df_2023 = df_2023[df_2023['REMOVAL_DATE'] < testing_date]
                    df_2024 = df_rep_ata6_rev[df_rep_ata6_rev['REMOVAL_DATE']>=testing_date]


                    df_stats.loc[ind] = [op] + [ata4] + [ata6] + [len(df_rep_ata6_rev[df_rep_ata6_rev['REMOVAL_DATE'] < validate_date])] + [len(df_2023)] + [len(df_2024)] + [len(df_rep_ata6)]
                    ind +=1
            except:
                continue

    df_stats['range'] = df_stats['train'] + df_stats['validate'] + df_stats['test']
    
    return df_stats























