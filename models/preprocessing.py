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


rep_use_columns = ['PART_NO', 'PART_SN', 'INSTALL_DATE', 'AC_SN',
       'AC_MODEL', 'OPERATOR_CODE', 'ATA_NUMBER', 
       'REMOVAL_DATE', 'REMOVAL_TYPE_CODE', 
       'QPA', 'FLIGHT_HOURS', 'FLIGHT_CYCLES']
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
    # Step 1: FLIGHT_HOURS > 0 필터링
    df_filtered = df_input[df_input['FLIGHT_HOURS'] > 0].copy()
    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered['RowID'] = df_filtered.index  # 내부 고유 ID

    # Step 2: REMOVAL과 INSTALL 날짜 매칭 (same-day 우선, 이후 가장 가까운 future install)
    # install_lists = df_filtered[['RowID', 'INSTALL_DATE', 'REMOVAL_DATE']]
    chains = [pd.DataFrame(columns=df_filtered.columns) for _ in range(qpa)]

    for _, r in df_filtered.iterrows():
        min_day = MAX_DAY
        # print(f'{r.RowID} Install: {r.INSTALL_DATE}')
        for i, chain in enumerate(chains):
            if len(chain) == 0: # and min_day == MAX_DAY:
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


##########################################################################################

def table_diff(df1, df2):
    temp_left = df1.merge(df2, on=df1.columns.tolist(), how='left', indicator=True)
    result = temp_left[temp_left['_merge']=='left_only'].drop(columns=['_merge'])
    return result

##########################################################################################
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

def related_at_install_series(df_rep, df_util, set_ata, ata_list, chk_date, testing_date):
    all_tests = []
    all_chks = []
    acs = df_rep.groupby('AC_SN').count().index
    for ac in tqdm(acs):
        df_ac = df_rep.groupby('AC_SN').get_group(ac)
        df_ac = df_ac.sort_values(by='INSTALL_DATE')
        df_target = df_ac[df_ac['INSTALL_DATE'] >= chk_date]
        df_target = df_target[df_target['REMOVAL_DATE'] < testing_date]
        df_target = df_target[df_target['FLIGHT_HOURS'] >0]
        if len(df_target) ==0:
            continue
        try:
            df_target = df_target.groupby('ATA_NUMBER').get_group(set_ata)
            # print(ac, len(df_target))
        except:
            # print(ac, 'NO', set_ata)
            continue
        try:
            df_util_ac = df_util.groupby('ac_sn').get_group(ac)
        except:
            continue
        

        ## Re-align and get series of removal info. by QPA
        if df_target['QPA'].iloc[0] >1:
            chain_list = build_chains_by_qpa(df_target, qpa=df_target['QPA'].iloc[0])
        else:
            chain_list = [df_target]

        for df_test in chain_list:
            df_chks = []
            df_test[['INSTALL_DATE', 'REMOVAL_DATE']] = df_test[['INSTALL_DATE', 'REMOVAL_DATE']].apply(pd.to_datetime)
            df_test = df_test.sort_values(by='INSTALL_DATE')
            
            if len(df_test) < 2:
                # print(ac, len(df_test))
                # display(df_test)
                continue
            df_test['prev_fh']= df_test['FLIGHT_HOURS'].shift(1)
            df_test['prev_fh'].iloc[0] = df_test['FLIGHT_HOURS'].iloc[0]
            df_m = pd.merge_asof(df_test[['INSTALL_DATE']], df_util_ac, left_on='INSTALL_DATE', right_on='AU_DATE', direction='backward')

            for i in range(len(df_test)):
                inst_d = df_test['INSTALL_DATE'].iloc[i]
                inst_cum_fh = df_m['CUMULATIVE_FLIGHT_HOURS'].iloc[i]


                df_chk = df_ac[df_ac['INSTALL_DATE']< inst_d]
                df_chk2 = df_chk[df_chk['REMOVAL_DATE']>  inst_d]
                j = len(df_chk2)


                df_chk2 = pd.merge(left=df_chk2, right=df_util_ac, how='inner', left_on=['AC_SN', 'INSTALL_DATE'], right_on=['ac_sn', 'AU_DATE'])
                df_chk2['FH_CUM'] = inst_cum_fh - df_chk2['CUMULATIVE_FLIGHT_HOURS_y']

                if (df_chk2['FH_CUM'] < 0).any():
                    print('CHK', inst_cum_fh, acs)
                    print(df_util_ac[df_util_ac['AU_DATE']== df_test['INSTALL_DATE'].iloc[i]]['CUMULATIVE_FLIGHT_HOURS'])
                    return df_chk2, df_test
                
                df_chk2['CUMULATIVE_FLIGHT_HOURS'] = df_chk2['CUMULATIVE_FLIGHT_HOURS_x']
                # df_chk2['FH_CUM'] = (inst_d-df_chk2['INSTALL_DATE'])

                ## If there is no removal records for ATA6s, take the last removal_date as their installation
                atas_ac = df_ac.groupby('ATA_NUMBER').count().index
                for ata_ac in atas_ac:
                    df_t = df_ac.groupby('ATA_NUMBER').get_group(ata_ac)
                    if np.max(df_t['REMOVAL_DATE']) <= inst_d:
                        df_chk2.loc[j] = df_t.iloc[np.argmax(df_t['REMOVAL_DATE'])]
                        df_chk2['FH_CUM'].loc[j] = inst_cum_fh - df_chk2['CUMULATIVE_FLIGHT_HOURS'].loc[j]
                        if df_chk2['FH_CUM'].loc[j] < 0:
                            
                            print('CHK2', inst_cum_fh, acs)
                            print(df_util_ac[df_util_ac['AU_DATE']== df_test['INSTALL_DATE'].iloc[i]]['CUMULATIVE_FLIGHT_HOURS'])
                            return df_chk2, df_test
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
                        df_chk2.loc[idx_add] = [df_chk2.iloc[0]['AC_SN']] + [a_t] + [df_chk2.iloc[0]['INSTALL_DATE']] +[0]
                        idx_add +=1

                # if len(df_chk2) != 3:
                    # display(df_chk2)
                df_chks.append(df_chk2)

            all_tests.append(df_test)
            all_chks.append(df_chks)

    return all_tests, all_chks


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

















