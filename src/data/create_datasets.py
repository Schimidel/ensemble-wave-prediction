import numpy as np
import pandas as pd
import glob
import pickle
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from src.features import features as feat 

def create_new(folders, typ, buoy, dest, lag):
    df_boia             = pd.read_csv(buoy)
    df_boia['Datetime'] = pd.to_datetime(df_boia['Datetime'])
    dict_features       = {}
    dict_target         = {}
    dict_rel            = {}
    list_dates          = []
    first_train_date    = pd.to_datetime(folders[0].split('/')[-2] + ' 03:00:00')

    max_number_pred     = lag*8
    for j in tqdm(range(0,max_number_pred), desc='Processing dataset...'):
        lista_cols_feat = []
        lista_cols_tgt  = []
        lista_cols_rel  = []
        for fold in folders:
            proces_folders = glob.glob(fold+typ+'/*')
            proces_folders.sort()    
            for file in proces_folders:
                if not file.endswith('.csv'):  # Ignora arquivos que não são CSV
                    continue
                df    = pd.read_csv(file, encoding='utf-8', sep=';', decimal=',').drop(['Unnamed: 0'], axis=1)
                value = df[['time','deterministic']].iloc[j]['deterministic']
                hour  = df[['time','deterministic']].iloc[j]['time']
                if value == 0 or np.isnan(value) == True:
                    value = lista_cols_feat[-1]
                try:
                    real = df_boia.loc[df_boia['Datetime'] == pd.to_datetime(hour)]['Wvht'].values[0]
                except:
                    real = real
                erro     = real-value
                erro_rel = np.abs(real-value)/np.abs(real)
                lista_cols_feat.append(value)
                lista_cols_tgt.append(erro)
                lista_cols_rel.append(erro_rel)

        dict_features[f'feat_{j}'] = lista_cols_feat
        dict_target[f'tgt_{j}']    = lista_cols_tgt
        dict_rel[f'rel_{j}']       = lista_cols_rel
        list_dates.append(hour)
        if j == 0:
            first_hour_predict = pd.to_datetime(hour)

    df_final_target   =  pd.DataFrame(dict_target)
    df_final_features =  pd.DataFrame(dict_features)
    df_final_rel      =  pd.DataFrame(dict_rel)

    df_final_target[0:-1].to_csv(f'{dest}noaa_data_target.csv', encoding='utf-8', sep=';', decimal=',')
    df_final_features[0:-1].to_csv(f'{dest}noaa_data_features.csv', encoding='utf-8', sep=';', decimal=',')
    df_final_rel[0:-1].to_csv(f'{dest}noaa_data_relative.csv', encoding='utf-8', sep=';', decimal=',')

    save_name         = f'{dest}first_hour_predict.pkl'
    with open(save_name, 'wb') as fp:
        pickle.dump(first_hour_predict, fp) 

    save_name         = f'{dest}first_hour_train.pkl'
    with open(save_name, 'wb') as fp:
        pickle.dump(first_train_date, fp) 

def dispatch(ori, dest, buoy_path, name, lag):
    ori     = feat.format_path(ori)
    dest    = feat.format_path(dest)
    boia    = buoy_path
    typ     = f'processed_{name}'

    df_boia             = pd.read_csv(boia)
    df_boia['Datetime'] = pd.to_datetime(df_boia['Datetime'])

    min_boia = df_boia['Datetime'].min()
    max_boia = df_boia['Datetime'].max()
    max_boia = max_boia - dt.timedelta(days=lag)

    # ============ NOVAS LINHAS PARA DEBUG ============
    import os  # Adicione no topo do arquivo se não existir
    print()
    print("\n--- DEBUG: Verificando caminho ---")
    print(f"Caminho completo (ori): {os.path.abspath(ori)}")  # Mostra o caminho absoluto
    print(f"Conteúdo do diretório: {os.listdir(ori)}")  # Lista arquivos/pastas dentro de 'ori'
    print("---------------------------------\n")
    # ================================================

    folders  = glob.glob(f'{ori}/*/')
    folders.sort()

    print()
    print("Conteúdo de folders:", folders)
    print()
      # Printa os valores da pasta
    min_noaa = pd.to_datetime(folders[0].split('/')[-2])
    max_noaa = pd.to_datetime(folders[-1].split('/')[-2])

    print()
    print(min_noaa)
    print(max_noaa)
    print()

    min_date = str(max(min_boia,min_noaa))[0:10].split('-')
    min_date = min_date[0]+min_date[1]+min_date[2]
    max_date = str(min(max_boia,max_noaa))[0:10].split('-')
    max_date = max_date[0]+max_date[1]+max_date[2]

    print("\n--- DEBUG: Datas ---")
    print(f"min_date: {min_date}")  # Exemplo: '20180428'
    print(f"max_date: {max_date}")  # Exemplo: '20180428'
    print(f"Pasta esperada: {ori+min_date+'/'}")  # Exemplo: './data/raw/noaa/santos/20180428/'
    print("-------------------\n")
    
    if len(folders) == 1:
      pass  # Não filtra
    
    else:
      folders = folders[folders.index(ori+min_date+'/'):folders.index(ori+max_date+'/')+1]
    
    print()
    print('aux')
    print(folders)
    print()

    aux = folders[0].split('/')[-2]   
    df_noaa_first = pd.read_csv(folders[0]+typ+'/'+f'processed_{aux}_lead_00.csv', encoding='utf-8', sep=';', decimal=',').drop(['Unnamed: 0'], axis=1) 
    if pd.to_datetime(df_noaa_first['time'][0]) < df_boia['Datetime'][0]:
        folders = folders[1:]

    last_date   = folders[-1].split('/')[-2]
    noaa_result = folders[-1]+typ+'/'+f'processed_{last_date}_lead_00.csv'
    df_noaa     = pd.read_csv(noaa_result, encoding='utf-8', sep=';', decimal=',').drop(['Unnamed: 0'], axis=1)

    pth_noaa    = feat.format_path(f'/home/storage/minuzzi/Machine_learning/ensemble-wave-prediction/data/raw/noaa/{name}/')
    df_noaa.to_csv(f'{pth_noaa}noaa_forecast.csv', encoding='utf-8', sep=';', decimal=',')

    save_name         = f'{dest}boia.pkl'
    with open(save_name, 'wb') as fp:
        pickle.dump(boia, fp) 

    create_new(folders, typ, boia, dest, lag)

    print('###############################################')
    print('###############################################')
    print('##                                           ##')
    print('##                Finished                   ##')
    print('##                                           ##')
    print('###############################################')
    print('###############################################')      
