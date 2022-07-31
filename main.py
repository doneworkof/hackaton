# Импорт библеотек
import joblib
from funcs import *
import pickle
import pandas as pd
import numpy as np
import json

# Загрузка модели и БД поддержки
# (для заполнения NaN ячеек)
model = joblib.load('model.joblib')
support = pd.read_excel('encoded.xlsx').drop([
    'Viability (%)', 'Surface area (um2)'], axis=1)

# Загрузка скейлеров и кодировщиков
with open('scalers.bf', 'rb') as f:
    scalers = pickle.load(f)

with open('encoders.bf', 'rb') as f:
    encoders = pickle.load(f)

# С помощью этой функции можно получить
# информацию о том, какие значения могут
# содержаться в данных столбцах
# (предназначено для категориальных)
def get_variety(col):
    if support[col].dtype != 'object':
        return support[col].describe()
    return support[col].unique()

# Функция, которая позволяет пользователю
# посмотреть, какие колонки будут задействованы
# в предсказании
def get_columns():
    return support.columns

# предсказание
def predict(df):
    # Копия
    df = df.copy()
    n = len(df)
    # Удаление столбцов, которые
    # не нужны для предсказания
    to_remove = [
        col for col in df
        if col not in support.columns
    ]
    print('Removing:', to_remove)
    df.drop(to_remove, axis=1, inplace=True)
    # Добавление нужных столбцов
    to_add = list(set(support.columns) - set(df.columns))
    for col in to_add:
        df[col] = [np.nan] * n
    
    if df['Nanoparticle'].isna().any():
        raise Exception('Nanoparticle column must be without NaN!')
    # Заполнение столбцов молекулярного веса и электроотрицательности
    cols = ['Molecular weight (g/mol)', 'Electronegativity']
    funcs = [get_mol_weights, get_electroneg]

    for col, func in zip(cols, funcs):
        ids = np.where(df[col].isna())[0]
        df.loc[ids, col] = func(df['Nanoparticle'][ids])

    # Соединение с БД поддержки
    df = pd.concat([df, support], axis=0)
    df.reset_index(drop=True, inplace=True)
    
    # Логарифмирование концентрации
    df['Concentration (ug/mL)'] = np.log10(df['Concentration (ug/mL)'])

    # Заполнение колонок значениями от
    # таких-же наноматериалов
    to_fill_with_sim = [
        'Molecular weight (g/mol)',
        'Size_in_Medium (nm)',
        'Diameter (nm)',
        'Density (g/cm3)',
        'Electronegativity',
        'Size_in_Water (nm)',
        'Zeta_in_Water (mV)',
        'Zeta_in_Medium (mV)',
        'is_inorganic'
    ]
    
    for col in df.columns:
        if col in to_fill_with_sim:
            fill_na_with_similar(df, col, 'Nanoparticle')
        fill_with_mode(df[col])

    # Разъединение БД
    df = df.iloc[:n].filter(get_columns())

    # Описание формы НЧ, логарифмирование значения
    get_bounds(df, 'Diameter (nm)', u_surf=False, volume=False)
    df['Surface area (um2)'] = np.log10(df['Surface area (um2)'])

    # Кодирование и скалирование
    for col in df.columns:
        print(f'Processing {col}...')
        if col in encoders:
            variety = get_variety(col)
            df[col] = df[col].apply(
                lambda x: encoders[col].transform([x])[0] if x in variety else np.nan)
            if df[col].isna().sum() == len(df):
                df.loc[0, col] = support[col].mode().iloc[0]
            fill_with_mode(df[col])
        if col in scalers:
            df[col] = scalers[col].transform(
                df[col].values.reshape(-1, 1))

    # Удаление столбца
    df.drop('Nanoparticle', axis=1, inplace=True)
    
    # Предсказание
    pred = model.predict(df).reshape(-1, 1)
    
    # Обратное скалирование полученных значений
    return scalers['Viability (%)'].inverse_transform(pred).reshape(-1)
