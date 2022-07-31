import numpy as np
from pymatgen.core.composition import Composition
from math import pi
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

# Вспомогательная функция для сбора данных
# о веществах с помощью pymatgen
def _get_comp_prop(formulas, func):
    values_d = {}
    for formula in formulas:
        try:
            comp = Composition(formula)
            values_d[formula] = func(comp)
        except:
            values_d[formula] = np.nan
    values = []
    for formula in formulas:
        values.append(values_d[formula])
    return np.array(values)

# Функция получения молекулярной массы для веществ
def get_mol_weights(formulas):
    return _get_comp_prop(formulas, lambda x: x.weight)

# Функция получения электроотрицательности для веществ
def get_electroneg(formulas):
   return _get_comp_prop(formulas, lambda x: x.average_electroneg)

# Функция для получения описания формы НЧ
# (Предпологаем, что форма НЧ -- сфера)
def get_bounds(df, as_diameter, surf_area=True, volume=True, u_surf=True):
    if surf_area:
        df['Surface area (um2)'] = 4 * pi * np.square(df[as_diameter]) / 10 ** 6
    if volume:
        df['Volume (um3)'] = 4 / 3 * pi * df[as_diameter] ** 3 / 10 ** 9
    if u_surf:
        df['U_Surface area (um-1)'] = df['Surface area (um2)'] / df['Volume (um3)']

# Заполнение пропусков в БД с помощью поиска подобного
# параметра у таких же материалов
def fill_na_with_similar(df, column, key='Material type'):
    where_nan = np.where(df[column].isna())[0]
    if len(where_nan) == 0:
        return 1
    modes = {}
    success = 0
    for idx in where_nan:
        elem = df[key][idx]
        if elem not in modes:
            not_nan = df[column][df[key] == elem].dropna()
            if len(not_nan) == 0:
                continue
            mode = not_nan.mode()[0]
            modes[elem] = mode
        df.loc[idx, column] = modes[elem]
        success += 1
    return success / len(where_nan)

# Функция ограничения
def clamp(val, minimum, maximum):
    return max(minimum, min(maximum, val))

# Функция для заполнения БД модой
def fill_with_mode(obj):
    obj.fillna(obj.mode().iloc[0], inplace=True)

# Соревнование моделей (с использованием кросс-валидации)
def get_winner_model(regressors, df, to_drop, target):
    scores = np.zeros((len(regressors),))
    for i in range(0, 5):
        shuffled = df.sample(frac=1)
        x, y = shuffled.drop(to_drop, axis=1), shuffled[target]

        print('SHUFFLING:', i)

        local_scores = []
        
        for reg in regressors:
            score = cross_val_score(reg(), x, y)
            local_scores.append(np.sum(score))
        
        scores += np.array(local_scores)

    winner = np.argmax(scores)
    return regressors[winner]

# Удобная функция для удаления выбросов
def delete_where(df, wh):
    df.drop(np.where(wh)[0], inplace=True)
    df.reset_index(inplace=True, drop=True)

# Кодирование БД
def encode(df):
    encoders = {}
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            enc = LabelEncoder().fit(df[col])
            df[col] = enc.transform(df[col])
            encoders[col] = enc
    return df, encoders

# Скалирование БД по колонкам
def scale_by_columns(df, scaler):
    df = df.copy()
    scalers = {}
    for col in df.columns:
        vals = df[col].values.reshape(-1, 1)
        scalers[col] = scaler().fit(vals)
        df[col] = scalers[col].transform(vals)
    return df, scalers
    
# Получение максимального значения каждой
# категориальной переменной
def get_cat_max(df, categorical):
    return dict([
        (col, df[col].max()) for col in categorical
    ])

# Функция преобразования строки из
# таблицы в массив numpy (путём one-hot кодировки)
def row_to_array(row, cat_max):
    arr = []
    for idx in row.index:
        if idx not in cat_max:
            arr.append(row[idx])
            continue
        subarr = [0] * (cat_max[idx] + 1)
        subarr[int(row[idx])] = 1
        arr += subarr
    return np.array(arr)
