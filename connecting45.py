import pandas as pd
import numpy as np
import sklearn as sl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn import metrics
import statistics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
#Загрузка БД
db1=pd.read_excel('Database_1.xlsx')
db2=pd.read_excel('Database_2.xlsx')
db3=pd.read_excel('Database_3.xlsx')
db4=pd.read_excel('Database_4.xlsx')
db5=pd.read_excel('Database_5.xlsx')
#Заполнение пропусков
#Функция
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
def clamp(val, minimum, maximum):
    return max(minimum, min(maximum, val))
#БД4
db4['Material type'].replace("don't remember", None, inplace=True)
fill_na_with_similar(db4, 'Material type', key = 'Elements')
fill_na_with_similar(db4, 'Core size (nm)')
fill_na_with_similar(db4, 'Hydro size (nm)')
fill_na_with_similar(db4, 'Surface charge (mV)')
fill_na_with_similar(db4, 'Surface area (m2/g)')
fill_na_with_similar(db4, 'Exposure dose (ug/mL)')
fill_na_with_similar(db4, 'Number of atoms')
fill_na_with_similar(db4, 'Molecular weight (g/mol)')
db4['Viability (%)'] = db4['Viability (%)'].apply(lambda x: clamp(x, 0, 200))
#БД5
db5.rename(columns={'material':'Material type'}, inplace = True)
fill_na_with_similar(db5, 'core_size')
fill_na_with_similar(db5, 'surf_charge')
fill_na_with_similar(db5, 'cell_line')
db5['viability'] = db5['viability'].apply(lambda x: clamp(x, 0, 200))
#Переименование столбцов
db4.rename(columns={
'Exposure dose (ug/mL)':'Concentration (ug/mL)',
'Surface charge (mV)':'Zeta potential (mV)'},inplace=True)
db5.rename(columns={
'dose':'Concentration (ug/mL)',
'core_size':'Core size (nm)',
'hydro_size':'Hydro size (nm)',
'surf_charge':'Zeta potential (mV)',
'surf_area':'Surface area (m2/g)',
'cell_line':'Cell line',
'cell_species':'Human(H)/Animal(A) cells',
'cell_origin':'Cell-organ/tissue source',
'time':'Exposure time (h)',
'viability':'Viability (%)',
'toxicity':'Toxicity'},inplace=True)
#Добавление столбцов
#Молярная масса
db5['Molecular weight (g/mol)']=np.where(db5['Material type']=='Al2O3', '101,96', np.nan)
for i in range (len(db5)):
    if(db5['Material type'].loc[i]=='ZnO'):
        db5.loc[i, 'Molecular weight (g/mol)']='81,38'
    if(db5['Material type'].loc[i]=='Fe2O3'):
        db5.loc[i, 'Molecular weight (g/mol)']='81,38'
    if(db5['Material type'].loc[i]=='Fe3O4'):
        db5.loc[i, 'Molecular weight (g/mol)']='231,533'
    if(db5['Material type'].loc[i]=='SiO2'):
        db5.loc[i, 'Molecular weight (g/mol)']='60,08'
    if(db5['Material type'].loc[i]=='TiO2'):
        db5.loc[i, 'Molecular weight (g/mol)']='79,866'
#Электроотр.
db5['Electronegativity']=np.where(db5['Material type']=='Al2O3', '1,61', np.nan)
for i in range (len(db5)):
    if(db5['Material type'].loc[i]=='ZnO'):
        db5.loc[i, 'Electronegativity']='1,65'
    if(db5['Material type'].loc[i]=='Fe2O3'):
        db5.loc[i, 'Electronegativity']='1,83'
    if(db5['Material type'].loc[i]=='Fe3O4'):
        db5.loc[i, 'Electronegativity']='1,83'
    if(db5['Material type'].loc[i]=='SiO2'):
        db5.loc[i, 'Electronegativity']='1,9'
    if(db5['Material type'].loc[i]=='TiO2'):
        db5.loc[i, 'Electronegativity']='1,54'
#Ионный радиус
db5['Ionic radius']=np.where(db5['Material type']=='Al2O3', '53,5', np.nan)
for i in range (len(db5)):
    if(db5['Material type'].loc[i]=='ZnO'):
        db5.loc[i, 'Ionic radius']='74'
    if(db5['Material type'].loc[i]=='Fe2O3'):
        db5.loc[i, 'Ionic radius']='55'
    if(db5['Material type'].loc[i]=='Fe3O4'):
        db5.loc[i, 'Ionic radius']='69'
    if(db5['Material type'].loc[i]=='SiO2'):
        db5.loc[i, 'Ionic radius']='40'
    if(db5['Material type'].loc[i]=='TiO2'):
        db5.loc[i, 'Ionic radius']='68'
#Время
db4['Exposure time (h)']=np.array(np.nan)
#Cell origin
db4['Cell-organ/tissue source']=np.array(np.nan)
#Density
db5['Density (g/cm3)']=np.array(np.nan)
#Сопоставление
cols=list(set(db4.columns)&set(db5.columns))
print(*cols,sep='\n')
#Соединение
df4=db4.filter(cols)
df5=db5.filter(cols)
db45=pd.concat([df4,df5], axis=0)
db45.reset_index(inplace=True,drop=True)
db45.head()
db45.to_excel('Database_45.xlsx', index = False)
