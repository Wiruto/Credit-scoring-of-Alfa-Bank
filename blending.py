# ИМПОРТ БИБЛИОТЕК

import shutil
import os

# работа с регулярными выражениями
import re

# библиотеки для работы с табличными данными
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fastparquet as fp

# генерация случайных чисел
import random
from random import randint
from sklearn.utils import shuffle

# библиотеки для построения графики
import seaborn as sns
import matplotlib.pyplot as plt #для визуализации
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots
import nbformat

# библиотеки для математических преобразований с массивами данных
import numpy as np
import mlx.core as mx
from sklearn import model_selection
from sklearn.model_selection import train_test_split

# библиотеки для работы с функциями(частичная передача аргументов в функцию)
from functools import partial

# библиотеки для работы со статистическими характеристиками
from scipy import stats
import statistics
from collections import Counter

# библиотеки для работы с pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

# проверка временного ряда на статичность
from statsmodels.tsa.stattools import adfuller

# Импортируем DBSCAN-кластеризацию
from sklearn.cluster import DBSCAN

# вставить картинку в Jupiter Notebook
from IPython.display import Image

# линейные модели машинного обучения
from sklearn import linear_model

# ансамбли моделей машинного обучения
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# поиск гиперпараметров модели
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
import optuna

 # метрики
from sklearn import metrics

# библиотека для стандартизации данных
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# сохранить полученные модели
import joblib
from joblib import dump, load

# сборщик мусора
import gc

# для ограничения времени выполнения функции
import signal
import func_timeout

# для отслеживания времени выполнения функции
import time

# очистить output
from IPython.display import clear_output

# ФУНКЦИИ ДЛЯ ПРЕОБРАЗОВАНИЯ ДАННЫХ

# функция torow_transformer

# Назначение: Преобразование признака столбца в признаки строки
#               с сохранением обратной последовательности в признаке.
#               (извлечение последних операций клиента)

# Внешние переменные функции: DataFrame, n_last
#   DataFrame - исоходый DataFrame
#   n_last - необходимое счисло послдених операций клиента
#   Структура DataFrame:
#       1. id
#       2. feature1
#       3. feature2
#       4. feature3
#       ...

# Результат работы функции: New DataFrame
#   Признаки New DataFrame:
#       1. id
#       2. feature1.1
#       3. feature1.2
#       4. feature1.3
#       ...
#       5. feature1.N
#       6. feature2.1
#       7. feature2.2
#       ...
#       8. feature2.N
#       ...
#   где featureX.1 - соотвествует последней операции клиента,
#       featureX.2 - соотвествует предпоследней операции клиента,
#       .....

# 2. dict_features - словарь (карта) признаков,
#    в котором отображается сокращения признаков и их расщифровка

# Алгоритм работы функции:
# 1. извлекаем признаки из данных DataFrame
# 3. формируем карту признаков dict_features:
#    3.1. каждый признак кодируется следующим образом: 'fn',
#         где n - порядковый номер признака
#    3.2. полные имена признаков задаются следующим образом: 'feature_N',
#         где N - порядок клиентской операции (большему N соотвествует, более ранняя операция)

# 4. Преобразуем данные к массиву
# 5. Группируем массив для каждого клиента
# 6. К групированному массиву прменяем следующие преобразования:
#    6.1. Обращаем порядок клиенских операций
#    6.2. выбираем посление n_last операций
#    6.3. Если число клиенски операций меньше чем n_last, дополняем их нулями
# 7. Преобразуем полученные данные к DataFrame

# Описание локальных переменных функции:
# 1. pd_data - исходный DataFrame
# 2. n_last - необходимое число последних операций клиента
# 3. list_id - список для хранения id клиентов внутри функции
# 4. list_features - список для хранения признаков исхдного dataframe внутри функции
# 5. dict_features - локальная карта признаков
# 6. rn_id - список количества операций для каждого клиента
# 6. array_data - данные преобразоанные к numpy-массиву
# 6. split_array - сгрупированные по клиентам данные преобразованные

# обьявлем функцию
def torow_func(dict_params):
    pd_data = dict_params['data']
    n_last = dict_params['n_last']

    # извлекаем список "id" клиентов
    list_id = pd_data['id'].unique().tolist()
    
    # извлекаем список признаков из данных        
    list_features = pd_data.columns.drop(['id','rn'])

    # формируем словарь для зашифрованных признаков
    dict_features = {}
    
    # заполним словарь dict_features
    num_f=0
    for feature in list_features:
        # шифруем признак: fk = "feature_agg_function"
        for num_feature in range(1,n_last+1):
            dict_features['f'+str(num_feature+num_f)] = feature+'_'+str(num_feature)
        num_f+=n_last
    

    # формируем словарь rn_id
    rn_id = pd_data.groupby('id')['id'].count().to_list()

    # для улучшения производительности преобразуем DataFrame в array-массив
    array_data = np.array(pd_data.iloc[:,2:]).transpose()

    # "порежем массив" по длине кредитной истории клиента
    split_array = np.array_split(array_data, np.cumsum(rn_id),axis=1)
    
    # определим порядок последующих преобразований в функции
    def transform_array(array_id):
        # обратим порядок клиентских операций 
        reverse_array_id = array_id[::,::-1]
        # выбрем после n операций клиента
        list_n_last = reverse_array_id[::,:n_last]
        # если клиенских операций было меньше чем n_last
        # дополним недастающие нулями и преобразуем данные к строке
        if len(list_n_last[0])<n_last:
            full_list_n_last = np.hstack((list_n_last,np.zeros((list_n_last.shape[0],n_last-len(list_n_last[0])),dtype='int64')))
            # преобразуем список к строке
            full_list_n_last = full_list_n_last.reshape(-1)
        else:
            full_list_n_last = list_n_last.reshape(-1)
        return full_list_n_last

    # применим transform_array преобразование к списку split_array
    list_data = np.array(list(map(transform_array,split_array)))[:-1]
    
    # преобразуем полученные данные к dataframe
    dataframe = pd.DataFrame(data=list_data, columns=dict_features.keys())

    # добавим столбец id
    dataframe.insert(0,'id',list_id)
    
    return dataframe, dict_features,rn_id


# преобразуем функции в инструмент для преобразования данных (Transformer)
torow_transformer = FunctionTransformer(torow_func)


# функция diff_feature

# Назначение: Определение дифференциальных характеристик ряда 

# Внешние переменные функции: 
#           1.Series/np.array/list


# Результат работы функции: 
# 1. diff_list - Список из значений:
#                   1.1. speed - скорость изменения ряда;
#                   1.2. accel - ускорение изменения ряда;
#                   1.3. bias - смещение ряда;
#                   1.4. pulse - импульс ряда;

# обьявлем функцию
def diff_feature(data):
    # преобразуем данные к numpy массиву
    data = np.array(data)
    # расчитаем необходимые характеристики
    speed = round(float(np.diff(data,1).mean()),2)
    accel = round(float(np.diff(data,2).mean()),2)
    bias = round(float(np.diff(data,1).sum()),2)
    pulse = round(float(np.diff(data,2).sum()),2)
    # сформируем из найденных значений в список
    diff_list = [speed,accel,bias,pulse]
    return diff_list



# функция statistic_features

# Назначение: Извлечение основных статистических характеристик
#             из признаков в исходном DataFrame.

# Внешние переменные функции: DataFrame
#   Признаки DataFrame:
#       1. id
#       2. feature1
#       3. feature2
#       4. feature3
#       ...

# Результат работы функции: 
# 1. dataframe - таблица с данными. 
#    Признаки dataframe:
#       1. id
#       2. feature1_mean
#       3. fearture1_hmean
#       4. feature1_std
#       5. feature1_min
#       6. feature1_25%
#       7. feature1_50%
#       8. feature1_75%
#       9. feature1_max
#       10. feature1_mode
#       11. feature1_frequency_mode
#       12. feature2_mean
#       ...
    
# 2. dict_features - словарь (карта) признаков,
#    в котором отображается сокращения признаков и их расщифровка

# Алгоритм работы функции:
# 1. извлекаем признаки из данных
# 2. формируем карту признаков:
#    2.1. каждый признак кодируется следующим образом: 'fn' где n - порядковый номер признака
#    2.2. полные имена признаков задаются следующим образом:
#         2.2.1 если в исходном dataframe признак бинарный, то: "Исходное имя признака"+"binary"
#         2.2.2 если в исходном dataframe признак не бинарный, то: "Исходное имя признака"+"Статистическая характеристика"
# 3. для каждого клиента по каждому признаку из исходного dataframe расчитываем статистические характеристики
# 4. записываем полученные значение в новый dataframe

# Описание локальных переменных функции:
# 1. dict_agg_function - словарь из агригирующих функций
#       keys: имена для обращения к функциям:
#       values: lamda-функция, соотвествующей статистической характристики
# 2. list_features - список для хранения признаков исхдного dataframe внутри функции
# 3. list_id - список для хранения id клиентов внутри функции
# 4. dict_features - локальная карта признаков
# 5. k - номер признака в dict_features на текущей итерации
# 6. dataframe - результирующий dataframe

# обьявлем функцию
def statistic_features(pd_data):
    # формируем список из функций для статистических преобразований
    # предусмотрим работу функций на случай, если в массиве данных всего 1 строка
    dict_agg_function = {
    'ptp' : lambda x: 0 if len(x) <= 3 else np.ptp(x),
    'mean': lambda x: 0 if len(x) <= 3 else x.mean(), 
    'gmean' : lambda x: stats.gmean(x),   
    'hmean': lambda x: stats.gmean(x),
    'pmean25': lambda x: stats.pmean(x,25),
    'pmean50': lambda x: stats.pmean(x,50),
    'pmean75': lambda x: stats.pmean(x,75),
    'expectile25': lambda x: stats.expectile(x,0.25),
    'expectile50': lambda x: stats.expectile(x),
    'expectile75': lambda x: stats.expectile(x,0.75),
    'moment': lambda x: stats.moment(x),
    'std': lambda x: 0 if len(x) <= 3 else np.std(x),
    'min': lambda x: min(x),
    '20%': lambda x: x.mean() if len(x) <= 3 else np.percentile(x,q=20),
    '30%': lambda x: x.mean() if len(x) <= 3 else np.percentile(x,q=30),
    '40%': lambda x: x.mean() if len(x) <= 3 else np.percentile(x,q=40),
    '50%': lambda x: x.mean() if len(x) <= 3 else np.percentile(x,q=50),
    '60%': lambda x: x.mean() if len(x) <= 3 else np.percentile(x,q=60),
    '70%': lambda x: x.mean() if len(x) <= 3 else np.percentile(x,q=70),
    'max': lambda x: max(x),
    'mode': lambda x: statistics.mean(statistics.multimode(x)),
    'frequency_mode': lambda x: round(list(x).count(statistics.multimode(x)[0])*len(statistics.multimode(x))/len(x),2),
    'cov' : lambda x: 0 if len(x) <= 3 else np.cov(x),
    'histogram' : lambda x: 0 if len(x) <= 3 else np.histogram(x)[1].mean(), 
    'speed': lambda x: 0 if len(x) <= 3 else diff_feature(x)[0],
    'accel': lambda x: 0 if len(x) <= 3 else diff_feature(x)[1],
    'bias': lambda x: 0 if len(x) <= 3 else diff_feature(x)[2],
    'pulse': lambda x: 0 if len(x) <= 3  else diff_feature(x)[3]
    } 

    # напишем функцию для преобразования массива до статистических характеристик
    def stat_func(array_data): 
        # сформируем лист под результаты преобразования
        list_for_result = []
        # запишем все статистические харкетристики из словаря dict_agg_function
        for func in dict_agg_function.values():
            list_for_result.append(func(array_data))
        return np.array(list_for_result).round(3)

    # напишем функцию для применения функции stat_func к списку
    def submap(list_data):
        # расчитаем количество операция клиента
        max_rn = len(list_data[0])
        # получим статистические характристики массива
        list_stat_features = np.array(list(map(stat_func,list_data))).reshape(-1)
        return np.hstack((max_rn,list_stat_features))
    
    # извлекаем список "id" клиентов
    list_id = pd_data['id'].unique().tolist()

    # извлекаем список признаков из данных        
    list_features = pd_data.columns.drop(['id','rn'])
    
    # формируем словарь для зашифрованных признаков
    dict_features = {'f1':'count'}
    k=1 # порядковый номер защифрованного признака

    # заполним словарь dict_features
    for feature in list_features:
        # шифруем признак: fk = "feature_agg_function"
        for key_function in dict_agg_function.keys():
            k+=1
            dict_features['f'+str(k)] = feature+'_'+key_function

    # формируем список rn_id
    rn_id = pd_data.groupby('id')['id'].count().to_list()

    # для улучшения производительности преобразуем DataFrame в array-массив
    array_data = np.array(pd_data.iloc[:,2:]).transpose()

    # "порежем массив" по длине кредитной истории клиента
    split_array = np.array_split(array_data, np.cumsum(rn_id),axis=1)[:-1]

    # получем статические характеристики признаков
    stat_features = np.array(list(map(submap,split_array)))
    
    # Сформируем dataframe из полученных данных
    dataframe = pd.DataFrame(data=stat_features, columns=dict_features.keys())

    # добавим столбец id
    dataframe.insert(0,'id',list_id)

    return dataframe, dict_features

# преобразуем функции в инструмент для преобразования данных (Transformer)
stat_transformer = FunctionTransformer(statistic_features)




# функция corr_transform_to_force

# Назначение: из матрицы взаимных корреляций
#             выделить не корелирующие признаки

# Внешние переменные функции: 
#           1. df.corr() - матрица корреляций
#           2. threshold - порог значимости корреляции:
#               значение коэффициента корреляции, больше которого
#               признаки считаются скоррелированными.

# Пояснение: 
# Под силой корреляции будем понимать следующее: если коэффициент 
# коррелиции между признаками больше значения threshold, то принимаем,
# что между признаками сильная корреляционная связь значение коэффициента 
# коррелияции заменяем на 1, иначе корреляционная связь слабая и значение 
# коээфициента корреляции заменяем на 0

# Результат работы функции: 
# 1. corr_matrix - матрица корреляций(отражает силу корреляции)
# 2. list_ncorr_features - список не скореллированных признаков
# 3. corr_force - сила корреляции всей матрицы: отношение числа скоррелированных 
# признаков к числу всех признаков в матрице

# Описание локальных переменных функции:
# 1. coor_force - функция преобазующая значение
#        коэффициента корряляции в силу корреляции
# 2. corr_matrix - матрица отражающая силу корряляции между признаками
# 3. max_corr - максимальное число взаимных корреляций между признаками
# 4. list_ncorr_features - список не коррелируемых признаков


# обьявлем функцию
def corr_transform_to_force(matrix,threshold=0.7):
    list_features = matrix.index.tolist()
    
    
    # создадим функцию для разметки матрицы корреляции
    # 1 - корреляция признаков выше порога значимости threshold
    # 0 - корреляция признаков ниже порога значимости threshold
    corr_force = lambda x: 1 if x >threshold else 0
    # выполним разметку матрицы корреляции
    corr_matrix = matrix.map(lambda x: corr_force(x))
    
    # алгоритм отбора не коррелиарных признаков:
    #   1. Найдем признак с наибольшим числом взаимных корреляций
    #   2. удалим найденный признак
    #   3. составим матрицу корреляций из отсавшися признаков
    #   4. повторяем пункты 1-3 до тех пор пока в матрице не останутся 
    #       не коррелированные признаки

    # ищем наибольшее число взаимных корреляций среди признаков
    max_corr = corr_matrix.sum().max()

    while max_corr > 1:
        # определяем признак с наибольшим числом взаимных корреляций
        max_corr_feature = corr_matrix.sum()[corr_matrix.sum()==corr_matrix.sum().max()].index[0]
        # удалем признак из матрицы корреляций
        corr_matrix = corr_matrix.drop(max_corr_feature).drop(max_corr_feature,axis=1)
        max_corr = corr_matrix.sum().max()
    # запишем не скоррелированные признаки в список
    list_ncorr_features = corr_matrix.index.tolist()
    # найдем силу корреляции всей матрицы как отношение
    # количества скоррелированных признаков к всмеу количеству признаков
    corr_force = round(1-len(list_ncorr_features)/len(list_features),3)
    return corr_matrix, list_ncorr_features, corr_force


# функция search_DBSCAN_parameters

# Назначение: Для подбора eps и min_samples параметров,
#               функция "прогоняет" DBSCAN кластеризацию 
#               с параметрами eps и min_samples
#               примающими значения из заданного диапазона.

# Внешние переменные функции: 
#           1. data - dataframe для кластеризации
#           2. r1 - начало диапазона
#           3. r2 - конец диапазона  
#           4. n - предпалагамое число кластеров      

# Результат работы функции: 
# 1. data_cluster - кластеризация данных при различных 
#       значениях параметров eps и min_samples

# Описание локальных переменных функции:
# 1. parametr_range - диапазон изменения параметров
# 2. dataframe_columns - колонки в результирующем dataframe
# 3. data_cluster - результрующий dataframe
# 4. index_cluster - текущая позиция в data_cluster
# 5. clustering - кластеризатор
# 6. list_cluster_values - список для заполнения текущими 
#                          значениями data_cluster

# обьявлем функцию
def search_DBSCAN_parameters(dataframe,r1,r2,n=3):
    # задаем диапозон измениния параметров
    parameter_range = range(r1,r2)
    # формируем заготовку для результирующего dataframe
    dataframe_columns = ['eps','min_samples',-1,0,1]
    # проверим что задано не меньше минимального количества кластеров
    if n<=3: 
        data_cluster = pd.DataFrame(columns=dataframe_columns)
    else: 
        for claster in range(4,n+1):
            dataframe_columns.append(claster-2)
        data_cluster = pd.DataFrame(columns=dataframe_columns)
    # задаем начально значение индекса в data_cluster
    index_cluster = 0

    # для подсчета обьектов в кластерах создадим dataframe
    dataframe_count = pd.DataFrame()
    
    # "прогоняем" DBSCAN кластеризациию по диапазону параметров
    for eps in parameter_range:
        
        for min_samples in parameter_range:
            print('current eps:',eps,'  current min_samples:', min_samples, end='\r')
            # запускаем кластеризацию с текущими параметрами
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(dataframe)
            # добавлем к данным столбец с разметкой
            dataframe_count['clater'] = clustering.labels_
            # формируем пустой список для заполнения
            list_cluster_values = []
            # добавлеям в список текущие параметры
            list_cluster_values.append(eps)
            list_cluster_values.append(min_samples)
            # добавлем в список количество обьктов в каждом кластере
            for column in dataframe_columns[2:]:
                list_cluster_values.append(len(dataframe_count['clater'][dataframe_count['clater']==column]))
                
            # заполняем dataframe  текущими данными
            data_cluster.loc[index_cluster] = list_cluster_values
            index_cluster +=1
            # сбрасываем dataframe_count
            dataframe_count = pd.DataFrame()
    return data_cluster



# функция generate_samples

# Назначение: для генерации индексов выборок данных

# Внешние переменные функции: 
#           1. max - определяет максимальное значение множества
#               из которого формируются выборки
#           2. n - количество выборок
#           3. k - мощность одной выборки
     

# Результат работы функции: 
# 1. samples_list - список с выбороками

# алгоритм работы:
# 1. задаем отрезок натурального ряда N мощностью max и добавлем в него 0.
#       In = N U {0}, I = {0,1,2,3,4,5,..,max}
# 2. если мощность множества In больше, необходимого количества элементов
#       cardo(In) > n x k , то из множества In формируем n случайных выборок
# размера k без повторения.
# 3. если мощность множества In меньше, необходимого количества элементов
#       cardo(In) < n x k , то из множества In формируем случайные выборки
# размера k без повторения, до тех пор пока не закончится множество In.
# После, добираем недостающее количество выборок случайными выборками 
# размера k из множества In с повторением (bootstrap метод).


# обьявлем функцию
def generate_samples(max,n,k,random_state = None):
    # создадим список под результат
    samples_list = []
    # формируем множество натуральных числе от 0 до max
    In = list(range(max+1))
    # Будем выполнять код пока не наберем необходимого количества выборок
    # нарушим порядок в множестве
    In = shuffle(In,random_state=random_state)
    # random.shuffle(In)

    # задаим границы извлечения данных из In
    In_start = 0
    In_end = k
    while len(samples_list) < n:
        # сформируем список под одну выборку
        sample = []
        # первые списки будем наполнять значениеми из множества In
        # без повторения, до тех пор пока все значения из множества In
        # не распределяться по выборкам
        if len(In)-In_end >= 0:
            sample.extend(In[In_start:In_end])
        else:                    
            # если элементов во множестве In недостаточно,
            # запоняем выборку "остатками" 
            sample.extend(In[In_start:])

            # остальные данные заполняем методом bootstrap
            # выполнем код пока не заполним выборку k значениями
            while len(sample) < k:
                # генерируем случайное число из диапазона от 0 до len(In)-1
                random_index = randint(0,len(In)-1)
                # добавляем значение из множества In с индексом random_index
                # в список index_list
                sample.append(In[random_index])

        # после того как мы набрали значения в выборку отправлем ее в samples_list
        samples_list.append(sample)
        # переходим к следующим данным в множестве In
        In_start+=k
        In_end+=k

    return samples_list



def my_train_test_split(X,y,random_state=42,train_size=0.8,):
    # если разбиение без стратификации

    # зададим число элементов в выборке train
    len_train = round(len(y)*train_size)
    # формируем множество натуральных чисел от 0 до max
    list_random_index = list(range(len(y)))
    # нарушим порядок в множестве
    list_random_index = shuffle(list_random_index,random_state=random_state)
    # формируем список индексов под train выборку
    train_samples = list_random_index[:len_train]
    # формируем список индексов под test выборку
    test_samples = list_random_index[len_train:]
    # выполнем код пока не заполним выборку k значениями
   
    X_train = X.iloc[train_samples]
    y_train = y.iloc[train_samples]
    X_test = X.iloc[test_samples]
    y_test = y.iloc[test_samples]

    return X_train, y_train, X_test, y_test




# функция class_1_percent_samples

# Назначение: для генерации индексов сбалансированных выборок

# Внешние переменные функции: 
#           1. data_target - массив из id и значений класса
#           2. class_1_percent - процент класса 1 в результирующей выборке
#           3. random_state - параметр для обеспечения воспроизваодимости функции
     

# Результат работы функции: 
# 1. samples_list - список со сблансированными выбороками

# обьявлем функцию
def class_1_percent_samples(data_target,class_1_percent,random_state = None):
    # приведем данные к нужной форме
    data_target = pd.DataFrame(data=np.array(data_target),columns =['id','flag'])
    
    # разделим клиентов  по признаку flag
    flag_0 = data_target[data_target['flag']==0].reset_index(drop=True)
    flag_1 = data_target[data_target['flag']==1].reset_index(drop=True)

    # определим класс большинства
    if flag_1.shape[0] > flag_0.shape[0]:
        majority_class = flag_1
        minority_class = flag_0
        # расчитаем необходимую величину выборки majority_class
        majority_class_size = round(minority_class.shape[0]*(class_1_percent)/(1-class_1_percent))
        # с помощью функции generate_samples сформируем выборку для majority_class
        samples_majority_class= generate_samples(majority_class.shape[0]-1,1,majority_class_size,random_state=random_state)
    else:
        majority_class = flag_0
        minority_class = flag_1
        # расчитаем необходимую величину выборки majority_class
        majority_class_size = round(minority_class.shape[0]*(1-class_1_percent)/(class_1_percent))
        # с помощью функции generate_samples сформируем выборку для majority_class
        samples_majority_class= generate_samples(majority_class.shape[0]-1,1,majority_class_size,random_state=random_state)
    
    # сформируем список выбороки с заданным процентом класс 1
    samples_list_id = minority_class['id'].values.tolist()+majority_class['id'].iloc[samples_majority_class[0]].tolist()

    return samples_list_id

# функция для удаления содержимого в папке
def delete_everything_in_folder(folder_path):
    shutil.rmtree(folder_path)
    os.mkdir(folder_path)

# ФУНКЦИИ ДЛЯ PIPELINE

# ФУНКЦИЯ to_base_model

# напишем функцию для преобразования данных от исходных  
# до признаков base models

def to_base_model(dataset):
    # сформируем списки признаков каждого подпространства
    date_features = ['id','rn','pre_since_opened','pre_since_confirmed','pre_pterm','pre_fterm',
                    'pre_till_pclose','pre_till_fclose','pclose_flag','fclose_flag']
    late_features = ['id','rn','pre_loans5','pre_loans530','pre_loans3060','pre_loans6090',
                    'pre_loans90','is_zero_loans5','is_zero_loans530','is_zero_loans3060',
                    'is_zero_loans6090','is_zero_loans90','pre_loans_total_overdue','pre_loans_max_overdue_sum']
    credit_features = ['id','rn','pre_loans_credit_limit','pre_loans_next_pay_summ','pre_loans_outstanding','pre_loans_credit_cost_rate']

    relative_features = ['id','rn','pre_util','pre_over2limit','pre_maxover2limit','is_zero_util',
                    'is_zero_over2limit','is_zero_maxover2limit']

    payments_features = ['id','rn','enc_paym_0','enc_paym_1','enc_paym_2','enc_paym_3','enc_paym_4','enc_paym_5','enc_paym_6',
                            'enc_paym_7','enc_paym_8','enc_paym_9','enc_paym_10','enc_paym_11','enc_paym_12','enc_paym_13',
                            'enc_paym_14', 'enc_paym_15', 'enc_paym_16', 'enc_paym_17','enc_paym_18','enc_paym_19','enc_paym_20',
                            'enc_paym_21','enc_paym_22', 'enc_paym_23','enc_paym_24']

    service_features = ['id','rn','enc_loans_account_holder_type','enc_loans_credit_status','enc_loans_credit_type','enc_loans_account_cur']

    # составим словарь небходимых данных и их признаков
    dict_torow_data = {
        'date_torow' : date_features,
        'late_torow' : late_features,
        'credit_torow' : credit_features,
        'relative_torow' : relative_features,
        'payments_torow': payments_features,
        'service_torow': service_features}
    # сформируем данные за последние 25 операций клиентов
    for space_features in dict_torow_data.keys():
        clear_output()
        # добавим индекацию процесса
        print('Current space features:',space_features)
        # сформируем данные для преобразования
        data_to_transform = dataset[dict_torow_data[space_features]]
        print('Start transform')
        data_torow = torow_transformer.transform({'data':data_to_transform,'n_last':25})[0]
        # сохраним преобразованные данные на диск для быстрого воспроизведения
        print('Start save')
        fp.write('models/temp/'+space_features,data_torow)
        # удалим использованные данные
        del data_to_transform
        gc.collect()

    # составим словарь небходимых данных и их признаков
    dict_stat_data = {
        'date_stat' : date_features,
        'late_stat' : late_features,
        'credit_stat' : credit_features,
        'relative_stat' : relative_features,
        'payments_stat': payments_features,
        'service_stat': service_features}
    # сформируем данные за последние 25 операций клиентов
    for space_features in dict_stat_data.keys():
        clear_output()
        # добавим индекацию процесса
        print('Current space features:',space_features)
        # сформируем данные для преобразования
        data_to_transform = dataset[dict_stat_data[space_features]]
        print('Start transform')
        data_stat = stat_transformer.transform(data_to_transform)[0]
        # сохраним преобразованные данные на диск для быстрого воспроизведения
        print('Start save')
        fp.write('models/temp/'+space_features,data_stat)
        # удалим использованные данные
        del data_to_transform, data_stat
        gc.collect()
    
    # сформируем словарь из данных
    dict_datasets = {
    'date_torow': fp.ParquetFile('models/temp/date_torow').to_pandas().set_index('id'),
    'late_torow': fp.ParquetFile('models/temp/late_torow').to_pandas().set_index('id'),
    'credit_torow': fp.ParquetFile('models/temp/credit_torow').to_pandas().set_index('id'),
    'relative_torow': fp.ParquetFile('models/temp/relative_torow').to_pandas().set_index('id'),
    'payments_torow': fp.ParquetFile('models/temp/payments_torow').to_pandas().set_index('id'),
    'service_torow': fp.ParquetFile('models/temp/service_torow').to_pandas().set_index('id'),
    'date_stat': fp.ParquetFile('models/temp/date_stat').to_pandas().set_index('id'),
    'late_stat': fp.ParquetFile('models/temp/late_stat').to_pandas().set_index('id'),
    'credit_stat': fp.ParquetFile('models/temp/credit_stat').to_pandas().set_index('id'),
    'relative_stat': fp.ParquetFile('models/temp/relative_stat').to_pandas().set_index('id'),
    'payments_stat': fp.ParquetFile('models/temp/payments_stat').to_pandas().set_index('id'),
    'service_stat': fp.ParquetFile('models/temp/service_stat').to_pandas().set_index('id'),
    }
    return dict_datasets

# ФУНКЦИЯ to_first_meta

# напишем функцию для преобразования данных от признаков base models
# до метапризнаков first metamodels

def to_first_meta(dict_datasets):
  # формируем пространство для метапризнаков
  list_spaces = ['date','late','credit','relative','payments','service']
  torow_models = ['LRTR','RFTR','GBTR']
  stat_models = ['LRST','RFST','GBST']
  
  # фомируем заготовки для результитрующих данных
  features_first_meta_torow = pd.DataFrame(index=dict_datasets['credit_torow'].index)
  features_first_meta_stat = pd.DataFrame(index=dict_datasets['credit_torow'].index)

  # указываем путь к директории в которой находятся обученные base модели
  directory = "models/base/"
  # Получаем список моделей в директории
  list_models = os.listdir(directory)

  # формируем метапризнаки данных torow
  for space in list_spaces:
    for type_model in torow_models:
      # для текущего подбпространства формируем список предобученных моделей 
      list_base_models = sorted([x for x in list_models if (re.search(space, x))and(re.search(type_model, x))])
      # делаем предсказание на каждой предобученной модели
      for name_model in list_base_models:
        # добавим индекацию процесса
        clear_output()
        print('Current feature: torow')
        print('Current space: ', space)
        print('Current type_model: ', type_model)
        print('Current name_model: ', name_model)
        # загружаем предобученную модель
        model_classifier = joblib.load('models/base/'+name_model)
        # загружаем list_n_last_features
        list_n_last_features = joblib.load('models/base/list_n_last_features/'+'list_n_last_features_'+type_model+'_'+name_model[5:-7]+'.joblib')
        if type_model=='LRTR':
          # загужаемый предобученный scaler
          scaler = joblib.load('models/base/scalers/'+'scaler_torow_'+name_model[5:-7]+'.joblib')
          features_first_meta_torow[name_model[:-7]] = model_classifier.predict_proba(scaler.transform(dict_datasets[space+'_torow'].to_numpy()[:,list_n_last_features]))[:,1]
        else:
          features_first_meta_torow[name_model[:-7]] = model_classifier.predict_proba(dict_datasets[space+'_torow'].to_numpy()[:,list_n_last_features])[:,1]

    del dict_datasets[space+'_torow']
    gc.collect()

  # формируем метапризнаки данных stat
  for space in list_spaces:
    for type_model in stat_models:
      # для текущего подбпространства формируем список предобученных моделей 
      list_base_models = sorted([x for x in list_models if (re.search(space, x))and(re.search(type_model, x))])
      # делаем предсказание на каждой предобученной модели
      for name_model in list_base_models:
        # добавим индекацию процесса
        clear_output()
        print('Current feature: stat')
        print('Current space: ', space)
        print('Current type_model: ', type_model)
        print('Current name_model: ', name_model)
        # загружаем предобученную модель
        model_classifier = joblib.load('models/base/'+name_model)
        # загружаем list_ncorr_features
        list_ncorr_features = joblib.load('models/base/list_ncorr_features/'+'list_ncorr_features_'+type_model+'_'+name_model[5:-7]+'.joblib')
        # загружаем 
        if type_model=='LRST':
          # загужаемый предобученный scaler
          scaler = joblib.load('models/base/scalers/'+'scaler_stat_'+name_model[5:-7]+'.joblib')
          features_first_meta_stat[name_model[:-7]] = model_classifier.predict_proba(scaler.transform(dict_datasets[space+'_stat'][list_ncorr_features]))[:,1]
        else:
          features_first_meta_stat[name_model[:-7]] = model_classifier.predict_proba(dict_datasets[space+'_stat'][list_ncorr_features])[:,1]

    del dict_datasets[space+'_stat']
    gc.collect()

  dict_meta_first = {
    'features_first_meta_torow': features_first_meta_torow,
    'features_first_meta_stat': features_first_meta_stat
    }

  # удалим содержимое папки models/temp/
  delete_everything_in_folder('models/temp/')

  return dict_meta_first

# ФУНКЦИЯ to_second_meta

# напишем функцию для преобразования данных от метапризнаков first metamodels
# до метапризнаков second metamodels

def to_second_meta(dict_datasets):
  # формируем пространство для метапризнаков
  space_models = ['LRTR','RFTR','GBTR','LRST','RFST','GBST']
  
  # фомируем заготовку для результитрующих данных
  features_second_meta = pd.DataFrame(index=dict_datasets['features_first_meta_stat'].index)

  # указываем путь к директории в которой находятся обученные base модели
  directory = "models/firstmeta/"
  # Получаем список моделей в директории
  list_models = os.listdir(directory)

  # формируем метапризнаки данных torow

  for model in space_models:
    # для текущего подбпространства формируем список предобученных моделей 
    list_first_meta_models = sorted([x for x in list_models if re.search(model, x)])
    # выберем неободимый dataset
    choose_dataset_name = lambda torow, stat, model: torow if model[-2:] == 'TR' else stat
    dataset_name = choose_dataset_name('features_first_meta_torow','features_first_meta_stat',model)

    # делаем предсказание на каждой предобученной модели
    for name_model in list_first_meta_models:
      # добавим индекацию процесса
      clear_output()
      print('Current name_model: ', name_model)
      # загружаем предобученную модель
      model_classifier = joblib.load('models/firstmeta/'+name_model)
      # загружаем list_best_features
      list_best_features = joblib.load('models/firstmeta/list_best_features/'+'list_best_features_'+name_model[:-7]+'.joblib')
      # запоняем dataframe мета признаками 
      features_second_meta[name_model[:-7]] = model_classifier.predict_proba(dict_datasets[dataset_name][list_best_features].to_numpy())[:,1]

  return features_second_meta


# ФУНКЦИЯ best_model

# напишем функцию для преобразования данных от метапризнаков second metamodels
# до предсказания результирующей метамодели

def best_model(data):
  # загрузим загрузим модель показавшую наилучший результат
  best_classifier = joblib.load('models/secondmeta/best_model.joblib')
  # загрузим selector для лучшей модели
  best_selector = joblib.load('models/secondmeta/selector/best_selector.joblib')
  list_best_features = best_selector.get_feature_names_out()

  # формируем данные для предсказания
  MX_data = data[list_best_features].to_numpy()

  # для метрик ROC AUC делаем предсказание модели в виде вероятности
  my_pred_proba = best_classifier.predict_proba(MX_data)[:,1]

  dataframe = pd.DataFrame(
    data=np.array([data.index,my_pred_proba]).transpose(),
    columns=['id','score'])
  dataframe['id'] = dataframe['id'].astype('int')

  dataframe.to_csv('prediction/prediction.csv',index=False)
  

  return dataframe


# ФУНКЦИЯ blendingClassifier

def blendingClassifier(data):
  # выполним преобразования для предсказания на базовых моделях 
  base_model_data = to_base_model(data)
  del data
  gc.collect()

  # выполним преобразования для предсказания на метамоделях первого порядка 
  first_meta_data = to_first_meta(base_model_data)
  del base_model_data
  gc.collect()

  # выполним преобразования для предсказания на метамоделях второго порядка 
  second_meta_data = to_second_meta(first_meta_data)
  del first_meta_data

  # выполним предсказание на результирующей метамодели 
  gc.collect()
  prediction = best_model(second_meta_data)
  
  # вывод сообщения с директорией сохранненого файла
  print('prediction saved to: prediction/prediction.csv')
  return prediction


