<center> <img src = "Images/logo.png" alt="drawing" style="width:400px;">

<center> <span style="background-size: 600px;background:White;color:RED;font-size: 60px;font-family: Comic Sans MS">Кредитный скоринг Альфа банка</span>

## Оглавление
1. [Описание проекта](#описание-проекта)
2. [Описание данных](#описание-данных)
3. [Принятые метрики](#принятые-метрики)
4. [Постановка задачи в рамках EDA](#постановка-задачи-в-рамках-eda)
5. [Постановка задачи в рамках Machine Learning](#постановка-задачи-в-рамках-machine-learning)
6. [Выводы](#выводы)
7. [Структура проекта](#структура-проекта)
8. [Установка проекта](#установка-проекта)
9. [Используемые зависимости](#используемые-зависимости)
10. [Авторы](#авторы)


## Описание проекта
Кредитный скоринг – важнейшая банковская задача. Стандартным подходом к ее решению   
является построение классических моделей машинного обучения, таких как логистическая   
регрессия и градиентный бустинг, на табличных данных, в том числе используя агрегации  
от каких-нибудь последовательных данных, например, транзакционных историй клиентов.   
Альтернативный подход заключается в использовании последовательных данных “как есть”,   
подавая их на вход рекуррентной нейронной сети.

В этом соревновании участникам предлагается решить задачу кредитного скоринга клиентов   
Альфа-Банка, используя только данные кредитных историй. [Источник](https://www.kaggle.com/competitions/alfa-bank-pd-credit-history)

## Описание данных

Датасет соревнования устроен таким образом, что кредиты для тренировочной выборки взяты   
за период в М месяцев, а кредиты для тестовой выборки взяты за последующие K месяцев.

Каждая запись кредитной истории содержит самую разнообразную информацию о прошлом кредите   
клиента, например, сумму, отношение клиента к кредиту, дату открытия и закрытия, информацию   
о просрочках по платежам и др. Все публикуемые данные тщательно анонимизированы.

Целевая переменная – бинарная величина, принимающая значения 0 и 1, где 1 соответствует   
дефолту клиента по кредиту.


## Принятые метрики

Метрика соревнования – $ROC$ $AUC$. Подробнее про метрику можно почитать, например, [здесь](https://dyakonov.org/2017/07/28/auc-roc-площадь-под-кривой-ошибок/).

## Постановка задачи в рамках $EDA$

Данные о клиентах представлены в виде заявок, номера которых соотностятся с датами подачи заявки.  
Большему номеру соответствует более поздняя дата заявки.

Разделим процесс преобразования признаков на три части: 

1. Разделим признаки на под пространства.

2. Преобразуем кредитные операции клиента в его признаки, сохранив последовательность операций.  
Такие методы кодирования как $Ordinal$ $Encoding$, $OneHot$ $Encoding$ не подойдут, так как уничтожится   
информация о последовательности операций.

3. Преобразование признаков как дискренного ряда.  
Для каждого признака в "границах" одного "id" рассчитать статистические показатели ряда, такие как:  
   - среднее значение ряда (математическое ожидание);  
   - средне гармоническое значение ряда;   
   - стандартное отклонение;  
   - минимальное значение;  
   - 25% квантиль;  
   - 50% квантиль;  
   - 75% квантиль;  
   - максимальное значение;  
   - мода ряда (среднее значение мод);  
   - частота появления моды (средненего значения мод);
   - и т.д. 

## Постановка задачи в рамках $Machine$ $Learning$

1. Для решения задачи построем блендинг моделей. 

2. В качестве базовых и метамоделей рассмотрим следующие классические модели классификации:
    - $linearmodel.LogisticRegression$ (Логистическая регрессия);
    - $RandomForestClassifier$ (Деревья решений);
    - $HistGradientBoostingClassifier$ (Градиентный бустинг).

3. В результате, преобразования данных было получено два пространства признаков ($torow$ и $stat$ признаки),  
состоящих из 6 подпространств:
    - $date$ $features$;
    - $late$ $payments$ $features$; 
    - $credit$ $features$;
    - $relative$ $features$;
    - $payments$ $features$;
    - $service$ $features$.

4. На первом этапе построения потроения блендинга, сфокусируем обучение базовых моделей,   
на каждом подпространстве в отдельности друго от друга.  

5. На втором этапе построения блендинга, обучим несколько групп метамоделей.   
Первая группа метамоделей в качестве метапризнаков использует предсказания базовых моделей,    
обученных на пространстве признаков $torow$.  
Вторая группа метамоделей в качестве метапризнаков использует предсказания базовых моделей,    
обученных на пространстве признаков $stat$.

6. На третьем этапе построения блендинга метамодель обучится на метапризнаках пространства 
$torow$ и $stat$.

<center> <img src = "Images/Blending.jpg" alt="drawing" style="width:1400px;">


## Выводы

<span style="color:Blue">

Качество полученной модели блендинга по метрике $ROC$ $AUC$:
- на тренировочном наборе: $ROC$ $AUC = 0.757$;
- на валидационном наборе: $ROC$ $AUC = 0.765$;
- на отложенном наборе: $ROC$ $AUC = 0.756$;
- на соревновательном ($kaggle$) наборе: $ROC$ $AUC = 0.743$;

<center> <img src = "Images/Kaggle.png" alt="drawing" style="width:1400px;">

**Мероприятия по возможному улучшению качества модели:**

1. Проанализировать данные на наличие выбросов и нерепрезентотивных данных. Скорее всего   
есть клиенты, данные которых "вводят в заблуждение" модель.  

2. Проанализировать влияние выбранных статистических характеристик, формирующих $dataset$ $stat$.  
Возможно, использование некоторых характеристики наоборот ухудшает качество модели.   
Возможно, есть иные характеристики числовых рядов которые были не учтены.  

3. Проанализирровать влияние размера обучающего набора на каждом этапе на качество результирующей модели.   
Возможно имеет смысл большую часть данных "потратить" на обучение базовых моделей, возможно наоборот.  

4. Проанализировать влияние подпространств признаков на результаты $first$ $metamodels$.  
Некоторые подпространства показали низкое качество на базовых моделеях, Возможно имеет смысл    
не использовать все подпространства признаков или разделить пространство признаков не на 6 выбранных     
подпространств, а с помощью методов класстеризации.   

5. Изучить вопрос выбранных базовых моделей. Возможно, имеет смысл сменить их или использовать не все.  
В проссе обучения на первом этапе, модель логистической регресии показала наихудшие результаты.

6. Проанализировать подбор гиперпараметров каждой модели. Сменить диапозоны параметров,  
выбрать дополнительные гиперпермараметры и т.д.

7. Сменить критерии "лучших" моделей. Возможно, имеет смысл использовать только один критерий для  
определения "лучших" моделей. В построенном блендинге на каждом этапе использовались все "лучшие"   
модели с предыдущего этапа. Возможно, есть смысл показывать метамодели результаты не от всех "лучших" моделей.

8. Увеличить или уменьшить глубину блендинга.  

9. Использовать не блендинг, а стекинг моделей.  

10. Использовать не блендинг, а нейронную сеть.  

</span>

## Структура проекта
* Images - папка с используемыми изображениями;
* [Notebook_v13.ipynb](https://github.com/Wiruto/Credit-scoring-of-Alfa-Bank/blob/master/Notebook_v13.ipynb) - jupyter-ноутбук, содержащий основной код проекта;
* README.md - краткое описание проекта;
* optuna_studies.db - результаты оптимизации моделей;
* requirements.txt - файл используемых библиотек и зависимостей.

## Используемые зависимости

Используемые библиотеки и зависимости представлены в файле: [requirements.txt](https://github.com/Wiruto/Credit-scoring-of-Alfa-Bank/blob/master/requirements.txt)

## Установка проекта

```
git clone https://github.com/Wiruto/Credit-scoring-of-Alfa-Bank
```
## Авторы

* [Колобов Виктор Валерьевич](https://github.com/Wiruto)


