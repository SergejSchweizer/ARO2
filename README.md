# ARO - predicting target based on sensoric data

Author: Sergej Schweizer

12.10.2021

---

## TOC:

# 1. Install packages


```python
#!pip install statsmodels mlxtend tensorflow_data_validation tensorflow xlrd seaborn
```

# 2. Import
---


```python
import pandas as pd
import numpy as np

import re
from typing import AnyStr, Callable, Tuple

import seaborn as sns 
import matplotlib.pyplot as plt
import datetime
import xlrd
import itertools

import tensorflow as tf
import tensorflow_data_validation as tfdv

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler, KBinsDiscretizer
from sklearn.feature_selection import RFE, SelectKBest, SelectFromModel, chi2, f_classif

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import statsmodels.api as sm
```

# 3. Load
---


```python
!head md_target_dataset.csv
```

    index;groups;target
    1;0;1.233766234
    2;0;2.467532468
    3;0;3.701298701
    4;0;4.935064935
    5;0;6.168831169
    6;0;7.402597403
    7;0;8.636363636
    8;0;9.87012987
    9;0;11.1038961



```python
df_raw_predictors = pd.read_csv(
    'md_raw_dataset.csv',
    sep=';',
    #index_col='Unnamed: 0'  # we  will need this column as
)
df_raw_predictors.rename({'Unnamed: 0': 'index_predictors'}, axis=1, inplace=True)

df_raw_target = pd.read_csv(
    'md_target_dataset.csv',
    sep=';',
    index_col='index',
    #usecols=['index','target'],
)
df_raw_predictors.shape, df_raw_target.shape
```




    ((9592, 35), (9589, 2))




```python
df_raw_predictors.head(5)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index_predictors</th>
      <th>when</th>
      <th>super_hero_group</th>
      <th>tracking</th>
      <th>place</th>
      <th>tracking_times</th>
      <th>crystal_type</th>
      <th>Unnamed: 7</th>
      <th>human_behavior_report</th>
      <th>human_measure</th>
      <th>...</th>
      <th>subprocess1_end</th>
      <th>reported_on_tower</th>
      <th>opened</th>
      <th>chemical_x</th>
      <th>raw_kryptonite</th>
      <th>argon</th>
      <th>pure_seastone</th>
      <th>crystal_supergroup</th>
      <th>Cycle</th>
      <th>groups</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>09/07/2020</td>
      <td>D</td>
      <td>84921</td>
      <td>1</td>
      <td>1</td>
      <td>group 27</td>
      <td>2</td>
      <td>3</td>
      <td>650</td>
      <td>...</td>
      <td>09/07/2020 13:27</td>
      <td>09/07/2020 13:37</td>
      <td>44021.58091</td>
      <td>15.850000</td>
      <td>693.0</td>
      <td>0.0</td>
      <td>49.51</td>
      <td>0</td>
      <td>2ª</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>09/07/2020</td>
      <td>D</td>
      <td>84941</td>
      <td>1</td>
      <td>1</td>
      <td>group 56</td>
      <td>1</td>
      <td>4</td>
      <td>700</td>
      <td>...</td>
      <td>09/07/2020 15:38</td>
      <td>09/07/2020 15:53</td>
      <td>44021.6737</td>
      <td>21.966667</td>
      <td>3570.0</td>
      <td>0.0</td>
      <td>99.94</td>
      <td>0</td>
      <td>2ª</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>09/07/2020</td>
      <td>D</td>
      <td>84951</td>
      <td>1</td>
      <td>1</td>
      <td>group 56</td>
      <td>2</td>
      <td>4</td>
      <td>800</td>
      <td>...</td>
      <td>09/07/2020 16:41</td>
      <td>09/07/2020 16:54</td>
      <td>44021.70867</td>
      <td>21.166667</td>
      <td>7950.0</td>
      <td>0.0</td>
      <td>91.49</td>
      <td>0</td>
      <td>2ª</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>09/07/2020</td>
      <td>D</td>
      <td>84971</td>
      <td>1</td>
      <td>1</td>
      <td>group 56</td>
      <td>7</td>
      <td>3</td>
      <td>700</td>
      <td>...</td>
      <td>09/07/2020 18:47</td>
      <td>09/07/2020 18:55</td>
      <td>09/07/2020 19:02</td>
      <td>15.250000</td>
      <td>807.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>2ª</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>09/07/2020</td>
      <td>D</td>
      <td>84981</td>
      <td>1</td>
      <td>1</td>
      <td>group 27</td>
      <td>17</td>
      <td>3</td>
      <td>700</td>
      <td>...</td>
      <td>09/07/2020 19:37</td>
      <td>09/07/2020 19:47</td>
      <td>09/07/2020 20:20</td>
      <td>20.566667</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>509.19</td>
      <td>0</td>
      <td>2ª</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>



# 4. EDA


```python
# Generate dataset statistics
comb_stats = tfdv.generate_statistics_from_dataframe(df_raw_predictors)
# Compare training with evaluation
tfdv.visualize_statistics(comb_stats)
```
![](https://github.com/SergejSchweizer/ARO2/blob/main/images/stats1.png?raw=true)





```python
# Generate dataset statistics
comb_stats = tfdv.generate_statistics_from_dataframe(df_raw_target)
# Compare training with evaluation
tfdv.visualize_statistics(comb_stats)
```
![](https://github.com/SergejSchweizer/ARO2/blob/main/images/stats2.png?raw=true)


# 5. Conclusions

* 1. groups columns in predictors and target dataframes seems to be similar, therefore we can try to map target to predictors
* 2. etherium_before_start has nearly 79% of missing values (NaN), this column can be deleted
* 3. multiple date columns has missing values, can we do imputation ? (mean, average ?)



```python
# Also we know now what kind of data we have !!!
DATE_COLUMNS = ['when', 'expected_start', 'start_process', 'start_subprocess1', 
                'start_critical_subprocess1','predicted_process_end',
                'process_end','subprocess1_end','reported_on_tower','opened']

CATEGORICAL_COLUMS = ['super_hero_group','crystal_type', 'crystal_supergroup', 'Cycle']


NUMERICAL_COLUMNS = [
    'tracking','place','tracking_times','Unnamed: 7','human_behavior_report', 'human_measure',
    'crystal_weight', 'expected_factor_x', 'previous_factor_x', 'first_factor_x',
    'expected_final_factor_x', 'final_factor_x', 'previous_adamantium', 'Unnamed: 17', 
    'chemical_x', 'raw_kryptonite', 'argon', 'pure_seastone' , 'groups', 'index_predictors']

ABSOLETE_COLUMNS = ['etherium_before_start']

TARGET_COLUMN = ['target']
```

# 6. Merge target with predictors
---

Groups within predictors and target seems to have same structure, only the groupid is different.


```python
def merge_based_on_group_occurences(df1: pd.DataFrame, df2: pd.DataFrame)-> pd.DataFrame:
    '''
    merge group ids to each other based on number of occurences
    '''
    
    # map group ids according to occurences
    df_temp = pd.merge(
        df1['groups'].value_counts().to_frame('occurences').reset_index(),
        df2['groups'].value_counts().to_frame('occurences').reset_index(),
        suffixes=('_predictor','_target'),
        on='occurences',
        how='inner'

    )
    
    df_temp.columns = ['group_id_predictors', 'occurences', 'group_id_target']
    return df_temp

df_groups = merge_based_on_group_occurences(
    df_raw_predictors,
    df_raw_target
)

df_groups.sort_values('group_id_predictors').head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group_id_predictors</th>
      <th>occurences</th>
      <th>group_id_target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>154</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.0</td>
      <td>149</td>
      <td>37</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.0</td>
      <td>149</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.0</td>
      <td>149</td>
      <td>45</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2.0</td>
      <td>144</td>
      <td>39</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2.0</td>
      <td>144</td>
      <td>34</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2.0</td>
      <td>144</td>
      <td>8</td>
    </tr>
    <tr>
      <th>150</th>
      <td>3.0</td>
      <td>81</td>
      <td>40</td>
    </tr>
    <tr>
      <th>144</th>
      <td>4.0</td>
      <td>103</td>
      <td>41</td>
    </tr>
    <tr>
      <th>133</th>
      <td>5.0</td>
      <td>114</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>



We just need to find most nearest group_id with respect to occurences


```python
def filter_group_ids_based_distance(df_groups: pd.DataFrame)-> pd.DataFrame:
    '''
    In case we have multiple target group ids,  find the closest one
    Assumption: same/similiar group ids = same groups in both dataframes
    '''
   
    df_groups_filterd = pd.DataFrame()

    for _, df_temp in df_groups.groupby('group_id_predictors'):
        idx = df_temp.iloc[0,0]
        ocs = df_temp.iloc[0,1]
        idx_target = df_groups.iloc[df_temp['group_id_target'].sub(idx).abs().idxmin()]['group_id_target']

        # build new dataframe with filtered group ids
        df_groups_filterd = df_groups_filterd.append(
            {'group_id_predictors': idx,
             'occurences': ocs,
             'group_id_target':idx_target},
            ignore_index=True).astype(int)

    return df_groups_filterd

df_groups_filter = filter_group_ids_based_distance(df_groups)
df_groups_filter.head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group_id_predictors</th>
      <th>group_id_target</th>
      <th>occurences</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>154</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>149</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>8</td>
      <td>144</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>40</td>
      <td>81</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>41</td>
      <td>103</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>30</td>
      <td>114</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>2</td>
      <td>112</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>32</td>
      <td>128</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>12</td>
      <td>134</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>3</td>
      <td>133</td>
    </tr>
  </tbody>
</table>
</div>



Now we concat (axis=1) predictors with target


```python
def concat_predictors_and_target(
    df_groups: pd.DataFrame, 
    df_predictors: pd.DataFrame, 
    df_target: pd.DataFrame)-> pd.DataFrame:
    '''
    concat predictors and target according to groups mapping
    '''

    df = pd.DataFrame()

    for idx, row in df_groups.iterrows():
        df = df.append(pd.concat(
            [
                df_predictors[df_predictors['groups'] == row['group_id_predictors']].reset_index(drop=True),
                df_target[df_target['groups'] == row['group_id_target']]['target'].reset_index(drop=True),
            ],
            ignore_index=True,
            axis=1,
            )
        )
    df.columns = df_predictors.columns.tolist() + ['target']
    
    return df



df = concat_predictors_and_target(
    df_groups_filter,
    df_raw_predictors,
    df_raw_target
)
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index_predictors</th>
      <th>when</th>
      <th>super_hero_group</th>
      <th>tracking</th>
      <th>place</th>
      <th>tracking_times</th>
      <th>crystal_type</th>
      <th>Unnamed: 7</th>
      <th>human_behavior_report</th>
      <th>human_measure</th>
      <th>...</th>
      <th>reported_on_tower</th>
      <th>opened</th>
      <th>chemical_x</th>
      <th>raw_kryptonite</th>
      <th>argon</th>
      <th>pure_seastone</th>
      <th>crystal_supergroup</th>
      <th>Cycle</th>
      <th>groups</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>09/07/2020</td>
      <td>D</td>
      <td>84921</td>
      <td>1</td>
      <td>1</td>
      <td>group 27</td>
      <td>2</td>
      <td>3</td>
      <td>650</td>
      <td>...</td>
      <td>09/07/2020 13:37</td>
      <td>44021.58091</td>
      <td>15.850000</td>
      <td>693.0</td>
      <td>0.000000</td>
      <td>49.51</td>
      <td>0</td>
      <td>2ª</td>
      <td>0.0</td>
      <td>1.233766</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>09/07/2020</td>
      <td>D</td>
      <td>84941</td>
      <td>1</td>
      <td>1</td>
      <td>group 56</td>
      <td>1</td>
      <td>4</td>
      <td>700</td>
      <td>...</td>
      <td>09/07/2020 15:53</td>
      <td>44021.6737</td>
      <td>21.966667</td>
      <td>3570.0</td>
      <td>0.000000</td>
      <td>99.94</td>
      <td>0</td>
      <td>2ª</td>
      <td>0.0</td>
      <td>2.467532</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>09/07/2020</td>
      <td>D</td>
      <td>84951</td>
      <td>1</td>
      <td>1</td>
      <td>group 56</td>
      <td>2</td>
      <td>4</td>
      <td>800</td>
      <td>...</td>
      <td>09/07/2020 16:54</td>
      <td>44021.70867</td>
      <td>21.166667</td>
      <td>7950.0</td>
      <td>0.000000</td>
      <td>91.49</td>
      <td>0</td>
      <td>2ª</td>
      <td>0.0</td>
      <td>3.701299</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>09/07/2020</td>
      <td>D</td>
      <td>84971</td>
      <td>1</td>
      <td>1</td>
      <td>group 56</td>
      <td>7</td>
      <td>3</td>
      <td>700</td>
      <td>...</td>
      <td>09/07/2020 18:55</td>
      <td>09/07/2020 19:02</td>
      <td>15.250000</td>
      <td>807.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0</td>
      <td>2ª</td>
      <td>0.0</td>
      <td>4.935065</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>09/07/2020</td>
      <td>D</td>
      <td>84981</td>
      <td>1</td>
      <td>1</td>
      <td>group 27</td>
      <td>17</td>
      <td>3</td>
      <td>700</td>
      <td>...</td>
      <td>09/07/2020 19:47</td>
      <td>09/07/2020 20:20</td>
      <td>20.566667</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>509.19</td>
      <td>0</td>
      <td>2ª</td>
      <td>0.0</td>
      <td>6.168831</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>131</th>
      <td>131</td>
      <td>27/03/2021</td>
      <td>A</td>
      <td>133831</td>
      <td>1</td>
      <td>1</td>
      <td>group 22</td>
      <td>9</td>
      <td>3</td>
      <td>690</td>
      <td>...</td>
      <td>27/03/2021 21:39</td>
      <td>27/03/2021 21:59</td>
      <td>25.900000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>88.61</td>
      <td>0</td>
      <td>1ª</td>
      <td>77.0</td>
      <td>223.235294</td>
    </tr>
    <tr>
      <th>132</th>
      <td>132</td>
      <td>28/03/2021</td>
      <td>A</td>
      <td>133841</td>
      <td>1</td>
      <td>2</td>
      <td>group 7</td>
      <td>20</td>
      <td>3</td>
      <td>480</td>
      <td>...</td>
      <td>28/03/2021 00:56</td>
      <td>28/03/2021 01:03</td>
      <td>37.166667</td>
      <td>NaN</td>
      <td>302.248697</td>
      <td>486.79</td>
      <td>0</td>
      <td>1ª</td>
      <td>77.0</td>
      <td>224.926471</td>
    </tr>
    <tr>
      <th>133</th>
      <td>133</td>
      <td>28/03/2021</td>
      <td>B</td>
      <td>133941</td>
      <td>1</td>
      <td>1</td>
      <td>group 80</td>
      <td>1</td>
      <td>1</td>
      <td>600</td>
      <td>...</td>
      <td>28/03/2021 14:18</td>
      <td>28/03/2021 14:33</td>
      <td>39.283333</td>
      <td>NaN</td>
      <td>787.991333</td>
      <td>741.17</td>
      <td>0</td>
      <td>1ª</td>
      <td>77.0</td>
      <td>226.617647</td>
    </tr>
    <tr>
      <th>134</th>
      <td>134</td>
      <td>28/03/2021</td>
      <td>B</td>
      <td>133981</td>
      <td>1</td>
      <td>1</td>
      <td>group 5</td>
      <td>14</td>
      <td>2</td>
      <td>500</td>
      <td>...</td>
      <td>28/03/2021 18:44</td>
      <td>28/03/2021 18:56</td>
      <td>44.216667</td>
      <td>NaN</td>
      <td>395.447094</td>
      <td>1115.60</td>
      <td>0</td>
      <td>1ª</td>
      <td>77.0</td>
      <td>228.308823</td>
    </tr>
    <tr>
      <th>135</th>
      <td>135</td>
      <td>29/03/2021</td>
      <td>C</td>
      <td>134021</td>
      <td>1</td>
      <td>1</td>
      <td>group 5</td>
      <td>16</td>
      <td>2</td>
      <td>800</td>
      <td>...</td>
      <td>29/03/2021 01:51</td>
      <td>29/03/2021 02:14</td>
      <td>52.666667</td>
      <td>1017.0</td>
      <td>754.424223</td>
      <td>888.46</td>
      <td>0</td>
      <td>1ª</td>
      <td>77.0</td>
      <td>230.000000</td>
    </tr>
  </tbody>
</table>
<p>9203 rows × 36 columns</p>
</div>



# 7. Exprolative Data Analyis
---
We begin our EDA with the target variable, key insights we should gain:
* what kind of distribution is it
* how many "mechanisms" have influence on the target variable (how many distributions)


```python
ax = plt.subplots(figsize=(10,7))
sns.scatterplot(
    x='index_predictors',
    y='target',
    data=df,
    hue='groups',   
)
```




    <AxesSubplot:xlabel='index_predictors', ylabel='target'>




    
![png](images/output_22_1.png)
    


* We have multiple linear relations of target and index (former "unknown: 0") of the predictors (< 175)
* Intercept is always zero
* coeficients may differ regardless to the groups variable
* Our task is to predict the target variable, !!!
* We need to find variables which explain the slope of linear regression

## 7.1 Analyze predictors
---

We begin with date columns



```python
df[DATE_COLUMNS]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>when</th>
      <th>expected_start</th>
      <th>start_process</th>
      <th>start_subprocess1</th>
      <th>start_critical_subprocess1</th>
      <th>predicted_process_end</th>
      <th>process_end</th>
      <th>subprocess1_end</th>
      <th>reported_on_tower</th>
      <th>opened</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>09/07/2020</td>
      <td>09/07/2020 13:10</td>
      <td>09/07/2020 13:08</td>
      <td>09/07/2020 13:11</td>
      <td>09/07/2020 13:13</td>
      <td>09/07/2020 13:41</td>
      <td>09/07/2020 13:28</td>
      <td>09/07/2020 13:27</td>
      <td>09/07/2020 13:37</td>
      <td>44021.58091</td>
    </tr>
    <tr>
      <th>1</th>
      <td>09/07/2020</td>
      <td>09/07/2020 15:08</td>
      <td>09/07/2020 15:11</td>
      <td>09/07/2020 15:16</td>
      <td>09/07/2020 15:18</td>
      <td>09/07/2020 15:49</td>
      <td>09/07/2020 15:39</td>
      <td>09/07/2020 15:38</td>
      <td>09/07/2020 15:53</td>
      <td>44021.6737</td>
    </tr>
    <tr>
      <th>2</th>
      <td>09/07/2020</td>
      <td>09/07/2020 16:15</td>
      <td>09/07/2020 16:16</td>
      <td>09/07/2020 16:20</td>
      <td>09/07/2020 16:22</td>
      <td>09/07/2020 16:54</td>
      <td>09/07/2020 16:42</td>
      <td>09/07/2020 16:41</td>
      <td>09/07/2020 16:54</td>
      <td>44021.70867</td>
    </tr>
    <tr>
      <th>3</th>
      <td>09/07/2020</td>
      <td>09/07/2020 18:22</td>
      <td>09/07/2020 18:24</td>
      <td>09/07/2020 18:31</td>
      <td>09/07/2020 18:33</td>
      <td>09/07/2020 19:02</td>
      <td>09/07/2020 18:47</td>
      <td>09/07/2020 18:47</td>
      <td>09/07/2020 18:55</td>
      <td>09/07/2020 19:02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>09/07/2020</td>
      <td>09/07/2020 19:14</td>
      <td>09/07/2020 19:12</td>
      <td>09/07/2020 19:16</td>
      <td>NaN</td>
      <td>09/07/2020 20:13</td>
      <td>09/07/2020 19:37</td>
      <td>09/07/2020 19:37</td>
      <td>09/07/2020 19:47</td>
      <td>09/07/2020 20:20</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>131</th>
      <td>27/03/2021</td>
      <td>27/03/2021 20:45</td>
      <td>27/03/2021 20:57</td>
      <td>27/03/2021 21:02</td>
      <td>27/03/2021 21:15</td>
      <td>27/03/2021 21:11</td>
      <td>27/03/2021 21:28</td>
      <td>27/03/2021 21:27</td>
      <td>27/03/2021 21:39</td>
      <td>27/03/2021 21:59</td>
    </tr>
    <tr>
      <th>132</th>
      <td>28/03/2021</td>
      <td>27/03/2021 22:20</td>
      <td>28/03/2021 00:02</td>
      <td>28/03/2021 00:10</td>
      <td>NaN</td>
      <td>28/03/2021 01:00</td>
      <td>28/03/2021 00:47</td>
      <td>28/03/2021 00:47</td>
      <td>28/03/2021 00:56</td>
      <td>28/03/2021 01:03</td>
    </tr>
    <tr>
      <th>133</th>
      <td>28/03/2021</td>
      <td>28/03/2021 13:30</td>
      <td>28/03/2021 13:25</td>
      <td>28/03/2021 13:29</td>
      <td>28/03/2021 13:58</td>
      <td>28/03/2021 14:13</td>
      <td>28/03/2021 14:09</td>
      <td>28/03/2021 14:09</td>
      <td>28/03/2021 14:18</td>
      <td>28/03/2021 14:33</td>
    </tr>
    <tr>
      <th>134</th>
      <td>28/03/2021</td>
      <td>28/03/2021 17:29</td>
      <td>28/03/2021 17:41</td>
      <td>28/03/2021 17:45</td>
      <td>NaN</td>
      <td>28/03/2021 18:27</td>
      <td>28/03/2021 18:30</td>
      <td>28/03/2021 18:29</td>
      <td>28/03/2021 18:44</td>
      <td>28/03/2021 18:56</td>
    </tr>
    <tr>
      <th>135</th>
      <td>29/03/2021</td>
      <td>29/03/2021 00:40</td>
      <td>29/03/2021 00:41</td>
      <td>29/03/2021 00:44</td>
      <td>29/03/2021 00:48</td>
      <td>29/03/2021 01:45</td>
      <td>29/03/2021 01:39</td>
      <td>29/03/2021 01:37</td>
      <td>29/03/2021 01:51</td>
      <td>29/03/2021 02:14</td>
    </tr>
  </tbody>
</table>
<p>9203 rows × 10 columns</p>
</div>



* We have Nans, different date formats (this can be seen also from visualize_statistics)
* Therefore we create helper functions (they should be moved to helper.py) for formating these coluns to generic datetime value (and type)


```python
# Functions for timestamp formating

def exel_timestamp_to_datetime(value: str)-> str:
    '''
    format exel timestamp to datetime str representations
    '''
    
    if re.match(r'\d.\d', str(value)):
        return xlrd.xldate_as_datetime(float(value), 0).strftime('%Y-%m-%d %H:%M:%S')
    else:
        return value

    
def wrong_timestamp_to_datetime(value: str)-> str:
    '''
    format wrong timestamp to datetime str representations
    '''
    
    if re.match(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}', str(value)):
        #print('bla')
        return datetime.datetime.strptime(value, '%y/%m/%Y %H:%M').strftime('%Y-%m-%d %H:%M:%S')
    else:
        return value

def date_to_datetime(value: str)-> str:
    '''
    format date to datetime str representations
    '''
    #print(value)
    if re.match(r'^\d{2}/\d{2}/\d{4}$', str(value)):
        return datetime.datetime.strptime(value, '%y/%m/%Y').strftime('%Y-%m-%d 00:00')
    else:
        return value

def convert_to_posix_timestamp(x):
    """Convert date objects to integers"""
    return x.timestamp()
```

### 7.1.1 Format date columns
---


```python
def format_date_columns(df :pd.DataFrame, date_columns: list)-> pd.DataFrame:
    '''
    format date columns
    '''

    df_formated = pd.DataFrame()

    for col in date_columns:
        # format data to same format
        df_formated[col] = df[col].apply(lambda x: exel_timestamp_to_datetime(x))
        df_formated[col] = df[col].apply(lambda x: wrong_timestamp_to_datetime(x))
        df_formated[col] = df[col].apply(lambda x: date_to_datetime(x))

        # format to datetime type
        df_formated[col] = pd.to_datetime(df_formated[col], errors='coerce')

        # fill all nans with columns medians
        df_formated[col].fillna(df_formated[col].median(), inplace=True)
        
    return df_formated
```

### 7.1.2 Generate new features based on date columns
---


```python
def generate_new_date_features(df: pd.DataFrame, date_columns: list)-> pd.DataFrame:
    '''
    generate new data features
    '''
    
    for col in date_columns:
        # split date to different columns
        df[col+'_year'] = df[col].dt.year
        df[col+'_month'] = df[col].dt.month
        #df_formated[col+'_week'] = df_formated[col].dt.week
        df[col+'_day'] = df[col].dt.dayofyear

        # get hour
        if not col == 'when':
            df[col+'_hour'] = df[col].dt.hour
            # get hour
            df[col+'_minutes'] = df[col].dt.minute
        
    return df_formated
    
def generate_new_diff_fieatures(df: pd.DataFrame, date_columns: list)-> pd.DataFrame:
    '''
    generate new features based on date difference
    
    '''

    combination_of_date_columns = list(itertools.combinations(date_columns, 2))

    # compute differences in minutes between ALL datetime columns
    for col1, col2 in combination_of_date_columns:
        column_name = f"{col1}_diff_{col2}" 
        df[column_name] = (df[col1] - df[col2]).astype('<m8[m]').astype(int)
        
    return df


    
def delete_date_columns(df: pd.DataFrame, date_columns : list)-> pd.DataFrame:
    '''
    delete absolte date columns
    '''
    
    df.drop(date_columns, axis=1, inplace=True)
    return df
    
```


```python
df_formated = format_date_columns(df, DATE_COLUMNS)
df_formated = generate_new_date_features(df_formated, DATE_COLUMNS)
df_formated = generate_new_diff_fieatures(df_formated, DATE_COLUMNS)
df_formated = delete_date_columns(df_formated, DATE_COLUMNS)
df_formated.head(5)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>when_year</th>
      <th>when_month</th>
      <th>when_day</th>
      <th>expected_start_year</th>
      <th>expected_start_month</th>
      <th>expected_start_day</th>
      <th>expected_start_hour</th>
      <th>expected_start_minutes</th>
      <th>start_process_year</th>
      <th>start_process_month</th>
      <th>...</th>
      <th>predicted_process_end_diff_process_end</th>
      <th>predicted_process_end_diff_subprocess1_end</th>
      <th>predicted_process_end_diff_reported_on_tower</th>
      <th>predicted_process_end_diff_opened</th>
      <th>process_end_diff_subprocess1_end</th>
      <th>process_end_diff_reported_on_tower</th>
      <th>process_end_diff_opened</th>
      <th>subprocess1_end_diff_reported_on_tower</th>
      <th>subprocess1_end_diff_opened</th>
      <th>reported_on_tower_diff_opened</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020</td>
      <td>7</td>
      <td>183</td>
      <td>2020</td>
      <td>9</td>
      <td>251</td>
      <td>13</td>
      <td>10</td>
      <td>2020</td>
      <td>9</td>
      <td>...</td>
      <td>13</td>
      <td>14</td>
      <td>4</td>
      <td>250815</td>
      <td>1</td>
      <td>-9</td>
      <td>250802</td>
      <td>-10</td>
      <td>250801</td>
      <td>250811</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020</td>
      <td>7</td>
      <td>183</td>
      <td>2020</td>
      <td>9</td>
      <td>251</td>
      <td>15</td>
      <td>8</td>
      <td>2020</td>
      <td>9</td>
      <td>...</td>
      <td>10</td>
      <td>11</td>
      <td>-4</td>
      <td>250943</td>
      <td>1</td>
      <td>-14</td>
      <td>250933</td>
      <td>-15</td>
      <td>250932</td>
      <td>250947</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020</td>
      <td>7</td>
      <td>183</td>
      <td>2020</td>
      <td>9</td>
      <td>251</td>
      <td>16</td>
      <td>15</td>
      <td>2020</td>
      <td>9</td>
      <td>...</td>
      <td>12</td>
      <td>13</td>
      <td>0</td>
      <td>251008</td>
      <td>1</td>
      <td>-12</td>
      <td>250996</td>
      <td>-13</td>
      <td>250995</td>
      <td>251008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020</td>
      <td>7</td>
      <td>183</td>
      <td>2020</td>
      <td>9</td>
      <td>251</td>
      <td>18</td>
      <td>22</td>
      <td>2020</td>
      <td>9</td>
      <td>...</td>
      <td>15</td>
      <td>15</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>-8</td>
      <td>-15</td>
      <td>-8</td>
      <td>-15</td>
      <td>-7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020</td>
      <td>7</td>
      <td>183</td>
      <td>2020</td>
      <td>9</td>
      <td>251</td>
      <td>19</td>
      <td>14</td>
      <td>2020</td>
      <td>9</td>
      <td>...</td>
      <td>36</td>
      <td>36</td>
      <td>26</td>
      <td>-7</td>
      <td>0</td>
      <td>-10</td>
      <td>-43</td>
      <td>-10</td>
      <td>-43</td>
      <td>-33</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 93 columns</p>
</div>



Yes, we have lot of correlated features (e.g Year), they will be filtered due to the feature selection process

### 7.1.3 Format categorical columns
---


```python
fig, axes = plt.subplots(1, 4, figsize=(15,7))
for idx, col in enumerate(CATEGORICAL_COLUMS):
    sns.histplot(x=col, data=df, ax=axes[idx],  alpha=0.65).set_title(col)
```


    
![png](images/output_35_0.png)
    


* We need to delete / impute row with the € Character in super_hero_group
* Super_hero_group, crystal_supergroup and Cycle I would suggest to translate to dummy variables
* In Cycle we need to rename values
* cristal type, I suggest to cut the group prefix and only leave the integer


```python
df['super_hero_group'].value_counts()
```




    B    2246
    D    2011
    A    1943
    C    1915
    G     378
    W     369
    Y     340
    ₢       1
    Name: super_hero_group, dtype: int64



* As we want to preserve as much data as possible, we impute instead of deleting 
* In this case we take the value with highest frequence


```python
df.loc[df['super_hero_group'] == '₢', 'super_hero_group'] = 'B'
```

* Replace non ASCI values in Cycle column


```python
df['Cycle'].value_counts()
```




    1ª     4717
    2ª     4226
    3ª      134
    131     126
    Name: Cycle, dtype: int64




```python
df['Cycle'].replace({'1ª':1, '2ª': 2, '3ª':3}, inplace=True)
```

* Drop 'group' prefix in crystal_type variable


```python
df['crystal_type'].replace({'group ': ''}, regex=True, inplace=True)
# copy crystal_type as int 
df_formated['crystal_type'] = df['crystal_type'].astype(int)
```

### 7.1.4 Create dummy variables
---


```python
DUMMY_COLUMNS = ['super_hero_group','crystal_supergroup', 'Cycle']

def generate_dummy_features(df_from: pd.DataFrame, df_to: pd.DataFrame, columns_list: list)-> pd.DataFrame:
    '''
    translate categorical variables to dummy variables    
    '''
    
    for col in columns_list:
        df_to =  pd.concat(
            [
                pd.get_dummies(df_from[col], prefix=col, drop_first=True),
                df_to
            ],axis=1
        )
        
    return df_to

df_formated = generate_dummy_features(df, df_formated, DUMMY_COLUMNS)
df_formated.head(5)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cycle_2</th>
      <th>Cycle_3</th>
      <th>Cycle_131</th>
      <th>crystal_supergroup_1</th>
      <th>super_hero_group_B</th>
      <th>super_hero_group_C</th>
      <th>super_hero_group_D</th>
      <th>super_hero_group_G</th>
      <th>super_hero_group_W</th>
      <th>super_hero_group_Y</th>
      <th>...</th>
      <th>predicted_process_end_diff_subprocess1_end</th>
      <th>predicted_process_end_diff_reported_on_tower</th>
      <th>predicted_process_end_diff_opened</th>
      <th>process_end_diff_subprocess1_end</th>
      <th>process_end_diff_reported_on_tower</th>
      <th>process_end_diff_opened</th>
      <th>subprocess1_end_diff_reported_on_tower</th>
      <th>subprocess1_end_diff_opened</th>
      <th>reported_on_tower_diff_opened</th>
      <th>crystal_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14</td>
      <td>4</td>
      <td>250815</td>
      <td>1</td>
      <td>-9</td>
      <td>250802</td>
      <td>-10</td>
      <td>250801</td>
      <td>250811</td>
      <td>27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>11</td>
      <td>-4</td>
      <td>250943</td>
      <td>1</td>
      <td>-14</td>
      <td>250933</td>
      <td>-15</td>
      <td>250932</td>
      <td>250947</td>
      <td>56</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>13</td>
      <td>0</td>
      <td>251008</td>
      <td>1</td>
      <td>-12</td>
      <td>250996</td>
      <td>-13</td>
      <td>250995</td>
      <td>251008</td>
      <td>56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>15</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>-8</td>
      <td>-15</td>
      <td>-8</td>
      <td>-15</td>
      <td>-7</td>
      <td>56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>36</td>
      <td>26</td>
      <td>-7</td>
      <td>0</td>
      <td>-10</td>
      <td>-43</td>
      <td>-10</td>
      <td>-43</td>
      <td>-33</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 104 columns</p>
</div>



### 7.1.5 Format numerical columns
---

Insights we gained before:
* crystal_weight has 2.76% nans  -> impute median
* previous_adamantium has 2.87% nans -> impute median
* raw_kryptonite has 47.53 % nans - delete this column ?
* argon has 3.50 % nans and 51 % zeros - are the zeros ok ?
* pure_seastone has 10 % nans - impute median


```python
#fig, axes = plt.subplots(len(NUMERICAL_COLUMNS), 1, figsize=(15,25))
#for  col, ax in zip(NUMERICAL_COLUMNS, axes):
#    sns.histplot(x=col, data=df, ax=ax,  alpha=0.65).set_title(col)

```


```python
for col in NUMERICAL_COLUMNS:
    # change columnt type to int
    df_formated[col] = df[col]
    # fill nans
    df_formated[col] = df_formated[col].fillna(df_formated[col].median())

#drop raw kryptonite
df_formated.drop('raw_kryptonite', axis=1, inplace=True)
```


```python
def generate_new_product_features(df: pd.DataFrame, date_columns: list)-> pd.DataFrame:
    '''
    generate new product features
    
    '''

    combination_of_date_columns = list(itertools.combinations(date_columns, 2))

    # compute differences in minutes between ALL datetime columns
    for col1, col2 in combination_of_date_columns:
        column_name = f"{col1}_mull_{col2}" 
        df[column_name] = df[col1].astype(int) * df[col2].astype(int)
        
    return df

#df_formated = generate_new_product_features(df_formated, df_formated.columns.tolist())
```


```python
df_formated.head(5)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cycle_2</th>
      <th>Cycle_3</th>
      <th>Cycle_131</th>
      <th>crystal_supergroup_1</th>
      <th>super_hero_group_B</th>
      <th>super_hero_group_C</th>
      <th>super_hero_group_D</th>
      <th>super_hero_group_G</th>
      <th>super_hero_group_W</th>
      <th>super_hero_group_Y</th>
      <th>...</th>
      <th>first_factor_x</th>
      <th>expected_final_factor_x</th>
      <th>final_factor_x</th>
      <th>previous_adamantium</th>
      <th>Unnamed: 17</th>
      <th>chemical_x</th>
      <th>argon</th>
      <th>pure_seastone</th>
      <th>groups</th>
      <th>index_predictors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1597.0</td>
      <td>1577.0</td>
      <td>1578.0</td>
      <td>0.0650</td>
      <td>1597.0</td>
      <td>15.850000</td>
      <td>0.0</td>
      <td>49.51</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1595.0</td>
      <td>1565.0</td>
      <td>1572.0</td>
      <td>0.0309</td>
      <td>1595.0</td>
      <td>21.966667</td>
      <td>0.0</td>
      <td>99.94</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1608.0</td>
      <td>1565.0</td>
      <td>1568.0</td>
      <td>0.0510</td>
      <td>1608.0</td>
      <td>21.166667</td>
      <td>0.0</td>
      <td>91.49</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1587.0</td>
      <td>1571.0</td>
      <td>1576.0</td>
      <td>0.0520</td>
      <td>1587.0</td>
      <td>15.250000</td>
      <td>0.0</td>
      <td>357.64</td>
      <td>0.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1599.0</td>
      <td>1579.0</td>
      <td>1590.0</td>
      <td>0.2800</td>
      <td>1599.0</td>
      <td>20.566667</td>
      <td>0.0</td>
      <td>509.19</td>
      <td>0.0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 123 columns</p>
</div>



# 8. Scale predictors
---



```python
# We use standart scaler, zero mean an unit stadart deviation
ss_scaler = StandardScaler()
df_ss = ss_scaler.fit_transform(df_formated)
df_ss = pd.DataFrame(df_ss, columns=df_formated.columns.tolist())
```


```python
#rs_scaler = preprocessing.RobustScaler()
#df_rs = rs_scaler.fit_transform(df)
#df_rs = pd.DataFrame(df_rs, columns=df.columns.tolist())
```


```python
# We check the destributions of our predictors
plt.figure(figsize=(25, 8))
ax = df_ss.boxplot()
#ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
```


    
![png](images/output_55_0.png)
    


# 9. Filter intra-correlations
---


```python
def filter_intra_correlations(
    df_corr: pd.DataFrame,
    corr_thresshold: float = 0.5,
)-> list:

    corr_cols = []
    for column in df_corr.columns.tolist():
        
        if column not in corr_cols:
            cor_target = abs(df_corr[column])
            cor_features = cor_target[ (cor_target>corr_thresshold) & (cor_target<1 )  ]
            #print(cor_features)
            #print(cor_features.index.tolist())
            corr_cols.extend(cor_features.index.tolist())
            #print(corr_cols)
    
    uncorelated_columns = [ x for x in df_corr.columns.tolist() if x not in  corr_cols ]
    #print(uncorelated_columns)#, corr_cols)
    return uncorelated_columns
```

## 9.1 Plot correlation map
---


```python
#correlation map
df_corr = df_ss.corr()

# Filter intracorrelated features
UNCORELATED_COLUMNS = filter_intra_correlations(
    df_corr,
    corr_thresshold=0.5
)

# add target to see corelations with our target variable
df_ss = pd.concat(
    [df_ss, df[TARGET_COLUMN].reset_index(drop=True)],
    axis=1,
    #ignore_index=True
)

# compute correlation matrix
df_corr = df_ss[UNCORELATED_COLUMNS + TARGET_COLUMN].corr()

# plot correlation map
plt.subplots(figsize=(25, 25))
sns.heatmap(
    df_corr, 
    annot=True, 
    linewidths=.5, 
    fmt= '.1f',
    cmap=plt.cm.PuBu
)
df_ss.drop(TARGET_COLUMN, axis=1, inplace=True)
```


    
![png](images/output_59_0.png)
    


What we have:
* The predictor "index_predictors" (former "unknown: 0" variable) has highest impact on target (0.9)
* "Cycle_2" and "groups" predictors have weak (both 0.1) impact on target (in connection with previous statement, we will be able to explain target)
* All other predictors have no influence on target

# 10. Modeling
---
* We will begin now with modeling, Our baseline will be multivariate regression with all predictors
* Then we will just pick up predictors with low p-value and try to analyze the mmodel performance again.
* Finaly stepwise regression analysis will unveil most important predictors (we can compare it with predictors selected by hand)


```python
# split data set in train and test
X_train, X_test, Y_train, Y_test = train_test_split(
    df_ss[UNCORELATED_COLUMNS],
    df[TARGET_COLUMN],
    test_size = 0.1,
    random_state = 123
)
```


```python
model_ols = sm.OLS(
    Y_train['target'].tolist(),
    X_train
).fit()
model_ols.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared (uncentered):</th>      <td>   0.235</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared (uncentered):</th> <td>   0.231</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>          <td>   61.63</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 15 Oct 2021</td> <th>  Prob (F-statistic):</th>           <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>10:24:54</td>     <th>  Log-Likelihood:    </th>          <td> -49843.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  8282</td>      <th>  AIC:               </th>          <td>9.977e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  8241</td>      <th>  BIC:               </th>          <td>1.001e+05</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    41</td>      <th>                     </th>              <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>              <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
                      <td></td>                         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Cycle_2</th>                                   <td>    4.6666</td> <td>    1.136</td> <td>    4.107</td> <td> 0.000</td> <td>    2.439</td> <td>    6.894</td>
</tr>
<tr>
  <th>Cycle_3</th>                                   <td>   -2.2022</td> <td>    1.145</td> <td>   -1.923</td> <td> 0.055</td> <td>   -4.448</td> <td>    0.043</td>
</tr>
<tr>
  <th>Cycle_131</th>                                 <td>   -1.5795</td> <td>    1.150</td> <td>   -1.374</td> <td> 0.169</td> <td>   -3.833</td> <td>    0.674</td>
</tr>
<tr>
  <th>crystal_supergroup_1</th>                      <td>    0.8299</td> <td>    1.454</td> <td>    0.571</td> <td> 0.568</td> <td>   -2.021</td> <td>    3.681</td>
</tr>
<tr>
  <th>super_hero_group_B</th>                        <td>    0.3770</td> <td>    1.416</td> <td>    0.266</td> <td> 0.790</td> <td>   -2.399</td> <td>    3.153</td>
</tr>
<tr>
  <th>super_hero_group_C</th>                        <td>   -0.3632</td> <td>    1.383</td> <td>   -0.263</td> <td> 0.793</td> <td>   -3.074</td> <td>    2.348</td>
</tr>
<tr>
  <th>super_hero_group_D</th>                        <td>    0.2389</td> <td>    1.401</td> <td>    0.171</td> <td> 0.865</td> <td>   -2.507</td> <td>    2.984</td>
</tr>
<tr>
  <th>super_hero_group_G</th>                        <td>    0.1993</td> <td>    1.260</td> <td>    0.158</td> <td> 0.874</td> <td>   -2.270</td> <td>    2.668</td>
</tr>
<tr>
  <th>super_hero_group_W</th>                        <td>   -0.1690</td> <td>    1.261</td> <td>   -0.134</td> <td> 0.893</td> <td>   -2.640</td> <td>    2.302</td>
</tr>
<tr>
  <th>super_hero_group_Y</th>                        <td>    0.6957</td> <td>    1.243</td> <td>    0.560</td> <td> 0.576</td> <td>   -1.740</td> <td>    3.131</td>
</tr>
<tr>
  <th>when_year</th>                                 <td>    1.0825</td> <td>    0.902</td> <td>    1.200</td> <td> 0.230</td> <td>   -0.686</td> <td>    2.851</td>
</tr>
<tr>
  <th>expected_start_month</th>                      <td>    0.3558</td> <td>    1.622</td> <td>    0.219</td> <td> 0.826</td> <td>   -2.824</td> <td>    3.536</td>
</tr>
<tr>
  <th>expected_start_hour</th>                       <td>    0.2332</td> <td>    1.098</td> <td>    0.212</td> <td> 0.832</td> <td>   -1.919</td> <td>    2.386</td>
</tr>
<tr>
  <th>expected_start_minutes</th>                    <td>    0.0142</td> <td>    1.233</td> <td>    0.012</td> <td> 0.991</td> <td>   -2.404</td> <td>    2.432</td>
</tr>
<tr>
  <th>start_process_year</th>                        <td>    1.0825</td> <td>    0.902</td> <td>    1.200</td> <td> 0.230</td> <td>   -0.686</td> <td>    2.851</td>
</tr>
<tr>
  <th>start_subprocess1_minutes</th>                 <td>    0.2206</td> <td>    1.317</td> <td>    0.167</td> <td> 0.867</td> <td>   -2.362</td> <td>    2.803</td>
</tr>
<tr>
  <th>start_critical_subprocess1_minutes</th>        <td>    0.4094</td> <td>    1.202</td> <td>    0.340</td> <td> 0.733</td> <td>   -1.947</td> <td>    2.766</td>
</tr>
<tr>
  <th>predicted_process_end_minutes</th>             <td>    0.9975</td> <td>    1.217</td> <td>    0.819</td> <td> 0.413</td> <td>   -1.389</td> <td>    3.384</td>
</tr>
<tr>
  <th>process_end_minutes</th>                       <td>   -0.0563</td> <td>    1.203</td> <td>   -0.047</td> <td> 0.963</td> <td>   -2.414</td> <td>    2.301</td>
</tr>
<tr>
  <th>reported_on_tower_minutes</th>                 <td>    0.2945</td> <td>    1.122</td> <td>    0.262</td> <td> 0.793</td> <td>   -1.905</td> <td>    2.494</td>
</tr>
<tr>
  <th>opened_minutes</th>                            <td>    0.3059</td> <td>    1.146</td> <td>    0.267</td> <td> 0.790</td> <td>   -1.941</td> <td>    2.553</td>
</tr>
<tr>
  <th>when_diff_expected_start</th>                  <td>   -1.2856</td> <td>    1.838</td> <td>   -0.699</td> <td> 0.484</td> <td>   -4.889</td> <td>    2.318</td>
</tr>
<tr>
  <th>when_diff_start_critical_subprocess1</th>      <td>   -0.1349</td> <td>    1.408</td> <td>   -0.096</td> <td> 0.924</td> <td>   -2.894</td> <td>    2.625</td>
</tr>
<tr>
  <th>expected_start_diff_start_process</th>         <td>    0.1463</td> <td>    1.311</td> <td>    0.112</td> <td> 0.911</td> <td>   -2.423</td> <td>    2.716</td>
</tr>
<tr>
  <th>expected_start_diff_start_subprocess1</th>     <td>    0.4930</td> <td>    1.508</td> <td>    0.327</td> <td> 0.744</td> <td>   -2.464</td> <td>    3.450</td>
</tr>
<tr>
  <th>expected_start_diff_predicted_process_end</th> <td>   -0.4095</td> <td>    1.190</td> <td>   -0.344</td> <td> 0.731</td> <td>   -2.742</td> <td>    1.923</td>
</tr>
<tr>
  <th>start_process_diff_process_end</th>            <td>    1.2723</td> <td>    1.159</td> <td>    1.098</td> <td> 0.272</td> <td>   -1.000</td> <td>    3.544</td>
</tr>
<tr>
  <th>start_process_diff_opened</th>                 <td>   -0.8889</td> <td>    1.560</td> <td>   -0.570</td> <td> 0.569</td> <td>   -3.947</td> <td>    2.169</td>
</tr>
<tr>
  <th>start_subprocess1_diff_reported_on_tower</th>  <td>   -0.3672</td> <td>    1.354</td> <td>   -0.271</td> <td> 0.786</td> <td>   -3.021</td> <td>    2.287</td>
</tr>
<tr>
  <th>crystal_type</th>                              <td>   -1.7649</td> <td>    1.325</td> <td>   -1.332</td> <td> 0.183</td> <td>   -4.362</td> <td>    0.832</td>
</tr>
<tr>
  <th>tracking</th>                                  <td>    1.0466</td> <td>    1.114</td> <td>    0.940</td> <td> 0.347</td> <td>   -1.137</td> <td>    3.230</td>
</tr>
<tr>
  <th>place</th>                                     <td>    1.5919</td> <td>    1.233</td> <td>    1.291</td> <td> 0.197</td> <td>   -0.825</td> <td>    4.009</td>
</tr>
<tr>
  <th>tracking_times</th>                            <td>   -0.9672</td> <td>    1.124</td> <td>   -0.860</td> <td> 0.390</td> <td>   -3.171</td> <td>    1.237</td>
</tr>
<tr>
  <th>Unnamed: 7</th>                                <td>    1.0470</td> <td>    1.102</td> <td>    0.951</td> <td> 0.342</td> <td>   -1.112</td> <td>    3.206</td>
</tr>
<tr>
  <th>human_behavior_report</th>                     <td>    0.8072</td> <td>    1.119</td> <td>    0.721</td> <td> 0.471</td> <td>   -1.387</td> <td>    3.001</td>
</tr>
<tr>
  <th>human_measure</th>                             <td>    0.6480</td> <td>    1.235</td> <td>    0.525</td> <td> 0.600</td> <td>   -1.773</td> <td>    3.069</td>
</tr>
<tr>
  <th>crystal_weight</th>                            <td>   -0.2075</td> <td>    1.206</td> <td>   -0.172</td> <td> 0.863</td> <td>   -2.572</td> <td>    2.157</td>
</tr>
<tr>
  <th>previous_factor_x</th>                         <td>   -0.8908</td> <td>    1.231</td> <td>   -0.724</td> <td> 0.469</td> <td>   -3.304</td> <td>    1.523</td>
</tr>
<tr>
  <th>expected_final_factor_x</th>                   <td>    1.4889</td> <td>    1.461</td> <td>    1.019</td> <td> 0.308</td> <td>   -1.375</td> <td>    4.352</td>
</tr>
<tr>
  <th>chemical_x</th>                                <td>    1.7594</td> <td>    1.277</td> <td>    1.377</td> <td> 0.168</td> <td>   -0.745</td> <td>    4.263</td>
</tr>
<tr>
  <th>groups</th>                                    <td>    3.5896</td> <td>    1.185</td> <td>    3.030</td> <td> 0.002</td> <td>    1.267</td> <td>    5.912</td>
</tr>
<tr>
  <th>index_predictors</th>                          <td>   54.7921</td> <td>    1.104</td> <td>   49.610</td> <td> 0.000</td> <td>   52.627</td> <td>   56.957</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1514.758</td> <th>  Durbin-Watson:     </th> <td>   0.079</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>3923.996</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 1.000</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 5.716</td>  <th>  Cond. No.          </th> <td>2.25e+15</td>
</tr>
</table>



* R² with all predictors is 0.236  (different formula then in skikit.LinearRegression ?)
* We can delete variables with p > 0.05 (alpha for significancy)

## 10.1 Linear Regression with all predictors
---


```python
model = LinearRegression()

model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

df_metrics = pd.DataFrame()
df_metrics = df_metrics.append(pd.Series(
    {
        'Predictors':len(X_test.columns.tolist()),
        'Test data R²':model.score(X_test, Y_test),
        'Test data MSE': mean_squared_error(Y_test, Y_pred),
    },
     name='All Predictors',
    )
)
df_metrics
df_metrics.index=['All Predictors']
df_metrics
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predictors</th>
      <th>Test data MSE</th>
      <th>Test data R²</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>All Predictors</th>
      <td>42.0</td>
      <td>370.880318</td>
      <td>0.893228</td>
    </tr>
  </tbody>
</table>
</div>



## 10.2  Linear Regression with low p value predictors
---


```python
LOW_P_COLUMNS = ['Cycle_2', 'groups', 'index_predictors']

model_ols = sm.OLS(
    Y_train['target'].tolist(),
    X_train[LOW_P_COLUMNS]
).fit()
model_ols.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared (uncentered):</th>      <td>   0.232</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared (uncentered):</th> <td>   0.232</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>          <td>   834.6</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 15 Oct 2021</td> <th>  Prob (F-statistic):</th>           <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>10:24:54</td>     <th>  Log-Likelihood:    </th>          <td> -49856.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  8282</td>      <th>  AIC:               </th>          <td>9.972e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  8279</td>      <th>  BIC:               </th>          <td>9.974e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>              <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>              <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Cycle_2</th>          <td>    4.8989</td> <td>    1.095</td> <td>    4.474</td> <td> 0.000</td> <td>    2.753</td> <td>    7.045</td>
</tr>
<tr>
  <th>groups</th>           <td>    3.2954</td> <td>    1.092</td> <td>    3.018</td> <td> 0.003</td> <td>    1.155</td> <td>    5.436</td>
</tr>
<tr>
  <th>index_predictors</th> <td>   54.6582</td> <td>    1.097</td> <td>   49.834</td> <td> 0.000</td> <td>   52.508</td> <td>   56.808</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1723.459</td> <th>  Durbin-Watson:     </th> <td>   0.083</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>4611.608</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 1.117</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 5.894</td>  <th>  Cond. No.          </th> <td>    1.03</td>
</tr>
</table>




```python
model = LinearRegression()

model.fit(X_train[LOW_P_COLUMNS], Y_train)
Y_pred = model.predict(X_test[LOW_P_COLUMNS])

df_metrics = df_metrics.append(pd.Series(
    {
        'Predictors':len(X_test[LOW_P_COLUMNS].columns.tolist()),
        'Test data R²':model.score(X_test[LOW_P_COLUMNS], Y_test),
        'Test data MSE': mean_squared_error(Y_test, Y_pred)
    },
     name='Low P Predictors',
    )
)
df_metrics
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predictors</th>
      <th>Test data MSE</th>
      <th>Test data R²</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>All Predictors</th>
      <td>42.0</td>
      <td>370.880318</td>
      <td>0.893228</td>
    </tr>
    <tr>
      <th>Low P Predictors</th>
      <td>3.0</td>
      <td>393.196898</td>
      <td>0.886803</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

## 10.3 Stepwise Regression (Sequential feature selection)
---


```python
model = LinearRegression()
sfs = SFS(
    model,
    k_features='parsimonious', 
    #verbose=1,
    forward=True, 
    floating=False, 
    scoring='r2',#'neg_mean_squared_error',#'r2',
    cv=10,
    n_jobs=-1,
)
sfs = sfs.fit(X_train, Y_train)
fig = plot_sfs(sfs.get_metric_dict(), kind='std_err', figsize=(15, 10))
```


    
![png](images/output_72_0.png)
    


* Here we can se how much variation of the target we can explain with certain number of features
* Best predictors according to forward feature selection are the same as selected by hand (Cycle2, groups, index_predictors)
* Using these 3 Predictors we can explain 87% of target variation


```python
pd.DataFrame.from_dict(sfs.get_metric_dict()).T[:10]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_idx</th>
      <th>cv_scores</th>
      <th>avg_score</th>
      <th>feature_names</th>
      <th>ci_bound</th>
      <th>std_dev</th>
      <th>std_err</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>(41,)</td>
      <td>[0.858596301346529, 0.8537260622998766, 0.8516...</td>
      <td>0.866412</td>
      <td>(index_predictors,)</td>
      <td>0.011075</td>
      <td>0.014911</td>
      <td>0.00497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(0, 41)</td>
      <td>[0.866277423773977, 0.8647235783139688, 0.8540...</td>
      <td>0.873531</td>
      <td>(Cycle_2, index_predictors)</td>
      <td>0.010796</td>
      <td>0.014536</td>
      <td>0.004845</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(0, 40, 41)</td>
      <td>[0.8664788211890816, 0.8658006873282414, 0.854...</td>
      <td>0.875889</td>
      <td>(Cycle_2, groups, index_predictors)</td>
      <td>0.010937</td>
      <td>0.014726</td>
      <td>0.004909</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(0, 10, 40, 41)</td>
      <td>[0.8675021119289829, 0.8677341528423116, 0.855...</td>
      <td>0.877433</td>
      <td>(Cycle_2, when_year, groups, index_predictors)</td>
      <td>0.01073</td>
      <td>0.014447</td>
      <td>0.004816</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(0, 10, 39, 40, 41)</td>
      <td>[0.868458793897442, 0.8690934041044777, 0.8555...</td>
      <td>0.878699</td>
      <td>(Cycle_2, when_year, chemical_x, groups, index...</td>
      <td>0.010556</td>
      <td>0.014213</td>
      <td>0.004738</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(0, 10, 21, 39, 40, 41)</td>
      <td>[0.8697353685074956, 0.8696996178891218, 0.856...</td>
      <td>0.879368</td>
      <td>(Cycle_2, when_year, when_diff_expected_start,...</td>
      <td>0.010344</td>
      <td>0.013928</td>
      <td>0.004643</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(0, 10, 21, 29, 39, 40, 41)</td>
      <td>[0.8699938422035802, 0.8695063677972124, 0.856...</td>
      <td>0.879916</td>
      <td>(Cycle_2, when_year, when_diff_expected_start,...</td>
      <td>0.010421</td>
      <td>0.014031</td>
      <td>0.004677</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(0, 1, 10, 21, 29, 39, 40, 41)</td>
      <td>[0.8708099350817892, 0.8699158394660369, 0.857...</td>
      <td>0.880483</td>
      <td>(Cycle_2, Cycle_3, when_year, when_diff_expect...</td>
      <td>0.010378</td>
      <td>0.013973</td>
      <td>0.004658</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(0, 1, 2, 10, 21, 29, 39, 40, 41)</td>
      <td>[0.8713613010625635, 0.8702514898720555, 0.858...</td>
      <td>0.880945</td>
      <td>(Cycle_2, Cycle_3, Cycle_131, when_year, when_...</td>
      <td>0.010336</td>
      <td>0.013917</td>
      <td>0.004639</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(0, 1, 2, 3, 10, 21, 29, 39, 40, 41)</td>
      <td>[0.8719853420989705, 0.8709885003166612, 0.858...</td>
      <td>0.881292</td>
      <td>(Cycle_2, Cycle_3, Cycle_131, crystal_supergro...</td>
      <td>0.010218</td>
      <td>0.013757</td>
      <td>0.004586</td>
    </tr>
  </tbody>
</table>
</div>



We take 17  predictors from SFS list


```python
STS17_COLUMNS = ['Cycle_2', 'Cycle_3', 'Cycle_131', 'crystal_supergroup_1', 'super_hero_group_C', 
                 'super_hero_group_D', 'when_year', 'expected_start_month', 'when_diff_expected_start', 
                 'start_process_diff_process_end', 'crystal_type', 'tracking', 'place', 'Unnamed: 7', 
                 'chemical_x', 'groups', 'index_predictors']
```


```python
model = LinearRegression()

model.fit(X_train[STS17_COLUMNS], Y_train)
Y_pred = model.predict(X_test[STS17_COLUMNS])

df_metrics = df_metrics.append(pd.Series(
    {
        'Predictors':len(X_test[STS17_COLUMNS].columns.tolist()),
        'Test data R²':model.score(X_test[STS17_COLUMNS], Y_test),
        'Test data MSE': mean_squared_error(Y_test, Y_pred)
    },
     name='STS 17 Predictors',
    )
)
df_metrics
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predictors</th>
      <th>Test data MSE</th>
      <th>Test data R²</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>All Predictors</th>
      <td>42.0</td>
      <td>370.880318</td>
      <td>0.893228</td>
    </tr>
    <tr>
      <th>Low P Predictors</th>
      <td>3.0</td>
      <td>393.196898</td>
      <td>0.886803</td>
    </tr>
    <tr>
      <th>STS 17 Predictors</th>
      <td>17.0</td>
      <td>370.927563</td>
      <td>0.893214</td>
    </tr>
  </tbody>
</table>
</div>



## 10.4 Plot some examples for Test data set 
---
We fit our Regression model with 3 low P predictors


```python
# concat test predictors and test target together
df_test = pd.concat(
    [X_test.reset_index(drop=True), Y_test.reset_index(drop=True)],
    axis=1,
    )

```


```python
def plot_n_groups(number_groups: int, fit_columns: list)-> None:
    '''
    Plot target points along with predicted regression and MSE dist for n groups
    
    '''

    # take 5 random groups for ploting
    GROUPS_LIST = df_test.sample(number_groups)['groups'].values.tolist()

    # fit on train data
    model = LinearRegression()
    model.fit(X_train[fit_columns], Y_train)

    fig, axes = plt.subplots(number_groups, 2, figsize=(10,10), gridspec_kw={'width_ratios': [2, 1]}, constrained_layout=True )

    for ax, group_idx in zip(axes, GROUPS_LIST):

        # Get dataset of certain group
        df_test_group = df_test[np.isclose(df_test['groups'],  group_idx)][fit_columns + TARGET_COLUMN].reset_index(drop=True)

        # predict on  test
        Y_pred_target = model.predict(df_test_group[fit_columns])

        # real target
        sns.scatterplot(
            x='index_predictors',
            y='target',
            data=df_test_group,
            #color='red',
            ax=ax[0],
            alpha=0.65,
        )
        # reg line
        ax[0].plot(df_test_group['index_predictors'], Y_pred_target, color = "blue")

        # MSE hist
        sns.kdeplot(
            x=df_test_group['target']-Y_pred_target[:,0],
            #color='green',
            ax=ax[1],
            #alpha=0.65,
            #hist=False,
        )
        # title
        ax[1].set_title('MSE')
        ax[1].set_xlabel('')


        # title
        ax[0].set_title( f"R² Score:{ r2_score(df_test_group['target'], Y_pred_target) } MSE: {mean_squared_error(df_test_group['target'], Y_pred_target)}")
        
plot_n_groups(5, LOW_P_COLUMNS)
```


    
![png](images/output_80_0.png)
    


* Regression line does not fit optimal
* Better Predictors could help
* Maybe interactions of predictors cound improve R²


```python
df_metrics
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predictors</th>
      <th>Test data MSE</th>
      <th>Test data R²</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>All Predictors</th>
      <td>42.0</td>
      <td>370.880318</td>
      <td>0.893228</td>
    </tr>
    <tr>
      <th>Low P Predictors</th>
      <td>3.0</td>
      <td>393.196898</td>
      <td>0.886803</td>
    </tr>
    <tr>
      <th>STS 17 Predictors</th>
      <td>17.0</td>
      <td>370.927563</td>
      <td>0.893214</td>
    </tr>
  </tbody>
</table>
</div>



Next steps could be:
* create Interactions of predictors (e.g. products)
* try Ridge regression (guess it will point to same predictors like STS)
* try PCA to compress multiple predictors to few.


```python

```
