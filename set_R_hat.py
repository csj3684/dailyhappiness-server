import functions

import numpy as np
import scipy.stats as ss
import copy
import pandas as pd
from patsy import dmatrices
import time
from collections import OrderedDict
import random
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import math
from matplotlib import pyplot as plt
from numba import jit
import datetime



weather_category = ['sunny', 'cloudy', 'rainy', 'snowy']

data = pd.read_csv('C:/Users/CAU/Desktop/2019_2/capstone_project/Test_data/user_info_sample.csv', engine='python')

n = 30

user_id = data.user_id[0:n]
time_min = data.time_min[0:n]
time_max = data.time_max[0:n]
cost = data.cost[0:n]


user_info = pd.DataFrame(index = user_id, columns = ['time_min', 'time_max', 'cost', 'applicable_missions', 'weekly_missions'])
for i in range(n):
    user_info.loc[user_id[i]] = [time_min[i], time_max[i], cost[i], None, None]

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

data = pd.read_csv('C:/Users/CAU/Desktop/2019_2/capstone_project/Test_data/mission_info_sample.csv', engine='python')

n = 30

mission_id = data.mission_id[0:n]
time_ = data.time[0:n]
cost = data.cost[0:n]

weather_category = ['sunny', 'cloudy', 'rainy', 'snowy']
weather = []
for i in range(n):
    weather.append([data.loc[i]['sunny'], data.loc[i]['cloudy'], data.loc[i]['rainy'], data.loc[i]['snowy']])


mission_info = pd.DataFrame(index = mission_id, columns = ['time', 'cost', 'weather'])
for i in range(n):
    mission_info.loc[mission_id[i]] = [time_[i], cost[i], None]
    mission_info.loc[mission_id[i]]['weather'] = pd.DataFrame(columns = weather_category, data = [weather[i]])

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

data = pd.read_csv('C:/Users/CAU/Desktop/2019_2/capstone_project/Test_data/rating_sample.csv', engine='python')

temperature_min = 14
temperature_max = 26
R_user_id = data.loc[:,'user_id'] # R 만들기 위한 배열들
R_mission_id = data.loc[:,'missions_id']#
R_weather = data.loc[:,'weather']#
R_temperature = data.loc[:,'temperature']#
R_rating = data.loc[:,'rating']#
R_data_num = len(R_user_id)


log = get_init_log(weather_category, user_id, mission_id)

classified_R = get_classified_R(user_id, mission_id, weather_category, temperature_min, temperature_max, R_user_id, R_mission_id, R_weather, R_temperature, R_rating, R_data_num)

start1 = time.time()

classified_R_hat = get_classified_R_hat_by_KNN(classified_R, log)
#classified_R_hat = get_classified_R_hat_by_Regression(classified_R)
#classified_R_hat = get_classified_R_hat_by_matrix_completion(classified_R)

print(time.time() - start1)

