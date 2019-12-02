import functions, set_R_hat

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

user_info = set_R_hat.user_info

mission_info = set_R_hat_mission_info

classified_R_hat = set_R_hat.classified_R_hat

weathers = ['sunny', 'sunny', 'sunny', 'rainy', 'snowy', 'snowy', 'snowy']

weekly_weather = functions.get_weekly_weather(weathers)
 
target_user_id = 'u1'

while True:
    action = input()
    if action == 1: 
        functions.set_user_applicable_missions(target_user_id, classified_R_hat, user_info, mission_info, weekly_weather)
    elif action == 2:
        set_weekly_mission(target_user_id, user_info, mission_info, functions.get_today_idx(), weekly_weather)
    elif action == 3:
        daily_mission = functions.get_daily_mission(target_user_id, user_info, today_weather)
        print("daily_mission", daily_mission)
    elif action == 4:
        update_user_applicable_missions(target_user_id, user_info, daily_mission, "done", functions.get_today_idx(), weekly_weather)
    elif action == 5:
        update_user_applicable_missions(target_user_id, user_info, daily_mission, "pass", functions.get_today_idx(), weekly_weather)
    else:
        print("Key Error")
        continue

 
