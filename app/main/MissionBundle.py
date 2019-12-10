from flask import Blueprint, request, render_template, flash, redirect, url_for,jsonify
from flask import current_app as app
from app.main.DB import DB
import mysql.connector
import pandas as pd
import numpy as np
import scipy.stats as ss
import copy
from patsy import dmatrices
import time
from collections import OrderedDict
import random
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import math
from matplotlib import pyplot as plt
import datetime
import json
from app.main.Weather import get_max_min_weekly_weather, get_weekly_weather_list


global user_info
global mission_info
global classified_R_hat
global classified_R
global original_R
global keep_info

missionBundlePage = Blueprint('missionBundlePage', __name__, url_prefix='/missionBundle')

def isNaN(a):
    return a != a

def get_original_R(user_id, mission_id, weather_category, temperature_min, temperature_max, R_user_id, R_mission_id, R_weather, R_temperature, R_rating, R_data_num):
    
    original_R = pd.DataFrame(index=weather_category, columns=['value'])

    for weather in original_R.index:
        original_R.loc[weather]['value'] = pd.DataFrame(index=user_id, columns=mission_id)

        R = original_R.loc[weather]['value']

        for i in R.index:
            for j in R.columns:
                R.loc[i][j] = -1

    for i in range(R_data_num):
        if R_rating[i] != -1:
            weather = R_weather[i]
            user_id = R_user_id[i]
            mission_id = R_mission_id[i]
            rating = R_rating[i]
            original_R.loc[weather]['value'].loc[user_id, mission_id] = rating
            other_weathers = list(original_R.index)
            other_weathers.remove(weather)
            for i in other_weathers:
                original_R.loc[i]['value'].loc[user_id, mission_id] = "Done"

    return original_R

def get_classified_R(user_id, mission_id, weather_category, temperature_min, temperature_max, R_user_id, R_mission_id, R_weather, R_temperature, R_rating, R_data_num):
    pd.set_option('display.max_rows', 500)
    
    classified_R = pd.DataFrame(index=weather_category, columns=['value'])
    original_R = pd.DataFrame(index=weather_category, columns=['value'])
    original_R.loc['sunny']['value'] = pd.DataFrame(index=user_id, columns=mission_id)

    for weather in classified_R.index:
        classified_R.loc[weather]['value'] = pd.DataFrame(index=user_id, columns=mission_id)
        #original_R.loc[weather]['value'] = pd.DataFrame(index=user_id, columns=mission_id)

        R = classified_R.loc[weather]['value']

        for i in R.index:
            for j in R.columns:
                R.loc[i][j] = -1

   
    for i in range(R_data_num):
        if R_rating[i] != -1:
            weather = R_weather[i]
            user_id = R_user_id[i]
            mission_id = R_mission_id[i]
            rating = R_rating[i]

            original_R.loc['sunny']['value'].loc[user_id, mission_id] = "Done"
            #original_R.loc[weather]['value'].loc[user_id, mission_id] = rating
               
            #other_weathers = list(original_R.index)
            #other_weathers.remove(weather)
            #for j in other_weathers:
            #    original_R.loc[j]['value'].loc[user_id, mission_id] = "Done"
            
            if (temperature_min <= R_temperature[i]) and (R_temperature[i] <= temperature_max):
          
                classified_R.loc[weather]['value'].loc[user_id, mission_id] = rating
               
                other_weathers = list(classified_R.index)
                other_weathers.remove(weather)
                for i in other_weathers:
                    classified_R.loc[i]['value'].loc[user_id, mission_id] = "Done"


    return classified_R, original_R

def get_init_classified_R_hat(classified_R):
    init_classified_R_hat = pd.DataFrame(index=classified_R.index, columns=['value'])

    for weather in init_classified_R_hat.index:
        user_id = classified_R.loc[weather]['value'].index
        mission_id = classified_R.loc[weather]['value'].columns
        init_classified_R_hat.loc[weather]['value'] = pd.DataFrame(index=user_id, columns=mission_id)

    return init_classified_R_hat

def get_init_log(weather_category, user_id, mission_id):
    init_log = pd.DataFrame(index=['regression', 'knn', 'matrix_completion'], columns=weather_category)
    for method in init_log.index:
        for weather in weather_category:
            init_log.loc[method][weather] = pd.DataFrame(index=user_id, columns=mission_id)
    return init_log

def get_coefficient(ols_result):
    coef = pd.DataFrame()
    coef['coefficient'] = ols_result.params
    return coef

def get_std_err(ols_result):
    std_err = pd.DataFrame()
    std_err['standard error'] = ols_result.bse
    return std_err

def get_t_statictics(ols_result):
    t_statistics = pd.DataFrame()
    t_statistics['t statistics'] = ols_result.tvalues
    return t_statistics

def get_vif(indep_variables):
    vif = pd.DataFrame()
    vif["VIF factor"] = [variance_inflation_factor(indep_variables.values, i) for i in range(indep_variables.shape[1])]
    vif["features"] = indep_variables.columns
    return vif

def get_formula(target, formula):
    FORMULA = target + ' ~ ' + formula[0]
    for i in range(1, len(formula)):
        FORMULA += (' + ' + formula[i])
    return FORMULA

def get_p_value(ols_result):
    p_value = pd.DataFrame()
    p_value['p_value'] = result.pvalues
    return p_value

def get_classified_R_hat_by_Regression(classified_R, log):
    classified_R_hat = get_init_classified_R_hat(classified_R)

    for weather in classified_R_hat.index:
        R = classified_R.loc[weather]['value']
        for i in R.index:
            for j in R.columns:
                log.loc['regression'][weather].loc[i][j] = pd.DataFrame(index=['ols_result', 'vif'], columns=['value'],
                                                                        data=[[[]], [[]]])
        R_hat = classified_R_hat.loc[weather]['value']
        R_refer = copy.deepcopy(R)
        R_refer_for_regression = copy.deepcopy(R)

        for user in R.index:
            target_user = user

            experienced_mission = []
            unexperienced_mission = []
            for mission in R.loc[target_user].index:
                if R.loc[target_user][mission] == -1:
                    unexperienced_mission.append(mission)
                else:
                    experienced_mission.append(mission)

            indep_user_idx = pd.DataFrame(index=R.index, columns=['idx'])
            indep_user_idx.drop(target_user, inplace=True)

            for indep_user in indep_user_idx.index:  # target user의 경험을 완전히 포함하지 못하는 user 제거
                for mission in experienced_mission:
                    if (R.loc[indep_user][mission] == -1) or (R.loc[indep_user][mission] == 'Done'):
                        indep_user_idx.drop(indep_user, inplace=True)
                        break;

            if indep_user_idx.index.size == 0:
                continue

            for mission in R_refer_for_regression.columns:
                if R.loc[target_user][mission] == -1:
                    R_refer_for_regression.drop(mission, axis=1, inplace=True)  # target user 가 안해본 mission 제거

            for mission in unexperienced_mission:
                target_mission = mission
                user_have_not_done_target_mission = []

                for indep_user in indep_user_idx.index:
                    if (R.loc[indep_user][target_mission] == -1) or (R.loc[indep_user][mission] == 'Done'):
                        user_have_not_done_target_mission.append(indep_user)
                        indep_user_idx.drop(indep_user, inplace=True)  # target_mission 을 안해본 user 제거

                if (indep_user_idx.index.size == 0) or (R_refer_for_regression.columns.size == 0):
                    for user in user_have_not_done_target_mission:  # target mission 을 안해봐서 제거된 user 복구
                        indep_user_idx.loc[user] = None
                    continue;

                formula = copy.deepcopy(list(R_refer_for_regression.columns))

                R_refer_for_regression.loc[:, target_mission] = R_refer.loc[:, target_mission]  # target mission 복구

                available = False
                breaker = False

                while available == False:
                    if formula == []:
                        breaker = True
                        log.loc['regression'][weather].loc[target_user, target_mission].loc['vif', 'value'].append(vif)
                        break
                    available = True
                    FORMULA = get_formula(target_mission, formula)

                    R_for_regression = pd.DataFrame(index=indep_user_idx.index, columns=R_refer_for_regression.columns)
                    for user in R_for_regression.index:  # 회귀에 필요한 미션과 유저들로 DataFrame 생성
                        R_for_regression.loc[user] = R_refer_for_regression.loc[user]

                    y, X = dmatrices(FORMULA, R_for_regression, return_type='dataframe')
                    vif = get_vif(X)
                    log.loc['regression'][weather].loc[target_user, target_mission].loc['vif', 'value'].append(vif)
                    for i in vif.index[1:]:
                        if float(vif.loc[i, 'VIF Factor']) > 10:
                            available = False
                            formula.remove(vif.loc[i, 'feature'])

                if breaker == True:
                    R_hat.loc[target_user_id][target_mission] = None
                    R_refer_for_regression.drop(target_mission, axis=1, inplace=True)  # target mission 삭제
                    for i in user_have_not_done_target_mission:  # target mission 을 안해봐서 제거된 user 복구
                        indep_user_idx.loc[i] = None
                    continue

                available = False

                while available == False:
                    if formula == []:
                        breaker = True
                        log.loc['regression'][weather].loc[target_user, target_mission].loc[
                            'ols_result', 'value'].append(ols_result)
                        break
                    available = True
                    FORMULA = get_formula(target_mission, formula)
                    ols_result = smf.ols(FORMULA, data=R_for_regression).fit()
                    p_value = get_p_value(result)
                    log.loc['regression'][weather].loc[target_user, target_mission].loc['ols_result', 'value'].append(
                        ols_result)
                    for i in p_value.index[1:]:
                        if p_value.loc[i, 'p_value'] > 0.1:
                            available = False
                            formula.remove(i)

                if breaker == True:
                    R_hat.loc[target_user_id][target_mission] = None
                    R_refer_for_regression.drop(target_mission, axis=1, inplace=True)  # target mission 삭제
                    for i in user_have_not_done_target_mission:  # target mission 을 안해봐서 제거된 user 복구
                        indep_user_idx.loc[i] = None
                    continue

                parameters = copy.deepcopy(ols_result.params)
                expected_Y = parameters.loc['Intercept']
                parameters.drop('Intercept', inplace=True)

                for k in parameters.keys():
                    expected_Y += (R_refer.loc[target_user_id][k] * parameters.loc[k])

                R_hat.loc[target_user][target_mission] = expected_Y

                R_refer_for_regression.drop(target_mission, axis=1, inplace=True)  # target mission 삭제
                for i in user_have_not_done_target_mission:  # target mission 을 안해봐서 제거된 user 복구
                    indep_user_idx.loc[i] = None

            for k in unexperienced_mission:
                R_refer_for_regression.loc[:, k] = R_refer.loc[:, k]  # target user 가 안해본 mission 복구

        for i in R_hat.index:
            for j in R_hat.columns:
                if R_hat.loc[i][j] < 0:
                    R_hat.loc[i][j] = 0

    return classified_R_hat

# KNN 함수들

def get_distance_and_default_weight(user1, user2):  # user 파라미터 = R.loc[user_id]
    shared_distance = 0
    distance2_for_1 = 0
    distance1_for_2 = 0
    tolarance = 5
    penalty_cofficient = 1
    penalty_for_1 = 0;
    penalty_for_2 = 0;
    NaN = True

    for mission in user1.index:
        if (user1[mission] == -1 or user1[mission] == "Done") and (user2[mission] == -1 or user2[mission] == "Done"):
            continue

        elif (user1[mission] == -1 or user1[mission] == "Done") and (user2[mission] != -1 and user2[mission] != "Done"):
            penalty_for_1 += 1

        elif (user1[mission] != -1 and user1[mission] != "Done") and (user2[mission] == -1 or user2[mission] == "Done"):
            penalty_for_2 += 1

        elif (user1[mission] != -1 and user1[mission] != "Done") and (
                user2[mission] != -1 and user2[mission] != "Done"):
            shared_distance += (np.power(int(user1[mission]) - int(user2[mission]), 2))
            NaN = False

    shared_distance = np.sqrt(shared_distance)
    if NaN == True:
        return {'distance': -1, 'weight': -1}, {'distance': -1, 'weight': -1}
    else:
        return {'distance': shared_distance * (np.power(1.1, penalty_for_1 * penalty_cofficient)), 'weight': -1}, {
            'distance': shared_distance * (np.power(1.1, penalty_for_2 * penalty_cofficient)), 'weight': -1}

def get_D(R):
    # R = DataFrame : index = user_id, columns = mission_id ,data = {'weather' : string, 'temperature' : float, 'rating' : int}
    # distance_matrix = DataFrame : index = user_id, columns = user_id, data = {'distance' : float, 'weight' : float}

    distance_matrix = pd.DataFrame(index=R.index, columns=R.index)
    for i in range(R.index.size):
        for j in range(i + 1, R.index.size):
            distance_matrix.loc[R.index[i]][R.index[j]], distance_matrix.loc[R.index[j]][
                R.index[i]] = get_distance_and_default_weight(R.iloc[i], R.iloc[j])

    return distance_matrix

def availability_for_representative_by_N(log, scale, k, x_mode, n_mode, x, n, sigma):
    log.loc[x]['accepted percentile'] = (str(int(
        round((scale * n_mode * np.exp((-1) * (((x - x_mode) ** 2) / (2 * (sigma ** 2))))) / n_mode * 100, 0))) + "%")
    log.loc[x]['n'] = n
    log.loc[x]['mask'] = round(scale * n_mode * np.exp((-1) * (((x - x_mode) ** 2) / (2 * (sigma ** 2)))), 1)

    if n <= (scale * n_mode * np.exp((-1) * (((x - x_mode) ** 2) / (2 * (sigma ** 2))))):
        if x_mode == x:
            log.loc[x]['P/F'] = "-"
        else:
            log.loc[x]['P/F'] = "PASS"
        return True
    else:
        log.loc[x]['P/F'] = "False"
        return False

def availability_for_representative_by_Customized(scale, k, x_mode, n_mode, x, n, sigma):
    if x < x_mode:
        if (n < np.exp(k * x) + scale * n_mode - np.exp(k * x_mode)):
            return True
        else:
            return False
    else:
        if (n < np.exp(k * (2 * x_mode - x)) + scale * n_mode - np.exp(k * x_mode)):
            return True
        else:
            return False

def get_modes(discrete_distribution):
    # discrete_distribution = dataFrame : index = rating, columns = ['n'], data = rating에 대한 도수 : float
    # mode = DataFrame : index = rating, columns = ['n'], data = 도수 : int

    mode = pd.DataFrame(columns=['n'])

    max_n = 0

    for i in discrete_distribution.index:
        if discrete_distribution.loc[i]['n'] > max_n:
            max_n = discrete_distribution.loc[i]['n']

    for i in discrete_distribution.index:
        if discrete_distribution.loc[i]['n'] == max_n:
            mode.loc[i] = max_n

    return mode

def get_representative_value(log, discrete_distribution, scale, k, sigma, user_id, mission_id):
    # representative_value = DataFrame : index = int, columns = ['value'], data = rating : int
    # log = DataFrame : index = rating, columns = ['accepted percentile', 'n', 'mask', 'P/F'], data = [mask 비율 : float, rating 도수 : float, mask : float, 'P/F' : String]

    breaker = False
    mode = get_modes(discrete_distribution)

    log.loc['representative']['value'] = pd.DataFrame(columns=['value'])
    representative_value = log.loc['representative']['value']
    log.loc['mode']['value'] = mode
    log.loc['graph']['value'] = pd.DataFrame(columns=['value'])
    for i in mode.index:

        graph_log = pd.DataFrame(index=discrete_distribution.index, columns=['accepted percentile', 'n', 'mask', 'P/F'])

        for j in discrete_distribution.index:

            if availability_for_representative_by_N(graph_log, scale, k, i, mode.loc[i]['n'], j,
                                                    discrete_distribution.loc[j]['n'], sigma) == False:
                breaker = True

        log.loc['graph']['value'].loc["graph" + str(i)] = [graph_log]

        if breaker == True:
            breaker = False
            continue

        representative_value.loc[representative_value.index.size] = i

    if representative_value.index.size > 3:
        return None

    elif representative_value.index.size == 3:
        if abs(representative_value.loc[0]['value'] - representative_value.loc[2]['value']) != 2:
            return None
        else:
            return representative_value.loc[1]['value']

    elif representative_value.index.size == 2:
        if abs(representative_value.loc[0]['value'] - representative_value.loc[1]['value']) != 1:
            return None
        else:
            return (representative_value.loc[0]['value'] + representative_value.loc[1]['value']) / 2

    elif representative_value.index.size == 1:
        return representative_value.loc[0]['value']

    else:
        return None

def find_k_nearest(log, ratings, distances_and_weights, user_idx, weight_tolerance):
    # k_nearest = DataFrame : index = rating, columns = ['n'], data = 도수 : float

    k = 0
    sum_influence = 0
    k_nearest = pd.DataFrame(index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], columns=['n'],
                             data=[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])

    while k < ratings.index.size:
        user_id = user_idx.index[k]
        if distances_and_weights.loc[user_id]['distance'] == 0:
            influence = 0.98
        else:
            influence = (1 / distances_and_weights.loc[user_id]['distance'] ** 2)

        if (influence / (sum_influence + influence)) < weight_tolerance:
            unavailable_weight = influence / (sum_influence + influence)
            break;

        sum_influence += influence
        k += 1

    for i in range(ratings.index.size):
        user_id = user_idx.index[i]

        if distances_and_weights.loc[user_id]['distance'] == 0:
            influence = 0.98
        else:
            influence = (1 / distances_and_weights.loc[user_id]['distance'] ** 2)

        distances_and_weights.loc[user_id]['weight'] = (influence / sum_influence)

    for i in range(k):
        user_id = user_idx.index[i]

        k_nearest.loc[ratings.loc[user_id]]['n'] += distances_and_weights.loc[user_id]['weight'] * 100

        log.loc['weight']['value'].loc[user_id] = None
        log.loc['weight']['value'].loc[user_id]['distance'] = distances_and_weights.loc[user_id]['distance']
        log.loc['weight']['value'].loc[user_id]['weight'] = distances_and_weights.loc[user_id]['weight']
        log.loc['weight']['value'].loc[user_id]['rating'] = ratings.loc[user_id]

    if k < ratings.index.size:  # 마지막 점 까지 도달해서 끝난게 아니라면 / 0.01 내려가는 다음 점까지는 추가 (출력용)

        log.loc['weight']['value'].loc[user_id] = None
        log.loc['weight']['value'].loc[user_id]['distance'] = distances_and_weights.loc[user_id]['distance']
        log.loc['weight']['value'].loc[user_id]['weight'] = distances_and_weights.loc[user_id]['weight']
        log.loc['weight']['value'].loc[user_id]['rating'] = ratings.loc[user_id]

    log.loc['k', 'value'] = k

    return k_nearest

def get_classified_R_hat_by_KNN(classified_R, log):
    classified_R_hat = get_init_classified_R_hat(classified_R)

    for weather in classified_R_hat.index:
        start3 = time.time()
        R = classified_R.loc[weather]['value']
        for i in R.index:
            for j in R.columns:
                log.loc['knn'][weather].loc[i][j] = pd.DataFrame(
                    index=['scale', 'sigma', 'weight_tolerance', 'k', 'weight', 'distribution'], columns=['value'])
                log.loc['knn'][weather].loc[i][j].loc['weight']['value'] = pd.DataFrame(
                    columns=['distance', 'weight', 'rating'])
                log.loc['knn'][weather].loc[i][j].loc['distribution']['value'] = pd.DataFrame(
                    index=['mode', 'k_nearest', 'graph', 'representative'], columns=['value'])
        R_hat = classified_R_hat.loc[weather]['value']

        D = get_D(R)

        # print(D)
        for user_id in R.index:
            start2 = time.time()
            target_user_id = user_id
            unexperienced_mission = []

            for mission in R.loc[target_user_id].index:
                if R.loc[target_user_id][mission] == -1:
                    unexperienced_mission.append(mission)

            for j in range(len(unexperienced_mission)):
                target_mission_id = unexperienced_mission[j]
                target_mission_R = copy.deepcopy(R.loc[:, target_mission_id])  # target_mission에 대한 user 정보
                others_distance = copy.deepcopy(D.loc[:, target_user_id])  # target_user 와의 거리

                others_distance.drop(target_user_id, inplace=True)
                target_mission_R.drop(target_user_id, inplace=True)

                # print(target_user_id)
                # print(target_mission_id)

                for i in others_distance.index:  # target_user 랑 거리를 측정할 수 없는 user들 제거
                    if others_distance.loc[i]['distance'] == -1:
                        target_mission_R.drop(i, inplace=True)
                # print("거리 없는 user 들 제거된 후\n", target_mission_R.index)
                for i in target_mission_R.index:  # target_mission 점수 없는 user들 제거
                    if (target_mission_R.loc[i] == -1) or target_mission_R.loc[i] == "Done":
                        target_mission_R.drop(i, inplace=True)
                # print("점수 없는 user 들 제거된 후\n", target_mission_R.index)

                if target_mission_R.index.size == 0:
                    R_hat.loc[target_user_id][target_mission_id] = None
                    continue

                sorted_user_by_distance = pd.DataFrame(index=target_mission_R.index, columns=['distance'])

                for i in target_mission_R.index:
                    sorted_user_by_distance.loc[i] = others_distance.loc[i]['distance']

                sorted_user_by_distance = sorted_user_by_distance.sort_values(by='distance')

                weight_tolerance = 0

                # print("다 제거된 후\n", target_mission_R.index)
                k_nearest = find_k_nearest(log.loc['knn'][weather].loc[target_user_id][target_mission_id],
                                           target_mission_R, others_distance, sorted_user_by_distance, weight_tolerance)
                log.loc['knn'][weather].loc[target_user_id][target_mission_id].loc['distribution']['value'].loc[
                    'k_nearest']['value'] = k_nearest

                scale = 1.2
                sigma = 2.56

                log.loc['knn'][weather].loc[target_user_id][target_mission_id].loc['scale']['value'] = scale
                log.loc['knn'][weather].loc[target_user_id][target_mission_id].loc['sigma']['value'] = sigma
                log.loc['knn'][weather].loc[target_user_id][target_mission_id].loc['weight_tolerance'][
                    'value'] = weight_tolerance

                log_tmp = log.loc['knn'][weather].loc[target_user_id][target_mission_id].loc['distribution']['value']
                R_hat.loc[target_user_id][target_mission_id] = get_representative_value(log_tmp, k_nearest, scale, -1,
                                                                                        sigma, target_user_id,
                                                                                        target_mission_id)

        for i in R.index:
            for j in R.columns:
                if R.loc[i][j] != -1:
                    R_hat.loc[i][j] = "Done"

    return classified_R_hat

class MatrixFactorization():
    def __init__(self, R, k, learning_rate, reg_param, epochs, verbose=False):
        self._R = R
        self._num_users, self._num_items = R.shape
        self._k = k
        self._learning_rate = learning_rate
        self._reg_param = reg_param
        self._epochs = epochs
        self._verbose = verbose

    def fit(self):
        interval = 200

        # init latent features
        self._P = np.random.normal(size=(self._num_users, self._k))
        self._Q = np.random.normal(size=(self._num_items, self._k))

        # init biases
        self._b_P = np.zeros(self._num_users)
        self._b_Q = np.zeros(self._num_items)
        #print(self._R[np.where(self._R != 0)])
        self._b = np.mean(self._R[np.where(self._R != 0)])

        # train while epochs
        self._training_process = []
        for epoch in range(self._epochs):

            # rating이 존재하는 index를 기준으로 training
            for i in range(self._num_users):
                for j in range(self._num_items):
                    if self._R[i, j] > 0:
                        self.gradient_descent(i, j, self._R[i, j])
            cost = self.cost()
            self._training_process.append((epoch, cost))

            # print status
            if self._verbose == True and ((epoch + 1) % interval == 0):
                print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))

    def cost(self):
        # xi, yi: R[xi, yi]는 nonzero인 value를 의미한다.
        # 참고: http://codepractice.tistory.com/90
        xi, yi = self._R.nonzero()
        predicted = self.get_complete_matrix()
        cost = 0
        for x, y in zip(xi, yi):
            cost += pow(self._R[x, y] - predicted[x, y], 2)
        return np.sqrt(cost) / len(xi)

    def gradient(self, error, i, j):

        dp = (error * self._Q[j, :]) - (self._reg_param * self._P[i, :])
        dq = (error * self._P[i, :]) - (self._reg_param * self._Q[j, :])
        return dp, dq

    def gradient_descent(self, i, j, rating):

        # get error
        prediction = self.get_prediction(i, j)
        error = rating - prediction

        # update biases
        self._b_P[i] += self._learning_rate * (error - self._reg_param * self._b_P[i])
        self._b_Q[j] += self._learning_rate * (error - self._reg_param * self._b_Q[j])

        # update latent feature
        dp, dq = self.gradient(error, i, j)
        self._P[i, :] += self._learning_rate * dp
        self._Q[j, :] += self._learning_rate * dq

    def get_prediction(self, i, j):

        return self._b + self._b_P[i] + self._b_Q[j] + self._P[i, :].dot(self._Q[j, :].T)

    def get_complete_matrix(self):

        return self._b + self._b_P[:, np.newaxis] + self._b_Q[np.newaxis:, ] + self._P.dot(self._Q.T)

    def print_results(self):

        print("User Latent P:")
        print(self._P)
        print("Item Latent Q:")
        print(self._Q.T)
        print("P x Q:")
        print(self._P.dot(self._Q.T))
        print("bias:")
        print(self._b)
        print("User Latent bias:")
        print(self._b_P)
        print("Item Latent bias:")
        print(self._b_Q)
        print("Final R matrix:")
        print(self.get_complete_matrix())
        print("Final RMSE:")
        print(self._training_process[self._epochs - 1][1])

def get_classified_R_hat_by_MatrixCompletion(classified_R, original_R):
    classified_R_hat = get_init_classified_R_hat(classified_R)
    # print(classified_R.loc['cloudy','value'])
    R_original = original_R.loc['sunny']['value']

    for weather in classified_R_hat.index:

        R = classified_R.loc[weather]['value']
        #R_original = original_R.loc[weather]['value']
        R_hat = classified_R_hat.loc[weather]['value']

        for user in R_hat.index:
            for mission in R_hat.columns:
                if (R.loc[user][mission] != "Done") and (R.loc[user][mission] != -1):
                    R_hat.loc[user][mission] = R.loc[user][mission]
                else:
                    R_hat.loc[user][mission] = 0

        # rating matrix - User X Item : (7 X 5)
        R_ = np.array(R_hat)

        #print(R_hat)
        # P, Q is (7 X k), (k X 5) matrix
        factorizer = MatrixFactorization(R_, k=5, learning_rate=0.01, reg_param=0.01, epochs=100, verbose=True)
        factorizer.fit()

        R_ = factorizer.get_complete_matrix()

        for i in range(R_hat.index.size):
            for j in range(R_hat.columns.size):
                if isNaN(R_original.iloc[i, j]) == False:
                    R_hat.iloc[i, j] = "Done"
                else:
                    R_hat.iloc[i, j] = R_[i, j]
                if R_[i, j] < 0:
                    R_hat.iloc[i, j] = 0


    return classified_R_hat

# 주간 시작 전처리

def back_up(check_point, real_g, applicable_missions, mission_info, weekly_cost, N):
    if check_point.index.size == 0:
        return 0

    i = check_point.iloc[check_point.index.size - 1]['index']
    accumulated_cost = check_point.iloc[check_point.index.size - 1]['accumulated_cost']
    real_accumulated_cost = copy.deepcopy(accumulated_cost)
    weather_condition = copy.deepcopy(check_point.iloc[check_point.index.size - 1]['weather_condition'])
    current_sack = copy.deepcopy(check_point.iloc[check_point.index.size - 1]['current_sack'])
    real_current_sack_size = current_sack.index.size
    h = check_point.iloc[check_point.index.size - 1]['h']

    check_point.drop(check_point.index[check_point.index.size - 1], inplace=True)

    return get_g_plus_h(check_point, weather_condition, applicable_missions, real_g, mission_info, i + 1, weekly_cost,
                        accumulated_cost, real_accumulated_cost, current_sack, h, real_current_sack_size, N)


def get_g_plus_h(check_point, weather_condition, applicable_missions, real_g, mission_info, i, weekly_cost,
                 accumulated_cost, real_accumulated_cost, current_sack, h, real_current_sack_size, N):
    global best

    huristic = 9 # 뒤에 가보진 미션들 중에서 현재 sack을 조정하여 담을 수 있는 최대평점으로 예상하는 수치

    if i == applicable_missions.index.size:
        return back_up(check_point, real_g, applicable_missions, mission_info, weekly_cost, N)

    while (True):
        mission_id = applicable_missions.iloc[i]['mission_id']
        cost = mission_info.loc[mission_id]['cost']
        g = applicable_missions.iloc[i]['g']
        weather = applicable_missions.iloc[i]['weather']
        required_weather_num = weather_condition.loc[weather]['value']

        if weekly_cost < (accumulated_cost + cost):
            break

        if (required_weather_num > 0) and ((mission_id in current_sack.index) == False):
            check_point.loc[check_point.index.size] = [i, accumulated_cost, weather_condition, current_sack, h]
            current_sack.loc[mission_id] = [weather, g, cost]
            h += g
            accumulated_cost += cost
            weather_condition.loc[weather]['value'] -= 1

        i += 1

        if current_sack.index.size == N:  # cost가 남은 경우
            if cost == 0:
                h = huristic * (N - real_current_sack_size)
                return real_g + h
            elif (((weekly_cost - real_accumulated_cost) / cost) * g) > (huristic * (N - real_current_sack_size)):
                h = huristic * (N - real_current_sack_size)
                return real_g + h
            else:
                h += min(huristic, (((weekly_cost - accumulated_cost) / cost) * g))
                h = max(h, ((weekly_cost - real_accumulated_cost) / cost) * g)
                return real_g + h

        if i == applicable_missions.index.size:
            return back_up(check_point, real_g, applicable_missions, mission_info, weekly_cost, N)

    if current_sack.index.size < N - 1:  # cost 다 써서 채웠는데 7개가 안되면 이번꺼 건너뛰고 다음꺼
        return get_g_plus_h(check_point, weather_condition, applicable_missions, real_g, mission_info, i + 1, weekly_cost, accumulated_cost, real_accumulated_cost, current_sack, h, real_current_sack_size, N)
        """ if cost == 0:
                    return 0
                elif real_g + min((huristic * N - current_sack.index.size) - (((weekly_cost - accumulated_cost) / cost) * g)) < best:
                    return 0
                else:"""

    # 단위무게로 쪼개서 마지막까지 채우기
    h += (((weekly_cost - accumulated_cost) / cost) * g)

    return real_g + h


def get_applicable_weekly_mission_set(weather_condition, user_id, mission_info, applicable_missions, current_sack, g,
                                      accumulated_cost, i, weekly_cost, N):
    global weekly_mission_set_candidate, best, visit_num, cutting_num

    visit_num += 1

    tolerance = 2 * N

    #print("best", best)
    #print("current_sack\n", current_sack)
    #print("\n")

    if current_sack.index.size == N:
        # print("sack size 도달")
        candidate_set = pd.DataFrame(data=[[copy.deepcopy(current_sack), g, accumulated_cost]],
                                     columns=['mission_set', 'total_g', 'total_cost'])
        weekly_mission_set_candidate = weekly_mission_set_candidate.append(candidate_set)
        weekly_mission_set_candidate = weekly_mission_set_candidate.reset_index(drop=True)
        if g > best:
            best = g
        #print("best", best)
        #print("current_sack\n", current_sack)
        #print("\n")
        return

    if i == applicable_missions.index.size:
        #print("마지막 미션 도달")
        if len(current_sack) < 7:
            return
        candidate_set = pd.DataFrame(data=[[copy.deepcopy(current_sack), g, accumulated_cost]],
                                     columns=['mission_set', 'total_g', 'total_cost'])
        weekly_mission_set_candidate = weekly_mission_set_candidate.append(candidate_set)
        weekly_mission_set_candidate = weekly_mission_set_candidate.reset_index(drop=True)
        if g > best:
            best = g
        #print("best", best)
        #print("current_sack\m", current_sack)
        #print("\n")
        return

    mission_id = applicable_missions.iloc[i]['mission_id']
    required_cost = mission_info.loc[mission_id]['cost']
    expected_R = applicable_missions.iloc[i]['g']
    weather = applicable_missions.iloc[i]['weather']
    required_weather_num = weather_condition.loc[weather]['value']

    if accumulated_cost + required_cost < weekly_cost:
        if required_weather_num > 0:
            if (mission_id in current_sack.index) == False:

                virtual_idx = i
                virtual_accmulated_cost = accumulated_cost
                virtual_current_sack = copy.deepcopy(current_sack)
                virtual_weather_condition = copy.deepcopy(weather_condition)
                check_point = pd.DataFrame(
                    columns=['index', 'accumulated_cost', 'weather_condition', 'current_sack', 'h'])
                # START = time.time()
                g_plus_h = get_g_plus_h(check_point, virtual_weather_condition, applicable_missions, g, mission_info,
                                        virtual_idx, weekly_cost, virtual_accmulated_cost, accumulated_cost,
                                        virtual_current_sack, 0, current_sack.index.size, N)
                # print("g + h spending time : ", time.time() - START)
                #print("g+h", g_plus_h)

                if g_plus_h <= best + tolerance:
                    cutting_num += 1
                    return

                g += expected_R
                accumulated_cost += required_cost
                weather_condition.loc[weather]['value'] -= 1
                current_sack.loc[mission_id] = [weather, expected_R, required_cost]

                get_applicable_weekly_mission_set(weather_condition, user_id, mission_info, applicable_missions,
                                                  current_sack, g, accumulated_cost, i + 1, weekly_cost, N)

                g -= expected_R
                accumulated_cost -= required_cost
                weather_condition.loc[weather]['value'] += 1
                current_sack.drop(mission_id, inplace=True)
            else:
                cutting_num += 1
            #    print(mission_id, "미션이 이미 있어요")
        else:
            cutting_num += 1
        #    print(mission_id, "날씨가 다 찼어요")
    else:
        cutting_num += 1
    #    print(mission_id, "못넣음 : cost 초과")

    #print("best", best)
    #print("current_sack\n", current_sack)
    #print("\n")
    get_applicable_weekly_mission_set(weather_condition, user_id, mission_info, applicable_missions, current_sack, g,
                                      accumulated_cost, i + 1, weekly_cost, N)


def get_weekly_mission_set_candidate(weather_condition, applicable_missions, user_id, mission_info, limited_cost,
                                     item_num):
    # mission_and_weather : index = mission_id, columns = weather, data = 0 or 1
    # weekly_mission_set_cadidate --> 'mission_set' = [] : String, mission_id

    global weekly_mission_set_candidate, best, visit_num, cutting_num

    weekly_mission_set_candidate = pd.DataFrame(columns=['mission_set', 'total_g', 'total_cost'])

    best = -1
    visit_num = 0
    cutting_num = 0

    current_sack = pd.DataFrame(columns=['weather', 'g', 'cost'])

    get_applicable_weekly_mission_set(weather_condition, user_id, mission_info, applicable_missions, current_sack, 0, 0,
                                      0, limited_cost, item_num)

    return weekly_mission_set_candidate

def get_applicable_missions(user_id1, classified_R_hat, user_info, mission_info, weekly_weather):
    # weekly_mission_set_candidate = columns = ['mission_set', 'cost']
    # 'mission_set' = [] : String, mission_id
    user_id = np.int64(user_id1)
    required_weather = []

    for weather in weekly_weather.index:
        for day in weekly_weather.columns:
            if weekly_weather.loc[weather][day] == 1:
                required_weather.append(weather)
                break;

    applicable_missions = pd.DataFrame(columns=['mission_id', 'g', 'cost', 'g per cost', 'weather'])

    for weather in required_weather:
        R_hat = classified_R_hat.loc[weather]['value']
        for mission in R_hat.columns:
            #print("type user_id", type(int(user_id)))
            #print("type index", R_hat.index)
            g = R_hat.loc[user_id,mission]
            cost = mission_info.loc[mission]['cost']
            if (g != "Done") and (g != None):
                
                #print("mission_id", mission)
                #print("g", g)
                #print("weather", weather)
                #print()

                #print("1")
                #print(user_info.loc[user_id, 'time_min'])
                #print(mission_info.loc[mission,'time'])
                #print(user_info.loc[user_id, 'time_max'])
                if (user_info.loc[user_id, 'time_min'] <= mission_info.loc[mission, 'time']) and (mission_info.loc[mission, 'time'] <= user_info.loc[user_id, 'time_max']):
                    if cost == 0:
                        applicable_missions.loc[applicable_missions.index.size] = [mission, g, cost, float('inf'),
                                                                                   weather]
                    else:
                        applicable_missions.loc[applicable_missions.index.size] = [mission, g, cost, float(g) / cost, weather]
                    #print(2)


    if applicable_missions.index.size == 0:
        print("knapsack에 쓰일 수 있는 미션 갯수 = 0")
        return None

    return applicable_missions

def sort_by_g_per_c(frame):
    frame.sort_values(by='g per cost', ascending=False, inplace=True)

    n = 0

    for i in range(frame.index.size):
        if frame.iloc[i]['cost'] != 0:
            n=i
            break

    if n != 0:
        tmp = frame.iloc[0:n].sort_values(by='g', ascending=False)

        t_list = []
        for i in tmp.index:
            t_list += [tmp.loc[i]]

        frame.iloc[0:n] = t_list

    tmp = frame.iloc[n:frame.index.size].sort_values(by='g per cost', ascending=False)

    t_list = []
    for i in tmp.index:
        t_list += [tmp.loc[i]]

    frame.iloc[n:frame.index.size] = t_list

def set_user_default_missions(target_user_id, classified_R_hat, user_info, mission_info, weekly_weather):

    user_id = np.int64(target_user_id)
    required_weather = []

    for weather in weekly_weather.index:
        for day in weekly_weather.columns:
            if weekly_weather.loc[weather][day] == 1:
                required_weather.append(weather)
                break

    default_missions = pd.DataFrame(columns=['mission_id', 'g', 'cost', 'g per cost', 'weather'])

    for weather in required_weather:
        R_hat = classified_R_hat.loc[weather]['value']
        for mission in R_hat.columns:
            g = R_hat.loc[user_id,mission]
            cost = mission_info.loc[mission]['cost']
            if (g != "Done") and (g != None):
                if cost == 0:
                    default_missions.loc[default_missions.index.size] = [mission, g, cost, float('inf'), weather]
                else:
                    default_missions.loc[default_missions.index.size] = [mission, g, cost, float(g) / cost, weather]
    
    a = user_info.loc[target_user_id]['time_min']
    b = user_info.loc[target_user_id]['time_max']
    c = user_info.loc[target_user_id]['cost']
    d = user_info.loc[target_user_id]['applicable_missions']
    e = user_info.loc[target_user_id]['weekly_missions']
    f = user_info.loc[target_user_id]['weekly_missions_log']
    g = user_info.loc[target_user_id]['daily_mission']
    i = user_info.loc[target_user_id]['recommendation_availability']

    user_info.loc[target_user_id] = [a, b, c, d, e, f, g, default_missions, i]

 
    if default_missions.index.size == 0:
        print("all missions done")
        return False

    return True


def set_user_applicable_missions(target_user_id, classified_R_hat, user_info, mission_info, weekly_weather):

    applicable_missions = get_applicable_missions(target_user_id, classified_R_hat, user_info, mission_info,
                                                  weekly_weather)

    if type(applicable_missions) == type(None):
        user_info.loc[target_user_id] = pd.DataFrame()
        return False
    a = user_info.loc[target_user_id]['time_min']
    b = user_info.loc[target_user_id]['time_max']
    c = user_info.loc[target_user_id]['cost']
    e = user_info.loc[target_user_id]['weekly_missions']
    f = user_info.loc[target_user_id]['weekly_missions_log']
    g = user_info.loc[target_user_id]['daily_mission']
    h = user_info.loc[target_user_id]['default_mission']
    i = user_info.loc[target_user_id]['recommendation_availability']

    user_info.loc[target_user_id] = [a, b, c, applicable_missions, e, f, g, h, i]
    
   

    return True

def update_user_applicable_missions(user_id,  mission_id, activity, today_idx, weekly_weather):
    global user_info, mission_info

    print("update> defalut\n", user_info.loc[user_id]['default_mission'])
    
    if user_info.loc[user_id, "recommendation_availability"] == True:
        if activity == "done":
            user_info.loc[user_id]['weekly_missions'].iloc[0].loc['mission_set'].drop(mission_id, inplace = True)

        for i in user_info.loc[user_id]['applicable_missions'].index:
            if user_info.loc[user_id]['applicable_missions'].loc[i]['mission_id'] == mission_id:
                user_info.loc[user_id]['applicable_missions'].drop(i, inplace = True)
        
    default_missions = user_info.loc[user_id]['default_mission']
    for i in default_missions.index:
        if default_missions.loc[i]['mission_id'] == mission_id:
            default_missions.drop(i, inplace = True)    

def set_weekly_mission(user_id, user_info, mission_info, today_idx, weekly_weather):
    
    global visit_num, cutting_num

    START = time.time()
            
    if get_mode(user_id) == "default":
        print("mode already default")
        return False

    applicable_missions = user_info.loc[user_id]['applicable_missions']

    sort_by_g_per_c(applicable_missions)
    applicable_missions.reset_index(drop = True, inplace = True)

    cost = user_info.loc[user_id]['cost']
    item_num = 7 - today_idx

    adjusted_weekly_weather = copy.deepcopy(weekly_weather)
    for i in range(today_idx):
        day = weekly_weather.columns[i]
        adjusted_weekly_weather.drop(day, axis=1, inplace=True)

    weather_condition = pd.DataFrame(index=weekly_weather.index, columns=['value'])
    for weather in weather_condition.index:
        weather_condition.loc[weather] = [sum(adjusted_weekly_weather.loc[weather])]
    #print("여기는 옴 1")
    weekly_mission_set_candidate = get_weekly_mission_set_candidate(weather_condition, applicable_missions, user_id,
                                                                    mission_info, cost, item_num)
    

    #print("여기는 끝 1")
    #print("weejkly_mission_set_candidate :\n", weekly_mission_set_candidate.iloc[0]['mission_set'])

    if weekly_mission_set_candidate.index.size == 0:
        print("No weekly_mission")
        print("Knapsack spending time", time.time() - START)
        return False

    
    print("mission num : ", applicable_missions.index.size)
    print("visit : ", visit_num)
    print("cutting : ", cutting_num)
    n = applicable_missions.index.size
    r = 7 - today_idx
    print("brute force : ", (math.factorial(n) / (math.factorial(r) * math.factorial(n - r))))


    weekly_mission_set_candidate = weekly_mission_set_candidate.sort_values(by="total_g", ascending=False)
    daily_missions = pd.DataFrame(weekly_mission_set_candidate.iloc[0]).T

    a = user_info.loc[user_id]['time_min']
    b = user_info.loc[user_id]['time_max']
    c = user_info.loc[user_id]['cost']
    d = user_info.loc[user_id]['applicable_missions']

    f = user_info.loc[user_id]['weekly_missions_log']
    g = user_info.loc[user_id]['daily_mission']
    h = user_info.loc[user_id]['default_mission']
    i = user_info.loc[user_id]['recommendation_availability']
    
    user_info.loc[user_id] = [a, b, c, d, daily_missions, f, g, h, i]

    print("Knapsack spending time", time.time() - START)
    return True

def get_daily_mission_by_default_mode(target_user_id, user_info, mission_info, today_weather):
    """if type(user_info.loc[target_user_id, 'applicable_missions']) != type(None):
        applicable_missions = user_info.loc[target_user_id, 'applicable_missions']
        for i in applicable_missions.index:
            if applicable_missions.loc[i, 'weather'] == today_weather:
                return applicable_missions.loc[i, 'mission_id']
    else:"""

    default_missions = user_info.loc[target_user_id, 'default_mission']
    if default_missions.index.size == 0:
        return None
    print(default_missions.sort_values(by = 'mission_id').reset_index(drop = True))
    rand_i = int(random.random() * default_missions.index.size)
    return default_missions.iloc[rand_i]['mission_id']

def get_daily_mission(target_user_id, user_info, today_weather):
    if get_mode(target_user_id) == "recommendation":
        print("get_daily_mission> mode = recommendation")
        daily_missions = user_info.loc[target_user_id]['weekly_missions'].iloc[0].loc['mission_set']
        
 
        for mission in daily_missions.index:
            if daily_missions.loc[mission]['weather'] == today_weather:
                return mission
    else:
        print("get_daily_mission> mode = default")
        return get_daily_mission_by_default_mode(target_user_id, user_info, mission_info, today_weather)




def back_up_for_cost(check_point, applicable_missions, mission_info, N):
    if check_point.index.size == 0:
        return float('inf')

    i = check_point.iloc[check_point.index.size - 1]['index']
    accumulated_cost = check_point.iloc[check_point.index.size - 1]['accumulated_cost']
    g = check_point.iloc[check_point.index.size - 1]['g']
    weather_condition = copy.deepcopy(check_point.iloc[check_point.index.size - 1]['weather_condition'])
    current_sack = copy.deepcopy(check_point.iloc[check_point.index.size - 1]['current_sack'])
    h = copy.deepcopy(check_point.iloc[check_point.index.size - 1]['h'])

    check_point.drop(check_point.index[check_point.index.size - 1], inplace=True)

    return get_c_plus_h(check_point, weather_condition, applicable_missions, mission_info, i, g, accumulated_cost,
                        current_sack, h, N)


def get_c_plus_h(check_point, weather_condition, applicable_missions, mission_info, i, g, real_cost, current_sack, h,
                 N):
    while (i < applicable_missions.index.size):
        mission_id = applicable_missions.iloc[i]['mission_id']
        cost = mission_info.loc[mission_id]['cost']
        weather = applicable_missions.iloc[i]['weather']
        required_weather_num = weather_condition.loc[weather]['value']

        if (required_weather_num > 0) and ((mission_id in current_sack.index) == False):
            check_point.loc[check_point.index.size] = [i, g, real_cost, weather_condition, current_sack, h]
            current_sack.loc[mission_id] = [weather, None, cost]
            h += cost
            weather_condition.loc[weather]['value'] -= 1

        i += 1

        if current_sack.size == N:
            return real_cost + h

    return back_up_for_cost(check_point, applicable_missions, mission_info, N)


def set_minimum_cost_mission_set(weather_condition, mission_info, applicable_missions, current_sack, g,
                                 accumulated_cost, i, N):
    global minimum_cost_mission_set, best_min

    if accumulated_cost >= best_min:
        return

    if current_sack.index.size == N:
        candidate_set = pd.DataFrame(data=[[copy.deepcopy(current_sack), g, accumulated_cost]],
                                     columns=['mission_set', 'total_g', 'total_cost'])
        minimum_cost_mission_set = minimum_cost_mission_set.append(candidate_set)
        minimum_cost_mission_set = minimum_cost_mission_set.reset_index(drop=True)
        if accumulated_cost < best_min:
            best_min = accumulated_cost
        print("sack size 도달")
        print("cost", accumulated_cost)
        print("best", best_min)
        print("current_sack")
        print(current_sack, end="\n\n\n")
        return

    if i == applicable_missions.index.size:
        if len(current_sack) < N:
            return
        candidate_set = pd.DataFrame(data=[[copy.deepcopy(current_sack), g, accumulated_cost]],
                                     columns=['mission_set', 'total_g', 'total_cost'])
        minimum_cost_mission_set = minimum_cost_mission_set.append(candidate_set)
        minimum_cost_mission_set = minimum_cost_mission_set.reset_index(drop=True)
        if accumulated_cost < best_min:
            best_min = accumulated_cost
        print("모든 미션 순회")
        print("cost", accumulated_cost)
        print("best", best_min)
        print("current_sack")
        #print(current_sack, end="\n\n\n")
        return

    mission_id = applicable_missions.iloc[i]['mission_id']
    required_cost = mission_info.loc[mission_id]['cost']
    expected_R = applicable_missions.iloc[i]['g']
    weather = applicable_missions.iloc[i]['weather']
    required_weather_num = weather_condition.loc[weather]['value']

    if applicable_missions.index.size - i + current_sack.index.size < N:
        return

    v_accumulated_cost = accumulated_cost
    v_i = i
    v_current_sack = copy.deepcopy(current_sack)
    v_current_sack_size = current_sack.index.size
    v_weather_condition = copy.deepcopy(weather_condition)

    while v_i < applicable_missions.index.size:
        v_mission_id = applicable_missions.iloc[v_i]['mission_id']
        v_required_cost = mission_info.loc[v_mission_id]['cost']
        v_accumulated_cost += v_required_cost
        v_i += 1
        v_current_sack_size += 1

    if v_current_sack_size < N:
        return

    """while (v_current_sack_size < N) and (v_i < applicable_missions.index.size):
        v_mission_id = applicable_missions.iloc[v_i]['mission_id']
        v_required_cost = mission_info.loc[v_mission_id]['cost']
        v_expected_R = applicable_missions.iloc[v_i]['g']
        v_weather = applicable_missions.iloc[v_i]['weather']
        v_required_weather_num = weather_condition.loc[v_weather]['value']

        if v_required_weather_num > 0:
            if (v_mission_id in v_current_sack.index) == False:
                v_weather_condition.loc[v_weather]['value'] -= 1
                v_current_sack.loc[mission_id] = [v_weather, None, v_required_cost]
                v_accumulated_cost += v_required_cost
                v_current_sack_size += 1

        v_i += 1

    if (v_accumulated_cost >= best_min) or (v_i == applicable_missions.index.size):
        return"""

    if (v_accumulated_cost >= best_min):
        return

    if required_weather_num > 0:
        if (mission_id in current_sack.index) == False:
            virtual_idx = i
            virtual_accmulated_cost = accumulated_cost
            virtual_current_sack = copy.deepcopy(current_sack)
            virtual_weather_condition = copy.deepcopy(weather_condition)
            # check_point = pd.DataFrame(columns = ['index', 'g', 'accumulated_cost', 'weather_condition', 'current_sack', 'h'])
            # c_plus_h = get_c_plus_h(check_point, virtual_weather_condition, applicable_missions, mission_info, virtual_idx, g, virtual_accmulated_cost, virtual_current_sack, 0, N)
            # print("g+h", g_plus_h)
            # print("best", best)

            # if c_plus_h > best_min:
            #    return

            g += expected_R
            accumulated_cost += required_cost
            weather_condition.loc[weather]['value'] -= 1
            current_sack.loc[mission_id] = [weather, expected_R, required_cost]

            set_minimum_cost_mission_set(weather_condition, mission_info, applicable_missions, current_sack, g,
                                         accumulated_cost, i + 1, N)

            g -= expected_R
            accumulated_cost -= required_cost
            weather_condition.loc[weather]['value'] += 1
            current_sack.drop(mission_id, inplace=True)

    set_minimum_cost_mission_set(weather_condition, mission_info, applicable_missions, current_sack, g,
                                 accumulated_cost, i + 1, N)

def get_minimum_cost_mission_set(target_user_id, user_info, mission_info, today_idx, weekly_weather):
    applicable_missions = user_info.loc[target_user_id]['applicable_missions']
    applicable_missions = applicable_missions.sort_values(by='cost')

    global minimum_cost_mission_set, best_min

    best_min = float('inf')

    minimum_cost_mission_set = pd.DataFrame(columns=['mission_set', 'total_g', 'total_cost'])

    item_num = 7 - today_idx

    adjusted_weekly_weather = copy.deepcopy(weekly_weather)
    for i in range(today_idx):
        day = weekly_weather.columns[i]
        adjusted_weekly_weather.drop(day, axis=1, inplace=True)

    weather_condition = pd.DataFrame(index=weekly_weather.index, columns=['value'])
    for weather in weather_condition.index:
        weather_condition.loc[weather] = [sum(adjusted_weekly_weather.loc[weather])]

    current_sack = pd.DataFrame(columns=['weather', 'g', 'cost'])

    set_minimum_cost_mission_set(weather_condition, mission_info, applicable_missions, current_sack, 0, 0, 0, item_num)

    if minimum_cost_mission_set.index.size == 0:
        return None

    minimum_cost_mission_set = minimum_cost_mission_set.sort_values(by="total_cost")
    # minimum_cost_mission_set = minimum_cost_mission_set's logs
    return pd.DataFrame(minimum_cost_mission_set.iloc[0]).T 

    # return pd.DataFrame(minimum_cost_mission_set.iloc[minimum_cost_mission_set.index.size - 1]).T, minimum_cost_mission_set

def get_weekly_weather(weathers, weather_category):
    weekly_weather = pd.DataFrame(index=weather_category, columns=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                                  data=[[0, 0, 0, 0, 0, 0, 0], ])
    for day_idx in range(weekly_weather.columns.size):
        weather = weathers[day_idx]
        weekly_weather.loc[weather][day_idx] = 1

    return weekly_weather

def get_today_idx():
    return datetime.datetime.today().weekday()

def get_mode(target_user_id):
    if user_info.loc[target_user_id, 'recommendation_availability'] == True:
        return "recommendation"
    else:
        return "default"

def set_mode(target_user_id, mode):
    if mode == "recommendation":
        user_info.loc[target_user_id, 'recommendation_availability'] = True
        print("mode to recommendation\n\n")
        return
    if mode == "default":
        user_info.loc[target_user_id, 'recommendation_availability'] = False
        user_info.loc[target_user_id, 'applicable_missions'] = pd.DataFrame()
        print("mode to default\n\n")
        return
        
def add_new_user(target_user_id):
    for weather in weather_category:
        classified_R.loc[weather, 'value'].loc[np.int64(target_user_id)] = None
        classified_R_hat.loc[weather, 'value'].loc[np.int64(target_user_id)] = None
    user_info.loc[np.int64(target_user_id)] = None

def init_keep_info():
    global keep_info
    keep_info = pd.DataFrame(columns = ['mission_id'])
    return

def add_keep_info(target_user_id, mission_id):
    global keep_info
    if target_user_id not in keep_info.index:
        target_user_id.loc[target_user_id] = [[]]

    if mission_id not in keep_info.loc[target_user_id, 'mission_id']:
        keep_info.loc[target_user_id, 'mission_id'].append(mission_id) 

def add_default_mission(mission_id, rating, cost, g_per_cost, weather):
    global user_info
    for i in user_info.index:
        user_info.lo[i, 'default_mission'] = [mission_id, rating, cost, g_per_cost, weather]

#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm#



def calculate_R_hat(where):
    pd.set_option('display.max_rows', 1000)
    global user_info, mission_info, classified_R_hat, classified_R, original_R, weather_category, keep_info
    db=DB()
    db.dbConnect()
    db.setCursorDic()


    sql="UPDATE User set isWeekFirst=1"
    try:
        db.curs.execute(sql)
        db.conn.commit()
    except mysql.connector.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))
    weather_category = ['sunny', 'cloudy']
    '''
    유저와 관련된 정보들 가져오기
    '''
    sql = "SELECT userIndex as user_id,1 as time_min, time_affordable as time_max, expense_affordable as cost,NULL as applicable_missions, NULL as weekly_missions FROM User"
    try:
        db.curs.execute(sql)
        user_list_db = db.curs.fetchall()
    except mysql.connector.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    if (where == 1):
        user_info = pd.DataFrame(data=user_list_db, columns=['user_id', 'time_min', 'time_max', 'cost', 'applicable_missions', 'weekly_missions', 'weekly_missions_log', 'daily_mission', 'default_mission', 'recommendation_availability'])
        user_info.set_index('user_id', inplace=True, drop=True)

    weather_category = ['sunny', 'cloudy']

    '''
        미션과 관련된 정보들 불러오기
        '''
    sql = "SELECT missionID as mission_id, missionTime as time, expense as cost FROM Mission"
    try:
        db.curs.execute(sql)
        mission_list_db = db.curs.fetchall()
    except mysql.connector.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    mission_info = pd.DataFrame(data=mission_list_db, columns=['mission_id', 'time', 'cost'])
    mission_info.set_index('mission_id', inplace=True)

    temperature_dic = get_max_min_weekly_weather()
    temperature_min = int(temperature_dic['min'])
    temperature_max = int(temperature_dic['max'])

    '''
        미션 평가들 가져오기
        '''
    sql = "SELECT user, mission, weather,temperature, rating FROM MissionEvaluation"
    try:
        db.curs.execute(sql)
        mission_evaluation_list_db = db.curs.fetchall()
    except mysql.connector.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    mission_evaluation_df = pd.DataFrame(data=mission_evaluation_list_db,
                                         columns=['user', 'mission', 'weather', 'temperature', 'rating'])


    sql = "SELECT count(*) as cnt FROM MissionEvaluation"
    try:
        db.curs.execute(sql)
        _cnt = db.curs.fetchone()
    except mysql.connector.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    R_user_id = list(mission_evaluation_df.loc[:, 'user'])
    R_mission_id = list(mission_evaluation_df.loc[:, 'mission'])
    R_weather = list(mission_evaluation_df.loc[:, 'weather'])
    R_temperature = list(mission_evaluation_df.loc[:, 'temperature'])
    R_rating = list(mission_evaluation_df.loc[:, 'rating'])
    R_data_num = _cnt['cnt']


    log = get_init_log(weather_category, user_info.index, mission_info.index)
    
    START = time.time()
   
    classified_R, original_R = get_classified_R(user_info.index, mission_info.index, weather_category, temperature_min,
                                              temperature_max, R_user_id, R_mission_id, R_weather, R_temperature, R_rating,
                                              R_data_num)
    print("classified_R, original_R", time.time() - START)
   

    #classified_R_hat = get_classified_R_hat_by_KNN(classified_R, log)
    # classified_R_hat = get_classified_R_hat_by_Regression(classified_R)
    classified_R_hat = get_classified_R_hat_by_MatrixCompletion(classified_R, original_R)



@missionBundlePage.route('/set', methods=['GET', 'POST'])
def setMissionBundle():
    #app.main.set_R_hat.set_R_hat()
    return 'success'

'''
안드로이드에서 미션을 가져올 때 호출
'''
@missionBundlePage.route('/get', methods=['GET', 'POST'])
def getMissionBundle():
    db=DB()
    db.dbConnect()
    db.setCursorDic()
    _userIndex = request.form['userIndex']
    print("getMissionBundle 함수 호출: ", _userIndex)
    #사용자가 21개의 미션 중 몇 번째 미션을 가지고 와야하는지 변수를 가져옴
    sql = f"SELECT missionOrder FROM User WHERE userIndex = {_userIndex}"
    try:
        db.curs.execute(sql)
        row = db.curs.fetchone()
        _missionOrder = int(row['missionOrder'])
        print("missionOrder : " , _missionOrder)
    except Exception as e:
        print("missionOrder 가져오기 오류 , ",e);
    # -----------------------------------------------------------------

    #MissionBundle에서 userIndex와 missionOrder로 mission을 가져온다.
    sql = f"SELECT missionIndex, missionName FROM MissionBundle join Mission on MissionBundle.missionIndex = Mission.missionID WHERE userIndex = {_userIndex} and missionOrder={_missionOrder}"
    try:
        db.curs.execute(sql)
        row = db.curs.fetchone()

    except Exception as e:
        print("missionIndex, missionName 가져오기 오류 , ",e)
    # -----------------------------------------------------------------


    db.dbDisconnect()
    print(row)
    return json.dumps(row).encode('utf-8')

@missionBundlePage.route('/get1', methods=['GET', 'POST'])
def getDailyMission():

    print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ get> get start ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ\n\n")
    global user_info, mission_info, classified_R_hat, classified_R, original_R, weather_category


    weather_category = ['sunny', 'cloudy']
    weathers = get_weekly_weather_list()

    weekly_weather = get_weekly_weather(weathers, weather_category)
    print("weekly_weather\n", weekly_weather)
    print("\n\n")
    target_user_id = int(request.form['userIndex'])

    db=DB()
    db.dbConnect()
    db.setCursorDic()

    sql = "select count(*) as number from MissionEvaluation where user = %s and date = %s"
    try:
        db.curs.execute(sql, (target_user_id, datetime.datetime.now().strftime('%Y-%m-%d')))
        row = db.curs.fetchone()
    except Exception as e:
        print("오늘 미션 했는지 가져오기 오류 , ", e)
        print("\n\n")

    if row['number']>0:
        sql ="SELECT content as missionName,-1 as missionID FROM HappySaying order by rand() LIMIT 1"
        try:
            db.curs.execute(sql)
            row = db.curs.fetchone()
        except Exception as e:
            print("행복에 관련된 명언 가져오기 오류 , ", e)
            print("\n\n")
        return json.dumps(row).encode('utf-8')
        
    sql = "SELECT isWeekFirst FROM User WHERE userIndex =%s"
    try:
        db.curs.execute(sql, (target_user_id,))
        row = db.curs.fetchone()
        is_week_first = row['isWeekFirst']
    except Exception as e:
        print("is_week_first 가져오기 오류 , ", e)
        print("\n\n")
    
    print("target", target_user_id)

    #print("R", classified_R.loc['sunny']['value'].loc[target_user_id])
    #print("R", classified_R.loc['cloudy']['value'].loc[target_user_id])
    
    #print("origin_R", original_R.loc['sunny']['value'].loc[target_user_id])
    #print("origin_R", original_R.loc['cloudy']['value'].loc[target_user_id])

    print("R_hat", classified_R_hat.loc['sunny']['value'].loc[target_user_id])
    print("R_hat", classified_R_hat.loc['cloudy']['value'].loc[target_user_id])

    if is_week_first==1:

        print("is week first\n\n")

        set_mode(target_user_id, "recommendation")
    
        if isNaN(classified_R.loc['sunny', 'value'].loc[target_user_id].iloc[0]) == True:
            exit = {'missionID' : -2, 'missionName' : "No Mission"}
            return json.dumps(exit).encode('utf-8')

        print("get> set_user_default_missions start\n\n")
        if set_user_default_missions(target_user_id, classified_R_hat, user_info, mission_info, weekly_weather) == False:
            print("There's no mission\n\n")
            exit = {'missionID' : -2, 'missionName' : "No Mission"}
            return json.dumps(exit).encode('utf-8')
        
        print("get> set_user_applicable_missions start\n\n")
       
        if set_user_applicable_missions(target_user_id, classified_R_hat, user_info, mission_info, weekly_weather) == False:
            set_mode(target_user_id, "default")
            print("get> applicablee == None\n\n")
        
        print("get> set_weekly_mission start\n\n")


        if set_weekly_mission(target_user_id, user_info, mission_info, datetime.datetime.today().weekday(), weekly_weather) == False:
            set_mode(target_user_id, "default")
            print("get> weekly == None\n\n")

    
    print("get> default_mission\n", user_info.loc[target_user_id]['default_mission'].sort_values(by = 'mission_id').reset_index(drop = True))
    print("\n\n")

    if get_mode(target_user_id) == "recommendation":
        print("get> applicable_mission\n", user_info.loc[target_user_id]['applicable_missions'].sort_values(by = 'mission_id').reset_index(drop = True))
        print("\n\n")
        print("get> weekly_mission\n", user_info.loc[target_user_id]['weekly_missions'].iloc[0].loc['mission_set'])
        print("\n\n")
        

    today_weather = weathers[datetime.datetime.today().weekday()]

    daily_mission = get_daily_mission(target_user_id, user_info, today_weather)
    print("get> daily mission : ",daily_mission)
    print("\n\n")
   

    sql = "SELECT * FROM Mission WHERE missionID = %s"
    try:
        db.curs.execute(sql, (int(daily_mission),))
        row = db.curs.fetchone()
    except Exception as e:
        print("미션 가져오기 가져오기 오류 , ", e)
        exit = {'missionID' : -2, 'missionName' : "No Mission"}
        row = exit

    sql = "UPDATE User SET isWeekFirst=0 WHERE userIndex = %s"
    try:
        db.curs.execute(sql,(target_user_id,))
        db.conn.commit()
    except Exception as e:
        print("isWeekFirst 설정 오류 , ", e)
        print("\n\n")

    
    db.dbDisconnect()
    print(row)
    print("\n\n")
    print("get> get end")
    return json.dumps(row).encode('utf-8')


'''
미션을 바꾸거나 하루가 지나서 missionOrder를 증가시킬 때 부르는 함수
post를 통해서 count가 넘어오면 count를 증가시킨다.
dislike도 넘어오면(싫어해서 미션을 넘긴거면) 미션 평가를 넣음
'''
@missionBundlePage.route('/increment', methods=['GET', 'POST'])
def incrementMission():
    global user_info, mission_info
    db=DB()
    db.dbConnect()
    db.setCursorDic()
    _userIndex = str(request.form['userIndex'])
    mission_id = request.form['mission']
    weathers = get_weekly_weather_list()
    weather_category = ['sunny', 'cloudy']
    weekly_weather = get_weekly_weather(weathers, weather_category)
    today_idx = get_today_idx()
    target_user_id = int(_userIndex)
    print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ pass start ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ\n\n")
    #count가 넘어왔으면,
    if 'count' in request.form.keys():
        print("pass> count 증가")
        print("\n\n")
        '''
        sql = f"UPDATE User set count=count+1 WHERE userIndex = {_userIndex}"
        try:
            db.curs.execute(sql)
            success = {'count': int(request.form['count'])+1}
        except Exception as e:
            print("count 증가 오류 , ", e)
            print("\n\n")
            success = {'success': 'False'}'''
        success = {'count': int(request.form['count'])}

        print("pass> update_user_applicable_missions start")
        update_user_applicable_missions(target_user_id,  np.int64(mission_id), 'pass', today_idx, weekly_weather)
        print("pass> applicable_mission\n", user_info.loc[target_user_id]['applicable_missions'])
        print("\n\n")

        if get_mode(target_user_id) == "recommendation":
            #print("pass> availability == True")
            START = time.time()
            minimum_cost_mission_set = get_minimum_cost_mission_set(target_user_id, user_info, mission_info, today_idx, weekly_weather)
            print("pass> minimum cost spending time", time.time() - START)
            if type(minimum_cost_mission_set) == type(None):   
                set_mode(target_user_id, "default")
                print('pass> No weekly_mission_set on weekly_weather')
                print("\n\n")
            else:
                minimum_cost = minimum_cost_mission_set.iloc[0]['total_cost']
                print("pass> success and minimum cost is ", minimum_cost)
                print("\n\n")

                if minimum_cost > user_info.loc[target_user_id, 'cost']:
                    print(minimum_cost - user_info.loc[target_user_id, 'cost'], 'pass> more required')
                    print("\n\n")

        if get_mode(target_user_id) == "recommendation":
            #print("pass> mode == recommendation")
            START = time.time()
            print("pass> set_weekly_mission start")
            set_weekly_mission(target_user_id, user_info, mission_info, datetime.datetime.today().weekday(), weekly_weather)
            print("pass> set_weekly_mission spending time", time.time() - START)
            print("pass> weekly_mission\n", user_info.loc[target_user_id]['weekly_missions'].iloc[0].loc['mission_set'])
        else:
            #print("pass> mode == default")
            print("pass> weekly missions == None")
        print("\n\n")
        print("pass> pass end")

        

    #dislike가 넘어왔으면,
    if 'dislike' in request.form.keys():
        missionNumber = str(request.form['mission'])
        id = _userIndex+"."+missionNumber
        sql = "INSERT INTO MissionEvaluation (evaluationIndex,user,mission,rating) VALUES(%s,%s,%s,1) ON DUPLICAE KEY SET rating=1"
        try:
            db.curs.execute(sql,(id,_userIndex,missionNumber))

        except Exception as e:
            print("dislike 삽입 오류 , ", e)
            print("\n\n")

        update_user_applicable_missions(target_user_id,  np.int64(mission_id), 'pass', today_idx, weekly_weather)
        print("pass> applicable_mission\n", user_info.loc[target_user_id]['applicable_missions'])
        print("\n\n")
        if get_mode(target_user_id) == "recommendation":
            START = time.time()
            minimum_cost_mission_set = get_minimum_cost_mission_set(target_user_id, user_info, mission_info, today_idx, weekly_weather)
            print("pass> minimum cost spending time", time.time() - START)
            if type(minimum_cost_mission_set) == type(None):
                print('pass> No weekly_mission_set on weekly_weather')
                print("\n\n")
            else:
                minimum_cost = minimum_cost_mission_set.iloc[0]['total_cost']
                print("pass> success and minimum cost is ", minimum_cost)
                print("\n\n")

                if minimum_cost > user_info.loc[target_user_id, 'cost']:
                    print(minimum_cost - user_info.loc[target_user_id, 'cost'], 'pass> more required')
                    print("\n\n")

        if get_mode(target_user_id) == "recommendation":
            START = time.time()
            set_weekly_mission(target_user_id, user_info, mission_info, datetime.datetime.today().weekday(), weekly_weather)
            print("pass> set_weekly_mission spending time", time.time() - START)
            print("pass> weekly_mission\n", user_info.loc[target_user_id]['weekly_missions'].iloc[0].loc['mission_set'])
        else:
            print("pass> applicable missions == None")    
        print("\n\n")


    db.dbDisconnect()

    #add_keep_info(target_user_id, np.int64(mission_id))

    return json.dumps(success, default=json_default).encode('utf-8')

@missionBundlePage.route('/getSurveyMission', methods=['GET', 'POST'])
def getSurveyMission():
    print("getSurveyMission")
    db=DB()
    db.dbConnect()
    db.setCursorDic()

    # missionOrder 하나 증가시킴
    sql = "SELECT missionID, missionName from Mission LIMIT 0 , 20"
    try:
        db.curs.execute(sql)
        rows = db.curs.fetchall()
    except Exception as e:
        print("missionOrder 증가 오류 , ", e)
        print("\n\n")

    return json.dumps(rows, default=json_default).encode('utf-8')

global user_info, mission_info, classified_R_hat

def user_rating_init(user_id, mission_list, weather_list, temperature_list, rating_list, data_num):
    global classified_R, weather_category
    temperature_dic = get_max_min_weekly_weather()
    t_min = int(temperature_dic['min'])
    t_max = int(temperature_dic['max'])

    
    for i in range(data_num):
        weather = weather_list[i]
        mission_id = mission_list[i]
        rating = rating_list[i]
        temperature = temperature_list[i]
        
        if (t_min <= temperature) and (temperature <= t_max):
            classified_R.loc[weather, 'value'].loc[np.int64(user_id), np.int64(mission_id)] = rating
            other_weathers = list(classified_R.index)
            other_weathers.remove(weather)
            for j in other_weathers:
                classified_R.loc[j, 'value'].loc[np.int64(user_id), np.int64(mission_id)] = "Done"

def json_default(value):
    if isinstance(value, datetime.date):
        return value.strftime('%Y-%m-%d')
    raise TypeError('not JSON serializable')


#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#





