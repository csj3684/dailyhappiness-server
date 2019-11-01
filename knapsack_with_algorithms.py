#!/usr/bin/env python
# coding: utf-8

# In[19]:


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

def isNaN(a):
    return a != a


# In[20]:


def show_R_hat_log(log_for_R_hat, method, user_id, mission_id):
    log = log_for_R_hat.loc[method]['value']
 
    if method == "KNN":
        print("user : ", user_id)
        print("mission : ", mission_id)
        
        mode = log.loc['graph_info']['value'].loc[user_id][mission_id][0]
        scale = log.loc['scale']['value']
        sigma = log.loc['sigma']['value']
        discrete_distribution = log.loc['distribution']['value']
        
        j = 1

        for i in mode.index:
            print("candidate : ", i)
            x = np.arange(-1,9, 0.1)
            y = scale * mode.loc[i]['n'] * np.exp((-1) * (((x + 1 - i) ** 2) / (2 * (sigma ** 2))))
            discrete_distribution.plot.bar()
            plt.plot(x, y)
            plt.show()
            print(log.loc['graph_info']['value'].loc[user_id][mission_id][j])
            j += 1
            
    elif method == "regression":
        print(log.loc[user_id][mission_id])
        
        
def show_knapsack(knapsack):
    for i in knapsack.index:
        print(i, "\tg :", round(knapsack.loc[i]['daily_missions']['expected_rating'],2), "\tcost :", round(knapsack.loc[i]['daily_missions']['total_cost'], 2), "\tmissions :", knapsack.loc[i]['daily_missions']['mission_id'])
    print()
    
def show_user_knapsack(log_for_knapsack, user_id):
    print("user ID : ", user_id, "\nlimited cost : ", log_for_knapsack.loc[user_id]['limited_cost'], "\nN : ", log_for_knapsack.loc[user_id]['N'])
    print()
    frame = pd.DataFrame(columns = ['g', 'cost', 'missions'])
    for i in range(len(log_for_knapsack.loc[user_id]['mission_batches'])):
        frame.loc[i] = [log_for_knapsack.loc[user_id]['mission_batches'][i]['expected_rating'], log_for_knapsack.loc[user_id]['mission_batches'][i]['total_cost'], log_for_knapsack.loc[user_id]['mission_batches'][i]['mission_id']]
        
    print(frame)
    print()
        
    


# In[21]:


# Matrix Completion 클래스들

class MatrixFactorization():
    def __init__(self, R, k, learning_rate, reg_param, epochs, verbose=False):
        """
        :param R: rating matrix
        :param k: latent parameter
        :param learning_rate: alpha on weight update
        :param reg_param: beta on weight update
        :param epochs: training epochs
        :param verbose: print status
        """

        self._R = R
        self._num_users, self._num_items = R.shape
        self._k = k
        self._learning_rate = learning_rate
        self._reg_param = reg_param
        self._epochs = epochs
        self._verbose = verbose


    def fit(self):
        interval = 200
        """
        training Matrix Factorization : Update matrix latent weight and bias

        참고: self._b에 대한 설명
        - global bias: input R에서 평가가 매겨진 rating의 평균값을 global bias로 사용
        - 정규화 기능. 최종 rating에 음수가 들어가는 것 대신 latent feature에 음수가 포함되도록 해줌.

        :return: training_process
        """

        # init latent features
        self._P = np.random.normal(size=(self._num_users, self._k))
        self._Q = np.random.normal(size=(self._num_items, self._k))

        # init biases
        self._b_P = np.zeros(self._num_users)
        self._b_Q = np.zeros(self._num_items)
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

            #print status
            if self._verbose == True and ((epoch + 1) % interval == 0):
                print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))


    def cost(self):
        """
        compute root mean square error
        :return: rmse cost
        """

        # xi, yi: R[xi, yi]는 nonzero인 value를 의미한다.
        # 참고: http://codepractice.tistory.com/90
        xi, yi = self._R.nonzero()
        predicted = self.get_complete_matrix()
        cost = 0
        for x, y in zip(xi, yi):
            cost += pow(self._R[x, y] - predicted[x, y], 2)
        return np.sqrt(cost) / len(xi)


    def gradient(self, error, i, j):
        """
        gradient of latent feature for GD

        :param error: rating - prediction error
        :param i: user index
        :param j: item index
        :return: gradient of latent feature tuple
        """

        dp = (error * self._Q[j, :]) - (self._reg_param * self._P[i, :])
        dq = (error * self._P[i, :]) - (self._reg_param * self._Q[j, :])
        return dp, dq


    def gradient_descent(self, i, j, rating):
        """
        graident descent function

        :param i: user index of matrix
        :param j: item index of matrix
        :param rating: rating of (i,j)
        """

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
        """
        get predicted rating: user_i, item_j
        :return: prediction of r_ij
        """
        return self._b + self._b_P[i] + self._b_Q[j] + self._P[i, :].dot(self._Q[j, :].T)


    def get_complete_matrix(self):
        """
        computer complete matrix PXQ + P.bias + Q.bias + global bias

        - PXQ 행렬에 b_P[:, np.newaxis]를 더하는 것은 각 열마다 bias를 더해주는 것
        - b_Q[np.newaxis:, ]를 더하는 것은 각 행마다 bias를 더해주는 것
        - b를 더하는 것은 각 element마다 bias를 더해주는 것

        - newaxis: 차원을 추가해줌. 1차원인 Latent들로 2차원의 R에 행/열 단위 연산을 해주기위해 차원을 추가하는 것.

        :return: complete matrix R^
        """
        return self._b + self._b_P[:, np.newaxis] + self._b_Q[np.newaxis:, ] + self._P.dot(self._Q.T)


    def print_results(self):
        """
        print fit results
        """

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
        print(self._training_process[self._epochs-1][1])


# In[22]:


# Regression 함수들

def get_formula(target_mission_id, experienced_mission):
    FORMULA = target_mission_id + " ~"
    for i in range(len(experienced_mission)):
        FORMULA = FORMULA + " " + experienced_mission[i]
        if i != len(experienced_mission) - 1 :
            FORMULA = FORMULA + " +"
    return FORMULA

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
        
def get_p_value(ols_result):
    p_value = pd.DataFrame()
    p_value['p_ value'] = ols_result.pvalues
    return p_value

def IsAvailable(ols_result):
    result = False
    #summary = ols_result.summary()
    
    return True


# In[23]:


# KNN 함수들

def get_distance_and_default_weight(user1, user2): # user 파라미터 = R.loc[user_id]
    shared_distance = 0
    distance2_for_1 = 0
    distance1_for_2 = 0
    tolarance = 5
    penalty_cofficient = 1
    penalty_for_1 = 0;
    penalty_for_2 = 0;
    NaN = True
    
    for i in user1.index:
        if (user1[i]['rating'] == -1) and (user2[i]['rating'] == -1):

            continue
        elif (user1[i]['rating'] == -1) and (user2[i]['rating'] != -1):
            penalty_for_1 += 1

        elif (user1[i]['rating'] != -1) and (user2[i]['rating'] == -1):
            penalty_for_2 += 1

        else:
            if user1[i]['weather'] != user2[i]['weather']:
                penalty_for_1 += 1
                penalty_for_2 += 1

            elif abs(user1[i]['temperature'] - user2[i]['temperature']) > tolarance:
                penalty_for_1 += 1
                penalty_for_2 += 1

            else:
                shared_distance += ( np.power(user1[i]['rating'] - user2[i]['rating'], 2) +
                                     np.power(user1[i]['temperature'] - user2[i]['temperature'],2))
                NaN = False

    shared_distance = np.sqrt(shared_distance)
    if NaN == True:
        return {'distance' : -1, 'weight' : -1}, {'distance' : -1, 'weight' : -1}
    else:
        return {'distance' : shared_distance * (np.power(1.1, penalty_for_1 * penalty_cofficient)), 'weight' : -1}, {'distance' : shared_distance * (np.power(1.1, penalty_for_2 * penalty_cofficient)), 'weight' : -1 }
        
def get_D(R): # R = DataFrame(index = user_id, columns = mission_id ,data = {날씨, 기온, rating}
    distance_matrix = pd.DataFrame(index = R.index, columns = R.index)
    for i in range(R.index.size):
        for j in range(i+1, R.index.size):
            distance_matrix.loc[R.index[i]][R.index[j]], distance_matrix.loc[R.index[j]][R.index[i]] = get_distance_and_default_weight(R.iloc[i], R.iloc[j]) 
    
    return distance_matrix

def availability_for_representative_by_N(scale, k, x_mode, n_mode, x, n, sigma, log):
  
    log.loc[x]['accepted percentile'] = (str(int(round((scale * n_mode * np.exp((-1) * (((x - x_mode) ** 2) / (2 * (sigma ** 2))))) / n_mode * 100, 0))) + "%")
    log.loc[x]['n'] = n
    log.loc[x]['mask'] = round(scale * n_mode * np.exp((-1) * (((x - x_mode) ** 2) / (2 * (sigma ** 2)))), 1)
    
    if n <= (scale * n_mode * np.exp((-1) * (((x - x_mode) ** 2) / (2 * (sigma ** 2))))):
        if x_mode == x :
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
    
    mode = pd.DataFrame(columns = ['n'])
    
    max_n = 0
    
    for i in discrete_distribution.index:
        if discrete_distribution.loc[i]['n'] > max_n:
            max_n = discrete_distribution.loc[i]['n']
        
    for i in discrete_distribution.index:
        if discrete_distribution.loc[i]['n'] == max_n:
            mode.loc[i] = max_n
    
    return mode

def get_representative_value(discrete_distribution, scale, k, sigma, log_for_R_hat, user_id, mission_id):
    
    breaker = False
    mode = get_modes(discrete_distribution)
    representative_value = pd.DataFrame(columns = ['value'])

    log_for_R_hat.loc[user_id][mission_id] = []
    log_for_R_hat.loc[user_id][mission_id].append(mode)
    
    for i in mode.index:
        
        log = pd.DataFrame(index = discrete_distribution.index, columns = ['accepted percentile', 'n', 'mask', 'P/F'])
       
        for j in discrete_distribution.index:
            if availability_for_representative_by_N(scale, k, i, mode.loc[i]['n'], j, discrete_distribution.loc[j]['n'], sigma, log) == False:
            ##if availability_for_representative_by_Customized(scale, k, i, mode.loc[i]['n'], j, discrete_distribution.loc[j]['n'], sigma) == False:
                breaker = True
                #break
        
        log_for_R_hat.loc[user_id][mission_id].append(log)
       
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

def find_k_nearest(ratings, distances_and_weights, user_idx):

    k = 0
    sum_influence = 0
    k_nearest = pd.DataFrame(index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], columns = ['n'], data = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])    
    
    while k < ratings.index.size:
        if distances_and_weights.loc[user_idx.index[k]]['distance'] == 0:
            influence = 3
        else:
            influence = (1 / distances_and_weights.loc[user_idx.index[k]]['distance'] ** 2)
            
        sum_influence += influence

        if influence / sum_influence < 0.001:
            break;

        k += 1


    for i in range(ratings.index.size):
        distances_and_weights.loc[user_idx.index[i]]['weight'] = (influence / sum_influence)


    for i in range(k):
        k_nearest.loc[ratings.iloc[i]['rating']] += distances_and_weights.loc[user_idx.index[i]]['weight'] * 100

    return k_nearest


def show_progress(current_sack, g, accumulated_cost, missions, i):
    global weekly_cost
    global best
    global N
    global time_min
    global time_max
    global sacks
    
    if sacks.index.size == 0:
        print("sacks : empty")
    else:
        print(sacks)
    print()
    print("current_sack : ", end="")
    q = 0
    k = 0
    while k < i:
        if q == len(current_sack):
            print("X", end=" ")
            k += 1
            continue

        if current_sack[q] == missions.index[k]:
            print(current_sack[q], end=" ")
            q += 1
        else:
            print("X", end=" ")

        k += 1
    print()
    print("g : ", g)

    print("accumulated_cost : ", accumulated_cost)

    print("best : ", best)

# sacks = dataFrame, current_sack = [], missions = dataFrame
def get_sacks(current_sack, g, accumulated_cost, missions, i):
    global weekly_cost
    global best
    global N
    global time_min
    global time_max
    global sacks

    #show_progress(current_sack, g, accumulated_cost, missions, i)

    if len(current_sack) == N:
        #print("sack size 도달")
        sack = pd.DataFrame(data=[[copy.deepcopy(current_sack), g, accumulated_cost]], columns=['sack', 'g', 'cost'])
        sacks = sacks.append(sack)
        sacks = sacks.reset_index(drop=True)
        return

    if i == missions.index.size:
        #print("마지막 미션 도달")
        sack = pd.DataFrame(data=[[copy.deepcopy(current_sack), g, accumulated_cost]], columns=['sack', 'g', 'cost'])
        sacks = sacks.append(sack)
        sacks = sacks.reset_index(drop=True)
        return

    # g + h 계산
    h = 0
    virtual_accumulated_cost = accumulated_cost
    virtual_index = i

    # 다 담을 수 있는데 까지 최대한 담고
    while (((weekly_cost - virtual_accumulated_cost) / missions.iloc[virtual_index].required_cost) > 1):
        h += missions.iloc[virtual_index].expected_R
        virtual_accumulated_cost += missions.iloc[virtual_index].required_cost
        virtual_index += 1
        
        if (virtual_index == missions.index.size) or (len(current_sack) + virtual_index - i == N):
            break

    # 안담아지면 단위무게로 쪼개서 마지막까지 채우기
    if (virtual_index < missions.index.size) and (len(current_sack) + virtual_index - i != N):
        h += ((weekly_cost - virtual_accumulated_cost) / missions.iloc[virtual_index].required_cost) * missions.iloc[virtual_index].expected_R
    #print("g+h : ", g + h)
    if g + h < best:
        #print("더이상 해볼 필요 없음 optimal < best")
        return


    if accumulated_cost + missions.iloc[i].required_cost < weekly_cost:
        if (time_min <= missions.iloc[i].required_time) and (missions.iloc[i].required_time <= time_max):
            g += missions.iloc[i].expected_R
            if g > best:
                best = g
            accumulated_cost += missions.iloc[i].required_cost
            current_sack.append(missions.index[i])

            #print(missions.index[i], "넣음")
            get_sacks(current_sack, g, accumulated_cost, missions, i + 1)
            g -= missions.iloc[i].expected_R
            accumulated_cost -= missions.iloc[i].required_cost
            del current_sack[-1]
        #else:
            #print()
            #print(missions.index[i], "못넣음 : 시간 범위 초과")
    #else:
        #print()
        #print(missions.index[i], "못넣음 : cost 초과")

    get_sacks(current_sack, g, accumulated_cost, missions, i + 1)


def get_daily_missions_sacks(T_min, T_max, weekly_limited_cost, weekly_item_num):
    
    global weekly_cost, best, N, time_min, time_max, sacks
    global mission_id, expected_R, required_time, required_cost

    weekly_cost = weekly_limited_cost
    best = 0
    N = weekly_item_num
    time_min = T_min
    time_max = T_max

    mission = pd.DataFrame(index=mission_id, columns=['expected_R', 'required_time', 'required_cost', 'R_per_cost'])

    for i in range(len(mission_id)):
        print(expected_R[i])
        print(required_time[i])
        print(required_cost[i])
        print((float)(expected_R[i]) / required_cost[i])
        print([expected_R[i], required_time[i], required_cost[i], (float)(expected_R[i]) / required_cost[i]])
        mission.loc[mission_id[i]] = [expected_R[i], required_time[i], required_cost[i], (float)(expected_R[i]) / required_cost[i]]

    # dataframe = index : mission_id, columns : expected_R, required_time, required_cost (expected_R 내림차순 정렬)
    mission = mission.sort_values(by='R_per_cost', ascending=False)

    sacks = pd.DataFrame(columns=['sack', 'g', 'cost'])

    get_sacks(current_sack=[], g=0, accumulated_cost=0, missions=mission, i=0)

    sacks = sacks.sort_values(by='g', ascending=False)

    return sacks

                                    
            





def get_R_hat_by_Regression(R, users_idx, missions_idx, data_num, log_for_R_hat):
    R_hat = pd.DataFrame(index = users_idx, columns = missions_idx)
    
    log_for_R_hat.loc['regression']['value'] = pd.DataFrame(index = users_idx, columns = missions_idx)
    log = log_for_R_hat.loc['regression']['value']
    
    for i in R.index:
        tmp = []
        for j in R.columns:
            tmp.append(R.loc[i][j]['rating'])
        R_hat.loc[i] = tmp
    
    R_refer = copy.deepcopy(R_hat)
    
    R_refer_for_regression = copy.deepcopy(R_hat)

    for i in range(len(users_idx)):
        target_user_id = users_idx[i]
        experienced_mission = []
        unexperienced_mission = []
        for j in R.loc[target_user_id].index:
            if R.loc[target_user_id][j]['rating'] == -1:
                unexperienced_mission.append(j)
            else:
                experienced_mission.append(j)
        
        indep_user_idx = pd.DataFrame(index = R.index, columns = ['idx'])
                
        for k in indep_user_idx.index:                                    # target user의 경험을 완전히 포함하지 못하는 user 제거
            for l in range(len(experienced_mission)): 
                if R.loc[k][experienced_mission[l]]['rating'] == -1:
                    indep_user_idx.drop(k, inplace = True)                
                    break;
     
        for k in R_refer_for_regression.columns:                                   
            if R.loc[target_user_id][k]['rating'] == -1:
                R_refer_for_regression.drop(k, axis = 1, inplace = True)                 # target user 가 안해본 mission 제거 
             
        for j in range(len(unexperienced_mission)):

            target_mission_id = unexperienced_mission[j] 
            user_have_not_done_target_mission = []
            
            for k in indep_user_idx.index:
                if R.loc[k][target_mission_id]['rating'] == -1:
                    user_have_not_done_target_mission.append(k)
                    indep_user_idx.drop(k, inplace = True)           # target_mission 을 안해본 user 제거 
                    

            if (indep_user_idx.index.size == 0) or (R_refer_for_regression.columns.size == 0):
                for i in user_have_not_done_target_mission:          # target mission 을 안해봐서 제거된 user 복구
                    indep_user_idx.loc[i] = None
                continue;        
        

        
            R_refer_for_regression.loc[:, target_mission_id] = R_refer.loc[:, target_mission_id] # target mission 복구
            
            FORMULA = get_formula(target_mission_id, experienced_mission) # FORMULA = "Y ~ M_1 + M_2 + ... + M_K"

            R_for_regression = pd.DataFrame(index = indep_user_idx.index, columns = R_refer_for_regression.columns)

            for user in R_for_regression.index:                       # 회귀에 필요한 미션과 유저들로 DataFrame 생성
                R_for_regression.loc[user] = R_refer_for_regression.loc[user]
                
            ols_result = smf.ols(FORMULA, data = R_for_regression).fit()
            
            log.loc[target_user_id][target_mission_id] = ols_result.summary()

            
            #vif = get_vif(indep_variables = X)
            
            if IsAvailable(ols_result) == True:

                parameters = copy.deepcopy(ols_result.params)
                expected_Y = parameters.loc['Intercept']
                parameters.drop('Intercept', inplace = True)

                for k in parameters.keys():
                    expected_Y += ( R_refer.loc[target_user_id][k] * parameters.loc[k] ) 

                R_hat.loc[target_user_id][target_mission_id] = expected_Y

            else :
                R_hat.loc[target_user_id][target_mission_id] = None
            
            R_refer_for_regression.drop(target_mission_id, axis = 1, inplace = True) # target mission 삭제
            for i in user_have_not_done_target_mission:               # target mission 을 안해봐서 제거된 user 복구
                    indep_user_idx.loc[i] = None
        
        for k in unexperienced_mission:                                  
            R_refer_for_regression.loc[:, k] = R_refer.loc[:, k]           # target user 가 안해본 mission 복구
        
    for i in R_hat.index:
        for j in R_hat.columns:
            if R_hat.loc[i][j] < 0 :
                R_hat.loc[i][j] = 0
    return R_hat





"""
regression
R_hat 리턴용

R_refer 복사
R_reer_for_regression 복사
user_idx 복사

	for user
		user 거르기
		target user가 안해본 미션 없애기
			for mission
				target mission 안하 사람 없애기
				target mission 복구

				회귀 및 R_hat 채우기

				target mission 제거
				target mission 안한 사람 없앤거 복귀
		target user가 안해본 미션 복구
		걸렀던 user 복구

"""





def get_R_hat_by_KNN(R, k, users_idx, missions_idx, data_num, log_for_R_hat):
    R_hat = pd.DataFrame(index = users_idx, columns = missions_idx)
    log_for_R_hat.loc['KNN']['value'] = pd.DataFrame(index = ['scale', 'sigma', 'distribution', 'graph_info'], columns = ['value'])
    log_for_R_hat.loc['KNN']['value'].loc['graph_info']['value'] = pd.DataFrame(index = users_idx, columns = missions_idx)
    log = log_for_R_hat.loc['KNN']['value']
    
    for i in range(data_num):
        R_hat.loc[users_id[i]][missions_id[i]] = R.loc[users_id[i]][missions_id[i]]['rating']
    
    D = get_D(R)

    for i in range(len(users_idx)):
        target_user_id = users_idx[i]
        unexperienced_mission = []   

        for mission in R.loc[target_user_id].index:
            if R.loc[target_user_id][mission]['rating'] == -1:
                unexperienced_mission.append(mission)

        for j in range(len(unexperienced_mission)):
            target_mission_id = unexperienced_mission[j]
            target_mission_R = copy.deepcopy(R.loc[:, target_mission_id]) # target_mission에 대한 user 정보
            others_distance = copy.deepcopy(D.loc[:, target_user_id]) # target_user 와의 거리

            others_distance.drop(target_user_id, inplace = True)

            for i in others_distance.index: # target_user 랑 거리를 측정할 수 없는 user들 제거
                if others_distance.loc[i]['distance'] == -1:
                    target_mission_R.drop(i, inplace = True)

            for i in target_mission_R.index: # target_mission 점수 없는 user들 제거
                if target_mission_R.loc[i]['rating'] == -1:
                    target_mission_R.drop(i, inplace = True)

            if target_mission_R.index.size == 0:
                R_hat.loc[target_user_id][target_mission_id] = None
                continue
            
            sorted_user_by_distance = pd.DataFrame(index = target_mission_R.index, columns = ['distance'])
            
            for i in target_mission_R.index:
                sorted_user_by_distance.loc[i] = others_distance.loc[i]['distance']
            
            sorted_user_by_distance = sorted_user_by_distance.sort_values(by = 'distance')
            
            k_nearest = find_k_nearest(target_mission_R, others_distance, sorted_user_by_distance)
            
            scale = 1.2
            
            sigma = 2.56
    
            log.loc['scale'] = scale
            log.loc['sigma'] = sigma
            log.loc['distribution']['value'] = k_nearest
  
            R_hat.loc[target_user_id][target_mission_id] = get_representative_value(k_nearest, scale, k, sigma, log.loc['graph_info']['value'], target_user_id, target_mission_id)
            
            #print(log.loc[target_user_id][target_mission_id][0])   
            
    return R_hat





def get_R_hat_by_MatrixCompletion(R, users_idx, missions_idx, data_num, log_for_R_hat):
    
    R_hat = pd.DataFrame(index = users_idx, columns = missions_idx)

    for i in range(data_num):
        R_hat.loc[users_id[i]][missions_id[i]] = R.loc[users_id[i]][missions_id[i]]['rating']
    
    # rating matrix - User X Item : (7 X 5)
    R_ = np.array(R_hat)

    
    # P, Q is (7 X k), (k X 5) matrix
    factorizer = MatrixFactorization(R_, k = 5, learning_rate=0.01, reg_param=0.01, epochs= 200, verbose=True)
    factorizer.fit()

    R_ = factorizer.get_complete_matrix()

    
    for i in range(R_hat.index.size):
        for j in range(R_hat.columns.size):
            if R_[i, j] < 0:
                R_hat.iloc[i][j] = 0 
            elif R_hat.iloc[i][j] == -1:
                R_hat.iloc[i][j] = R_[i, j]
    
    return R_hat





def get_knapsack(R_hat, users_idx, log_for_knapsack):
    
    global mission_id, expected_R, required_time, required_cost
    
    knapsack = pd.DataFrame(index = users_idx, columns = ['daily_missions'])
    
    for i in range(len(users_idx)):
        target_user_id = users_idx[i]

        target_user_rating = copy.deepcopy(R_hat.loc[target_user_id])
        for i in target_user_rating.index:
            if isNaN(target_user_rating[i]) == True:
                target_user_rating.drop(i, inplace = True)

        mission_id = list(target_user_rating.index)
        expected_R  = list(target_user_rating)
        required_time = []
        required_cost = []
        for i in range(len(mission_id)):
            required_time.append(20)
            required_cost.append(random.random())

        sacks = get_daily_missions_sacks(T_min = 0, T_max = 50, weekly_limited_cost = 7, weekly_item_num = 7)

        mission_batches = []
        
        for i in range(sacks.index.size):
            mission_batches.append({'mission_id': sacks.iloc[i].sack, 'expected_rating' : sacks.iloc[i].g, 'total_cost' : sacks.iloc[i].cost})
        
        log_for_knapsack.loc[target_user_id]['mission_batches'] = mission_batches
        log_for_knapsack.loc[target_user_id]['limited_cost'] = 7
        log_for_knapsack.loc[target_user_id]['N'] = 7
        
        knapsack.loc[target_user_id]['daily_missions'] = mission_batches[0]

    return knapsack


#------------------------------------------------ main ----------------------------------------------------#


start = time.time()
data = pd.read_csv('/home/csj3684/2019_2/capstone_project/Test_data/rating_sample.csv', engine='python')
users_id = data.loc[:,'users_id']
missions_id = data.loc[:,'missions_id']
weather = data.loc[:,'weather']
temperature = data.loc[:,'temperature']
rating = data.loc[:,'rating']
data_num = len(users_id)

users_idx = list(OrderedDict.fromkeys(users_id))
missions_idx = list(OrderedDict.fromkeys(missions_id))
R = pd.DataFrame(index = users_idx, columns = missions_idx)

log_for_R_hat = pd.DataFrame(index = ['regression', 'KNN', 'matrix_completion'], columns = ['value'])


for i in range(data_num):
    R.loc[users_id[i]][missions_id[i]] = {'weather' : weather[i], 'temperature' : temperature[i], 'rating' : rating[i]}

R_hat = get_R_hat_by_KNN(R, 3, users_idx, missions_idx, data_num, log_for_R_hat)

print("R_hat 구하기 : ",time.time() - start, "초")

R_hat

show_R_hat_log(log_for_R_hat, "KNN", 'u1', 'M2')

start = time.time()

log_for_knapsack = pd.DataFrame(index = users_idx, columns = ['mission_batches', 'limited_cost', 'N'])

knapsack = get_knapsack(R_hat, users_idx, log_for_knapsack)

print("knapsack 구하기 : ",time.time() - start, "초\n")

show_knapsack(knapsack)

for i in range(5):
    show_user_knapsack(log_for_knapsack, users_idx[i])







