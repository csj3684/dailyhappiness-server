
def isNaN(a):
    return a != a

def get_classified_R(user_id, mission_id, weather_category, temperature_min, temperature_max, R_user_id, R_mission_id, R_weather, R_temperature, R_rating, R_data_num):
    
    classified_R = pd.DataFrame(index = weather_category, columns = ['value'])
    for weather in classified_R.index:
        classified_R.loc[weather]['value'] = pd.DataFrame(index = user_id, columns = mission_id)
          
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
            
            if (temperature_min <= R_temperature[i]) and (R_temperature[i] <= temperature_max):
                classified_R.loc[weather]['value'].loc[user_id][mission_id] = rating
                other_weathers = list(classified_R.index)
                other_weathers.remove(weather)
                for i in other_weathers:
                    classified_R.loc[i]['value'].loc[user_id][mission_id] = "Done"
                
            
    return classified_R
    
def get_init_classified_R_hat(classified_R):
    
    init_classified_R_hat = pd.DataFrame(index = classified_R.index, columns = ['value'])
    
    for weather in init_classified_R_hat.index:
        user_id = classified_R.loc[weather]['value'].index
        mission_id = classified_R.loc[weather]['value'].columns
        init_classified_R_hat.loc[weather]['value'] = pd.DataFrame(index = user_id, columns = mission_id)
        
    return init_classified_R_hat

def get_init_log(weather_category, user_id, mission_id):
    init_log = pd.DataFrame(index = ['regression', 'knn', 'matrix_completion'], columns = weather_category)
    for method in init_log.index:
        for weather in weather_category:
            init_log.loc[method][weather] = pd.DataFrame(index = user_id, columns = mission_id)
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
        FORMULA +=  (' + ' + formula[i])
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
                log.loc['regression'][weather].loc[i][j] = pd.DataFrame(index = ['ols_result', 'vif'], columns = ['value'], data = [[[]],[[]]])    
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
            
            indep_user_idx = pd.DataFrame(index = R.index, columns = ['idx'])
            indep_user_idx.drop(target_user, inplace = True)

            for indep_user in indep_user_idx.index:                                    # target user의 경험을 완전히 포함하지 못하는 user 제거
                for mission in experienced_mission: 
                    if (R.loc[indep_user][mission] == -1) or (R.loc[indep_user][mission] == 'Done'):
                        indep_user_idx.drop(indep_user, inplace = True)                
                        break;
                        
            if indep_user_idx.index.size == 0:
                continue
                
            for mission in R_refer_for_regression.columns:                                   
                if R.loc[target_user][mission] == -1:
                    R_refer_for_regression.drop(mission, axis = 1, inplace = True)      # target user 가 안해본 mission 제거 
            
            for mission in unexperienced_mission:
                target_mission = mission
                user_have_not_done_target_mission = []

                for indep_user in indep_user_idx.index:
                    if (R.loc[indep_user][target_mission] == -1) or (R.loc[indep_user][mission] == 'Done'):
                        user_have_not_done_target_mission.append(indep_user)
                        indep_user_idx.drop(indep_user, inplace = True)           # target_mission 을 안해본 user 제거 

                if (indep_user_idx.index.size == 0) or (R_refer_for_regression.columns.size == 0):
                    for user in user_have_not_done_target_mission:          # target mission 을 안해봐서 제거된 user 복구
                        indep_user_idx.loc[user] = None
                    continue;       
                    
                formula = copy.deepcopy(list(R_refer_for_regression.columns))    
                    
                R_refer_for_regression.loc[:, target_mission] = R_refer.loc[:, target_mission] # target mission 복구

                available = False
                breaker = False
                
                while available == False:
                    if formula == []:
                        breaker = True
                        log.loc['regression'][weather].loc[target_user, target_mission].loc['vif', 'value'].append(vif)
                        break
                    available = True
                    FORMULA = get_formula(target_mission, formula)
                    
                    R_for_regression = pd.DataFrame(index = indep_user_idx.index, columns = R_refer_for_regression.columns)
                    for user in R_for_regression.index:                       # 회귀에 필요한 미션과 유저들로 DataFrame 생성
                        R_for_regression.loc[user] = R_refer_for_regression.loc[user]

                    y, X = dmatrices(FORMULA, R_for_regression, return_type = 'dataframe')
                    vif = get_vif(X)
                    log.loc['regression'][weather].loc[target_user, target_mission].loc['vif', 'value'].append(vif)
                    for i in vif.index[1:]:
                        if float(vif.loc[i, 'VIF Factor']) > 10:
                            available = False
                            formula.remove(vif.loc[i, 'feature'])
                            
                if breaker == True:
                    R_hat.loc[target_user_id][target_mission] = None
                    R_refer_for_regression.drop(target_mission, axis = 1, inplace = True) # target mission 삭제
                    for i in user_have_not_done_target_mission:                           # target mission 을 안해봐서 제거된 user 복구
                            indep_user_idx.loc[i] = None        
                    continue
                    
                available = False          

                while available == False:
                    if formula == []:
                        breaker = True
                        log.loc['regression'][weather].loc[target_user, target_mission].loc['ols_result', 'value'].append(ols_result)
                        break
                    available = True
                    FORMULA = get_formula(target_mission, formula)
                    ols_result = smf.ols(FORMULA, data = R_for_regression).fit()
                    p_value = get_p_value(result)
                    log.loc['regression'][weather].loc[target_user, target_mission].loc['ols_result', 'value'].append(ols_result)
                    for i in p_value.index[1:]:
                        if p_value.loc[i, 'p_value'] > 0.1:
                            available = False
                            formula.remove(i)
                            
                if breaker == True:
                    R_hat.loc[target_user_id][target_mission] = None
                    R_refer_for_regression.drop(target_mission, axis = 1, inplace = True) # target mission 삭제
                    for i in user_have_not_done_target_mission:                           # target mission 을 안해봐서 제거된 user 복구
                            indep_user_idx.loc[i] = None        
                    continue
                
                parameters = copy.deepcopy(ols_result.params)
                expected_Y = parameters.loc['Intercept']
                parameters.drop('Intercept', inplace = True)

                for k in parameters.keys():
                    expected_Y += ( R_refer.loc[target_user_id][k] * parameters.loc[k] ) 

                R_hat.loc[target_user][target_mission] = expected_Y

                R_refer_for_regression.drop(target_mission, axis = 1, inplace = True) # target mission 삭제
                for i in user_have_not_done_target_mission:               # target mission 을 안해봐서 제거된 user 복구
                        indep_user_idx.loc[i] = None

            for k in unexperienced_mission:                                  
                R_refer_for_regression.loc[:, k] = R_refer.loc[:, k]           # target user 가 안해본 mission 복구

        for i in R_hat.index:
            for j in R_hat.columns:
                if R_hat.loc[i][j] < 0 :
                    R_hat.loc[i][j] = 0
    
    return classified_R_hat

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
    
    for mission in user1.index:
        if (user1[mission] == -1 or user1[mission] == "Done") and (user2[mission] == -1 or user2[mission] == "Done"):
            continue
            
        elif (user1[mission] == -1 or user1[mission] == "Done") and (user2[mission] != -1 and user2[mission] != "Done"):
            penalty_for_1 += 1
            
        elif (user1[mission] != -1 and user1[mission] != "Done") and (user2[mission] == -1 or user2[mission] == "Done"):
            penalty_for_2 += 1
            
        elif (user1[mission] != -1 and user1[mission] != "Done") and (user2[mission] != -1 and user2[mission] != "Done"):
            shared_distance += ( np.power(int(user1[mission]) - int(user2[mission]), 2) ) 
            NaN = False

    shared_distance = np.sqrt(shared_distance)
    if NaN == True:
        return {'distance' : -1, 'weight' : -1}, {'distance' : -1, 'weight' : -1}
    else:
        return {'distance' : shared_distance * (np.power(1.1, penalty_for_1 * penalty_cofficient)), 'weight' : -1}, {'distance' : shared_distance * (np.power(1.1, penalty_for_2 * penalty_cofficient)), 'weight' : -1 }
        
def get_D(R): 
    # R = DataFrame : index = user_id, columns = mission_id ,data = {'weather' : string, 'temperature' : float, 'rating' : int}
    # distance_matrix = DataFrame : index = user_id, columns = user_id, data = {'distance' : float, 'weight' : float}
    
    distance_matrix = pd.DataFrame(index = R.index, columns = R.index)
    for i in range(R.index.size):
        for j in range(i+1, R.index.size):
            distance_matrix.loc[R.index[i]][R.index[j]], distance_matrix.loc[R.index[j]][R.index[i]] = get_distance_and_default_weight(R.iloc[i], R.iloc[j]) 
    
    return distance_matrix


def availability_for_representative_by_N(log, scale, k, x_mode, n_mode, x, n, sigma):
  
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
    # discrete_distribution = dataFrame : index = rating, columns = ['n'], data = rating에 대한 도수 : float
    # mode = DataFrame : index = rating, columns = ['n'], data = 도수 : int
    
    mode = pd.DataFrame(columns = ['n'])
    
    max_n = 0
    
    for i in discrete_distribution.index:
        if discrete_distribution.loc[i]['n'] > max_n:
            max_n = discrete_distribution.loc[i]['n']
        
    for i in discrete_distribution.index:
        if discrete_distribution.loc[i]['n'] == max_n:
            mode.loc[i] = max_n
    
    return mode


def get_representative_value(log, discrete_distribution, scale, k, sigma,user_id, mission_id):
    #representative_value = DataFrame : index = int, columns = ['value'], data = rating : int
    #log = DataFrame : index = rating, columns = ['accepted percentile', 'n', 'mask', 'P/F'], data = [mask 비율 : float, rating 도수 : float, mask : float, 'P/F' : String]
    
    breaker = False
    mode = get_modes(discrete_distribution)
    
    log.loc['representative']['value'] = pd.DataFrame(columns = ['value'])
    representative_value = log.loc['representative']['value']
    log.loc['mode']['value'] = mode
    log.loc['graph']['value'] = pd.DataFrame(columns = ['value'])
    for i in mode.index:
        
        graph_log = pd.DataFrame(index = discrete_distribution.index, columns = ['accepted percentile', 'n', 'mask', 'P/F'])
       
        for j in discrete_distribution.index:
            
            if availability_for_representative_by_N(graph_log, scale, k, i, mode.loc[i]['n'], j, discrete_distribution.loc[j]['n'], sigma) == False:
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
    #k_nearest = DataFrame : index = rating, columns = ['n'], data = 도수 : float
    
    k = 0
    sum_influence = 0
    k_nearest = pd.DataFrame(index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], columns = ['n'], data = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])    

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
        
            
    if k < ratings.index.size:            # 마지막 점 까지 도달해서 끝난게 아니라면 / 0.01 내려가는 다음 점까지는 추가 (출력용)

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
                log.loc['knn'][weather].loc[i][j] = pd.DataFrame(index = ['scale', 'sigma', 'weight_tolerance', 'k', 'weight', 'distribution'], columns = ['value'])    
                log.loc['knn'][weather].loc[i][j].loc['weight']['value'] = pd.DataFrame(columns = ['distance', 'weight', 'rating'])
                log.loc['knn'][weather].loc[i][j].loc['distribution']['value'] = pd.DataFrame(index = ['mode', 'k_nearest', 'graph', 'representative'], columns = ['value'])
        R_hat = classified_R_hat.loc[weather]['value']
    
        D = get_D(R)

        #print(D)
        for user_id in R.index:
            start2 = time.time()
            target_user_id = user_id
            unexperienced_mission = []   

            for mission in R.loc[target_user_id].index:
                if R.loc[target_user_id][mission] == -1:
                    unexperienced_mission.append(mission)

            for j in range(len(unexperienced_mission)):
                target_mission_id = unexperienced_mission[j]
                target_mission_R = copy.deepcopy(R.loc[:, target_mission_id]) # target_mission에 대한 user 정보
                others_distance = copy.deepcopy(D.loc[:, target_user_id]) # target_user 와의 거리
                
                others_distance.drop(target_user_id, inplace = True)
                target_mission_R.drop(target_user_id, inplace = True)
                
                #print(target_user_id)
                #print(target_mission_id)
                
                for i in others_distance.index: # target_user 랑 거리를 측정할 수 없는 user들 제거
                    if others_distance.loc[i]['distance'] == -1:
                        target_mission_R.drop(i, inplace = True)
                #print("거리 없는 user 들 제거된 후\n", target_mission_R.index)
                for i in target_mission_R.index: # target_mission 점수 없는 user들 제거
                    if (target_mission_R.loc[i] == -1) or target_mission_R.loc[i] == "Done":
                        target_mission_R.drop(i, inplace = True)
                #print("점수 없는 user 들 제거된 후\n", target_mission_R.index)
    
                if target_mission_R.index.size == 0:
                    R_hat.loc[target_user_id][target_mission_id] = None
                    continue

                sorted_user_by_distance = pd.DataFrame(index = target_mission_R.index, columns = ['distance'])

                for i in target_mission_R.index:
                    sorted_user_by_distance.loc[i] = others_distance.loc[i]['distance']

                sorted_user_by_distance = sorted_user_by_distance.sort_values(by = 'distance')

                weight_tolerance = 0    
        
                #print("다 제거된 후\n", target_mission_R.index)
                k_nearest = find_k_nearest(log.loc['knn'][weather].loc[target_user_id][target_mission_id], target_mission_R, others_distance, sorted_user_by_distance, weight_tolerance)
                log.loc['knn'][weather].loc[target_user_id][target_mission_id].loc['distribution']['value'].loc['k_nearest']['value'] = k_nearest
                
                scale = 1.2
                sigma = 2.56
                
                log.loc['knn'][weather].loc[target_user_id][target_mission_id].loc['scale']['value'] = scale
                log.loc['knn'][weather].loc[target_user_id][target_mission_id].loc['sigma']['value'] = sigma
                log.loc['knn'][weather].loc[target_user_id][target_mission_id].loc['weight_tolerance']['value'] = weight_tolerance

                log_tmp = log.loc['knn'][weather].loc[target_user_id][target_mission_id].loc['distribution']['value']
                R_hat.loc[target_user_id][target_mission_id] = get_representative_value(log_tmp, k_nearest, scale, -1, sigma, target_user_id, target_mission_id)

        
    
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
        print(self._training_process[self._epochs-1][1])


def get_classified_R_hat_by_MatrixCompletion(classified_R):
    
    classified_R_hat = get_init_classified_R_hat(classified_R)

    for weather in classified_R_hat.index:
        R = classified_R.loc[weather]['value']
        R_hat = classified_R_hat.loc[weather]['value']
        
        for user in R_hat.index:
            for mission in R_hat.columns:
                if (R.loc[user][mission] != "Done") and (R.loc[user][mission] != -1):
                    R_hat.loc[user][mission] = R.loc[user][mission]
                else:
                    R_hat.loc[user][mission] = 0
    
        # rating matrix - User X Item : (7 X 5)
        R_ = np.array(R_hat)


        # P, Q is (7 X k), (k X 5) matrix
        factorizer = MatrixFactorization(R_, k = 5, learning_rate=0.01, reg_param=0.01, epochs= 200, verbose=True)
        factorizer.fit()

        R_ = factorizer.get_complete_matrix()


        for i in range(R_hat.index.size):
            for j in range(R_hat.columns.size):
                if R.iloc[i][j] != -1:
                    R_hat.iloc[i][j] = "Done"
                else:
                    R_hat.iloc[i][j] = R_[i, j]
                if R_[i, j] < 0:
                    R_hat.iloc[i][j] = 0 
            
    
    return classified_R_hat

# 주간 시작 전처리

def back_up(check_point, real_g, applicable_missions, mission_info, weekly_cost, N):
    if check_point.index.size == 0:
        return 0
    
    i = check_point.iloc[check_point.index.size - 1]['index']
    accumulated_cost = check_point.iloc[check_point.index.size - 1]['accumulated_cost']
    weather_condition = copy.deepcopy(check_point.iloc[check_point.index.size - 1]['weather_condition'])
    current_sack = copy.deepcopy(check_point.iloc[check_point.index.size - 1]['current_sack'])
    h = check_point.iloc[check_point.index.size - 1]['h']
    
    check_point.drop(check_point.index[check_point.index.size - 1], inplace = True)
    
    return get_g_plus_h(check_point, weather_condition, applicable_missions, real_g, mission_info, i + 1, weekly_cost, accumulated_cost, current_sack, h, N)

def get_g_plus_h(check_point, weather_condition, applicable_missions, real_g, mission_info, i, weekly_cost, accumulated_cost, current_sack, h, N):

    if i == applicable_missions.index.size:
        return back_up(check_point, real_g, applicable_missions, mission_info, weekly_cost, N)
        
    while (True):
        mission_id = applicable_missions.iloc[i]['mission_id']
        cost = mission_info.loc[mission_id]['cost']
        g = applicable_missions.iloc[i]['g']
        weather = applicable_missions.iloc[i]['weather']
        required_weather_num = weather_condition.loc[weather]['value']
        
        if (weekly_cost - accumulated_cost) <= cost:
            break
            
        if (required_weather_num > 0) and ((mission_id in current_sack.index) == False):
            check_point.loc[check_point.index.size] = [i, accumulated_cost, weather_condition, current_sack, h]
            current_sack.loc[mission_id] = [weather, g, cost]
            h += g
            accumulated_cost += cost
            weather_condition.loc[weather]['value'] -= 1
            
        i += 1

        if current_sack.index.size == N: # 7번째까지 다 담고 남아있는 cost에 대해 마지막 g/c로 다채우기
            h += min(10, (((weekly_cost - accumulated_cost) / cost) * g))
            return real_g + h

        if i == applicable_missions.index.size:
            return back_up(check_point, real_g, applicable_missions, mission_info, weekly_cost, N)

    if current_sack.index.size < 6: # cost 다 써서 채웠는데 7개가 안되면 이번꺼 건너뛰고 다음꺼
        return get_g_plus_h(check_point, weather_condition, applicable_missions, real_g, mission_info, i + 1, weekly_cost, accumulated_cost, current_sack, h, N)
        
    # 단위무게로 쪼개서 마지막까지 채우기
    h += (((weekly_cost - accumulated_cost) / cost) * g)

    return real_g + h
    
def get_applicable_weekly_mission_set(weather_condition, user_id, mission_info, applicable_missions, current_sack, g, accumulated_cost, i, weekly_cost, N):
  
    global weekly_mission_set_candidate, best
    
    #print("current_sack : ", current_sack)
    #print("\n")
    
    if current_sack.index.size == N:
        #print("sack size 도달")
        candidate_set = pd.DataFrame(data=[[copy.deepcopy(current_sack), g, accumulated_cost]], columns=['mission_set', 'total_g', 'total_cost'])
        weekly_mission_set_candidate = weekly_mission_set_candidate.append(candidate_set)
        weekly_mission_set_candidate = weekly_mission_set_candidate.reset_index(drop=True)
        if g > best:
            best = g
        #print("current_sack : ", current_sack)
        #print("\n")
        return

    if i == applicable_missions.index.size:
        #print("마지막 미션 도달")
        if len(current_sack) < 7:
            return
        candidate_set = pd.DataFrame(data=[[copy.deepcopy(current_sack), g, accumulated_cost]], columns=['mission_set', 'total_g', 'total_cost'])
        weekly_mission_set_candidate = weekly_mission_set_candidate.append(candidate_set)
        weekly_mission_set_candidate = weekly_mission_set_candidate.reset_index(drop=True)
        if g > best:
            best = g
        #print("current_sack : ", current_sack)
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
                check_point = pd.DataFrame(columns = ['index', 'accumulated_cost', 'weather_condition', 'current_sack', 'h'])
                g_plus_h = get_g_plus_h(check_point, virtual_weather_condition, applicable_missions, g, mission_info, virtual_idx, weekly_cost, virtual_accmulated_cost, virtual_current_sack, 0, N)
                #print("g+h", g_plus_h)
                #print("best", best)
                
                if g_plus_h < best:
                    return

                g += expected_R
                accumulated_cost += required_cost
                weather_condition.loc[weather]['value'] -= 1
                current_sack.loc[mission_id] = [weather, expected_R, required_cost]
                                  
                get_applicable_weekly_mission_set(weather_condition, user_id, mission_info, applicable_missions, current_sack, g, accumulated_cost, i + 1, weekly_cost, N)

                g -= expected_R
                accumulated_cost -= required_cost
                weather_condition.loc[weather]['value'] += 1
                current_sack.drop(mission_id, inplace = True)
            #else:
                
                #print("미션이 이미 있어요")
        #else:
            
            #print("날씨가 다 찼어요
    #else:
        
        #print(mission_id, "못넣음 : cost 초과")
    
    #print("current_sack 탈출방면")
    #print(current_sack)
    get_applicable_weekly_mission_set(weather_condition, user_id, mission_info, applicable_missions, current_sack, g, accumulated_cost, i + 1, weekly_cost, N)
   
    
def get_weekly_mission_set_candidate(weather_condition, applicable_missions, user_id, mission_info, limited_cost, item_num):
    # mission_and_weather : index = mission_id, columns = weather, data = 0 or 1
    # weekly_mission_set_cadidate --> 'mission_set' = [] : String, mission_id
    
    global weekly_mission_set_candidate, best
                                   
    weekly_mission_set_candidate = pd.DataFrame(columns=['mission_set', 'total_g', 'total_cost'])
                                      
    best = -1    
    
    current_sack = pd.DataFrame(columns = ['weather', 'g', 'cost'])
    
    get_applicable_weekly_mission_set(weather_condition, user_id, mission_info, applicable_missions, current_sack, 0, 0, 0, limited_cost, item_num)
            
    return weekly_mission_set_candidate


def get_applicable_missions(user_id, classified_R_hat, user_info, mission_info, weekly_weather):
    # weekly_mission_set_candidate = columns = ['mission_set', 'cost']
    # 'mission_set' = [] : String, mission_id
    
    required_weather = []
    
    for weather in weekly_weather.index:
        for day in weekly_weather.columns:
            if weekly_weather.loc[weather][day] == 1:
                required_weather.append(weather)
                break;
                
    applicable_missions = pd.DataFrame(columns = ['mission_id', 'g', 'cost', 'g per cost', 'weather'])

    for weather in required_weather:
        R_hat = classified_R_hat.loc[weather]['value']
        for mission in R_hat.columns: 
            if mission_info.loc[mission]['weather'].iloc[0][weather] == 1:
                g = R_hat.loc[user_id][mission]
                cost = mission_info.loc[mission]['cost']
                if (g != "Done") and (g != None):
                    if (user_info.loc[user_id]['time_min'] <= mission_info.loc[mission]['time']) and (mission_info.loc[mission]['time'] <= user_info.loc[user_id]['time_max']):
                        if cost == 0:
                            applicable_missions.loc[applicable_missions.index.size] = [mission, g, cost, float('inf'), weather]
                        else:
                            applicable_missions.loc[applicable_missions.index.size] = [mission, g, cost, float(g) / cost, weather]
    
    if applicable_missions.index.size == 0:
        print("knapsack에 쓰일 수 있는 미션 갯수 = 0")
        return None
    
    return applicable_missions
    
    
def sort_by_g_per_c(frame):
    
    frame.sort_values(by = 'g per cost', ascending = False, inplace = True)

    for n in range(frame.index.size):
        if frame.iloc[n]['cost'] != 0:
            break
            
    if n != 0:
        tmp = frame.iloc[0:n].sort_values(by = 'g', ascending = False)

        t_list = []
        for i in tmp.index:
            t_list += [tmp.loc[i]]

        frame.iloc[0:n] = t_list

    tmp = frame.iloc[n:frame.index.size].sort_values(by = 'g per cost', ascending = False)

    t_list = []
    for i in tmp.index:
        t_list += [tmp.loc[i]]

    frame.iloc[n:frame.index.size] = t_list

def set_user_applicable_missions(target_user_id, classified_R_hat, user_info, mission_info, weekly_weather):
    applicable_missions = get_applicable_missions(target_user_id, classified_R_hat, user_info, mission_info, weekly_weather)
    
    if type(applicable_missions) == type(None):
        return False
    
    user_info.loc[target_user_id]["applicable_missions"] = applicable_missions
    
    return True

def update_user_applicable_missions(user_id, user_info, mission_id, activity, today_idx, weekly_weather):
    applicable_missions = user_info.loc[user_id]['applicable_missions']
    daily_missions = user_info.loc[user_id]['weekly_missions'].iloc[0].loc['mission_set']
    
    if activity == "done":
        daily_missions.drop(mission_id, inplace = True)
        for i in applicable_missions.index:
            if applicable_missions.loc[i]['mission_id'] == mission_id:
                applicable_missions.drop(i, inplace = True)
                    
        
    elif activity == "pass":
        for i in applicable_missions.index:
            if applicable_missions.loc[i]['mission_id'] == mission_id:
                applicable_missions.drop(i, inplace = True)
                
        set_weekly_mission(user_id, user_info, mission_info, today_idx, weekly_weather)
        
def set_weekly_mission(user_id, user_info, mission_info, today_idx, weekly_weather):
    
    applicable_missions = user_info.loc[user_id]['applicable_missions']
    sort_by_g_per_c(applicable_missions)
    
    cost = user_info.loc[user_id]['cost']
    item_num = 7 - today_idx
    
    adjusted_weekly_weather = copy.deepcopy(weekly_weather)
    for i in range(today_idx):
        day = weekly_weather.columns[i]
        adjusted_weekly_weather.drop(day, axis = 1, inplace = True)
        
    
    weather_condition = pd.DataFrame(index = weekly_weather.index, columns = ['value'])
    for weather in weather_condition.index:
        weather_condition.loc[weather] = [sum(adjusted_weekly_weather.loc[weather])]
    
    weekly_mission_set_candidate = get_weekly_mission_set_candidate(weather_condition, applicable_missions, user_id, mission_info, cost, item_num)

    if weekly_mission_set_candidate.index.size == 0:
        return False
    
    weekly_mission_set_candidate = weekly_mission_set_candidate.sort_values(by = "total_g", ascending = False)
    daily_missions = pd.DataFrame(weekly_mission_set_candidate.iloc[0]).T
    
    user_info.loc[user_id]['weekly_missions'] = daily_missions
    
    return True

def get_daily_mission(user_id, user_info, today_weather):
    daily_missions = user_info.loc[user_id]['weekly_missions'].iloc[0].loc['mission_set']
    
    if daily_missions.index.size == 0:
        print("주간 미션들 다시 받아오세요 ~")
        
    for mission in daily_missions.index:
        if daily_missions.loc[mission]['weather'] == today_weather:
            return mission
        
def set_minimum_cost_mission_set(weather_condition, mission_info, applicable_missions, current_sack, g, accumulated_cost, i, N):
    global minimum_cost_mission_set, best_min
        
    if current_sack.index.size == N:
        candidate_set = pd.DataFrame(data=[[copy.deepcopy(current_sack), g, accumulated_cost]], columns=['mission_set', 'total_g', 'total_cost'])
        minimum_cost_mission_set = minimum_cost_mission_set.append(candidate_set)
        minimum_cost_mission_set = minimum_cost_mission_set.reset_index(drop=True)
        if accumulated_cost < best_min:
            best_min = accumulated_cost
        print("sack size 도달")
        print("cost", accumulated_cost)
        print("best", best_min)
        print("current_sack")
        print(current_sack, end = "\n\n\n")
        return

    if i == applicable_missions.index.size:
        if len(current_sack) < 7:
            return
        candidate_set = pd.DataFrame(data=[[copy.deepcopy(current_sack), g, accumulated_cost]], columns=['mission_set', 'total_g', 'total_cost'])
        minimum_cost_mission_set = minimum_cost_mission_set.append(candidate_set)
        minimum_cost_mission_set = minimum_cost_mission_set.reset_index(drop=True)
        if accumulated_cost < best_min:
            best_min = accumulated_cost
        print("모든 미션 순회")
        print("cost", accumulated_cost)
        print("best", best_min)
        print("current_sack")
        print(current_sack, end = "\n\n\n")
        return
    
    
    virtual_idx = i
    virtual_accumulated_cost = accumulated_cost
    virtual_weather_condition = copy.deepcopy(weather_condition)
    virtual_current_size = current_sack.index.size
    virtual_current_sack = list(current_sack.index)
    
    while (virtual_current_size < N) and (virtual_idx < applicable_missions.index.size):
        virtual_mission_id = applicable_missions.iloc[virtual_idx]['mission_id']
        virtual_required_cost = mission_info.loc[virtual_mission_id]['cost']
        virtual_weather = applicable_missions.iloc[virtual_idx]['weather']
        virtual_required_weather_num = virtual_weather_condition.loc[virtual_weather]['value']
        
        if virtual_required_weather_num > 0:
            if (virtual_mission_id in virtual_current_sack) == False:
                virtual_accumulated_cost += virtual_required_cost
                virtual_weather_condition.loc[virtual_weather]['value'] -= 1
                virtual_current_size += 1
                virtual_current_sack.append(virtual_mission_id)
        
        virtual_idx += 1
        
    print("virtual_idx", virtual_idx)
    print("applicable_missions.index.size", applicable_missions.index.size)    
    
    mission_id = applicable_missions.iloc[i]['mission_id']
    required_cost = mission_info.loc[mission_id]['cost']
    expected_R = applicable_missions.iloc[i]['g']
    weather = applicable_missions.iloc[i]['weather']
    required_weather_num = weather_condition.loc[weather]['value']    
        
    print("cost", accumulated_cost)
    print("virtual_accumulated_cost", virtual_accumulated_cost)
    print("best", best_min)
    print("current_sack")
    print(current_sack, end = "\n\n\n") 
        
    if virtual_current_size < N:
        print("virtual_current_size", virtual_current_size)
        print("N", N)
        return
    
    if virtual_accumulated_cost > best_min:
        return
    
    if required_weather_num > 0:
        if (mission_id in current_sack.index) == False:
            
            g += expected_R
            accumulated_cost += required_cost
            weather_condition.loc[weather]['value'] -= 1
            current_sack.loc[mission_id] = [weather, expected_R, required_cost]

            set_minimum_cost_mission_set(weather_condition, mission_info, applicable_missions, current_sack, g, accumulated_cost, i + 1, N)

            g -= expected_R
            accumulated_cost -= required_cost
            weather_condition.loc[weather]['value'] += 1
            current_sack.drop(mission_id, inplace = True)
            
    set_minimum_cost_mission_set(weather_condition, mission_info, applicable_missions, current_sack, g, accumulated_cost, i + 1, N)

    
def get_minimum_cost_mission_set(target_user_id, user_info, mission_info, today_idx, weekly_weather):
    applicable_missions = user_info.loc[target_user_id]['applicable_missions']
    applicable_missions = applicable_missions.sort_values(by = 'cost')
    
    global minimum_cost_mission_set, best_min
    
    best_min = float('inf')
    
    minimum_cost_mission_set = pd.DataFrame(columns=['mission_set', 'total_g', 'total_cost'])
    
    item_num = 7 - today_idx
    
    adjusted_weekly_weather = copy.deepcopy(weekly_weather)
    for i in range(today_idx):
        day = weekly_weather.columns[i]
        adjusted_weekly_weather.drop(day, axis = 1, inplace = True)
        
    weather_condition = pd.DataFrame(index = weekly_weather.index, columns = ['value'])
    for weather in weather_condition.index:
        weather_condition.loc[weather] = [sum(adjusted_weekly_weather.loc[weather])]
    
    current_sack = pd.DataFrame(columns = ['weather', 'g', 'cost'])
    
    set_minimum_cost_mission_set(weather_condition, mission_info, applicable_missions, current_sack, 0, 0, 0, item_num)

    if minimum_cost_mission_set.index.size == 0:
        return None, None
    
    minimum_cost_mission_set = minimum_cost_mission_set.sort_values(by = "total_cost")
    
    return pd.DataFrame(minimum_cost_mission_set.iloc[0]).T, minimum_cost_mission_set
    
    #return pd.DataFrame(minimum_cost_mission_set.iloc[minimum_cost_mission_set.index.size - 1]).T, minimum_cost_mission_set
    
def get_weekly_weather(weathers):
    weekly_weather = pd.DataFrame(index = weather_category, columns = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], data = [[0, 0, 0, 0, 0, 0, 0],])
    for day_idx in range(weekly_weather.columns.size):
        weather = weathers[day_idx]
        weekly_weather.loc[weather][day_idx] = 1
        
    return weekly_weather