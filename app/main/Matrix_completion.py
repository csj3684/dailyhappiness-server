import numpy as np
import pandas as pd
from patsy import dmatrices
import scipy.stats as ss
import copy
import time
from collections import OrderedDict


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

            # print status
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
        print(self._training_process[self._epochs - 1][1])


def init_evaluations(data):
    for i in range(1, len(data)):
        row = data.loc[i]
        evaluations[str(row[0])] = {}
        for j in range(len(row) - 3):
            evaluations[str(row[0])][str(data.loc[0][3 + j])] = {'weather': str(row[1]), 'temperature': int(row[2]),
                                                                 'rating': int(row[3 + j])}
    return evaluations


def get_json_dict(evaluations, R, R_hat):
    users = []
    missions = []
    ratings = []
    evaluations = OrderedDict(sorted(evaluations.items()))
    for user, i in zip(evaluations.keys(), range(len(evaluations.keys()))):
        evaluations[user] = OrderedDict(sorted(evaluations[user].items()))
        for mission, j in zip(evaluations[user].keys(), range(len(evaluations[user].keys()))):
            users.append(user)
            missions.append(mission)
            if evaluations[user][mission]['rating'] == -1:
                ratings.append(R_hat[i][j])
            else:
                ratings.append(evaluations[user][mission]['rating'])
    json_dict = {'users': users, 'missions': missions, 'ratings': ratings}

    return json_dict


# run example


users = []
missions = []
ratings = []
json_dict = {}
evaluations = {}  # evauations['user_id']['mission_id']['weather', 'temperature', 'rating']

# 정보 읽어오기
# data = pd.read_csv('/home/csj3684/2019_2/capstone_project/Test_data/integrated_sample_data.csv', engine='python')
data = pd.read_csv('C:/Users/csj36/Desktop/git local/2019_2/capstone_project/Test_data/40by40.csv', engine='python')

# evaluations 입력
# evaluations['user_id']['mission_id']['weather', 'temperature', 'rating']
evaluations = init_evaluations(data)
R = []

for user, i in zip(evaluations.keys(), range(len(evaluations.keys()))):
    R.append([])
    for mission, j in zip(evaluations[user].keys(), range(len(evaluations[user].keys()))):
        R[i].append(evaluations[user][mission]['rating'])

if __name__ == "__main__":
    # rating matrix - User X Item : (7 X 5)
    R_hat = np.array(R)

    # P, Q is (7 X k), (k X 5) matrix
    factorizer = MatrixFactorization(R_hat, k=5, learning_rate=0.01, reg_param=0.01, epochs=200, verbose=True)
    factorizer.fit()

    json_dict = get_json_dict(evaluations, R, factorizer.get_complete_matrix())

    for i in range(len(evaluations.keys()) * len(evaluations.keys())):
        print(json_dict['users'][i], "\t", json_dict['missions'][i], "\t", json_dict['ratings'][i])


