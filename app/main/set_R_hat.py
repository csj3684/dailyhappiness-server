#import functions
from app.main.DB import DB
from app.main.Weather import get_max_min_weekly_weather
import app.main.functions as functions
import pymysql
import pandas as pd


import time

user_info=0
mission_info=0
classified_R_hat=0

value =pd.DataFrame(data=[[1,2,3,4,5],[6,7,8,9,10]])

def set_R_hat():
    DB.dbConnect()
    DB.setCursorDic()

    global user_info
    global mission_info
    global classified_R_hat

    weather_category = ['sunny', 'cloudy', 'rainy','snowy']
    '''
    유저와 관련된 정보들 가져오기
    '''
    sql = "SELECT userIndex as user_id,1 as time_min, time_affordable as time_max, expense_affordable as cost,NULL as applicable_missions, NULL as weekly_missions FROM User"
    try:
        DB.curs.execute(sql)
        user_list_db = DB.curs.fetchall()
    except pymysql.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    user_info = pd.DataFrame(data=user_list_db,
                             columns=['user_id', 'time_min', 'time_max', 'cost', 'applicable_missions',
                                      'weekly_missions'])
    user_info.set_index('user_id', inplace=True, drop=True)

    weather_category = ['sunny', 'cloudy']

    '''
        미션과 관련된 정보들 불러오기
        '''
    sql = "SELECT missionID as mission_id, missionTime as time, expense as cost FROM Mission"
    try:
        DB.curs.execute(sql)
        mission_list_db = DB.curs.fetchall()
    except pymysql.Error as e:
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
        DB.curs.execute(sql)
        mission_evaluation_list_db = DB.curs.fetchall()
    except pymysql.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    mission_evaluation_df = pd.DataFrame(data=mission_evaluation_list_db,
                                         columns=['user', 'mission', 'weather', 'temperature', 'rating'])

    sql = "SELECT count(*) as cnt FROM MissionEvaluation"
    try:
        DB.curs.execute(sql)
        _cnt = DB.curs.fetchone()
    except pymysql.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    R_user_id = list(mission_evaluation_df.loc[:, 'user'])
    R_mission_id = list(mission_evaluation_df.loc[:, 'mission'])
    R_weather = list(mission_evaluation_df.loc[:, 'weather'])
    R_temperature = list(mission_evaluation_df.loc[:, 'temperature'])
    R_rating = list(mission_evaluation_df.loc[:, 'rating'])
    R_data_num = _cnt['cnt']
    

    #print(R_user_id)

    log = functions.get_init_log(weather_category, user_info.index, mission_info.index)

    classified_R = functions.get_classified_R(user_info.index, mission_info.index, weather_category, temperature_min, temperature_max, R_user_id, R_mission_id, R_weather, R_temperature, R_rating, R_data_num)
    #print(classified_R.loc['cloudy', 'value'])
    start1 = time.time()

    #classified_R_hat = functions.get_classified_R_hat_by_KNN(classified_R, log)
    #classified_R_hat = get_classified_R_hat_by_Regression(classified_R)
    classified_R_hat = functions.get_classified_R_hat_by_MatrixCompletion(classified_R)

    print(time.time() - start1)

    return


def function():
    global user_info
    global mission_info
    global classified_R_hat

    user_info = user_info+1
    return

def printaa():
    print(user_info)

    return user_info

def get_R_hat():
    global classified_R_hat
    return classified_R_hat

def get_user_info():
    global user_info
    return user_info

def get_mission_info():
    global mission_info
    return mission_info

def get_value():
    global value
    return value

def set_value(a):
    global value
    value = a
    return