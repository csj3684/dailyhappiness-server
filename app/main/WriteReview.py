from flask import Blueprint, request, render_template, flash, redirect, url_for,jsonify
from flask import current_app as app
from app.main.DB import DB
from app.main.Weather import getTodaysWeather,getTemperature,getWeather,get_weekly_weather_list
from werkzeug import secure_filename
import math
import json
from datetime import datetime
import pandas as pd
from app.main.MissionBundle import calculate_R_hat, user_rating_init,get_weekly_weather,get_today_idx,update_user_applicable_missions
import numpy as np
import mysql.connector
import json


writeReviewPage = Blueprint('writeReviewPage', __name__, url_prefix='/writeReview')
filename = ""
@writeReviewPage.route('/image', methods=['GET', 'POST'])
def uploadImage():
    print("uploadImage 호출")

    try:
        file = request.files['upload']
        file.save(app.root_path+"/static/img/" + secure_filename(file.filename))
        global filename
        filename = "https://dailyhappiness.xyz/static/img/" + secure_filename(file.filename)
        success = {'success': True}
    except Exception as e:
        success = {'success': False}
        print("image 저장 실패 : "+e)


    return json.dumps(success).encode('utf-8')


'''
review에서 이미지를 제외하고 업로드 하는 부분
'''
@writeReviewPage.route('/review', methods=['GET', 'POST'])
def uploadReview():

    print("uploadImage 호출")
    db=DB()
    db.dbConnect()
    db.setCursorDic()

    weathers = get_weekly_weather_list()
    weather_category = ['sunny', 'cloudy']
    weekly_weather = get_weekly_weather(weathers, weather_category)
    today_idx = get_today_idx()

    print(request.form)

    userIndex = request.form['userIndex']
    missionIndex = request.form['missionIndex']
    missionRating = request.form['missionRating']
    location_lat = request.form['locationlat']
    location_lon = request.form['locationlon']
    rs = grid(location_lat,location_lon) # x,y 좌표

    now = datetime.now()
    now_date = now.strftime('%Y-%m-%d')

    content = request.form['content']
    getTodaysWeather(rs)
    weather = getWeather()
    temperature = getTemperature()

    sql = "SELECT grade FROM User WHERE userIndex = %s"
    try:
        db.curs.execute(sql, (userIndex,))
        row = db.curs.fetchone()
        current_grade = row['grade']
    except Exception as e:
        print(e)

    sql = "UPDATE User SET missionCount = missionCount+1, isWeekFirst = 0 WHERE userIndex = %s"
    try:
        db.curs.execute(sql,(userIndex,))
    except Exception as e:
        print(e)
        db.conn.rollback()
        success = {'success': False}
        return success

    sql = "UPDATE User " \
          "SET grade= " \
          "CASE " \
          "WHEN missionCount BETWEEN 0 AND 2 THEN 1 " \
          "WHEN missionCount BETWEEN 2 AND 3 THEN 2 " \
          "WHEN missionCount BETWEEN 3 AND 4 THEN 3 " \
          "WHEN missionCount BETWEEN 4 AND 5 THEN 4 " \
          "WHEN missionCount BETWEEN 5 AND 6 THEN 5 " \
          "WHEN missionCount BETWEEN 6 AND 7 THEN 6 " \
          "WHEN missionCount BETWEEN 7 AND 8 THEN 7 " \
          "WHEN missionCount BETWEEN 211 AND 240 THEN 8 " \
          "WHEN missionCount BETWEEN 241 AND 270 THEN 9 " \
          "WHEN missionCount BETWEEN 271 AND 300 THEN 10 " \
          "ELSE 11 " \
          "END " \
          "WHERE userIndex = %s;"
    try:
        db.curs.execute(sql, (userIndex,))
        success = {'success': True}
    except Exception as e:
        print(e)
        db.conn.rollback()
        success = {'success': False}
        return success

    sql = "SELECT grade FROM User WHERE userIndex = %s"
    try:
            db.curs.execute(sql, (userIndex,))
            row = db.curs.fetchone()
            after_grade = row['grade']
    except Exception as e:
        print(e)
        db.conn.rollback()
    if current_grade != after_grade:
        if row['grade']==2:
            success['level-up'] = "회색 클로버로 레벨업 했습니다."
        elif row['grade']==3:
            success['level-up'] = "갈색 클로버로 레벨업 했습니다."
        elif row['grade']==4:
            success['level-up'] = "연노랑 클로버로 레벨업 했습니다."
        elif row['grade']==5:
            success['level-up'] = "에메랄드 클로버로 레벨업 했습니다."
        elif row['grade']==5:
            success['level-up'] = "파랑 클로버로 레벨업 했습니다."
        elif row['grade']==5:
            success['level-up'] = "선홍 클로버로 레벨업 했습니다."
        elif row['grade']==6:
            success['level-up'] = "연보라 클로버로 레벨업 했습니다."
        elif row['grade']==7:
            success['level-up'] = "하늘 클로버로 레벨업 했습니다."
        elif row['grade']==8:
            success['level-up'] = "노랑 클로버로 레벨업 했습니다."

    sql = "INSERT INTO MissionEvaluation(evaluationIndex, user, mission, rating, weather, date, comment, picture, temperature) VALUES(%s, %s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE rating=%s, weather=%s, date = %s,comment=%s,picture=%s,temperature=%s"
    print(type(userIndex+"."+missionIndex), type(userIndex), type(missionIndex), type(missionRating),type(weather),type(now_date),type(content),type(filename),type(temperature))
    try:
        db.curs.execute(sql, (userIndex+"."+missionIndex, userIndex, missionIndex, missionRating,weather,now_date,content,filename,temperature,missionRating,weather,now_date,content,filename,temperature))
        db.conn.commit()
        success['success'] =  True
    except Exception as e:
        print(e)

        success['success'] =  False
    print(success)
    #update_user_applicable_missions(int(userIndex), np.int64(missionIndex), 'done', today_idx, weekly_weather)



    db.dbDisconnect()
    return json.dumps(success).encode('utf-8')


def grid(v1, v2) :
    v1 = float(v1)
    v2 = float(v2)
    RE = 6371.00877 # 지구 반경(km)
    GRID = 5.0 # 격자 간격(km)
    SLAT1 = 30.0 # 투영 위도1(degree)
    SLAT2 = 60.0 # 투영 위도2(degree)
    OLON = 126.0 # 기준점 경도(degree)
    OLAT = 38.0 # 기준점 위도(degree)
    XO = 43 # 기준점 X좌표(GRID)
    YO = 136 # 기1준점 Y좌표(GRID)
    DEGRAD = math.pi / 180.0
    RADDEG = 180.0 / math.pi
    re = RE / GRID
    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD
    olon = OLON * DEGRAD
    olat = OLAT * DEGRAD
    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = math.pow(sf, sn) * math.cos(slat1) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / math.pow(ro, sn); rs = {}
    ra = math.tan(math.pi * 0.25 + (v1) * DEGRAD * 0.5)
    ra = re * sf / math.pow(ra, sn)
    theta = v2 * DEGRAD - olon
    if theta > math.pi : theta -= 2.0 * math.pi
    if theta < -math.pi : theta += 2.0 * math.pi
    theta *= sn
    rs['x'] = math.floor(ra * math.sin(theta) + XO + 0.5)
    rs['y'] = math.floor(ro - ra * math.cos(theta) + YO + 0.5)

    return rs

@writeReviewPage.route('/survey', methods=['GET', 'POST'])
def writeSurveyReview():
    print("write Survey Review.py")
    
    userIndex = request.form['userIndex']
    missionID = request.form['missionID']
    rating = request.form['rating']
    isLast = int(request.form['isLast'])
    print("islast", isLast)
    id = str(userIndex)+"."+str(missionID)
    getTodaysWeather({'x': 59, 'y': 125})
    todaysWeather = getWeather()
    todaysTemperature = getTemperature()
    target_user_id = int(request.form['userIndex'])

    db= DB()
    db.dbConnect()
    db.setCursorDic()

    sql ="INSERT INTO MissionEvaluation (evaluationIndex, user, mission,rating, weather, temperature) " \
         "VALUES (%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE " \
         "rating=%s, weather=%s, temperature=%s"
    try:
        db.curs.execute(sql, (id, userIndex,missionID,rating,todaysWeather,todaysTemperature,rating,todaysWeather,todaysTemperature))
        db.conn.commit()
        row = {'end' : 0}
    except Exception as e:
        print(e)
    
    if isLast != 0:
        print("isisLast", isLast)
        sql ="UPDATE User SET didSurvey=1 WHERE userIndex = %s"
        try:
            db.curs.execute(sql, (userIndex,))
            db.conn.commit()
            row = {'end': 1}
        except Exception as e:
            print(e)

        sql = "SELECT user, mission, weather,temperature, rating FROM MissionEvaluation WHERE user = %s"
        try:
            db.curs.execute(sql,(userIndex,))
            mission_evaluation_list_db = db.curs.fetchall()
        except mysql.connector.Error as e:
            print("Error %s" ,e)

        mission_evaluation_df = pd.DataFrame(data=mission_evaluation_list_db,
                                             columns=['user', 'mission', 'weather', 'temperature', 'rating'])

        sql = "SELECT count(*) as cnt FROM MissionEvaluation"
        try:
            db.curs.execute(sql)
            _cnt = db.curs.fetchone()
        except mysql.connector.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
        """
        #user_id = list(mission_evaluation_df.loc[:, 'user'])
        mission_list = list(mission_evaluation_df.loc[:, 'mission'])
        weather_list = list(mission_evaluation_df.loc[:, 'weather'])
        temperature_list = list(mission_evaluation_df.loc[:, 'temperature'])
        rating_list = list(mission_evaluation_df.loc[:, 'rating'])
        data_num = len(rating_list)
        #data_num = _cnt['cnt']
        user_rating_init(target_user_id, mission_list, weather_list, temperature_list, rating_list, data_num)
        """
        print("write_review> calculating_R_hat")
        calculate_R_hat(0)
        print("write_review> R_hat_complete")
    return json.dumps(row).encode('utf-8')