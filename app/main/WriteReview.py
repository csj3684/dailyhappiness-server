
from flask import Blueprint, request, render_template, flash, redirect, url_for,jsonify
from flask import current_app as app
from app.main.DB import DB
from app.main.Weather import getTodaysWeather,getTemperature,getWeather
from werkzeug import secure_filename
import math
import json
from datetime import datetime

import pymysql
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
        filename = "https://dailyhappiness.xyz"+"/static/img/" + secure_filename(file.filename)
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
    DB.dbConnect()
    DB.setCursorDic()

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

    sql = "INSERT INTO MissionEvaluation(evaluationIndex, user, mission, rating, weather, date, comment, picture, temperature) VALUES(%s, %s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE rating=%s, weather=%s, date = %s,comment=%s,picture=%s,temperature=%s"
    print(type(userIndex+"."+missionIndex), type(userIndex), type(missionIndex), type(missionRating),type(weather),type(now_date),type(content),type(filename),type(temperature))
    try:
        DB.curs.execute(sql, (userIndex+"."+missionIndex, userIndex, missionIndex, missionRating,weather,now_date,content,filename,temperature,missionRating,weather,now_date,content,filename,temperature))
        DB.conn.commit()
        success = {'success': True}
    except Exception as e:
        print(e)

        success = {'success': False}
    DB.dbDisconnect()
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
