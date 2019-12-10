from flask import Blueprint, request, render_template, flash, redirect, url_for,jsonify, session
from flask import current_app as app
from app.main.DB import DB
from app.main.Weather import getTodaysWeather,getTemperature,getWeather
from werkzeug import secure_filename
import math
import json
from datetime import datetime
import hashlib
from app.main.MissionBundle import calculate_R_hat, add_default_mission
import mysql.connector
import json


Manage = Blueprint('Manage', __name__, url_prefix='/manage')

@Manage.route('/show', methods=['GET', 'POST'])
def manageMissionCandidate():
    print("\n\nmanageMissionCanidate 호출\n\n")


    if 'mode' in request.args:
        mode = request.args.get('mode')
    else:
        mode = 'likes'
    if 'userID' in session:
        userID = session['userID']
        userIndex = session['userIndex']
        db = DB()
        db.dbConnect()
        db.setCursorDic()

        '''로그인이 되어 있으면 '''
        '''관리자일 경우에 '''

        if mode == 'likes':
            sql = "SELECT * FROM MissionCandidate WHERE likes>=0 and dislikes<=5 ORDER BY likes DESC"
        elif mode == 'dislikes':
            sql = "SELECT * FROM MissionCandidate ORDER BY dislikes DESC"
        elif mode == 'oldest':
            sql = "SELECT * FROM MissionCandidate ORDER BY date"
        elif mode == 'duplicate':
            sql = "SELECT * FROM MissionCandidate ORDER BY duplicateCount DESC"
        try:
            db.curs.execute(sql)
            missionCandidateList = db.curs.fetchall()
            #print(missionCandidateList)
        except mysql.connector.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            db.dbDisconnect()
            return "Error %d: %s" % (e.args[0], e.args[1])

        sql = "SELECT missionID, missionName, missionTime,expense FROM Mission ORDER BY missionID DESC"
        try:
            db.curs.execute(sql)
            missionList = db.curs.fetchall()
        except mysql.connector.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            db.dbDisconnect()
            return "Error %d: %s" % (e.args[0], e.args[1])
        db.dbDisconnect()

        return render_template('/manage/manage.html', missionCandidateList = missionCandidateList,missionList=missionList, userID = userID, userIndex = userIndex,mode = mode)
    else:
        
        return render_template('/manage/login.html')

@Manage.route('/login', methods=['GET', 'POST'])
def managerLogin():
    print("\n\nmanagerLogin 호출\n\n")
    id = request.form['id']
    password = request.form['password']
    encoded_password = hashlib.sha256(password.encode()).hexdigest()
    print("\n\n"+encoded_password+"\n\n")
    db = DB()
    db.dbConnect()
    db.setCursorDic()
    sql = "SELECT * FROM User WHERE id = %s and password=%s"
    try:
        db.curs.execute(sql, (id, encoded_password))
        row = db.curs.fetchone()
    except mysql.connector.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))
    print("관리자 로그인",row)


    #로그인 됨
    if row:
        #로그인은 되는데 관리자가 아니면
        if row['manager']==0:
            db.dbDisconnect()
            return render_template('/manage/login.html', message="관리자 아이디로 로그인해주세요")
        #로그인도 되고 관리자이면
        else:
            session['userID'] = row['id']
            session['userIndex'] = row['userIndex']
            db.dbDisconnect()
            
            return manageMissionCandidate()
    #로그인 안 됨
    else:
        try:
            sql = "SELECT EXISTS(SELECT id FROM User WHERE id = %s) AS success"
            db.curs.execute(sql,(id,))
            row = db.curs.fetchone()

            if row['success']==1:
                message = "비밀번호가 잘못되었습니다."
            else:
                message = "아이디가 잘못되었습니다."
        except mysql.connector.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
        return render_template('/manage/login.html', message=message)

@Manage.route('/toOfficial', methods=['GET', 'POST'])
def toOfficial():
    missionCandidateList = request.form.getlist("missionCandidate")
    db=DB()
    db.dbConnect()
    db.setCursorDic()
    for missionCandidateIndex in missionCandidateList:
        cost = request.form['cost-'+missionCandidateIndex]
        name = request.form['name-'+missionCandidateIndex]
        time = request.form['time-'+missionCandidateIndex]
        rating = request.form['rating-'+missionCandidateIndex]
        writer = request.form['writer-'+missionCandidateIndex]
        user = str(session['userIndex'])
        '''Mission table에 후보 미션을 넣는것'''
        sql = "INSERT INTO Mission(missionName, missionTime, expense) VALUES (%s,%s,%s)"
        try:
            db.curs.execute(sql, (name, time,cost))
            #DB.conn.commit()
        except mysql.connector.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            db.dbDisconnect()
            return "<h1>INSERT mission error</h1>"

        '''넣은 mission의 미션 번호를 가져오는 것'''
        sql = "SELECT missionID FROM Mission WHERE missionName = %s"
        try:
            db.curs.execute(sql, (name,))
            missionID = str(db.curs.fetchone()['missionID'])
        except mysql.connector.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            db.dbDisconnect()
            return "<h1>get mission id error</h1>"

        '''넣은 미션의 관리자 평가를 넣음'''
        #37.505343, 126.957080 : 중앙대학교 위도 경도
        #x,y :59, 125 관리자 위치는 중앙대학교로 함.
        getTodaysWeather({'x':59, 'y':125})
        todaysWeather = getWeather()
        todaysTemperature = getTemperature()
        sql = "INSERT INTO MissionEvaluation (evaluationIndex, user, mission, rating, weather, comment, picture, temperature) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
        try:
           
            db.curs.execute(sql, (user+"."+missionID, user, missionID, rating,todaysWeather, name, "https://dailyhappiness.xyz/static/img/no-image.jpg", todaysTemperature))

        except mysql.connector.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            db.dbDisconnect()
            return "<h1>insert manager's rating error</h1>"

        '''User에서 작성자의 missionCandidateCount를 증가시켜준다.'''
        sql = "UPDATE User set missionCandidateCount = missionCandidateCount+1 WHERE userIndex = %s"
        try:
            db.curs.execute(sql, (writer,))
        except mysql.connector.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            db.dbDisconnect()
            return "<h1>update mission candidate count error</h1>"
        
        sql = "UPDATE User set isWeekFirst =1"
        try:
            db.curs.execute(sql)
        except mysql.connector.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            db.dbDisconnect()
            return "<h1>update isFirst error</h1>"

        '''Mission 테이블에 넣은 미션은 MissionCandidate 테이블에서 지운다.'''
        sql = "DELETE FROM MissionCandidate WHERE missionCandidateIndex = %s"
        try:
            db.curs.execute(sql, (missionCandidateIndex,))
            db.conn.commit()
        except mysql.connector.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            db.conn.rollback()
            db.dbDisconnect()
            return "<h1>delete mission candidate error</h1>"
    

    if rating == 0:
        add_default_mission(missionID, rating, cost, 0, todaysWeather)
    elif cost == 0:
        add_default_mission(missionID, rating, cost, float('inf'), todaysWeather)
    else:
        add_default_mission(missionID, rating, cost, float(rating) / cost, todaysWeather)
        
   
    print("calculate_R_hat() start")
    calculate_R_hat(0)
    db.dbDisconnect()
    return render_template('/manage/complete.html')

@Manage.route('/delete', methods=['GET', 'POST'])
def delete():
    missionCandidateList = request.form.getlist("missionCandidate")
    db=DB()
    db.dbConnect()
    db.setCursorDic()
    for missionCandidateIndex in missionCandidateList:

        sql = "DELETE FROM MissionCandidate WHERE missionCandidateIndex = %s"
        try:
            db.curs.execute(sql, (missionCandidateIndex,))
            db.conn.commit()

        except mysql.connector.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            db.conn.rollback()
            db.dbDisconnect()
            return "<h1>delete mission candidate error</h1>"
    db.dbDisconnect()
    return render_template('/manage/complete.html')

def json_default(value):
    if isinstance(value, datetime.date):
        return value.strftime('%Y-%m-%d')
    raise TypeError('not JSON serializable')
