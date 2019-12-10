from flask import Blueprint, request, render_template, flash, redirect, url_for,jsonify, session
from flask import current_app as app
from app.main.DB import DB
from app.main.Weather import getTodaysWeather,getTemperature,getWeather
from werkzeug import secure_filename
import math
import json
from datetime import datetime
import hashlib

import pymysql
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

        DB.dbConnect()
        DB.setCursorDic()

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
            DB.curs.execute(sql)
            missionCandidateList = DB.curs.fetchall()
            #print(missionCandidateList)
        except pymysql.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            DB.dbDisconnect()
            return "Error %d: %s" % (e.args[0], e.args[1])

        sql = "SELECT missionID, missionName, missionTime,expense FROM Mission ORDER BY missionID DESC"
        try:
            DB.curs.execute(sql)
            missionList = DB.curs.fetchall()
        except pymysql.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            DB.dbDisconnect()
            return "Error %d: %s" % (e.args[0], e.args[1])
        DB.dbDisconnect()

        return render_template('/manage/manage.html', missionCandidateList = missionCandidateList,missionList=missionList, userID = userID, userIndex = userIndex,mode = mode)
    else:
        print("로그인 template")
        return render_template('/manage/login.html')

@Manage.route('/login', methods=['GET', 'POST'])
def managerLogin():
    print("\n\nmanagerLogin 호출\n\n")
    id = request.form['id']
    password = request.form['password']
    encoded_password = hashlib.sha256(password.encode()).hexdigest()
    print("\n\n"+encoded_password+"\n\n")
    DB.dbConnect()
    DB.setCursorDic()
    sql = "SELECT * FROM User WHERE id = %s and password=%s"
    try:
        DB.curs.execute(sql, (id, encoded_password))
        row = DB.curs.fetchone()
    except pymysql.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))
    print(row)


    #로그인 됨
    if row:
        #로그인은 되는데 관리자가 아니면
        if row['manager']==0:
            DB.dbDisconnect()
            return render_template('/manage/login.html', message="관리자 아이디로 로그인해주세요")
        #로그인도 되고 관리자이면
        else:
            session['userID'] = row['id']
            session['userIndex'] = row['userIndex']
            DB.dbDisconnect()
            print("redirect manage")
            return manageMissionCandidate()
    #로그인 안 됨
    else:
        try:
            sql = "SELECT EXISTS(SELECT id FROM User WHERE id = %s) AS success"
            DB.curs.execute(sql,(id))
            row = DB.curs.fetchone()

            if row['success']==1:
                message = "비밀번호가 잘못되었습니다."
            else:
                message = "아이디가 잘못되었습니다."
        except pymysql.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
        return render_template('/manage/login.html', message=message)

@Manage.route('/toOfficial', methods=['GET', 'POST'])
def toOfficial():
    missionCandidateList = request.form.getlist("missionCandidate")
    DB.dbConnect()
    DB.setCursorDic()
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
            DB.curs.execute(sql, (name, time,cost))
            #DB.conn.commit()
        except pymysql.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            DB.dbDisconnect()
            return "<h1>INSERT mission error</h1>"

        '''넣은 mission의 미션 번호를 가져오는 것'''
        sql = "SELECT missionID FROM Mission WHERE missionName = %s"
        try:
            DB.curs.execute(sql, (name))
            missionID = str(DB.curs.fetchone()['missionID'])
        except pymysql.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            DB.dbDisconnect()
            return "<h1>get mission id error</h1>"

        '''넣은 미션의 관리자 평가를 넣음'''
        #37.505343, 126.957080 : 중앙대학교 위도 경도
        #x,y :59, 125 관리자 위치는 중앙대학교로 함.
        getTodaysWeather({'x':59, 'y':125})
        todaysWeather = getWeather()
        todaysTemperature = getTemperature()
        sql = "INSERT INTO MissionEvaluation (evaluationIndex, user, mission, rating, weather, comment, picture, temperature) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
        try:
            print(type(user), type(missionID), type(rating), type(todaysWeather), type(todaysTemperature))
            DB.curs.execute(sql, (user+"."+missionID, user, missionID, rating,todaysWeather, name, "https://dailyhappiness.xyz/static/img/no-image.jpg", todaysTemperature))

        except pymysql.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            DB.dbDisconnect()
            return "<h1>insert manager's rating error</h1>"

        '''User에서 작성자의 missionCandidateCount를 증가시켜준다.'''
        sql = "UPDATE User set missionCandidateCount = missionCandidateCount+1 WHERE userIndex = %s"
        try:
            DB.curs.execute(sql, (writer))
        except pymysql.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            DB.dbDisconnect()
            return "<h1>update mission candidate count error</h1>"

        '''Mission 테이블에 넣은 미션은 MissionCandidate 테이블에서 지운다.'''
        sql = "DELETE FROM MissionCandidate WHERE missionCandidateIndex = %s"
        try:
            DB.curs.execute(sql, (missionCandidateIndex))
            DB.conn.commit()
        except pymysql.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            DB.conn.rollback()
            DB.dbDisconnect()
            return "<h1>delete mission candidate error</h1>"
    DB.dbDisconnect()
    return render_template('/manage/complete.html')

@Manage.route('/delete', methods=['GET', 'POST'])
def delete():
    missionCandidateList = request.form.getlist("missionCandidate")
    DB.dbConnect()
    DB.setCursorDic()
    for missionCandidateIndex in missionCandidateList:

        sql = "DELETE FROM MissionCandidate WHERE missionCandidateIndex = %s"
        try:
            DB.curs.execute(sql, (missionCandidateIndex))
            DB.conn.commit()

        except pymysql.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            DB.conn.rollback()
            DB.dbDisconnect()
            return "<h1>delete mission candidate error</h1>"
    DB.dbDisconnect()
    return render_template('/manage/complete.html')

def json_default(value):
    if isinstance(value, datetime.date):
        return value.strftime('%Y-%m-%d')
    raise TypeError('not JSON serializable')
