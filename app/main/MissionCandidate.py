
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

'''
미션 후보들을 넣고 클라이언트로 가져가고 하는 일을 하는 파일
'''
MissionCandidate = Blueprint('MissionCandidate', __name__, url_prefix='/missionCandidate')

@MissionCandidate.route('/insert', methods=['GET', 'POST'])
def insertMissionCandidate():
    '''
    매개변수 : userIndex, missionName
    :return : 성공 여부
    '''
    print("insertMissionCandidate 호출")
    user = request.form['userIndex']
    mission = request.form['missionName']
    today = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    DB.dbConnect()
    DB.setCursorDic()
    print(mission)
    sql = "INSERT INTO MissionCandidate (userIndex,date,missionName,likes,dislikes,likesRatio,duplicateCount) VALUES (%s,%s,%s,0,0,0.0,0)"

    try:
        DB.curs.execute(sql,(user,today,mission))
        DB.conn.commit()
        success = {'success':True}
    except pymysql.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))
        success = {'success':False}

    #DB.dbDisconnect()
    return json.dumps(success).encode('utf-8')

@MissionCandidate.route('/getOne', methods=['GET', 'POST'])
def getOneMissionCandidate():
    print("\n\ngetOneMissionCandidate 호출\n\n")
    '''
    원하는 미션을 하나만 받아올 때 쓰이는 함수.
    '''
    userIndex = request.form['userIndex']
    missionCandidateIndex = request.form['missionCandidateIndex']

    DB.dbConnect()
    DB.setCursorDic()

    sql = f"SELECT " \
          f"MissionCandidate.missionName, " \
          f"MissionCandidate.missionCandidateIndex AS missionCandidateIndex, " \
          f"MissionCandidate.likes AS totalLikes, " \
          f"MissionCandidate.dislikes AS totalDislikes, " \
          f"MissionCandidate.duplicateCount AS totalDuplicateCount, " \
          f"IFNULL(MissionCandidateEvaluation.likes,0) As userLikes, " \
          f"IFNULL(MissionCandidateEvaluation.dislikes,0) As userDislikes, " \
          f"IFNULL(MissionCandidateEvaluation.duplicateCount,0) As userDuplicateCount, " \
          f"User.id AS user " \
          f"FROM MissionCandidate " \
          f"LEFT JOIN MissionCandidateEvaluation ON (MissionCandidate.missionCandidateIndex =MissionCandidateEvaluation.missionCandidateIndex and MissionCandidateEvaluation.userIndex = {userIndex}) " \
          f"WHERE MissionCandidate.missionCandidateIndex = {missionCandidateIndex} " \
          f"JOIN User ON MissionCandidate.userIndex = User.userIndex"
    try:
        DB.curs.execute(sql)
        rows = DB.curs.fetchall()
    except pymysql.Error as e:
        print("Error %d: %s %s" % (e.args[0], e.args[1]))
        success = {'success':False}

    print(rows)
    DB.dbDisconnect()
    return json.dumps(rows, default=json_default).encode('utf-8')


@MissionCandidate.route('/get', methods=['GET', 'POST'])
def getMissionCandidate():
    print("\n\ngetMissionCandidate 호출\n\n")
    '''
    클라이언트에서 미션 후보들을 받아올 때 쓰이는 함수
    최신순과 좋아요 순이 있다. 10개씩 받아온다.
    매개변수 : userIndex, missionCandidateCount, mode
    :return:
    '''
    userIndex = request.form['userIndex']
    missionCandidateCount = int(request.form['missionCandidateCount'])

    DB.dbConnect()
    DB.setCursorDic()



    #mode가 1이면 최신순, 0이면 좋아요가 많은 순
    mode = int(request.form['mode'])
    '''
    missionName : 미션 내용
    missionCandidateIndex : 미션 후보 인덱스
    totalLikes : 좋아요 수
    totalDislikes  : 싫어요 수
    totalDuplicateCount : 중복체크 수
    userLikes : 유저가 좋아요 눌렀는지
    userDislikes : 유저가 싫어요 눌렀는지
    userDuplicateCount : 유저가 중복 눌렀는지
    '''
    if mode == 1:
        sql = f"SELECT " \
              f"MissionCandidate.missionName, " \
              f"MissionCandidate.missionCandidateIndex AS missionCandidateIndex, " \
              f"MissionCandidate.likes AS totalLikes, " \
              f"MissionCandidate.dislikes AS totalDislikes, " \
              f"MissionCandidate.duplicateCount AS totalDuplicateCount, " \
              f"IFNULL(MissionCandidateEvaluation.likes,0) As userLikes, " \
              f"IFNULL(MissionCandidateEvaluation.dislikes,0) As userDislikes, " \
              f"IFNULL(MissionCandidateEvaluation.duplicateCount,0) As userDuplicateCount, " \
              f"User.id AS user " \
              f"FROM MissionCandidate " \
              f"LEFT JOIN MissionCandidateEvaluation ON (MissionCandidate.missionCandidateIndex =MissionCandidateEvaluation.missionCandidateIndex and MissionCandidateEvaluation.userIndex = {userIndex}) " \
              f"JOIN User ON MissionCandidate.userIndex = User.userIndex " \
              f"ORDER BY date DESC LIMIT {missionCandidateCount}, 20"
    elif mode ==0:
        sql = f"SELECT " \
              f"MissionCandidate.missionName, " \
              f"MissionCandidate.missionCandidateIndex AS missionCandidateIndex " \
              f"MissionCandidate.likes AS totalLikes, " \
              f"MissionCandidate.dislikes AS totalDislikes, " \
              f"MissionCandidate.duplicateCount AS totalDuplicateCount, " \
              f"IFNULL(MissionCandidateEvaluation.likes,0) As userLikes, " \
              f"IFNULL(MissionCandidateEvaluation.dislikes,0) As userDislikes, " \
              f"IFNULL(MissionCandidateEvaluation.duplicateCount,0) As userDuplicateCount, " \
              f"User.id AS user " \
              f"FROM MissionCandidate " \
              f"LEFT JOIN MissionCandidateEvaluation ON (MissionCandidate.missionCandidateIndex =MissionCandidateEvaluation.missionCandidateIndex and MissionCandidateEvaluation.userIndex = {userIndex})" \
              f"JOIN User ON MissionCandidate.userIndex = User.userIndex " \
              f"ORDER BY MissionCandidate.likes DESC LIMIT {missionCandidateCount}, 20"


    try:
        DB.curs.execute(sql)
        rows = DB.curs.fetchall()
    except pymysql.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))
        success = {'success':False}

    print(rows)
    DB.dbDisconnect()
    return json.dumps(rows, default=json_default).encode('utf-8')

@MissionCandidate.route('/increment', methods=['GET', 'POST'])
def evaluation():
    '''
    좋아요나 싫어요, 중복체크를 늘리거나 줄인다.
    매개변수 : userIndex, missionCandidateIndex, which => likes나 dislikes, count 중 하나의 값이 들어가 있음, value = 1 or -1
    which : 1이면 likes, 2이면 dislikes, 3이면 count
    (하나 늘리거나 줄이거나 그대로) 줄이는 경우는 클릭했다가 다시 클릭할 때
    :return: success : true or false
    '''
    print("\n\nevaluation 호출\n\n")
    userIndex = request.form['userIndex']
    missioncandidateIndex = request.form['missionCandidateIndex']
    which = int(request.form['which'])

    if which ==1 : # likes
        likes = int(request.form['value'])
        dislikes =0
        count =0
    elif which ==2 : # dislikes
        dislikes = int(request.form['value'])
        likes = 0
        count = 0
    elif which ==3 : # duplicate count
        count = int(request.form['value'])
        likes = 0
        dislikes = 0
    else:
        print("which 매개변수 값 오류")
        return json.dumps({'success':False}, default=json_default).encode('utf-8')

    DB.dbConnect()
    DB.setCursorDic()
    #MissionCandidate 테이블의 likes, dislikes, count 값을 바꾼다.
    sql = f"UPDATE MissionCandidate SET likes = likes+{likes}, dislikes = dislikes+{dislikes}, duplicateCount = duplicateCount + {count} WHERE missionCandidateIndex = {missioncandidateIndex}"

    try:
        DB.curs.execute(sql)
        success = {'success':True}
        print("\nMissionCandidate 테이블 좋아요, 싫어요 중복 바꾸기 완료\n")

    except pymysql.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))
        success = {'success':False}
        DB.dbDisconnect()
        return json.dumps(success, default=json_default).encode('utf-8')

    # MissionCandidateEvaluation 테이블에 likes, dislikes, duplicateCount 값을 넣는다.
    # 0은 아직 평가 안 됨. 1은 평가 함( 좋아요 누름, 싫어요 누름, 등 )

    id = userIndex+"."+missioncandidateIndex
    sql = f"INSERT INTO MissionCandidateEvaluation (id, missionCandidateIndex, userIndex, likes,dislikes,duplicateCount) " \
          f"VALUES ({id}, {missioncandidateIndex}, {userIndex}, {likes}, {dislikes}, {count}) ON DUPLICATE KEY " \
          f"UPDATE likes = likes+{likes}, dislikes = dislikes + {dislikes}, duplicateCount = duplicateCount+{count}"
    try:
        DB.curs.execute(sql)
        DB.conn.commit()
        success = {'success':True}
        print("\nMissionCandidateEvaluation 바꾸기 완료\n")
    except pymysql.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))
        success = {'success':False}

    DB.dbDisconnect()

    return json.dumps(success, default=json_default).encode('utf-8')

@MissionCandidate.route('/search', methods=['GET', 'POST'])
def search():
    keyword = request.form['keyword']
    DB.dbConnect()
    DB.setCursorDic()
    '''미션 후보에서 검색하기'''
    sql = "SELECT missionName FROM MissionCandidate WHERE missionName LIKE '{}'".format('%{0}%'.format(keyword))
    try:
        DB.curs.execute(sql)
        rows1 = DB.curs.fetchall()
    except pymysql.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    '''공식 미션에서 검색하기'''
    sql = "SELECT missionName FROM Mission WHERE missionName LIKE '{}'".format('%{0}%'.format(keyword))
    try:
        DB.curs.execute(sql)
        rows2 = DB.curs.fetchall()
    except pymysql.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))
    rows=[]
    if rows1:
        rows = rows+rows1
    if rows2:
        rows = rows+rows2

    if not rows:
        rows = [{'missionName':'검색 결과 없음.'}]
    return json.dumps(rows, default=json_default).encode('utf-8')

def json_default(value):
    if isinstance(value, datetime.date):
        return value.strftime('%Y-%m-%d')
    raise TypeError('not JSON serializable')
