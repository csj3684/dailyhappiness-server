from flask import Blueprint, request, render_template, flash, redirect, url_for,jsonify, session
from flask import current_app as app
from app.main.DB import DB
from app.main.Weather import getTodaysWeather,getTemperature,getWeather
from werkzeug import secure_filename
import math
import json
from datetime import datetime
import hashlib

import mysql.connector
import json

MissionKing = Blueprint('MissionKing', __name__, url_prefix='/missionKing')

@MissionKing.route('/update', methods=['GET', 'POST'])
def updateMissionKing():
    print("\nupdateMissionKing\n")
    db = DB()
    db.dbConnect()
    db.setCursorDic()


    sql = "SELECT userIndex, missionCount, grade FROM User ORDER BY missionCount DESC LIMIT 0, 10 "
    try:
        db.curs.execute(sql)
        rows = db.curs.fetchall()
        print("\n\n")
        print(rows)
        print("\n\n")
    except mysql.connector.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))
        success = {'success': 'False'}
    #미션왕 넣기
    for i in range(len(rows)):
        ranking = i+1
        row = rows[i]
        which = 1 #1이면 미션왕 0이면 추천왕
        id = str(which) + "." + str(ranking)
        userIndex = row['userIndex']
        count = row['missionCount']
        emblem ="https://dailyhappiness.xyz/static/img/emblem/grade"+str(row['grade'])+".png"
        sql = f"INSERT INTO MissionKing (idMissionKing,userIndex,which,ranking, number, emblem) " \
              f"VALUES (%s,%s,%s,%s,%s,%s) " \
              f"ON DUPLICATE KEY UPDATE userIndex = %s, number = %s, emblem=%s"
        try:
            db.curs.execute(sql,(id, userIndex,which,ranking,count,emblem,userIndex,count,emblem))
            db.conn.commit()
            success = {'success': 'True'}
        except mysql.connector.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            db.conn.rollback()
            success = {'success': 'False'}

    sql = "SELECT userIndex, missionCandidateCount, grade FROM User ORDER BY missionCandidateCount DESC LIMIT 0, 10 "
    try:
        db.curs.execute(sql)
        rows2 = db.curs.fetchall()
        print("\n\n")
        print(rows2)
        print("\n\n")
    except mysql.connector.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))
        success = {'success': 'False'}
    for i in range(len(rows2)):
        ranking = i+1
        row = rows2[i]
        which = 0 #1이면 미션왕 0이면 추천왕
        id = str(which) + "." + str(ranking)
        userIndex = row['userIndex']
        count = row['missionCandidateCount']
        emblem ="https://dailyhappiness.xyz/static/img/emblem/grade"+str(row['grade'])+".png"
        sql = f"INSERT INTO MissionKing (idMissionKing,userIndex,which,ranking, number, emblem) " \
              f"VALUES (%s,%s,%s,%s,%s,%s) " \
              f"ON DUPLICATE KEY UPDATE userIndex = %s, number = %s, emblem=%s"
        try:
            db.curs.execute(sql,(id, userIndex,which,ranking,count,emblem,userIndex,count,emblem))
            db.conn.commit()
            success = {'success': 'True'}
        except mysql.connector.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            db.conn.rollback()
            success = {'success': 'False'}
    db.dbDisconnect()
    return "<h1>success : " + success['success']+ "</h1>"



@MissionKing.route('/get', methods=['GET', 'POST'])
def getMissionKing():
    db = DB()
    db.dbConnect()
    db.setCursorDic()

    sql = "SELECT MissionKing.idMissionKing, MissionKing.userIndex, MissionKing.which, MissionKing.ranking,MissionKing.number, MissionKing.emblem, User.id FROM MissionKing JOIN User ON (MissionKing.userIndex = User.userIndex and User.id IS NOT NULL)"

    try:
        db.curs.execute(sql)
        rows = db.curs.fetchall()
    except mysql.connector.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    return json.dumps(rows, default=json_default).encode('utf-8')


def json_default(value):
    if isinstance(value, datetime.date):
        return value.strftime('%Y-%m-%d')
    raise TypeError('not JSON serializable')
