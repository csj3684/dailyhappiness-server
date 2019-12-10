from flask import Blueprint, request, render_template, flash, redirect, url_for,jsonify
import mysql.connector
import schedule
import time
from datetime import datetime
from app.main.DB import DB
from app.main.MissionBundle import calculate_R_hat, init_keep_info

mySchedule = Blueprint('mySchedule', __name__, url_prefix='/mySchedule')

@mySchedule.route('/week', methods=['GET', 'POST'])
def weekSchedule():
    schedule.every().monday.at("04:00").do(every_week_do)

    while True:
        schedule.run_pending()
        time.sleep(1)
    return 'a'


@mySchedule.route('/day', methods=['GET', 'POST'])
def daySchedule():
    schedule.every().day.at("00:00").do(resetCount)

    while True:
        schedule.run_pending()
        time.sleep(1)
    return 'a'

def resetCount():
    '''
    미션을 넘길 때 증가하는 카운트를 초기화 시킨다.
    :return:
    '''
    db =DB()
    db.dbConnect()
    db.setCursorDic()

    sql = "UPDATE User set count = 0, missionOrder = missionOrder+1"
    #missionOrder를 증가시키는 것은 뺄 수도 있다.
    try:
        db.curs.execute(sql)
        db.conn.commit()
    except mysql.connector.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

def every_week_do():
    db.dbConnect()
    db.setCursorDic()

    sql = "UPDATE User SET isWeekFirst = 1"
    try:
        db.curs.execute(sql)
        db.conn.commit()
    except Exception as e:
        print("reset is first 오류 , ", e)

    db.dbDisconnect()
    print("calculating_R_hat")
    calculate_R_hat(1)
    print("R_hat_complete")
    init_keep_info()
    return