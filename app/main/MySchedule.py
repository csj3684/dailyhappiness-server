from flask import Blueprint, request, render_template, flash, redirect, url_for,jsonify
import pymysql
import schedule
import time
from datetime import datetime
from app.main.DB import DB
from app.main.MissionBundle import calculate_R_hat

mySchedule = Blueprint('mySchedule', __name__, url_prefix='/mySchedule')

@mySchedule.route('/week', methods=['GET', 'POST'])
def weekSchedule():
    schedule.every(1).minutes.do(calculate_R_hat)

    while True:
        schedule.run_pending()
        time.sleep(1)
    return 'a'


@mySchedule.route('/day', methods=['GET', 'POST'])
def daySchedule():
    schedule.every(1).day.at("23:00").do(resetCount)

    while True:
        schedule.run_pending()
        time.sleep(1)
    return 'a'

def resetCount():
    '''
    미션을 넘길 때 증가하는 카운트를 초기화 시킨다.
    :return:
    '''

    DB.dbConnect()
    DB.setCursorDic()

    sql = "UPDATE User set count = 1, missionOrder = missionOrder+1"
    #missionOrder를 증가시키는 것은 뺄 수도 있다.
    try:
        DB.curs.execute(sql)
        DB.conn.commit()
    except pymysql.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))
