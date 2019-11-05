


from flask import Blueprint, request, render_template, flash, redirect, url_for,jsonify
from flask import current_app as app
from app.main.DB import DB
import pymysql
import pandas as pd
from collections import OrderedDict
from app.main.knapsack_with_algorithms import getKnapsack
from app.main.knapsack_with_algorithms import show_knapsack
import json

missionBundlePage = Blueprint('missionBundlePage', __name__, url_prefix='/missionBundle')

'''
cron job 에서 매일 월요일 아침에 호출
예상점수가 전부 다 들어가있는 2차원 배열을 받아서 knapsack 문제를 풀고 21개를 MissionBundle 테이블에 저장
'''
@missionBundlePage.route('/set', methods=['GET', 'POST'])
def setMissionBundle():
    pd.set_option('display.max_columns', 500)
    print("setMissionBundle 함수 호출")
    '''
    각 유저별로 knapsack을 가져온다.
    '''


    knapsack = getKnapsack()
    print(knapsack.loc[67]['daily_missions']['expected_rating'])

    DB.dbConnect()
    DB.setCursorDic()

    '''
    데이터베이스에 Mission Bundle에 저장하는 부분
    '''
    print( knapsack.index.values[:])
    for user in knapsack.index.values[:]:
        if knapsack.loc[user]['daily_missions']!=None:
            print(knapsack.loc[user])
            for i in range(len(knapsack.loc[user]['daily_missions']['mission_id'])):
                missionID = knapsack.loc[user]['daily_missions']['mission_id'][i]
                if str(type(knapsack.loc[user]['daily_missions']['expected_rating']))=="<class 'int'>" or str(type(knapsack.loc[user]['daily_missions']['expected_rating']))=="<class 'float'>":
                    expectedRating = knapsack.loc[user]['daily_missions']['expected_rating']
                else:
                    expectedRating = knapsack.loc[user]['daily_missions']['expected_rating'][i]
                order = i+1
                index = str(user)+"."+str(order)
                print("index:",index,"missionID: ",missionID.item(),", user : ",user.item(),"order: ",order,", expectedRating: ", expectedRating)
                sql = "INSERT INTO MissionBundle(userAndOrder, userIndex, missionOrder, missionIndex, expectedRating) VALUES(%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE missionIndex=%s, expectedRating = %s "
                try:

                    DB.curs.execute(sql,(index, user.item(), order,missionID.item(),expectedRating,missionID.item(),float(expectedRating)))
                    DB.conn.commit()
                except pymysql.Error as e:
                    print("Error %d: %s" % (e.args[0], e.args[1]))

    DB.dbDisconnect()
    return 'abc'



'''
안드로이드에서 미션을 가져올 때 호출
'''
@missionBundlePage.route('/get', methods=['GET', 'POST'])
def getMissionBundle():
    DB.dbConnect()
    DB.setCursorDic()
    _userIndex = request.form['userIndex']
    print("getMissionBundle 함수 호출: ", _userIndex)
    #사용자가 21개의 미션 중 몇 번째 미션을 가지고 와야하는지 변수를 가져옴
    sql = f"SELECT missionOrder FROM User WHERE userIndex = {_userIndex}"
    try:
        DB.curs.execute(sql)
        row = DB.curs.fetchone()
        _missionOrder = int(row['missionOrder'])
        print("missionOrder : " , _missionOrder)
    except Exception as e:
        print("missionOrder 가져오기 오류 , ",e);
    # -----------------------------------------------------------------

    #MissionBundle에서 userIndex와 missionOrder로 mission을 가져온다.
    sql = f"SELECT missionIndex, missionName FROM MissionBundle join Mission on MissionBundle.missionIndex = Mission.missionID WHERE userIndex = {_userIndex} and missionOrder={_missionOrder}"
    try:
        DB.curs.execute(sql)
        row = DB.curs.fetchone()

    except Exception as e:
        print("missionIndex, missionName 가져오기 오류 , ",e);
    # -----------------------------------------------------------------


    DB.dbDisconnect()
    print(row)
    return json.dumps(row).encode('utf-8')


'''
미션을 바꾸거나 하루가 지나서 missionOrder를 증가시킬 때 부르는 함수
post를 통해서 count가 넘어오면 count를 증가시킨다.
dislike도 넘어오면(싫어해서 미션을 넘긴거면) 미션 평가를 넣음
'''
@missionBundlePage.route('/increment', methods=['GET', 'POST'])
def incrementMission():
    DB.dbConnect()
    DB.setCursorDic()
    _userIndex = request.form['userIndex']

    #count가 넘어왔으면,
    if 'count' in request.form.keys():
        sql = f"UPDATE User set count=count+1 WHERE userIndex = {_userIndex}"
        try:
            DB.curs.execute(sql)
            DB.conn.commit()
        except Exception as e:
            print("count 증가 오류 , ", e)


    #missionOrder 하나 증가시킴
    sql = f"UPDATE User set missionOrder=missionOrder+1 WHERE userIndex = {_userIndex}"
    try:
        DB.curs.execute(sql)
        DB.conn.commit()
    except Exception as e:
        print("missionOrder 증가 오류 , ",e)

    DB.dbDisconnect()





