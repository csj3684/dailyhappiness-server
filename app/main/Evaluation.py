'''
데이터베이스 MissionEvaluation 테이블에서 데이터를 가져와서 비어있는 점수들을 evaluationList 에 채워넣는다.
'''
import json
import pandas as pd
from collections import OrderedDict
import math as math


from app.main.DB import DB



def getEvaluation():
    print("getEvaluation 함수 호출")



    DB.dbConnect()
    DB.setCursorDic()

    #몇 명의 유저가 있는지 확인해서 _user 변수에 넣음
    #------------------------------------------------
    sql = "SELECT user FROM MissionEvaluation ORDER BY user DESC limit 1"
    try:
        DB.curs.execute(sql)
        row = DB.curs.fetchone()
        _user = int(row['user'])
        print("마지막 유저 번호 : ", _user)
    except Exception as e:
        print("_user 가져오기 오류 , ",e)
    #------------------------------------------------

    #몇 개의 미션이 있는지 확인해서 _mission 변수에 넣음
    # ------------------------------------------------
    sql = "SELECT missionID FROM Mission ORDER BY missionID DESC limit 1"
    try:
        DB.curs.execute(sql)
        row = DB.curs.fetchone()
        _mission = int(row['missionID'])
        print("마지막 미션 번호 : " , _mission)
    except Exception as e:
        print("_mission 가져오기 오류 , ",e)
    # ------------------------------------------------
    #mission 가져옴-----------------------------------
    sql = "SELECT missionID FROM Mission"
    try:
        DB.curs.execute(sql)
        missions = DB.curs.fetchall()

    except Exception as e:
        print("mission 가져오기 오류 , ",e)
    #User 가져옴--------------------------------------
    sql = "SELECT userIndex FROM User"
    try:
        DB.curs.execute(sql)
        users = DB.curs.fetchall()

    except Exception as e:
        print("user 가져오기 오류 , ", e)
    #evaluationList 선언

    evaluationList = pd.DataFrame(data=[{"users_id": i['userIndex'], "missions_id":missionID['missionID'], "weather":-1,"temperature":-1, "rating":-1}  for i in users for missionID in missions],
                                  columns=['users_id','missions_id','weather','temperature','rating'],
                                  index=[[userIndex['userIndex'] for userIndex in users for i in missions], [missionID['missionID'] for i in users for missionID in missions]])
    '''            user_id     missions_id     weather     temperature     rating
    user1   1
            2
            3
            4
            5
    user2   1
            2
            3
            4
            5
    '''


    #sql 문을 불러옴
    sql = "SELECT user, mission, weather, temperature, rating FROM MissionEvaluation"
    try:
        DB.curs.execute(sql)
        rows=DB.curs.fetchall()
        #print(rows)
    except Exception as e:
        print("missionEvaluation 가져오기 오류 , ",e)


    #print(evaluationList)
    for row in rows:
        if row['weather']==None:
            row['weather'] =-1
        if row['temperature']==None:
            row['temperature'] =-1
        evaluationList.loc[(int(row['user']),int(row['mission'])),"weather"] = row["weather"]
        evaluationList.loc[(int(row['user']),int(row['mission'])), "temperature"] = row["temperature"]
        evaluationList.loc[(int(row['user']),int(row['mission'])), "rating"] = row["rating"]


    #print(evaluationList)
    DB.dbDisconnect()
    return evaluationList



