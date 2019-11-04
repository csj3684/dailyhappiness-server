'''
데이터베이스 MissionEvaluation 테이블에서 데이터를 가져와서 비어있는 점수들을 evaluationList 에 채워넣는다.
'''
import json

from app.main.DB import DB
from app.main.MissionEvaluation import MissionEvalutation


def expectRatings():
    print("expectRatings 함수 호출")
    evaluationList = getEvaluation()

    '''
    1. 회귀분석으로 예측하기
    '''



    '''
    2. KNN으로 예측하기
    '''

    '''
    3. matrix completion 으로 예측하기
    '''

    return evaluationList



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

    #evaluationList 선언

    evaluationList = [[0 for i in range(_mission)] for j in range(_user)]
    print("evaluationList 선언 완료\n", evaluationList)

    #2중 for 문을 돌면서 evaluationList 를 채운다.
    for userNumber in range(_user):
        for missionNumber in range(_mission):
            _userNumber = userNumber+1
            _missionNumber = missionNumber+1
            sql = f"SELECT IFNULL(rating, -1) as rating, IFNULL(weather, -1) as weather, IFNULL(temperature, -1) as temperature FROM MissionEvaluation WHERE user={_userNumber} and mission={_missionNumber}"

            try:
                DB.curs.execute(sql)
                row = DB.curs.fetchone()

                if row is None:
                    evaluationList[userNumber][missionNumber] = MissionEvalutation(_userNumber,_missionNumber,-1,-1,-1)
                else:
                    evaluationList[userNumber][missionNumber] = MissionEvalutation(_userNumber,_missionNumber,row['rating'],row['weather'],row['temperature'])
                #print(_userNumber, " , ", _missionNumber," : ",evaluationList[userNumber][missionNumber].rating)

            except Exception as e:
                print("evaluationList 채우기 오류: ",e)


    DB.dbDisconnect()
    return evaluationList



