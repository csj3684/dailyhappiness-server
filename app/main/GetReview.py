from flask import Blueprint, request, render_template, flash, redirect, url_for,jsonify
from flask import current_app as app
from app.main.DB import DB
import pymysql
import json
import datetime

getReviewPage = Blueprint('getReviewPage', __name__, url_prefix='/getReviews')
@getReviewPage.route('/', methods=['GET', 'POST'])
def getReviews():
    DB.dbConnect()
    DB.setCursorDic()


    print("getReviews 호출")
    userIndex = request.form['userIndex']
    getMine = request.form['getMine']
    print(type(request.form['reviewCount']))
    reviewCount = int(request.form['reviewCount'])
    print("getMine = ",type(getMine))


    if getMine=="true":
        sql = "SELECT evaluationIndex, user,id, mission,missionName, rating, weather, date, comment, picture, temperature ,User.grade as grade FROM MissionEvaluation join Mission on MissionEvaluation.mission = Mission.missionID join User on (MissionEvaluation.user = User.userIndex) WHERE MissionEvaluation.user={} and NOT date IS NULL ORDER BY date DESC LIMIT {} , 10".format(userIndex, reviewCount)
    else:
        sql = "SELECT evaluationIndex,user,id,mission,missionName, rating, weather, date, comment, picture, temperature,User.grade as grade FROM MissionEvaluation join Mission on MissionEvaluation.mission = Mission.missionID join User on (MissionEvaluation.user = User.userIndex) WHERE NOT date IS NULL ORDER BY date DESC LIMIT {} , 10".format(reviewCount)

    try:

        DB.curs.execute(sql)
        rows = DB.curs.fetchall()
    except pymysql.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    DB.dbDisconnect()

    rows = list({row['evaluationIndex'] : row for row in rows}.values())
    print(rows)
    temp = json.dumps(rows, default=json_default).encode('utf-8')
    return temp


def json_default(value):
    if isinstance(value, datetime.date):
        return value.strftime('%Y-%m-%d')
    raise TypeError('not JSON serializable')

