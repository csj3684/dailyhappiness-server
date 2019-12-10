

from flask import Blueprint, request, render_template, flash, redirect, url_for,jsonify
from flask import current_app as app
from app.main.DB import DB
import mysql.connector
import json

loginPage= Blueprint('loginPage', __name__, url_prefix='/login')

@loginPage.route('/', methods=['GET','POST'])
def login():
    if request.method =='POST':
        _id = request.form['id']
        _password = request.form['password']
        db = DB()
        db.dbConnect()
        db.setCursorDic()
        
        sql = "select * from User where id = %s and password = %s"
        
        try:
            db.curs.execute(sql, (_id, _password))

            rows = db.curs.fetchone()
            print("rows", rows)

        except mysql.connector.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
        if rows == None:
            return json.dumps({'error':1}).encode('utf-8')
        '''미션을 한 번도 안 했는지 확인인(설문조사 포)'''
        sql = "select user from MissionEvaluation where user = %s"
        try:
            db.curs.execute(sql, (_id,))

            row = db.curs.fetchone()
            print("row",row)
        except mysql.connector.Error as e:
            print("error")
            print("Error %d: %s" % (e.args[0], e.args[1]))

        if row:
            rows['isFirst'] = 0
        else:
            rows['isFirst'] = 1
        db.dbDisconnect()
        print("로그인",rows)
        rows['error'] = 0
        return json.dumps(rows).encode('utf-8')
    elif request.method =='GET':
        return 'GET'

        
        


