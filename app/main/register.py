from flask import Blueprint, request, render_template, flash, redirect, url_for,jsonify
from flask import current_app as app
from app.main.DB import DB
from app.main.MissionBundle import calculate_R_hat, add_new_user

import mysql.connector
import json

registerPage= Blueprint('registerPage', __name__, url_prefix='/register')

@registerPage.route('/', methods=['GET','POST'])
def register():
    print("register.py")
    
    db=DB()
    db.dbConnect()
    db.setCursorDic()
    if request.method =='POST':
        _id = request.form['id']
        _password = request.form['password']
        _age = request.form['age']
        _gender = request.form['gender']
    
        print(_id, _password, _age, _gender)


        sql = "INSERT INTO User(id, password, age, gender) VALUES(%s,%s,%s,%s)"
        
        try:
            db.curs.execute(sql, (_id, _password,_age,_gender))
            db.conn.commit()
            success = {'success':True}
        except mysql.connector.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))

            
            success = {'success':False}
        sql = "SELECT userIndex FROM User WHERE id = %s"
        try:
            db.curs.execute(sql, (_id,))
            row = db.curs.fetchone()
            user_id = row['userIndex']
            success = {'success':True}
        except mysql.connector.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))

        add_new_user(user_id)

        return json.dumps(success).encode('utf-8')
    elif request.method =='GET':
        return 'GET'

@registerPage.route('/idCheck', methods=['GET','POST'])
def checkId():
    _id = request.form['id']
    db = DB()
    db.dbConnect()
    db.setCursorDic()
    '''아이디 중복 체크'''
    sql = "SELECT id FROM User WHERE id=%s"
    try:
        db.curs.execute(sql, (_id,))
        row = db.curs.fetchone()
    except Exception as e:
        print(e)

    db.dbDisconnect()
    if row:
        success = {'duplicate': True}
        return json.dumps(success).encode('utf-8')
    else:
        success = {'duplicate': False}
        return json.dumps(success).encode('utf-8')

@registerPage.route('/mypage', methods=['GET','POST'])
def mypage():
    print("mypage 호출")
    user = request.form['userIndex']
    time = request.form['time_affordable']
    expense = request.form['expense_affordable']
    push = request.form['push_notification']

    db=DB()

    db.dbConnect()
    db.setCursorDic()

    sql = "UPDATE User set time_affordable=%s,expense_affordable=%s, push_notification =%s WHERE userIndex=%s"
    try:
        db.curs.execute(sql,(time,expense,push,user))
        db.conn.commit()
        success = {'duplicate': True}
    except Exception as e:
        print(e)
        success = {'duplicate': False}

    return json.dumps(success).encode('utf-8')


