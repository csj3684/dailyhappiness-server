
 

from flask import Blueprint, request, render_template, flash, redirect, url_for,jsonify
from flask import current_app as app
from app.main.DB import DB
import json

registerPage= Blueprint('registerPage', __name__, url_prefix='/register')

@registerPage.route('/', methods=['GET','POST'])
def register():
      DB.dbConnect()
      DB.setCursorDic()
      if request.method =='POST':
        _id = request.form['id']
        _password = request.form['password']
        _age = request.form['age']
        _gender = request.form['gender']
        
        print(_id, _password, _age, _gender)


        sql = "INSERT INTO User(id, password, age, gender) VALUES(%s,%s,%s,%s)"
        
        try:
            DB.curs.execute(sql, (_id, _password,_age,_gender))
            DB.conn.commit()
            success = {'success':True}
        except Exception as e:
            print(e)
            
            success = {'success':False}
        finally:
            DB.dbDisconnect()

        return json.dumps(success).encode('utf-8')
      elif request.method =='GET':
        return 'GET'

@registerPage.route('/idCheck', methods=['GET','POST'])
def checkId():
    _id = request.form['id']

    DB.dbConnect()
    DB.setCursorDic()
    '''아이디 중복 체크'''
    sql = "SELECT id FROM User WHERE id=%s"
    try:
        DB.curs.execute(sql, (_id))
        row = DB.curs.fetchone()
    except Exception as e:
        print(e)

    DB.dbDisconnect()
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

    DB.dbConnect()
    DB.setCursorDic()

    sql = "UPDATE User set time_affordable=%s,expense_affordable=%s, push_notification =%s WHERE userIndex=%s"
    try:
        DB.curs.execute(sql,(time,expense,push,user))
        DB.conn.commit()
        success = {'duplicate': True}
    except Exception as e:
        print(e)
        success = {'duplicate': False}

    return json.dumps(success).encode('utf-8')


