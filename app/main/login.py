

from flask import Blueprint, request, render_template, flash, redirect, url_for,jsonify
from flask import current_app as app
from app.main.DB import DB
import json

loginPage= Blueprint('loginPage', __name__, url_prefix='/login')

@loginPage.route('/', methods=['GET','POST'])
def login():
      if request.method =='POST':
        _id = request.form['id']
        _password = request.form['password']

        DB.dbConnect()
        DB.setCursorDic()
        
        sql = "select * from User where id = %s and password = %s"
        
        try:
            DB.curs.execute(sql, (_id, _password))

            rows = DB.curs.fetchone()

        except Exception as e:
            print(e)

        '''미션을 한 번도 안 했는지 확인인(설문조사 포)'''
        sql = "select user from MissionEvaluation where user = %s"
        try:
            DB.curs.execute(sql, (_id))

            row = DB.curs.fetchone()
            
        except Exception as e:
            print(e)

        if row:
            rows['isFirst'] = 0
        else:
            rows['isFirst'] = 1
        DB.dbDisconnect()
        print(rows)
        return json.dumps(rows).encode('utf-8')
      elif request.method =='GET':
        return 'GET'
        
        


