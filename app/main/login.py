

from flask import Blueprint, request, render_template, flash, redirect, url_for,jsonify
from flask import current_app as app
from app.main.DB import DB

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
        
            rows = DB.curs.fetchall()
        
        except Exception as e:
            print(e)
        finally:
            DB.dbDisconnect()
        
        
        return jsonify(rows)
      elif request.method =='GET':
        return
        
        


