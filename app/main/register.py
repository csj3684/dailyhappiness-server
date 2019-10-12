
 

from flask import Blueprint, request, render_template, flash, redirect, url_for,jsonify
from flask import current_app as app
from app.main.DB import DB
import json

registerPage= Blueprint('registerPage', __name__, url_prefix='/register')

@registerPage.route('/', methods=['GET','POST'])
def register():
      if request.method =='POST':
        _id = request.form['id']
        _password = request.form['password']
        _age = request.form['age']
        _gender = request.form['gender']
        
        print(_id, _password, _age, _gender)
        DB.dbConnect()
        DB.setCursorDic()
        
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
        return
               
        
        


