# file name : index.py
# pwd : /project_name/app/main/index.py
 
from flask import Blueprint, request, render_template, flash, redirect, url_for
from flask import current_app as app
from app.main.DB import DB
import json
from app.main.Weather import getTemperature,getWeather, getTodaysWeather,get_weekly_weather_list

import pandas as pd
import app.main.MissionBundle as R_hat_module

import urllib.request,re
import xml.etree.ElementTree as ET
# 추가할 모듈이 있다면 추가
 
main= Blueprint('main', __name__, url_prefix='/')
 
@main.route('/', methods=['GET'])
def index():
      
      # /main/index.html은 사실 /project_name/app/templates/main/index.html을 가리킵니다.
      return render_template('/main/index.html')


@main.route('/image', methods=['GET'])
def image():
    # /main/index.html은 사실 /project_name/app/templates/main/index.html을 가리킵니다.
    print("image 함수 호출")
    #print(request.form)
    filename = "/static/img/"+request.args.get('filename')
    return render_template('/main/image.html', filename = filename)

@main.route('/aa', methods=['GET'])
def aa():
    list = get_weekly_weather_list()

    return json.dumps(list).encode('utf-8')

@main.route('/get', methods=['GET'])
def bb():
    value = R_hat_module.get_user_info()
    print("get_user_info")
    print(value)
    return 'a'

def json_default(value):
    if isinstance(value, datetime.date):
        return value.strftime('%Y-%m-%d')
    raise TypeError('not JSON serializable')

def plus(number):
    number.iloc[1,1] = number.iloc[1,1]+1

    return