

from urllib.request import urlopen
import json
from datetime import datetime, timedelta
import urllib.request,re
import xml.etree.ElementTree as ET

from pyowm import OWM



temperature =-1
weather =-1
def getTemperature():
    if temperature ==-1:
        print("getTodaysWeather를 먼저 호출하세요")
    return temperature

def getWeather():
    if weather ==-1:
        print("getTodaysWeather를 먼저 호출하세요")
    return weather

def getTodaysWeather(rs):
    now = datetime.now()
    now_date = now.strftime('%Y%m%d')
    now_hour = int(now.strftime('%H'))
    global temperature
    global weather
    if now_hour < 6:
        base_date = str(int(now_date) - 1)
    else:
        base_date = now_date
    base_hour = get_base_time(now_hour)

    service_key = '6V5ITltjhLcj9eixxl5fDhAEwFcUbTiR%2FKJJBdoShEFbzjz3fzyiC4FpUeGkPJmcZTodxNH0ETVHDN5H4X%2FrrQ%3D%3D'
    num_of_rows = '10'
    _type = 'json'
    url = 'http://newsky2.kma.go.kr/service/SecndSrtpdFrcstInfoService2/ForecastSpaceData?serviceKey={}'\
          '&base_date={}&base_time={}&nx={}&ny={}&numOfRows={}&_type={}'.format(
            service_key, base_date, base_hour, rs['x'], rs['y'], num_of_rows, _type)

    response_body = urlopen(url).read().decode('utf8')
    jsonData = json.loads(response_body)

    parseWeatherData(jsonData)


def parseWeatherData(jsonData):
    global weather
    global temperature
    try:
        weather_info = jsonData['response']['body']['items']['item']
        #'sunny', 'cloudy', 'rainy', 'snowy'
        for i in weather_info:
            if i['category'] == 'SKY':
                if i['fcstValue'] == 1:
                    print("sunny")
                    weather= 'sunny' # 맑음
                elif i['fcstValue'] == 4 or 3:
                    print("cloudy")
                    weather= 'cloudy' # 흐림
                elif i['fcstValue'] == 2:
                    print("rainy")
                    weather = 'rainy'  # 흐림
            elif i['category'] == 'PTY':
                if i['fcstValue'] == 1 or 2:
                    print("rainy")
                    weather= 'rainy' # 비
                elif i['fcstValue'] ==3:
                    print("sonwy")
                    weather= 'snowy' # 눈
            elif i['category'] == 'T3H':
                temperature = i['fcstValue']
    except KeyError:
        print('getSkyInfo 실패!')

def get_base_time(hour):
    hour = int(hour)
    if hour < 3:
        temp_hour = '20'
    elif hour < 6:
        temp_hour = '23'
    elif hour < 9:
        temp_hour = '02'
    elif hour < 12:
        temp_hour = '05'
    elif hour < 15:
        temp_hour = '08'
    elif hour < 18:
        temp_hour = '11'
    elif hour < 21:
        temp_hour = '14'
    elif hour < 24:
        temp_hour = '17'
    return temp_hour + '00'

def get_max_min_weekly_weather():
    url = "http://www.weather.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=108"

    temperature_max = []
    temperature_min = []

    ufile = urllib.request.urlopen(url)
    contents = ufile.read().decode('utf-8')

    root = ET.fromstring(contents)

    for location in root.find('channel').find('item').find('description').find('body').findall('location'):
        for data in location.findall('data'):
            temperature_max.append(data.find('tmx').text)
            temperature_min.append(data.find('tmn').text)

    return {'max': max(temperature_max), 'min':min(temperature_min)}

def get_weekly_weather_list():
    today = datetime.today().date()#.strftime('%Y%m%d')
    today_day_index = today.weekday()
    weather_list = ['cloudy' for i in range(7)]
    openweather_key = "651e99c02e73a125832267efd3e2b11e"
    owm = OWM(openweather_key)
    fc = owm.three_hours_forecast('Korea')


    f = fc.get_forecast()
    lst = f.get_weathers()
    for weather in lst:

        date_str = weather.get_reference_time('date').strftime('%Y%m%d')
        date_ = datetime.strptime(date_str,'%Y%m%d').date()

        date_diff = date_ - today
        if today_day_index + date_diff.days>6:
            break
        weather_list[today_day_index+date_diff.days] = str_match(weather.get_status())

    if weather_list[6]==0:
        weather_list[6] = weather_list[5]

    return weather_list

def str_match(str):
    if str == 'Rain' or str == 'Thunderstorm' or str=='Drizzle' or str=='Rain' or str =='Snow':
        return 'cloudy'
    else:
        return 'sunny'


