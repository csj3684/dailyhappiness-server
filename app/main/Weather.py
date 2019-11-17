

from urllib.request import urlopen
import json
from datetime import datetime



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
        for i in weather_info:
            if i['category'] == 'SKY':
                if i['fcstValue'] == 1:
                    weather= 1 # 맑음
                elif i['fcstValue'] == 4:
                    weather= 4 # 흐림
            elif i['category'] == 'PTY':
                if i['fcstValue'] == 1 or 2:
                    weather= 2 # 비
                elif i['fcstValue'] ==3:
                    weather= 3 # 눈
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



