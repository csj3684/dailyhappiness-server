import functions, set_R_hat


weather_category = ['sunny', 'cloudy', 'rainy' , 'snowy']

import set_R_hat

user_info = set_R_hat.user_info

mission_info = set_R_hat.mission_info

classified_R_hat = set_R_hat.classified_R_hat

weathers = ['sunny', 'sunny', 'sunny', 'rainy', 'snowy', 'snowy', 'snowy']

weekly_weather = functions.get_weekly_weather(weathers, weather_category)
 
target_user_id = 'u1'



while True:
    today_weather = weathers[today_idx]
    print("action : ", end="")
    
    action = input()
    
    if action == '1': 
        print("get_applicable_mission : ", functions.set_user_applicable_missions(target_user_id, classified_R_hat, user_info, mission_info, weekly_weather))
    elif action == '2':
        print("set_weekly_mission : ", functions.set_weekly_mission(target_user_id, user_info, mission_info, functions.datetime.datetime.today().weekday(), weekly_weather))
    elif action == '3':
        daily_mission = functions.get_daily_mission(target_user_id, user_info, today_weather)
        print("daily_mission", daily_mission)
    elif action == "done":
        functions.update_user_applicable_missions(target_user_id, user_info, mission_info, daily_mission, "done", functions.datetime.datetime.today().weekday(), weekly_weather)
        today_idx += 1
    elif action == "pass":
        functions.update_user_applicable_missions(target_user_id, user_info, mission_info, daily_mission, "pass", functions.datetime.datetime.today().weekday(), weekly_weather)
    elif action == "exit":
        break
    elif action == "switch":
        target_user_id = input()
        continue
    elif action == "weekly_weather":
        print(weekly_weather)
    elif action == "daily_mission":
        print(daily_mission)
    elif action == "applicable_mission":
        print(user_info.loc[target_user_id]['applicable_missions'])
    elif action == "weekly_mission":
        print(user_info.loc[target_user_id]['weekly_missions'].iloc[0].loc['mission_set'])
    elif action == "idx":
        print(today_idx)
    elif action == "reset":
        today_idx = 0
        continue
    else:
        print("Key Error")
        continue

 