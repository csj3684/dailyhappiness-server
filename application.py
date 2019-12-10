import time
from app import app
import threading

from app.main.MySchedule import weekSchedule, daySchedule

from app.main.MissionBundle import calculate_R_hat, init_keep_info
print("application.py")
print("calculating_R_hat")
START = time.time()
calculate_R_hat(1)
print(time.time() - START)
print("R_hat_complete")
init_keep_info()
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'



weekThread = threading.Thread(target=weekSchedule)
weekThread.start()
dayThread = threading.Thread(target=daySchedule)
dayThread.setDaemon(True)
dayThread.start()

app.run(host="0.0.0.0",debug=True)

