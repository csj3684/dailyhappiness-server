from flask import Flask, url_for

app = Flask(__name__)

#파일 이름이 index.py 이므로
from app.main.index import main as main
from app.main.login import loginPage
from app.main.register import registerPage
from app.main.MissionBundle import missionBundlePage


app.register_blueprint(main)
app.register_blueprint(loginPage)    
app.register_blueprint(registerPage)
app.register_blueprint(missionBundlePage)

