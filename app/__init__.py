from flask import Flask, url_for

app = Flask(__name__)

#파일 이름이 index.py 이므로
from app.main.index import main as main
from app.main.login import loginPage
from app.main.register import registerPage
from app.main.MissionBundle import missionBundlePage
from app.main.WriteReview import writeReviewPage
from app.main.GetReview import getReviewPage
from app.main.MissionCandidate import MissionCandidate
from app.main.Manage import Manage
from app.main.MissionKing import MissionKing
from app.main.MySchedule import mySchedule


app.register_blueprint(main)
app.register_blueprint(loginPage)    
app.register_blueprint(registerPage)
app.register_blueprint(missionBundlePage)
app.register_blueprint(writeReviewPage)
app.register_blueprint(getReviewPage)
app.register_blueprint(MissionCandidate)
app.register_blueprint(Manage)
app.register_blueprint(MissionKing)
app.register_blueprint(mySchedule)