import mysql.connector

class DB:
    conn=0
    curs=0
    
    
    def dbConnect(self):
        self.conn = mysql.connector.connect(host='dailyhappiness.cglqv9cus9nr.ap-northeast-2.rds.amazonaws.com', user='admin', password='', db='dailyhappiness', charset='utf8')
        
      
    def setCursorDic(self):
        self.curs = self.conn.cursor(dictionary=True)
        
   
    def dbDisconnect(self):
        self.conn.close()

