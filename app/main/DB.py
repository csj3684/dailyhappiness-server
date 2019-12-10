import pymysql

class DB:
    conn=0
    curs=0
    
    @staticmethod
    def dbConnect():
        DB.conn = pymysql.connect(host='dailyhappiness.cglqv9cus9nr.ap-northeast-2.rds.amazonaws.com', user='admin', password='', db='dailyhappiness', charset='utf8')
        
    @staticmethod    
    def setCursorDic():
        DB.curs = DB.conn.cursor(pymysql.cursors.DictCursor)
        
    @staticmethod    
    def dbDisconnect():
        DB.conn.close()

