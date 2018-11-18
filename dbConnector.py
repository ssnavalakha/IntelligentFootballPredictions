import pymysql

HOST = "localhost"
USER = "root"
PASSWORD = "root123"
DB = "football_stats"

def getConnection(host = HOST, user = USER, password = PASSWORD, db = DB):
    # Connect to the database
    connection = pymysql.connect(host=host,
                                 user=user,
                                 password=password,
                                 db=db,
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
    return connection

