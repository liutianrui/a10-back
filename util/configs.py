#数据库配置
HOST = 'localhost'
PORT = '3306'
DATABASE = 'fault_diagnosis_database'
USERNAME = 'root'
PASSWORD = 'root'

DB_URI = "mysql+pymysql://{username}:{password}@{host}:{port}/{db}?charset=utf8".format(username=USERNAME,password=PASSWORD, host=HOST,port=PORT, db=DATABASE)

SQLALCHEMY_DATABASE_URI = DB_URI
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ECHO = True
