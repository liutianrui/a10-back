import pymysql
import csv
from collections import namedtuple

def get_data(filename):
    with open(filename) as f:
        f_csv = csv.reader(f)
        headings = next(f_csv)
        Row = namedtuple('Row', headings)
        for r in f_csv:
            yield Row(*r)

def execute_sql(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        print('执行成功')

def main():
    conn = pymysql.connect(
        host='localhost',
        user='root',
        passwd='root',
        db='fault_diagnosis_database',
        port=3306,
        charset="utf8")
    print(1)
    # 将CSV文件中的数据插入MySQL数据表中
    SQL_FORMAT = """insert into dataset1 values('{0}', '{1}', '{2}', '{3}', '{4}', '{5}', '{6}', '{7}', '{8}', '{9}', '{10}', '{11}', '{12}', '{13}', '{14}', '{15}', '{16}', '{17}', '{18}', '{19}', '{20}', '{21}', '{22}', '{23}', '{24}', '{25}', '{26}', '{27}', '{28}', '{29}', '{30}', '{31}', '{32}', '{33}', '{34}', '{35}',
     '{36}', '{37}', '{38}', '{39}', '{40}', '{41}', '{42}', '{43}', '{44}', '{45}', '{46}', '{47}', '{48}', '{49}', '{50}', '{51}', '{52}', '{53}', '{54}', '{55}', '{56}', '{57}', '{58}', '{59}', '{60}', '{61}', '{62}', '{63}', '{64}', '{65}', '{66}', '{67}', '{68}', '{69}', '{70}',
     '{71}', '{72}', '{73}', '{74}', '{75}', '{76}', '{77}', '{78}', '{79}', '{80}', '{81}', '{82}', '{83}', '{84}', '{85}', '{86}', '{87}', '{88}', '{89}', '{90}', '{91}', '{92}', '{93}', '{94}', '{95}', '{96}', '{97}', '{98}', '{99}', '{100}', '{101}', '{102}', '{103}', '{104}', '{105}', '{106}', '{107}','{108}')"""
    conn.autocommit(1)
    for t in get_data('../data/preprocess_train.csv'):
        '''
        print(t.sample_id, t.feature0, t.feature1, t.feature2, t.feature3, t.feature4, t.feature5, t.feature6,
              t.feature7, t.feature8, t.feature9, t.feature10, t.feature11, t.feature12, t.feature13, t.feature14,
              t.feature15, t.feature16, t.feature17, t.feature18, t.feature19, t.feature20, t.feature21, t.feature22,
              t.feature23, t.feature24, t.feature25, t.feature26, t.feature27, t.feature28, t.feature29, t.feature30,
              t.feature31, t.feature32,
              t.feature33, t.feature34, t.feature35, t.feature36, t.feature37, t.feature38, t.feature39, t.feature40,
              t.feature41, t.feature42, t.feature43, t.feature44, t.feature45, t.feature46, t.feature47, t.feature48,
              t.feature49, t.feature50, t.feature51, t.feature52, t.feature53, t.feature54, t.feature55, t.feature56,
              t.feature57, t.feature58, t.feature59, t.feature60, t.feature61, t.feature62, t.feature63, t.feature64,
              t.feature65, t.feature66, t.feature67, t.feature68, t.feature69, t.feature70, t.feature71, t.feature72,
              t.feature73, t.feature74, t.feature75, t.feature76,
              t.feature77, t.feature78, t.feature79, t.feature80, t.feature81, t.feature82, t.feature83, t.feature84,
              t.feature85, t.feature86, t.feature87, t.feature88, t.feature89, t.feature90, t.feature91, t.feature92,
              t.feature93, t.feature94, t.feature95, t.feature96, t.feature97, t.feature98, t.feature99, t.feature100,
              t.feature101, t.feature102, t.feature103, t.feature104, t.feature105, t.feature106, t.label)
        '''
        sql = SQL_FORMAT.format(t.sample_id, t.feature0, t.feature1, t.feature2, t.feature3, t.feature4, t.feature5,
                                t.feature6, t.feature7, t.feature8, t.feature9, t.feature10, t.feature11, t.feature12,
                                t.feature13, t.feature14, t.feature15, t.feature16, t.feature17, t.feature18,
                                t.feature19, t.feature20, t.feature21, t.feature22, t.feature23, t.feature24,
                                t.feature25, t.feature26, t.feature27, t.feature28, t.feature29, t.feature30,
                                t.feature31, t.feature32,
                                t.feature33, t.feature34, t.feature35, t.feature36, t.feature37, t.feature38,
                                t.feature39, t.feature40, t.feature41, t.feature42, t.feature43, t.feature44,
                                t.feature45, t.feature46, t.feature47, t.feature48, t.feature49, t.feature50,
                                t.feature51, t.feature52, t.feature53, t.feature54, t.feature55, t.feature56,
                                t.feature57, t.feature58, t.feature59, t.feature60, t.feature61, t.feature62,
                                t.feature63, t.feature64, t.feature65, t.feature66, t.feature67, t.feature68,
                                t.feature69, t.feature70, t.feature71, t.feature72, t.feature73, t.feature74,
                                t.feature75, t.feature76,
                                t.feature77, t.feature78, t.feature79, t.feature80, t.feature81, t.feature82,
                                t.feature83, t.feature84, t.feature85, t.feature86, t.feature87, t.feature88,
                                t.feature89, t.feature90, t.feature91, t.feature92, t.feature93, t.feature94,
                                t.feature95, t.feature96, t.feature97, t.feature98, t.feature99, t.feature100,
                                t.feature101, t.feature102, t.feature103, t.feature104, t.feature105, t.feature106,
                                t.label)
        print(sql)
        conn.ping(reconnect=True) # 每次连接之前，会检查当前连接是否已关闭，如果连接关闭则会重新进行连接
        execute_sql(conn, sql)

        conn.commit()  # 提交到数据库
        conn.close()  # 关闭数据库服务


if __name__ == '__main__':
    main()
