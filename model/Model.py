# 使用flask_sqlalchemy操作数据库需要定义实体类，实体类与真实数据库表进行映射
from util.flask_sql_init import db

class userinfo(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50))
    password = db.Column(db.String(50))

    def __init__(self, username, password):
        self.username = username
        self.password = password

# class Merge(db.Model):
class dataset(db.Model):
    sample_id = db.Column(db.Integer, primary_key=True)
    feature0 = db.Column(db.Text)
    feature1 = db.Column(db.Text)
    feature2 = db.Column(db.Text)
    feature3 = db.Column(db.Text)
    feature4 = db.Column(db.Text)
    feature5 = db.Column(db.Text)
    feature6 = db.Column(db.Text)
    feature7 = db.Column(db.Text)
    feature8 = db.Column(db.Text)
    feature9 = db.Column(db.Text)
    feature10 = db.Column(db.Text)
    feature11 = db.Column(db.Text)
    feature12 = db.Column(db.Text)
    feature13 = db.Column(db.Text)
    feature14 = db.Column(db.Text)
    feature15 = db.Column(db.Text)
    feature16 = db.Column(db.Text)
    feature17 = db.Column(db.Text)
    feature18 = db.Column(db.Text)
    feature19 = db.Column(db.Text)
    feature20 = db.Column(db.Text)
    feature21 = db.Column(db.Text)
    feature22 = db.Column(db.Text)
    feature23 = db.Column(db.Text)
    feature24 = db.Column(db.Text)
    feature25 = db.Column(db.Text)
    feature26 = db.Column(db.Text)
    feature27 = db.Column(db.Text)
    feature28 = db.Column(db.Text)
    feature29 = db.Column(db.Text)
    feature30 = db.Column(db.Text)
    feature31 = db.Column(db.Text)
    feature32 = db.Column(db.Text)
    feature33 = db.Column(db.Text)
    feature34 = db.Column(db.Text)
    feature35 = db.Column(db.Text)
    feature36 = db.Column(db.Text)
    feature37 = db.Column(db.Text)
    feature38 = db.Column(db.Text)
    feature39 = db.Column(db.Text)
    feature40 = db.Column(db.Text)
    feature41 = db.Column(db.Text)
    feature42 = db.Column(db.Text)
    feature43 = db.Column(db.Text)
    feature44 = db.Column(db.Text)
    feature45 = db.Column(db.Text)
    feature46 = db.Column(db.Text)
    feature47 = db.Column(db.Text)
    feature48 = db.Column(db.Text)
    feature49 = db.Column(db.Text)
    feature50 = db.Column(db.Text)
    feature51 = db.Column(db.Text)
    feature52 = db.Column(db.Text)
    feature53 = db.Column(db.Text)
    feature54 = db.Column(db.Text)
    feature55 = db.Column(db.Text)
    feature56 = db.Column(db.Text)
    feature57 = db.Column(db.Text)
    feature58 = db.Column(db.Text)
    feature59 = db.Column(db.Text)
    feature60 = db.Column(db.Text)
    feature61 = db.Column(db.Text)
    feature62 = db.Column(db.Text)
    feature63 = db.Column(db.Text)
    feature64 = db.Column(db.Text)
    feature65 = db.Column(db.Text)
    feature66 = db.Column(db.Text)
    feature67 = db.Column(db.Text)
    feature68 = db.Column(db.Text)
    feature69 = db.Column(db.Text)
    feature70 = db.Column(db.Text)
    feature71 = db.Column(db.Text)
    feature72 = db.Column(db.Text)
    feature73 = db.Column(db.Text)
    feature74 = db.Column(db.Text)
    feature75 = db.Column(db.Text)
    feature76 = db.Column(db.Text)
    feature77 = db.Column(db.Text)
    feature78 = db.Column(db.Text)
    feature79 = db.Column(db.Text)
    feature80 = db.Column(db.Text)
    feature81 = db.Column(db.Text)
    feature82 = db.Column(db.Text)
    feature83 = db.Column(db.Text)
    feature84 = db.Column(db.Text)
    feature85 = db.Column(db.Text)
    feature86 = db.Column(db.Text)
    feature87 = db.Column(db.Text)
    feature88 = db.Column(db.Text)
    feature89 = db.Column(db.Text)
    feature90 = db.Column(db.Text)
    feature91 = db.Column(db.Text)
    feature92 = db.Column(db.Text)
    feature93 = db.Column(db.Text)
    feature94 = db.Column(db.Text)
    feature95 = db.Column(db.Text)
    feature96 = db.Column(db.Text)
    feature97 = db.Column(db.Text)
    feature98 = db.Column(db.Text)
    feature99 = db.Column(db.Text)
    feature100 = db.Column(db.Text)
    feature101 = db.Column(db.Text)
    feature102 = db.Column(db.Text)
    feature103 = db.Column(db.Text)
    feature104 = db.Column(db.Text)
    feature105 = db.Column(db.Text)
    feature106 = db.Column(db.Text)
    label = db.Column(db.Integer)
    # stuno = db.Column(db.Integer)
    # location = db.Column(db.String(64))
    # time = db.Column(db.String(64))
    # consume = db.Column(db.Float)
    # rank = db.Column(db.Integer)
    def __init__(self, sample_id, feature0, feature1, feature2, feature3,
                 feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13,
                 feature14, feature15, feature16, feature17, feature18, feature19, feature20, feature21, feature22,
                 feature23, feature24, feature25, feature26, feature27, feature28, feature29, feature30, feature31,
                 feature32, feature33, feature34, feature35, feature36, feature37, feature38, feature39, feature40,
                 feature41, feature42, feature43, feature44, feature45, feature46, feature47, feature48, feature49,
                 feature50, feature51, feature52, feature53, feature54, feature55, feature56, feature57, feature58,
                 feature59, feature60, feature61, feature62, feature63, feature64, feature65, feature66, feature67,
                 feature68, feature69, feature70, feature71, feature72, feature73, feature74, feature75, feature76,
                 feature77, feature78, feature79, feature80, feature81, feature82, feature83, feature84, feature85,
                 feature86, feature87, feature88, feature89, feature90, feature91, feature92, feature93, feature94,
                 feature95, feature96, feature97, feature98, feature99, feature100, feature101, feature102, feature103,
                 feature104, feature105, feature106, label):
        self.sample_id = sample_id
        self.feature0 = feature0
        self.feature1 = feature1
        self.feature2 = feature2
        self.feature3 = feature3
        self.feature4 = feature4
        self.feature5 = feature5
        self.feature6 = feature6
        self.feature7 = feature7
        self.feature8 = feature8
        self.feature9 = feature9
        self.feature10 = feature10
        self.feature11 = feature11
        self.feature12 = feature12
        self.feature13 = feature13
        self.feature14 = feature14
        self.feature15 = feature15
        self.feature16 = feature16
        self.feature17 = feature17
        self.feature18 = feature18
        self.feature19 = feature19
        self.feature20 = feature20
        self.feature21 = feature21
        self.feature22 = feature22
        self.feature23 = feature23
        self.feature24 = feature24
        self.feature25 = feature25
        self.feature26 = feature26
        self.feature27 = feature27
        self.feature28 = feature28
        self.feature29 = feature29
        self.feature30 = feature30
        self.feature31 = feature31
        self.feature32 = feature32
        self.feature33 = feature33
        self.feature34 = feature34
        self.feature35 = feature35
        self.feature36 = feature36
        self.feature37 = feature37
        self.feature38 = feature38
        self.feature39 = feature39
        self.feature40 = feature40
        self.feature41 = feature41
        self.feature42 = feature42
        self.feature43 = feature43
        self.feature44 = feature44
        self.feature45 = feature45
        self.feature46 = feature46
        self.feature47 = feature47
        self.feature48 = feature48
        self.feature49 = feature49
        self.feature50 = feature50
        self.feature51 = feature51
        self.feature52 = feature52
        self.feature53 = feature53
        self.feature54 = feature54
        self.feature55 = feature55
        self.feature56 = feature56
        self.feature57 = feature57
        self.feature58 = feature58
        self.feature59 = feature59
        self.feature60 = feature60
        self.feature61 = feature61
        self.feature62 = feature62
        self.feature63 = feature63
        self.feature64 = feature64
        self.feature65 = feature65
        self.feature66 = feature66
        self.feature67 = feature67
        self.feature68 = feature68
        self.feature69 = feature69
        self.feature70 = feature70
        self.feature71 = feature71
        self.feature72 = feature72
        self.feature73 = feature73
        self.feature74 = feature74
        self.feature75 = feature75
        self.feature76 = feature76
        self.feature77 = feature77
        self.feature78 = feature78
        self.feature79 = feature79
        self.feature80 = feature80
        self.feature81 = feature81
        self.feature82 = feature82
        self.feature83 = feature83
        self.feature84 = feature84
        self.feature85 = feature85
        self.feature86 = feature86
        self.feature87 = feature87
        self.feature88 = feature88
        self.feature89 = feature89
        self.feature90 = feature90
        self.feature91 = feature91
        self.feature92 = feature92
        self.feature93 = feature93
        self.feature94 = feature94
        self.feature95 = feature95
        self.feature96 = feature96
        self.feature97 = feature97
        self.feature98 = feature98
        self.feature99 = feature99
        self.feature100 = feature100
        self.feature101 = feature101
        self.feature102 = feature102
        self.feature103 = feature103
        self.feature104 = feature104
        self.feature105 = feature105
        self.feature106 = feature106
        self.label = label
        # self.location = location
        # self.time = time
        # self.consume = consume
        # self.rank = rank


class Score_rank(db.Model):
    ID = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64))
    JAVA = db.Column(db.Integer)
    Python = db.Column(db.Integer)
    DS = db.Column(db.Integer)
    OS = db.Column(db.Integer)
    CS = db.Column(db.Integer)
    CN = db.Column(db.Integer)
    rank = db.Column(db.Integer)
    notice = db.Column(db.String(64))
    grade = db.Column(db.CHAR(200))

    def __init__(self, ID, name, JAVA, Python, DS, OS, CS, CN, rank, notice, grade):
        self.ID = ID
        self.name = name
        self.JAVA = JAVA
        self.Python = Python
        self.DS = DS
        self.OS = OS
        self.CS = CS
        self.CN = CN
        self.rank = rank
        self.notice = notice
        self.grade = grade
