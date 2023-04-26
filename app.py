from util import configs#引用数据库配置文件
from model.Model import *
from flask import *
from flask_cors import *
import Rf_Classifier_Plus
import jwt
from util.flask_sql_init import db #引用数据库启动文件
# 列表数据
# list_name = ['id', 'stuno', 'location', 'date', 'consume', 'rank', 'name']
# list_score_rank = ['ID', 'stu_name', 'JAVA', 'Python', 'DS', 'OS', 'CS', 'CN', 'rank', 'notice', 'grade']
# list_cn = ['图书馆', '校车', '教务处', '文印中心', '开水', '洗衣房', '超市', '校医院', '其他', '淋浴', '食堂']
# list_en = ['library', 'bus', 'office', 'printing', 'boiling', 'washingroom', 'supermarket', 'hospital', 'etc',
#            'bath', 'canteen']
list_fdd_data = ['sample_id','feature0', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14', 'feature15', 'feature16', 'feature17', 'feature18', 'feature19', 'feature20', 'feature21', 'feature22', 'feature23', 'feature24', 'feature25', 'feature26', 'feature27', 'feature28', 'feature29', 'feature30', 'feature31', 'feature32', 'feature33', 'feature34', 'feature35', 'feature36', 'feature37', 'feature38', 'feature39', 'feature40', 'feature41', 'feature42', 'feature43', 'feature44', 'feature45', 'feature46', 'feature47', 'feature48', 'feature49', 'feature50', 'feature51', 'feature52', 'feature53', 'feature54', 'feature55', 'feature56', 'feature57', 'feature58', 'feature59', 'feature60', 'feature61', 'feature62', 'feature63', 'feature64', 'feature65', 'feature66', 'feature67', 'feature68', 'feature69', 'feature70', 'feature71', 'feature72', 'feature73', 'feature74', 'feature75', 'feature76', 'feature77', 'feature78', 'feature79', 'feature80', 'feature81', 'feature82', 'feature83', 'feature84', 'feature85', 'feature86', 'feature87', 'feature88', 'feature89', 'feature90', 'feature91', 'feature92', 'feature93', 'feature94', 'feature95', 'feature96', 'feature97', 'feature98', 'feature99', 'feature100', 'feature101', 'feature102', 'feature103', 'feature104', 'feature105', 'feature106', 'label']
dict = {'SUCCESS': [200, 'success'], 'VERIFY_FAIL': [300, '温馨提示：账号和密码必须填写！'],
        'WRONG_PASSWORD': [400, '温馨提示：密码错误，请输入正确密码！']
    , 'NON_EXIST': [500, '温馨提示：该用户不存在，请注册！'], 'REQUEST_ERROR': [100, '请求错误！'], 'USER_EXIST': [600, '用户已存在，请登录！'],
        'NO_FOUND': [700, '查询无结果！']}

app = Flask(__name__)
CORS(app, supports_credentials=True)

# 加载配置文件
app.config.from_object(configs)

# db绑定app
db.init_app(app)


# 自定义响应类（便于向前端返回格式化的json数据）
class Respones_s():
    def success(data):
        """
        :return:成功时返回jsonify
        """
        code_l = dict['SUCCESS']
        return jsonify({'status': code_l[0], 'data': data, 'info': code_l[1]})

    def fail(code):
        """
        :return: 失败时返回jsonify
        """
        code_l = dict[code]
        return jsonify({'status': code_l[0], 'data': None, 'info': code_l[1]})

# 使用JWT加密：
def generate_jwt(payload, expiry, secret=None):
    """
    :param payload: dict载荷
    :param expiry: datetime 有效时间
    :param secret:  密钥
    :return: token
    """
    _payload = {'exp': expiry}
    _payload.update(payload)
    token = jwt.encode(_payload, secret, algorithm='HS256')
    return token

# 登录接口(完成)
@app.route('/user_login', methods=['POST', 'GET'])
def user_login():
    if request.method == "POST":
        username = request.get_json()['username']
        password = request.get_json()['password']
        print(username)
        print(password)
        data = userinfo.query.filter_by(username=username).all()

        print(data)
        if username is None and password is None:
            return Respones_s.fail('VERIFY_FAIL')
        elif data is not None and password == data[0].password:
            dict_secret = {'username': username, 'id': data[0].id}
            expire_time = '2h'
            secret_key = 'lalalahahahazhousil'
            token_secret = generate_jwt(dict_secret, expire_time, secret_key)
            return Respones_s.success(token_secret)
            # print(data[0].id)
        elif data is not None and password != data[0].password:
            return Respones_s.fail('WRONG_PASSWORD')
        else:
            return Respones_s.fail('NON_EXIST')
    return Respones_s.fail('REQUEST_ERROR')

# 注册接口（前端未开发）
@app.route("/register", methods=["GET", "POST"])
def register():
    code = 0
    if request.method == 'POST':
        username = request.get_json()['username']
        password = request.get_json()['password']
        data = userinfo.query.filter_by(username=username).all()
        if username is None and password is None:
            return Respones_s.fail('VERIFY_FAIL')
        elif data is not None and username == data[0].username:
            return Respones_s.fail('USER_EXIST')
        else:
            db.session.add(userinfo(username, password))
            db.session.commit()
            return Respones_s.success('注册成功！')
    return Respones_s.fail('REQUEST_ERROR')

# 算法接口
@app.route("/classify", methods=['GET'])
def classifier():
    """
    :return:返回模型每一个label的平均查准率和查全率和最终评价指标：macro_Precision, macro_Recall,macro_F1;
    :return: 返回测试集中每一个label的数量LABEL0, LABEL1, LABEL2, LABEL3, LABEL4, LABEL5
    """

    filepath = ".data/preprocess_train.csv"
    print("running...")
    macro_P, macro_R, macro_F1, LABEL0, LABEL1, LABEL2, LABEL3, LABEL4, LABEL5 = Rf_Classifier_Plus.classify(filepath)
    print("done")
    dict = {'macro_P': macro_P,'macro_R': macro_R, 'macro_F1': macro_F1,
            'LABEL0': LABEL0, 'LABEL1': LABEL1, 'LABEL2': LABEL2, 'LABEL3':LABEL3, 'LABEL4':LABEL4, 'LABEL5':LABEL5}
    # json_f = json.dumps(dict)
    return Respones_s.success(dict)


# # 预测结果/刷卡记录接口
# @app.route("/Fault_diagnosis_data/<stuno>/<page>", methods=['GET'])
# def findStuno(sample_id, page):
#     """
#     :param sample_id:
#     :return: 返回查询数据
#     """
#     dict_result = {}
#     list_t = []
#     list_result = []
#
#     # 查询某个样本的数据(ID)
#     try:
#         name = dataset.query.filter_by(ID=sample_id)[0].name
#     except Exception as e:
#         return Respones_s.fail('NO_FOUND')
#
#     data = dataset.query.filter_by(sample_id=sample_id).paginate(page=int(page), per_page=20, error_out=False)
#     current_page = data.page
#     total_page = data.pages
#     total_data = data.total
#     # print(data)
#     # print(type(data))
#     # print(data.items)
#     for i in data.items:
#         # list_tmp = [i.sample_id, i.stuno, i.location, i.time, i.consume, i.rank, name]
#         list_tmp = [i.sample_id, i.feature0, i.feature1, i.feature2, i.feature3, i.feature4, i.feature5, i.feature6, i.feature7, i.feature8, i.feature9, i.feature10, i.feature11, i.feature12, i.feature13, i.feature14, i.feature15, i.feature16, i.feature17, i.feature18, i.feature19, i.feature20, i.feature21, i.feature22, i.feature23, i.feature24, i.feature25, i.feature26, i.feature27, i.feature28, i.feature29, i.feature30, i.feature31, i.feature32, i.feature33, i.feature34, i.feature35, i.feature36, i.feature37, i.feature38, i.feature39, i.feature40, i.feature41, i.feature42, i.feature43, i.feature44, i.feature45, i.feature46, i.feature47, i.feature48, i.feature49, i.feature50, i.feature51, i.feature52, i.feature53, i.feature54, i.feature55, i.feature56, i.feature57, i.feature58, i.feature59, i.feature60, i.feature61, i.feature62, i.feature63, i.feature64, i.feature65, i.feature66, i.feature67, i.feature68, i.feature69, i.feature70, i.feature71, i.feature72, i.feature73, i.feature74, i.feature75, i.feature76, i.feature77, i.feature78, i.feature79, i.feature80, i.feature81, i.feature82, i.feature83, i.feature84, i.feature85, i.feature86, i.feature87, i.feature88, i.feature89, i.feature90, i.feature91, i.feature92, i.feature93, i.feature94, i.feature95, i.feature96, i.feature97, i.feature98, i.feature99, i.feature100, i.feature101, i.feature102, i.feature103, i.feature104, i.feature105, i.feature106, i.label]
#         list_t.append(list_tmp)
#     print(list_t)
#     # 将list数据转化成字典
#     for l in list_t:
#         list_dict = {}
#         for i in range(len(l)):
#             if i == 2:
#                 key = list_en[list_cn.index(l[i])]
#                 list_dict[list_name[i]] = key
#             else:
#                 list_dict[list_name[i]] = l[i]
#         list_result.append(list_dict)
#     print(list_result)
#     # dict_result['data'] = list_result
#
#     # json_f = json.dumps(dict_result)
#
#     return Respones_s.success(
#         {'list': list_result, 'current_page': current_page, 'total_page': total_page, 'total_data': total_data})



# 获取成绩列表接口
#获取数据列表接口
@app.route("/fault_diagnosis_data", methods=['GET'])
def score_list():
    """
    :return:返回学生成绩列表
    """
    list_t = []
    list_result = []

    sample_id = request.args.get('sample_id')#获取id参数
    label = request.args.get('label')
    page = request.args.get('page')

    # 查询成绩排名等级数据
    if sample_id is None and label is None:
        #数据库表dataset的数据
        data = dataset.query.paginate(page=int(page), per_page=20, error_out=False)
    else:
        if sample_id is not None:
            data = dataset.query.filter_by(sample_id=sample_id).paginate(page=int(page), per_page=20, error_out=False)
        elif label is not None:
            data = dataset.query.filter_by(label=label).paginate(page=int(page), per_page=20, error_out=False)
        else:
            data = dataset.query.filter_by(sample_id=sample_id,label=label).paginate(page=int(page), per_page=20, error_out=False)

    if data is None:
        return Respones_s.fail('NO_FOUND')
    current_page = data.page
    total_page = data.pages
    total_data = data.total

    for i in data.items:
        list_tmp = [i.sample_id, i.feature0, i.feature1, i.feature2, i.feature3, i.feature4, i.feature5, i.feature6, i.feature7, i.feature8, i.feature9, i.feature10, i.feature11, i.feature12, i.feature13, i.feature14, i.feature15, i.feature16, i.feature17, i.feature18, i.feature19, i.feature20, i.feature21, i.feature22, i.feature23, i.feature24, i.feature25, i.feature26, i.feature27, i.feature28, i.feature29, i.feature30, i.feature31, i.feature32, i.feature33, i.feature34, i.feature35, i.feature36, i.feature37, i.feature38, i.feature39, i.feature40, i.feature41, i.feature42, i.feature43, i.feature44, i.feature45, i.feature46, i.feature47, i.feature48, i.feature49, i.feature50, i.feature51, i.feature52, i.feature53, i.feature54, i.feature55, i.feature56, i.feature57, i.feature58, i.feature59, i.feature60, i.feature61, i.feature62, i.feature63, i.feature64, i.feature65, i.feature66, i.feature67, i.feature68, i.feature69, i.feature70, i.feature71, i.feature72, i.feature73, i.feature74, i.feature75, i.feature76, i.feature77, i.feature78, i.feature79, i.feature80, i.feature81, i.feature82, i.feature83, i.feature84, i.feature85, i.feature86, i.feature87, i.feature88, i.feature89, i.feature90, i.feature91, i.feature92, i.feature93, i.feature94, i.feature95, i.feature96, i.feature97, i.feature98, i.feature99, i.feature100, i.feature101, i.feature102, i.feature103, i.feature104, i.feature105, i.feature106, i.label]
        list_t.append(list_tmp)
        # list_t.append(list_tmp)

    # 将list数据转化成字典
    for l in list_t:
        list_dict = {}
        for i in range(len(l)):
            list_dict[list_fdd_data[i]] = l[i]
        list_result.append(list_dict)
    # dict_result['data'] = list_result
    # json_f = json.dumps(dict_result)
    # j = jsonify(dict_result)
    # print(json_f)
    return Respones_s.success(
        {'list': list_result, 'current_page': current_page, 'total_page': total_page, 'total_data': total_data})


# 获取成绩列表接口
# @app.route("/stu_score", methods=['GET'])
# def score_list():
#     """
#     :return:返回学生成绩列表
#     """
#     list_t = []
#     list_result = []
#
#     ID = request.args.get('ID')#获取id参数
#     name = request.args.get('name')
#     page = request.args.get('page')
#
#     # 查询成绩排名等级数据
#     if ID is None and name is None:
#         #数据库表score_rank的数据
#         data = Score_rank.query.paginate(page=int(page), per_page=20, error_out=False)
#     else:
#         if ID is not None:
#             data = Score_rank.query.filter_by(ID=ID).paginate(page=int(page), per_page=20, error_out=False)
#         elif name is not None:
#             data = Score_rank.query.filter_by(name=name).paginate(page=int(page), per_page=20, error_out=False)
#         else:
#             data = Score_rank.query.filter_by(ID=ID,name=name).paginate(page=int(page), per_page=20, error_out=False)
#
#     if data is None:
#         return Respones_s.fail('NO_FOUND')
#     current_page = data.page
#     total_page = data.pages
#     total_data = data.total
#
#     for i in data.items:
#         list_tmp = [i.ID, i.name, i.JAVA, i.Python, i.DS, i.OS, i.CS, i.CN, i.rank, i.notice, i.grade]
#         list_t.append(list_tmp)
#
#     # 将list数据转化成字典
#     for l in list_t:
#         list_dict = {}
#         for i in range(len(l)):
#             list_dict[list_score_rank[i]] = l[i]
#         list_result.append(list_dict)
#     # dict_result['data'] = list_result
#     # json_f = json.dumps(dict_result)
#     # j = jsonify(dict_result)
#     # print(json_f)
#     return Respones_s.success(
#         {'list': list_result, 'current_page': current_page, 'total_page': total_page, 'total_data': total_data})
#
#

# # 删除数据集接口
@app.route("/delDataset", methods=['POST',"GET"])
def delDataset():
    # 获取sample_id进行查询
    ID = request.get_json()['ID']
    db.session.query(dataset).filter_by(sample_id=ID).delete()
    db.session.commit()

    return Respones_s.success('SUCCESS')


if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(host='localhost', port=9999, debug=True)
