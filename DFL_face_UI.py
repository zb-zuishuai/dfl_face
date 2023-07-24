# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget, QMainWindow
from PyQt5.QtGui import QIcon, QFont, QFontDatabase
from PyQt5.QtCore import pyqtSignal, QThread
import os
import shutil
import time
from os.path import join, dirname
from DFLJPG import *
from UI import Ui_Form
from pathlib import Path
import tqdm
from LandmarksProcessor import *
from FaceType import *
from insightface.app import FaceAnalysis
import concurrent.futures


class MySignals(QThread):
    text_print = pyqtSignal(str)


class ExtractThread(QThread):
    update_signal = pyqtSignal(str)

    def __init__(self, arg_dict):
        super().__init__()
        self.num = 1
        self.original_folder = arg_dict['original_folder']  # 原图文件夹
        self.aligned_folder = arg_dict['aligned_folder']  # 保存人脸文件夹
        self.face_style = {'wf': FaceType.WHOLE_FACE,
                           'head': FaceType.HEAD,
                           'f': FaceType.FULL}[arg_dict['face_style']]
        self.jpeg_quality = int(arg_dict['jpeg_quality'])
        self.face_num = int(arg_dict['face_num'])
        self.need_debug = arg_dict['need_debug']
        self.max_worker = int(arg_dict['max_worker'])
        self.image_size = int(arg_dict['image_size'])
        self.det_size = int(arg_dict['det_size'])
        self.text = []
        if arg_dict['gpu']:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        self.faceAnalysis = FaceAnalysis(name='buffalo_l',
                                         providers=providers,
                                         allowed_modules=['detection', 'landmark_2d_106'])
        self.faceAnalysis.prepare(
            ctx_id=0, det_thresh=float(arg_dict['detection_threshold']),
            det_size=(self.det_size, self.det_size))  # ctx_id=0表示选显卡0

        os.makedirs(self.aligned_folder, exist_ok=True)

    # pyqt5用cv2很难保存，需要改写
    @staticmethod
    def cv2_imwrite(filename, img, *args):
        ret, buf = cv2.imencode(Path(filename).suffix, img, *args)
        if ret:
            try:
                with open(filename, "wb") as stream:
                    stream.write(buf)
            except:
                pass

    # 转68特征点
    @staticmethod
    def landmark106to68(pt106):
        landmark106to68 = [1, 10, 12, 14, 16, 3, 5, 7, 0, 23, 21, 19, 32, 30, 28, 26, 17,  # 脸颊17点
                           43, 48, 49, 51, 50,  # 左眉毛5点
                           102, 103, 104, 105, 101,  # 右眉毛5点
                           72, 73, 74, 86, 78, 79, 80, 85, 84,  # 鼻子9点
                           35, 41, 42, 39, 37, 36,  # 左眼睛6点
                           89, 95, 96, 93, 91, 90,  # 右眼睛6点
                           52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55, 65, 66, 62, 70, 69, 57, 60, 54  # 嘴巴20点
                           ]
        pt68 = []
        if len(pt106) != 106:
            return None
        for i in range(68):
            index = landmark106to68[i]
            pt68.append(pt106[index])
        return pt68

    def extract_to_dfl_img(self, img_path, img_name, aligned_path):
        img_mat = cv2.imread(img_path)
        heigh, width, _ = img_mat.shape  # (1920, 1080, 3)
        debug_image = None
        if self.need_debug == 'yes':
            debug_image = img_mat.copy()  # debug图
            os.makedirs(self.original_folder + '/aligned_debug', exist_ok=True)

        faces = self.faceAnalysis.get(img_mat)
        if faces is None or len(faces) == 0:  # 检测是否存在脸
            return None

        img_name = img_name.replace("png", "jpg").replace("PNG", "jpg")  # 后缀变动
        i = 0
        for face in faces:
            # print(face.bbox)#xmin,ymin,xmax,ymax = box
            face_x, face_y = face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]

            image_size = self.image_size
            landmark = self.landmark106to68(face.landmark_2d_106)
            image_to_face_mat = get_transform_mat(
                landmark, image_size, self.face_style)  # 脸型,变换矩阵
            # scale_x, scale_y = 0.5, 0.5  # 7-20添加缩放变换
            # image_to_face_mat = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])  # 7-20添加缩放变换

            face_image = cv2.warpAffine(
                img_mat, image_to_face_mat, (image_size, image_size), cv2.INTER_LANCZOS4)

            output_filepath = aligned_path + "/" + img_name.split('.')[0] + '_' + str(i) + '.jpg'  # 多人

            self.cv2_imwrite(output_filepath, face_image, [int(
                cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])  # 质量

            if self.need_debug == 'yes':
                output_debug_path = self.original_folder + '/aligned_debug/' + img_name  # debug保存路径
                rect = face.bbox.astype(np.int32)  # 矩形框坐标
                draw_rect_landmarks(
                    debug_image, rect, landmark, self.face_style, image_size, transparent_mask=True)  # 画人脸框
                self.cv2_imwrite(output_debug_path, debug_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])  # 保存debug图

            dflimg = DFLJPG.load(output_filepath)
            dflimg.set_face_type(FaceType.toString(self.face_style))
            face_image_landmarks = transform_points(landmark, image_to_face_mat)
            # 相似变换之后的人脸关键点
            dflimg.set_landmarks(face_image_landmarks.tolist())
            dflimg.set_source_filename(img_name)
            dflimg.set_source_rect(face.bbox)

            # 源人脸关键点
            dflimg.set_source_landmarks(landmark)
            dflimg.set_image_to_face_mat(image_to_face_mat)
            dflimg.save()
            i += 1
            if i > int(self.face_num):
                break

    def run(self):
        dir_path = self.original_folder
        aligned_path = self.aligned_folder
        start_time = time.time()

        # 扫描图片文件名
        image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            # 提交任务并获取对应的Future对象列表
            futures = [executor.submit(self.extract_to_dfl_img, dir_path + '/' + img_name, img_name, aligned_path)
                       for img_name in image_files]

            progress_bar = tqdm.tqdm(concurrent.futures.as_completed(futures),
                                     desc="人脸提取进度", leave=True, ascii=True, total=len(futures))
            # 处理每个已完成的任务
            for future in progress_bar:
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()

        # 打印总体运行时间
        end_time = time.time()
        processing_time = end_time - start_time
        print("Processing completed in {:.2f} seconds.".format(processing_time))
        self.update_signal.emit("人脸提取完毕！！！     耗时{:.2f}秒".format(processing_time))
        self.update_signal.emit("=" * 50)

    def run2(self):  # 单线程
        dir_path = self.original_folder
        aligned_path = self.aligned_folder
        start_time = time.time()
        total_images = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        with tqdm.tqdm(total=len(total_images), desc="人脸提取进度", leave=True, initial=0, ascii=True) as pbar:
            for img_name in total_images:
                try:
                    self.extract_to_dfl_img(dir_path + "/" + img_name, img_name, aligned_path)
                    pbar.update(1)
                except Exception as e:
                    traceback.print_exc()

        end_time = time.time()
        processing_time = end_time - start_time
        print("Processing completed in {:.2f} seconds.".format(processing_time))
        self.update_signal.emit("人脸提取完毕！！！     耗时{:.2f}秒".format(processing_time))
        self.update_signal.emit("=" * 50)


class CheckThread(QThread):  # 定位人脸并提取人脸边缘带
    update_signal = pyqtSignal(str)

    def __init__(self, data_src_dir, aligned_dir):  # 传参
        super().__init__()
        self.ori_path = data_src_dir
        self.face_path = aligned_dir

    def run(self):  # 纠错用的
        ori_path = self.ori_path
        face_path = self.face_path
        if not os.path.exists(ori_path) or not os.path.exists(face_path):
            self.update_signal.emit("请确认文件夹是否都存在")
        else:
            now_time = time.strftime('%H-%M-%S', time.localtime(time.time()))
            error_path = os.path.join(ori_path, 'error' + '-' + now_time)
            os.makedirs(error_path, exist_ok=True)
            self.update_signal.emit("创建error文件夹: " + error_path)
            error = []
            # face_dir = [img.split('_')[0] for img in os.listdir(face_path)]  # aligned全部文件名字
            face_dir = [DFLJPG.load(os.path.join(face_path, img)).get_dict()['source_filename'].split('.')[0] for img in
                        os.listdir(face_path)]

            for img in os.listdir(ori_path):  # 原图文件夹下的图片,要确保是图片
                if os.path.isfile(os.path.join(ori_path, img)):  # 判断是否是文件
                    if any([img.lower().endswith(ext) for ext in ['jpg', 'png', 'jpeg']]):  # [True,False,False]
                        name = img.split('.')[0]  # 图片名字
                        print(name)
                        if name not in face_dir:  # 如果原图片没有aligned下的人脸图
                            error.append(os.path.join(ori_path, img))
                            shutil.copy(os.path.join(ori_path, img), os.path.join(error_path, img))
            self.update_signal.emit("显示出错的图的路径：")
            for i in error:
                self.update_signal.emit(i)
            self.update_signal.emit("所有没切到脸的原图保存在下面的文件夹里：")
            self.update_signal.emit(error_path)
            self.update_signal.emit("处理完毕！！！")


def set_font_recursive(widget, font):
    # 设置当前部件的字体
    widget.setFont(font)

    # 如果当前部件是容器部件（如窗口、布局等），则递归设置其子部件的字体
    if isinstance(widget, QWidget):
        for child_widget in widget.findChildren(QWidget):
            set_font_recursive(child_widget, font)


class Stats(QMainWindow, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        font_id = QFontDatabase.addApplicationFont(r'字体\d.ttf')  # 替换自己的字体，路径需更换
        font_families = QFontDatabase.applicationFontFamilies(font_id)
        font_family = font_families[0] if font_families else '黑体'
        font = QFont(font_family, 12)
        font.setBold(True)  # 添加加粗效果
        set_font_recursive(self, font)  # 递归更改字体
        self.label.setFont(QFont(font_family, 24))  # 定制改变
        self.label_readme1.setFont(QFont(font_family, 14))
        self.label_readme2.setFont(QFont(font_family, 14))
        self.thread2 = None
        self.thread1 = None

        self.btn_input_path_tab0.clicked.connect(lambda: self.select_input_path(0))
        self.btn_output_path_tab0.clicked.connect(lambda: self.select_output_path(0))

        self.btn_input_path_tab1.clicked.connect(lambda: self.select_input_path(1))
        self.btn_output_path_tab1.clicked.connect(lambda: self.select_output_path(1))

        self.btn_start.clicked.connect(self.extract_run)
        self.open_btn_output_path_tab0.clicked.connect(self.open_save_dir)

        self.btn_start_1.clicked.connect(self.check_run)
        self.open_btn_output_path_tab1.clicked.connect(self.open_aligned_dir)

        self.ms = MySignals()
        self.ms.text_print.connect(self.printToGui)

        self.data_dir = ''
        self.save_dir = ''

        self.data_src_dir = ''
        self.aligned_dir = ''

    def open_save_dir(self):
        if os.path.exists(self.save_dir):
            try:
                os.startfile(self.save_dir)
            except FileNotFoundError:
                self.ms.text_print.emit('文件夹不存在!!!')
        else:
            self.ms.text_print.emit('文件夹不存在!!!')

    def open_aligned_dir(self):
        if os.path.exists(self.data_src_dir):
            try:
                os.startfile(self.data_src_dir)
            except FileNotFoundError:
                self.ms.text_print.emit('aligned文件夹不存在!!!')
        else:
            self.ms.text_print.emit('aligned文件夹不存在!!!')

    def printToGui(self, text):  # 输出日志
        self.infoBox.append(str(text))  # 添加
        self.infoBox.ensureCursorVisible()  # 光标

    def select_input_path(self, tab_index):  # 输入文件夹
        if tab_index == 0:
            input_path = QFileDialog.getExistingDirectory(self, "选择图片目录", "test_img")
            self.lineEdit_input_tab0.setText(input_path)
            self.ms.text_print.emit(f"输入路径为：%s" % input_path)
            self.data_dir = input_path
            os.makedirs(join(input_path, 'aligned'), exist_ok=True)
        elif tab_index == 1:
            input_path = QFileDialog.getExistingDirectory(self, "选择data_src/data_dst目录", "test_img")
            self.lineEdit_input_tab1.setText(input_path)
            self.ms.text_print.emit(f"输入路径为：%s" % input_path)
            self.data_src_dir = input_path

    def select_output_path(self, tab_index):  # 输出文件夹
        if tab_index == 0:
            output_path = QFileDialog.getExistingDirectory(self, "选择保存目录", dirname(self.data_dir))
            self.lineEdit_output_tab0.setText(output_path)
            self.ms.text_print.emit(f"输出路径为：%s" % output_path)
            self.save_dir = output_path
        elif tab_index == 1:
            output_path = QFileDialog.getExistingDirectory(self, "选择aligned目录", dirname(self.data_src_dir))
            self.lineEdit_output_tab1.setText(output_path)
            self.ms.text_print.emit(f"输出路径为：%s" % output_path)
            self.aligned_dir = output_path

    def update_infobox_ui(self, text, status_code):
        if status_code != 0:  # 0为进程结束的标记，1为进程未结束
            if text.strip() != "":
                self.ms.text_print.emit(text.replace("\n", ""))
        else:
            self.ms.text_print.emit("处理完成!!!")
            self.btn_start.setEnabled(True)
            self.btn_start.setText("开始处理")
            self.btn_start.setStyleSheet(
                '''QPushButton{background-color: rgb(188, 255, 205);border: 2px solid gray;border-radius: 5px;padding: 5px 10px;}
                QPushButton:hover {background-color: rgb(85, 170, 0);}''')

    def extract_run(self):
        if self.data_dir == '' or self.save_dir == '':
            self.ms.text_print.emit("请先选好原图文件夹和保存人脸文件夹")
        else:
            self.btn_start.setEnabled(False)
            self.btn_start.setText("正在运行中..")
            self.btn_start.setStyleSheet(
                '''QPushButton{background-color: #ffb7e9;border: 2px solid gray;border-radius: 5px;padding: 5px 10px;}
                QPushButton:hover {background-color: rgb(85, 170, 0);}''')

            need_debug_comboBox_choose = self.need_debug_comboBox.currentText()
            numworker_comboBox_choose = self.numworker_comboBox.currentText()
            face_style_comboBox_choose = self.face_style_comboBox.currentText()
            face_num_comboBox_choose = self.face_num_comboBox.currentText()
            jpeg_quality_comboBox_choose = self.jpeg_quality_comboBox.currentText()
            image_size_comboBox_choose = self.image_size_comboBox.currentText()
            det_size_comboBox_choose = self.det_size_comboBox.currentText()
            detection_threshold_comboBox_choose = self.detection_threshold_comboBox.currentText()
            gpu_choose = self.checkBox.isChecked()

            self.ms.text_print.emit('需要debug图吗:' + need_debug_comboBox_choose)
            self.ms.text_print.emit('线程数量选择:' + numworker_comboBox_choose)
            self.ms.text_print.emit('选择脸型：' + face_style_comboBox_choose)
            self.ms.text_print.emit('图片最大人数：' + face_num_comboBox_choose)
            self.ms.text_print.emit('图片质量：' + jpeg_quality_comboBox_choose)
            self.ms.text_print.emit('图片大小：' + image_size_comboBox_choose)
            self.ms.text_print.emit('人脸检测范围大小：' + det_size_comboBox_choose)
            self.ms.text_print.emit('人脸检测阈值：' + detection_threshold_comboBox_choose)
            self.ms.text_print.emit('GPU开启状态：' + str(gpu_choose) + '\n')
            agr_dict_ = {'original_folder': self.data_dir,
                         'aligned_folder': self.save_dir,
                         'face_style': face_style_comboBox_choose,
                         'jpeg_quality': jpeg_quality_comboBox_choose,
                         'face_num': face_num_comboBox_choose,
                         'need_debug': need_debug_comboBox_choose,
                         'max_worker': numworker_comboBox_choose,
                         'image_size': image_size_comboBox_choose,
                         'det_size': det_size_comboBox_choose,
                         'detection_threshold': detection_threshold_comboBox_choose,
                         'gpu': gpu_choose
                         }
            self.ms.text_print.emit(f"开始处理任务ing...")
            self.thread1 = ExtractThread(arg_dict=agr_dict_)
            self.thread1.update_signal.connect(self.printToGui)
            self.thread1.finished.connect(self.extract_finish)
            self.thread1.start()

    def extract_finish(self):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("开始处理")
        self.btn_start.setStyleSheet(
            '''QPushButton{background-color: rgb(188, 255, 205);border: 2px solid gray;border-radius: 5px;padding: 5px 10px;}
            QPushButton:hover {background-color: rgb(85, 170, 0);}''')

    def check_run(self):  # 纠错用的
        self.btn_start_1.setEnabled(False)
        self.btn_start_1.setText("正在检查中..")
        self.btn_start_1.setStyleSheet(
            '''QPushButton{background-color: #ffb7e9;border: 2px solid gray;border-radius: 5px;padding: 5px 10px;}
            QPushButton:hover {background-color: rgb(85, 170, 0);}''')
        self.thread2 = CheckThread(self.data_src_dir, self.aligned_dir)
        self.thread2.update_signal.connect(self.printToGui)
        self.thread2.finished.connect(self.check_finish)
        self.thread2.start()

    def check_finish(self):
        self.btn_start_1.setEnabled(True)
        self.btn_start_1.setText("开始处理")
        self.btn_start_1.setStyleSheet(
            '''QPushButton{background-color: rgb(188, 255, 205);border: 2px solid gray;border-radius: 5px;padding: 5px 10px;}
            QPushButton:hover {background-color: rgb(85, 170, 0);}''')


if __name__ == "__main__":
    app = QApplication([])
    app.setWindowIcon(QIcon('logo.jpg'))  # 加载logo
    stats = Stats()
    stats.show()
    app.exec_()
