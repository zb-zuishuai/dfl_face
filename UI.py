# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'yymain2.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(750, 800)
        Form.setMinimumSize(QtCore.QSize(750, 800))
        Form.setMaximumSize(QtCore.QSize(750, 800))
        Form.setStyleSheet("QWidget {\n"
"    border: none;\n"
"    background-color: #F8F8FF;\n"
"}")
        self.tabwidget_1 = QtWidgets.QTabWidget(Form)
        self.tabwidget_1.setGeometry(QtCore.QRect(50, 20, 561, 441))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.tabwidget_1.setFont(font)
        self.tabwidget_1.setStyleSheet("QTabWidget::tab-bar {\n"
"    alignment: center;\n"
"}\n"
"\n"
"QTabWidget::pane {\n"
"    background-color: transparent;\n"
"}\n"
"\n"
"QTabBar::tab {\n"
"    \n"
"    background-color: #f1f1f1;\n"
"    border: 1px solid #dcdcdc;\n"
"    border-bottom-color: #c2c7cb;\n"
"    min-width: 80px;\n"
"    padding: 8px;\n"
"    font-weight: bold;\n"
"}\n"
"\n"
"\n"
"QTabBar::tab:selected {\n"
"    background-color: rgb(188, 255, 205);\n"
"    border-bottom-color: #a7b0b3;\n"
"}\n"
"\n"
"QTabBar::tab:!selected:hover {\n"
"    background-color: #eaeaea;\n"
"}\n"
"QTabBar::tab:pressed{\n"
"    border-left:1px solid black;\n"
"    border-top:1px solid black;\n"
"    border-right:none;\n"
"    border-bottom:none;;\n"
"}")
        self.tabwidget_1.setObjectName("tabwidget_1")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.groupBox_13 = QtWidgets.QGroupBox(self.tab_4)
        self.groupBox_13.setGeometry(QtCore.QRect(0, 10, 541, 411))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_13.setFont(font)
        self.groupBox_13.setStyleSheet("QGroupBox {\n"
"    background-color: transparent;\n"
"    border: none;\n"
"}\n"
"")
        self.groupBox_13.setCheckable(False)
        self.groupBox_13.setChecked(False)
        self.groupBox_13.setObjectName("groupBox_13")
        self.groupBox = QtWidgets.QGroupBox(self.groupBox_13)
        self.groupBox.setGeometry(QtCore.QRect(0, 60, 541, 171))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setFlat(False)
        self.groupBox.setCheckable(False)
        self.groupBox.setChecked(False)
        self.groupBox.setObjectName("groupBox")
        self.numworker_comboBox = QtWidgets.QComboBox(self.groupBox)
        self.numworker_comboBox.setGeometry(QtCore.QRect(430, 30, 87, 22))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.numworker_comboBox.setFont(font)
        self.numworker_comboBox.setStyleSheet("QComboBox {\n"
"      border: 2px solid gray;\n"
"      border-radius: 5px;\n"
"      padding: 0 8px;\n"
"      background: rgb(255, 170, 255);\n"
"      font: bold;\n"
"      selection-background-color: darkgray;\n"
"  }")
        self.numworker_comboBox.setObjectName("numworker_comboBox")
        self.numworker_comboBox.addItem("")
        self.numworker_comboBox.addItem("")
        self.label_20 = QtWidgets.QLabel(self.groupBox)
        self.label_20.setGeometry(QtCore.QRect(330, 30, 81, 28))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        self.label_20.setFont(font)
        self.label_20.setStyleSheet("")
        self.label_20.setObjectName("label_20")
        self.label_19 = QtWidgets.QLabel(self.groupBox)
        self.label_19.setGeometry(QtCore.QRect(40, 30, 131, 28))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_19.setFont(font)
        self.label_19.setStyleSheet("")
        self.label_19.setObjectName("label_19")
        self.need_debug_comboBox = QtWidgets.QComboBox(self.groupBox)
        self.need_debug_comboBox.setGeometry(QtCore.QRect(180, 30, 87, 22))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.need_debug_comboBox.setFont(font)
        self.need_debug_comboBox.setStyleSheet("QComboBox {\n"
"      border: 2px solid gray;\n"
"      border-radius: 5px;\n"
"      padding: 0 8px;\n"
"      background: rgb(255, 170, 255);\n"
"      font: bold;\n"
"      selection-background-color: darkgray;\n"
"  }")
        self.need_debug_comboBox.setObjectName("need_debug_comboBox")
        self.need_debug_comboBox.addItem("")
        self.need_debug_comboBox.addItem("")
        self.face_num_comboBox = QtWidgets.QComboBox(self.groupBox)
        self.face_num_comboBox.setGeometry(QtCore.QRect(430, 60, 87, 22))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.face_num_comboBox.setFont(font)
        self.face_num_comboBox.setStyleSheet("QComboBox {\n"
"      border: 2px solid gray;\n"
"      border-radius: 5px;\n"
"      padding: 0 8px;\n"
"      background: rgb(255, 170, 255);\n"
"      font: bold;\n"
"      selection-background-color: darkgray;\n"
"  }")
        self.face_num_comboBox.setObjectName("face_num_comboBox")
        self.face_num_comboBox.addItem("")
        self.face_num_comboBox.addItem("")
        self.face_num_comboBox.addItem("")
        self.face_num_comboBox.addItem("")
        self.face_num_comboBox.addItem("")
        self.face_num_comboBox.addItem("")
        self.face_style_comboBox = QtWidgets.QComboBox(self.groupBox)
        self.face_style_comboBox.setGeometry(QtCore.QRect(180, 60, 87, 22))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.face_style_comboBox.setFont(font)
        self.face_style_comboBox.setStyleSheet("QComboBox {\n"
"      border: 2px solid gray;\n"
"      border-radius: 5px;\n"
"      padding: 0 8px;\n"
"      background: rgb(255, 170, 255);\n"
"      font: bold;\n"
"      selection-background-color: darkgray;\n"
"  }")
        self.face_style_comboBox.setObjectName("face_style_comboBox")
        self.face_style_comboBox.addItem("")
        self.face_style_comboBox.addItem("")
        self.face_style_comboBox.addItem("")
        self.label_21 = QtWidgets.QLabel(self.groupBox)
        self.label_21.setGeometry(QtCore.QRect(40, 60, 91, 28))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        self.label_21.setFont(font)
        self.label_21.setStyleSheet("")
        self.label_21.setObjectName("label_21")
        self.label_22 = QtWidgets.QLabel(self.groupBox)
        self.label_22.setGeometry(QtCore.QRect(330, 60, 81, 28))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        self.label_22.setFont(font)
        self.label_22.setStyleSheet("")
        self.label_22.setObjectName("label_22")
        self.jpeg_quality_comboBox = QtWidgets.QComboBox(self.groupBox)
        self.jpeg_quality_comboBox.setGeometry(QtCore.QRect(430, 90, 87, 22))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.jpeg_quality_comboBox.setFont(font)
        self.jpeg_quality_comboBox.setStyleSheet("QComboBox {\n"
"      border: 2px solid gray;\n"
"      border-radius: 5px;\n"
"      padding: 0 8px;\n"
"      background: rgb(255, 170, 255);\n"
"      font: bold;\n"
"      selection-background-color: darkgray;\n"
"  }")
        self.jpeg_quality_comboBox.setObjectName("jpeg_quality_comboBox")
        self.jpeg_quality_comboBox.addItem("")
        self.jpeg_quality_comboBox.addItem("")
        self.jpeg_quality_comboBox.addItem("")
        self.label_23 = QtWidgets.QLabel(self.groupBox)
        self.label_23.setGeometry(QtCore.QRect(330, 90, 81, 28))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        self.label_23.setFont(font)
        self.label_23.setStyleSheet("")
        self.label_23.setObjectName("label_23")
        self.label_24 = QtWidgets.QLabel(self.groupBox)
        self.label_24.setGeometry(QtCore.QRect(40, 90, 91, 28))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        self.label_24.setFont(font)
        self.label_24.setStyleSheet("")
        self.label_24.setObjectName("label_24")
        self.image_size_comboBox = QtWidgets.QComboBox(self.groupBox)
        self.image_size_comboBox.setGeometry(QtCore.QRect(180, 90, 87, 22))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.image_size_comboBox.setFont(font)
        self.image_size_comboBox.setStyleSheet("QComboBox {\n"
"      border: 2px solid gray;\n"
"      border-radius: 5px;\n"
"      padding: 0 8px;\n"
"      background: rgb(255, 170, 255);\n"
"      font: bold;\n"
"      selection-background-color: darkgray;\n"
"  }")
        self.image_size_comboBox.setObjectName("image_size_comboBox")
        self.image_size_comboBox.addItem("")
        self.image_size_comboBox.addItem("")
        self.image_size_comboBox.addItem("")
        self.image_size_comboBox.addItem("")
        self.image_size_comboBox.addItem("")
        self.image_size_comboBox.addItem("")
        self.label_25 = QtWidgets.QLabel(self.groupBox)
        self.label_25.setGeometry(QtCore.QRect(40, 120, 121, 28))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        self.label_25.setFont(font)
        self.label_25.setStyleSheet("")
        self.label_25.setObjectName("label_25")
        self.det_size_comboBox = QtWidgets.QComboBox(self.groupBox)
        self.det_size_comboBox.setGeometry(QtCore.QRect(180, 120, 87, 22))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.det_size_comboBox.setFont(font)
        self.det_size_comboBox.setStyleSheet("QComboBox {\n"
"      border: 2px solid gray;\n"
"      border-radius: 5px;\n"
"      padding: 0 8px;\n"
"      background: rgb(255, 170, 255);\n"
"      font: bold;\n"
"      selection-background-color: darkgray;\n"
"  }")
        self.det_size_comboBox.setObjectName("det_size_comboBox")
        self.det_size_comboBox.addItem("")
        self.det_size_comboBox.addItem("")
        self.det_size_comboBox.addItem("")
        self.label_27 = QtWidgets.QLabel(self.groupBox)
        self.label_27.setGeometry(QtCore.QRect(330, 120, 81, 28))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_27.setFont(font)
        self.label_27.setStyleSheet("")
        self.label_27.setObjectName("label_27")
        self.detection_threshold_comboBox = QtWidgets.QComboBox(self.groupBox)
        self.detection_threshold_comboBox.setGeometry(QtCore.QRect(430, 120, 87, 22))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.detection_threshold_comboBox.setFont(font)
        self.detection_threshold_comboBox.setStyleSheet("QComboBox {\n"
"      border: 2px solid gray;\n"
"      border-radius: 5px;\n"
"      padding: 0 8px;\n"
"      background: rgb(255, 170, 255);\n"
"      font: bold;\n"
"      selection-background-color: darkgray;\n"
"  }")
        self.detection_threshold_comboBox.setObjectName("detection_threshold_comboBox")
        self.detection_threshold_comboBox.addItem("")
        self.detection_threshold_comboBox.addItem("")
        self.detection_threshold_comboBox.addItem("")
        self.checkBox = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox.setGeometry(QtCore.QRect(330, 150, 111, 19))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox.setFont(font)
        self.checkBox.setObjectName("checkBox")
        self.groupBox_14 = QtWidgets.QGroupBox(self.groupBox_13)
        self.groupBox_14.setGeometry(QtCore.QRect(0, 230, 541, 111))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_14.setFont(font)
        self.groupBox_14.setCheckable(False)
        self.groupBox_14.setChecked(False)
        self.groupBox_14.setObjectName("groupBox_14")
        self.layoutWidget_6 = QtWidgets.QWidget(self.groupBox_14)
        self.layoutWidget_6.setGeometry(QtCore.QRect(10, 50, 511, 71))
        self.layoutWidget_6.setObjectName("layoutWidget_6")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.layoutWidget_6)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_16 = QtWidgets.QLabel(self.layoutWidget_6)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        self.label_16.setFont(font)
        self.label_16.setStyleSheet("")
        self.label_16.setObjectName("label_16")
        self.horizontalLayout_7.addWidget(self.label_16)
        self.lineEdit_output_tab0 = QtWidgets.QLineEdit(self.layoutWidget_6)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.lineEdit_output_tab0.setFont(font)
        self.lineEdit_output_tab0.setStyleSheet("border: 2px solid gray;\n"
"border-radius: 10px;")
        self.lineEdit_output_tab0.setObjectName("lineEdit_output_tab0")
        self.horizontalLayout_7.addWidget(self.lineEdit_output_tab0)
        self.btn_output_path_tab0 = QtWidgets.QPushButton(self.layoutWidget_6)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btn_output_path_tab0.setFont(font)
        self.btn_output_path_tab0.setStyleSheet("QPushButton {\n"
"    background-color: rgb(188, 255, 205);\n"
"    border: 2px solid gray;\n"
"    border-radius: 5px;\n"
"    padding: 5px 10px;\n"
"    font-weight: bold;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: rgb(85, 170, 0);\n"
"}\n"
"\n"
"QPushButton:pressed{\n"
"    border-left:2px solid black;\n"
"    border-top:2px solid black;\n"
"    border-right:none;\n"
"    border-bottom:none;;\n"
"}")
        self.btn_output_path_tab0.setObjectName("btn_output_path_tab0")
        self.horizontalLayout_7.addWidget(self.btn_output_path_tab0)
        self.layoutWidget_7 = QtWidgets.QWidget(self.groupBox_14)
        self.layoutWidget_7.setGeometry(QtCore.QRect(10, 20, 511, 41))
        self.layoutWidget_7.setObjectName("layoutWidget_7")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.layoutWidget_7)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_17 = QtWidgets.QLabel(self.layoutWidget_7)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        self.label_17.setFont(font)
        self.label_17.setStyleSheet("")
        self.label_17.setObjectName("label_17")
        self.horizontalLayout_8.addWidget(self.label_17)
        self.lineEdit_input_tab0 = QtWidgets.QLineEdit(self.layoutWidget_7)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.lineEdit_input_tab0.setFont(font)
        self.lineEdit_input_tab0.setStyleSheet("border: 2px solid gray;\n"
"border-radius: 10px;")
        self.lineEdit_input_tab0.setText("")
        self.lineEdit_input_tab0.setObjectName("lineEdit_input_tab0")
        self.horizontalLayout_8.addWidget(self.lineEdit_input_tab0)
        self.btn_input_path_tab0 = QtWidgets.QPushButton(self.layoutWidget_7)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btn_input_path_tab0.setFont(font)
        self.btn_input_path_tab0.setStyleSheet("QPushButton {\n"
"    background-color: rgb(188, 255, 205);\n"
"    border: 2px solid gray;\n"
"    border-radius: 5px;\n"
"    padding: 5px 10px;\n"
"    font-weight: bold;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: rgb(85, 170, 0);\n"
"}\n"
"\n"
"QPushButton:pressed{\n"
"    border-left:2px solid black;\n"
"    border-top:2px solid black;\n"
"    border-right:none;\n"
"    border-bottom:none;;\n"
"}")
        self.btn_input_path_tab0.setAutoDefault(False)
        self.btn_input_path_tab0.setFlat(False)
        self.btn_input_path_tab0.setObjectName("btn_input_path_tab0")
        self.horizontalLayout_8.addWidget(self.btn_input_path_tab0)
        self.btn_start = QtWidgets.QPushButton(self.groupBox_13)
        self.btn_start.setGeometry(QtCore.QRect(179, 350, 161, 41))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btn_start.setFont(font)
        self.btn_start.setStyleSheet("QPushButton {\n"
"    background-color: rgb(188, 255, 205);\n"
"    border: 2px solid gray;\n"
"    border-radius: 5px;\n"
"    padding: 5px 10px;\n"
"    font-weight: bold;\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(85, 170, 0);\n"
"}\n"
"\n"
"QPushButton:pressed{\n"
"    border-left:2px solid black;\n"
"    border-top:2px solid black;\n"
"    border-right:none;\n"
"    border-bottom:none;;\n"
"}\n"
"")
        self.btn_start.setObjectName("btn_start")
        self.open_btn_output_path_tab0 = QtWidgets.QPushButton(self.groupBox_13)
        self.open_btn_output_path_tab0.setGeometry(QtCore.QRect(390, 350, 131, 41))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.open_btn_output_path_tab0.setFont(font)
        self.open_btn_output_path_tab0.setStyleSheet("QPushButton {\n"
"    background-color: rgb(188, 255, 205);\n"
"    border: 2px solid gray;\n"
"    border-radius: 5px;\n"
"    padding: 5px 10px;\n"
"    font-weight: bold;\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(85, 170, 0);\n"
"}\n"
"\n"
"QPushButton:pressed{\n"
"    border-left:2px solid black;\n"
"    border-top:2px solid black;\n"
"    border-right:none;\n"
"    border-bottom:none;;\n"
"}\n"
"font: 11pt \"幼圆\";")
        self.open_btn_output_path_tab0.setObjectName("open_btn_output_path_tab0")
        self.label_readme1 = QtWidgets.QLabel(self.groupBox_13)
        self.label_readme1.setGeometry(QtCore.QRect(20, 20, 541, 51))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_readme1.setFont(font)
        self.label_readme1.setStyleSheet("\n"
"color: rgb(255, 0, 255);\n"
"")
        self.label_readme1.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignHCenter)
        self.label_readme1.setObjectName("label_readme1")
        self.label_readme1.raise_()
        self.groupBox.raise_()
        self.groupBox_14.raise_()
        self.btn_start.raise_()
        self.open_btn_output_path_tab0.raise_()
        self.tabwidget_1.addTab(self.tab_4, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_2.setGeometry(QtCore.QRect(0, 10, 541, 371))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setStyleSheet("QGroupBox {\n"
"    background-color: transparent;\n"
"    border: none;\n"
"}")
        self.groupBox_2.setObjectName("groupBox_2")
        self.groupBox_15 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_15.setGeometry(QtCore.QRect(0, 150, 531, 101))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_15.setFont(font)
        self.groupBox_15.setObjectName("groupBox_15")
        self.layoutWidget_8 = QtWidgets.QWidget(self.groupBox_15)
        self.layoutWidget_8.setGeometry(QtCore.QRect(10, 50, 511, 51))
        self.layoutWidget_8.setObjectName("layoutWidget_8")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.layoutWidget_8)
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_18 = QtWidgets.QLabel(self.layoutWidget_8)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        self.label_18.setFont(font)
        self.label_18.setStyleSheet("")
        self.label_18.setObjectName("label_18")
        self.horizontalLayout_9.addWidget(self.label_18)
        self.lineEdit_output_tab1 = QtWidgets.QLineEdit(self.layoutWidget_8)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.lineEdit_output_tab1.setFont(font)
        self.lineEdit_output_tab1.setStyleSheet("border: 2px solid gray;\n"
"border-radius: 10px;")
        self.lineEdit_output_tab1.setObjectName("lineEdit_output_tab1")
        self.horizontalLayout_9.addWidget(self.lineEdit_output_tab1)
        self.btn_output_path_tab1 = QtWidgets.QPushButton(self.layoutWidget_8)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btn_output_path_tab1.setFont(font)
        self.btn_output_path_tab1.setStyleSheet("QPushButton {\n"
"    background-color: rgb(188, 255, 205);\n"
"    border: 2px solid gray;\n"
"    border-radius: 5px;\n"
"    padding: 5px 10px;\n"
"    font-weight: bold;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: rgb(85, 170, 0);\n"
"}\n"
"\n"
"QPushButton:pressed{\n"
"    border-left:2px solid black;\n"
"    border-top:2px solid black;\n"
"    border-right:none;\n"
"    border-bottom:none;;\n"
"}")
        self.btn_output_path_tab1.setObjectName("btn_output_path_tab1")
        self.horizontalLayout_9.addWidget(self.btn_output_path_tab1)
        self.layoutWidget_9 = QtWidgets.QWidget(self.groupBox_15)
        self.layoutWidget_9.setGeometry(QtCore.QRect(10, 20, 511, 33))
        self.layoutWidget_9.setObjectName("layoutWidget_9")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.layoutWidget_9)
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_26 = QtWidgets.QLabel(self.layoutWidget_9)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        self.label_26.setFont(font)
        self.label_26.setStyleSheet("")
        self.label_26.setObjectName("label_26")
        self.horizontalLayout_10.addWidget(self.label_26)
        self.lineEdit_input_tab1 = QtWidgets.QLineEdit(self.layoutWidget_9)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.lineEdit_input_tab1.setFont(font)
        self.lineEdit_input_tab1.setStyleSheet("border: 2px solid gray;\n"
"border-radius: 10px;")
        self.lineEdit_input_tab1.setText("")
        self.lineEdit_input_tab1.setObjectName("lineEdit_input_tab1")
        self.horizontalLayout_10.addWidget(self.lineEdit_input_tab1)
        self.btn_input_path_tab1 = QtWidgets.QPushButton(self.layoutWidget_9)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btn_input_path_tab1.setFont(font)
        self.btn_input_path_tab1.setStyleSheet("QPushButton {\n"
"    background-color: rgb(188, 255, 205);\n"
"    border: 2px solid gray;\n"
"    border-radius: 5px;\n"
"    padding: 5px 10px;\n"
"    font-weight: bold;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: rgb(85, 170, 0);\n"
"}\n"
"\n"
"\n"
"QPushButton:pressed{\n"
"    border-left:2px solid black;\n"
"    border-top:2px solid black;\n"
"    border-right:none;\n"
"    border-bottom:none;;\n"
"}")
        self.btn_input_path_tab1.setObjectName("btn_input_path_tab1")
        self.horizontalLayout_10.addWidget(self.btn_input_path_tab1)
        self.btn_start_1 = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_start_1.setGeometry(QtCore.QRect(200, 260, 161, 41))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btn_start_1.setFont(font)
        self.btn_start_1.setStyleSheet("QPushButton {\n"
"    background-color: rgb(188, 255, 205);\n"
"    border: 2px solid gray;\n"
"    border-radius: 5px;\n"
"    padding: 5px 10px;\n"
"    font-weight: bold;\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(85, 170, 0);\n"
"}\n"
"\n"
"QPushButton:pressed{\n"
"    border-left:2px solid black;\n"
"    border-top:2px solid black;\n"
"    border-right:none;\n"
"    border-bottom:none;;\n"
"}")
        self.btn_start_1.setObjectName("btn_start_1")
        self.open_btn_output_path_tab1 = QtWidgets.QPushButton(self.groupBox_2)
        self.open_btn_output_path_tab1.setGeometry(QtCore.QRect(400, 260, 131, 41))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.open_btn_output_path_tab1.setFont(font)
        self.open_btn_output_path_tab1.setStyleSheet("QPushButton {\n"
"    background-color: rgb(188, 255, 205);\n"
"    border: 2px solid gray;\n"
"    border-radius: 5px;\n"
"    padding: 5px 10px;\n"
"    font-weight: bold;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: rgb(85, 170, 0);\n"
"}\n"
"\n"
"QPushButton:pressed{\n"
"    border-left:2px solid black;\n"
"    border-top:2px solid black;\n"
"    border-right:none;\n"
"    border-bottom:none;;\n"
"}")
        self.open_btn_output_path_tab1.setObjectName("open_btn_output_path_tab1")
        self.label_readme2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_readme2.setGeometry(QtCore.QRect(10, 40, 561, 61))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_readme2.setFont(font)
        self.label_readme2.setStyleSheet("\n"
"color: rgb(255, 0, 255);")
        self.label_readme2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_readme2.setObjectName("label_readme2")
        self.tabwidget_1.addTab(self.tab, "")
        self.groupBox_4 = QtWidgets.QGroupBox(Form)
        self.groupBox_4.setGeometry(QtCore.QRect(50, 730, 551, 51))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setObjectName("groupBox_4")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.groupBox_4)
        self.textBrowser_2.setGeometry(QtCore.QRect(10, 20, 531, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.textBrowser_2.setFont(font)
        self.textBrowser_2.setStyleSheet("background-color: rgba(255, 255, 255, 0);\n"
"border-color: rgba(255, 255, 255, 0);")
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.groupBox_5 = QtWidgets.QGroupBox(Form)
        self.groupBox_5.setGeometry(QtCore.QRect(50, 470, 661, 261))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setObjectName("groupBox_5")
        self.infoBox = QtWidgets.QTextBrowser(self.groupBox_5)
        self.infoBox.setGeometry(QtCore.QRect(10, 20, 631, 231))
        self.infoBox.setStyleSheet("border: 2px solid gray;\n"
"border-radius: 10px;\n"
"background-color: transparent;\n"
"\n"
"")
        self.infoBox.setObjectName("infoBox")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(630, 100, 41, 361))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(22)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setStyleSheet("")
        self.label.setObjectName("label")

        self.retranslateUi(Form)
        self.tabwidget_1.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "快速切脸V2（作者：●—●，论坛号：fanda）"))
        self.groupBox_13.setTitle(_translate("Form", "补充说明"))
        self.groupBox.setTitle(_translate("Form", "可选选项（默认也行）"))
        self.numworker_comboBox.setItemText(0, _translate("Form", "1"))
        self.numworker_comboBox.setItemText(1, _translate("Form", "2"))
        self.label_20.setText(_translate("Form", "<html><head/><body><p><span style=\" font-weight:600;\">线程数量</span></p></body></html>"))
        self.label_19.setText(_translate("Form", "<html><head/><body><p><span style=\" font-weight:600;\">保留debug图？</span></p></body></html>"))
        self.need_debug_comboBox.setItemText(0, _translate("Form", "no"))
        self.need_debug_comboBox.setItemText(1, _translate("Form", "yes"))
        self.face_num_comboBox.setItemText(0, _translate("Form", "6"))
        self.face_num_comboBox.setItemText(1, _translate("Form", "5"))
        self.face_num_comboBox.setItemText(2, _translate("Form", "4"))
        self.face_num_comboBox.setItemText(3, _translate("Form", "3"))
        self.face_num_comboBox.setItemText(4, _translate("Form", "2"))
        self.face_num_comboBox.setItemText(5, _translate("Form", "1"))
        self.face_style_comboBox.setItemText(0, _translate("Form", "wf"))
        self.face_style_comboBox.setItemText(1, _translate("Form", "head"))
        self.face_style_comboBox.setItemText(2, _translate("Form", "f"))
        self.label_21.setText(_translate("Form", "<html><head/><body><p><span style=\" font-weight:600;\">脸型选择</span></p></body></html>"))
        self.label_22.setText(_translate("Form", "<html><head/><body><p><span style=\" font-weight:600;\">人数上限</span></p></body></html>"))
        self.jpeg_quality_comboBox.setItemText(0, _translate("Form", "100"))
        self.jpeg_quality_comboBox.setItemText(1, _translate("Form", "95"))
        self.jpeg_quality_comboBox.setItemText(2, _translate("Form", "90"))
        self.label_23.setText(_translate("Form", "<html><head/><body><p><span style=\" font-weight:600;\">图片质量</span></p></body></html>"))
        self.label_24.setText(_translate("Form", "<html><head/><body><p><span style=\" font-weight:600;\">图片大小</span></p></body></html>"))
        self.image_size_comboBox.setItemText(0, _translate("Form", "512"))
        self.image_size_comboBox.setItemText(1, _translate("Form", "256"))
        self.image_size_comboBox.setItemText(2, _translate("Form", "384"))
        self.image_size_comboBox.setItemText(3, _translate("Form", "416"))
        self.image_size_comboBox.setItemText(4, _translate("Form", "1024"))
        self.image_size_comboBox.setItemText(5, _translate("Form", "768"))
        self.label_25.setText(_translate("Form", "<html><head/><body><p><span style=\" font-weight:600;\">人脸检测范围</span></p></body></html>"))
        self.det_size_comboBox.setItemText(0, _translate("Form", "640"))
        self.det_size_comboBox.setItemText(1, _translate("Form", "256"))
        self.det_size_comboBox.setItemText(2, _translate("Form", "320"))
        self.label_27.setText(_translate("Form", "<html><head/><body><p>检测阈值</p></body></html>"))
        self.detection_threshold_comboBox.setItemText(0, _translate("Form", "0.6"))
        self.detection_threshold_comboBox.setItemText(1, _translate("Form", "0.7"))
        self.detection_threshold_comboBox.setItemText(2, _translate("Form", "0.8"))
        self.checkBox.setText(_translate("Form", "开启GPU"))
        self.groupBox_14.setTitle(_translate("Form", "必选选项"))
        self.label_16.setText(_translate("Form", "<html><head/><body><p><span style=\" font-weight:600;\">结果保存文件夹：</span></p></body></html>"))
        self.lineEdit_output_tab0.setPlaceholderText(_translate("Form", "选择一个保存结果的文件夹"))
        self.btn_output_path_tab0.setText(_translate("Form", "保存文件夹"))
        self.label_17.setText(_translate("Form", "<html><head/><body><p><span style=\" font-weight:600;\">待处理输入路径：</span></p></body></html>"))
        self.lineEdit_input_tab0.setPlaceholderText(_translate("Form", "选择数据集文件"))
        self.btn_input_path_tab0.setText(_translate("Form", "输入文件夹"))
        self.btn_start.setText(_translate("Form", "开始处理"))
        self.open_btn_output_path_tab0.setText(_translate("Form", "打开文件夹"))
        self.label_readme1.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600; color:#ff00ff;\">不要出现中文图片名字，不要出现中文目录</span></p><p><br/></p></body></html>"))
        self.tabwidget_1.setTabText(self.tabwidget_1.indexOf(self.tab_4), _translate("Form", "切脸功能"))
        self.groupBox_2.setTitle(_translate("Form", "补充说明"))
        self.groupBox_15.setTitle(_translate("Form", "必选选项"))
        self.label_18.setText(_translate("Form", "<html><head/><body><p><span style=\" font-weight:600;\">aligned 的文件夹：</span></p></body></html>"))
        self.lineEdit_output_tab1.setPlaceholderText(_translate("Form", "选择一个保存结果的文件夹"))
        self.btn_output_path_tab1.setText(_translate("Form", "人脸文件夹"))
        self.label_26.setText(_translate("Form", "<html><head/><body><p><span style=\" font-weight:600;\">原图的文件夹路径：</span></p></body></html>"))
        self.lineEdit_input_tab1.setPlaceholderText(_translate("Form", "选择视频文件"))
        self.btn_input_path_tab1.setText(_translate("Form", "原图文件夹"))
        self.btn_start_1.setText(_translate("Form", "开始处理"))
        self.open_btn_output_path_tab1.setText(_translate("Form", "进入文件夹"))
        self.label_readme2.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">对比data_dst中的原图与aligned中的人脸数据。</span></p><p align=\"center\"><span style=\" font-weight:600;\">(1)找出未切脸的原图;(2)复制到error文件夹中。</span></p></body></html>"))
        self.tabwidget_1.setTabText(self.tabwidget_1.indexOf(self.tab), _translate("Form", "检查功能"))
        self.groupBox_4.setTitle(_translate("Form", "算法"))
        self.textBrowser_2.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'黑体\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:8pt;\">人脸检测算法：</span><span style=\" font-family:\'SimSun\'; font-size:9pt;\">https://github.com/deepinsight/insightface</span></p></body></html>"))
        self.groupBox_5.setTitle(_translate("Form", "运行日志"))
        self.label.setText(_translate("Form", "本\n"
"软\n"
"件\n"
"免\n"
"费\n"
"勿\n"
"上\n"
"当"))