# from PyQt5.QtWidgets import QApplication, QLabel
# from PyQt5.QtGui import QFont, QFontDatabase
#
# if __name__ == '__main__':
#     app = QApplication([])
#
#     # 加字体文件
#     font_id = QFontDatabase.addApplicationFont('a.ttf')
#
#     # 获取字体名称
#     font_families = QFontDatabase.applicationFontFamilies(font_id)
#     font_family = font_families[0] if font_families else 'Arial'
#
#     # 创建字体对象
#     custom_font = QFont(font_family, 12)
#
#     # 应用字体到控件
#     label = QLabel('快速切脸无敌')
#     label.setFont(custom_font)
#     label.show()
#
#     app.exec_()

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton
from PyQt5.QtGui import QFont, QFontDatabase


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.button = None
        self.label = None
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('PyQt5 Example')

        self.label = QLabel(self)
        self.label.setGeometry(100, 50, 200, 30)
        self.label.setText("显示")
        font_id = QFontDatabase.addApplicationFont(r'字体\a.ttf')  # 替换自己的字体，路径需更换
        font_families = QFontDatabase.applicationFontFamilies(font_id)
        font_family = font_families[0] if font_families else '黑体'
        self.label.setFont(QFont(font_family, 12))  # 设置字体为a.ttf

        self.button = QPushButton('Change Text', self)
        self.button.setGeometry(100, 100, 100, 30)
        self.button.clicked.connect(self.changeText)

    def changeText(self):
        self.label.setText("什么字体")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()

    window.show()
    sys.exit(app.exec_())
