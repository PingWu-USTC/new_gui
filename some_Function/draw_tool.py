# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'draw_tool.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(523, 994)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.splitter_26 = QtWidgets.QSplitter(Form)
        self.splitter_26.setOrientation(QtCore.Qt.Vertical)
        self.splitter_26.setObjectName("splitter_26")
        self.label = QtWidgets.QLabel(self.splitter_26)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(36)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.splitter_21 = QtWidgets.QSplitter(self.splitter_26)
        self.splitter_21.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_21.setObjectName("splitter_21")
        self.label_22 = QtWidgets.QLabel(self.splitter_21)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(24)
        self.label_22.setFont(font)
        self.label_22.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_22.setObjectName("label_22")
        self.comboBox_4 = QtWidgets.QComboBox(self.splitter_21)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_4.sizePolicy().hasHeightForWidth())
        self.comboBox_4.setSizePolicy(sizePolicy)
        self.comboBox_4.setMinimumSize(QtCore.QSize(237, 45))
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(20)
        self.comboBox_4.setFont(font)
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.splitter_16 = QtWidgets.QSplitter(self.splitter_26)
        self.splitter_16.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_16.setObjectName("splitter_16")
        self.label_18 = QtWidgets.QLabel(self.splitter_16)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(12)
        self.label_18.setFont(font)
        self.label_18.setObjectName("label_18")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.splitter_16)
        self.lineEdit_3.setMinimumSize(QtCore.QSize(0, 24))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lineEdit_3.setFont(font)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.label_19 = QtWidgets.QLabel(self.splitter_16)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(12)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.splitter_16)
        self.lineEdit_4.setMinimumSize(QtCore.QSize(0, 24))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lineEdit_4.setFont(font)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.splitter_22 = QtWidgets.QSplitter(self.splitter_26)
        self.splitter_22.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_22.setObjectName("splitter_22")
        self.label_20 = QtWidgets.QLabel(self.splitter_22)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(18)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.label_21 = QtWidgets.QLabel(self.splitter_22)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(18)
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.splitter_23 = QtWidgets.QSplitter(self.splitter_26)
        self.splitter_23.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_23.setObjectName("splitter_23")
        self.splitter_17 = QtWidgets.QSplitter(self.splitter_23)
        self.splitter_17.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_17.setObjectName("splitter_17")
        self.label_23 = QtWidgets.QLabel(self.splitter_17)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(12)
        self.label_23.setFont(font)
        self.label_23.setObjectName("label_23")
        self.comboBox_5 = QtWidgets.QComboBox(self.splitter_17)
        self.comboBox_5.setMinimumSize(QtCore.QSize(214, 33))
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(14)
        self.comboBox_5.setFont(font)
        self.comboBox_5.setObjectName("comboBox_5")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.splitter_18 = QtWidgets.QSplitter(self.splitter_23)
        self.splitter_18.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_18.setObjectName("splitter_18")
        self.label_24 = QtWidgets.QLabel(self.splitter_18)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(12)
        self.label_24.setFont(font)
        self.label_24.setObjectName("label_24")
        self.comboBox_6 = QtWidgets.QComboBox(self.splitter_18)
        self.comboBox_6.setMinimumSize(QtCore.QSize(214, 33))
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(14)
        self.comboBox_6.setFont(font)
        self.comboBox_6.setObjectName("comboBox_6")
        self.comboBox_6.addItem("")
        self.splitter_24 = QtWidgets.QSplitter(self.splitter_26)
        self.splitter_24.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_24.setObjectName("splitter_24")
        self.splitter_19 = QtWidgets.QSplitter(self.splitter_24)
        self.splitter_19.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_19.setObjectName("splitter_19")
        self.label_25 = QtWidgets.QLabel(self.splitter_19)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(12)
        self.label_25.setFont(font)
        self.label_25.setObjectName("label_25")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.splitter_19)
        self.lineEdit_5.setMinimumSize(QtCore.QSize(0, 24))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lineEdit_5.setFont(font)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.splitter_20 = QtWidgets.QSplitter(self.splitter_24)
        self.splitter_20.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_20.setObjectName("splitter_20")
        self.label_26 = QtWidgets.QLabel(self.splitter_20)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(12)
        self.label_26.setFont(font)
        self.label_26.setObjectName("label_26")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.splitter_20)
        self.lineEdit_6.setMinimumSize(QtCore.QSize(0, 24))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lineEdit_6.setFont(font)
        self.lineEdit_6.setText("")
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.label_2 = QtWidgets.QLabel(self.splitter_26)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(36)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.splitter_15 = QtWidgets.QSplitter(self.splitter_26)
        self.splitter_15.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_15.setObjectName("splitter_15")
        self.label_8 = QtWidgets.QLabel(self.splitter_15)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(24)
        self.label_8.setFont(font)
        self.label_8.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.comboBox = QtWidgets.QComboBox(self.splitter_15)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox.sizePolicy().hasHeightForWidth())
        self.comboBox.setSizePolicy(sizePolicy)
        self.comboBox.setMinimumSize(QtCore.QSize(237, 45))
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(20)
        self.comboBox.setFont(font)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.splitter_14 = QtWidgets.QSplitter(self.splitter_26)
        self.splitter_14.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_14.setObjectName("splitter_14")
        self.label_9 = QtWidgets.QLabel(self.splitter_14)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(24)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.pushButton = QtWidgets.QPushButton(self.splitter_14)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(18)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.splitter_14)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(18)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.splitter_25 = QtWidgets.QSplitter(self.splitter_26)
        self.splitter_25.setMinimumSize(QtCore.QSize(0, 30))
        self.splitter_25.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_25.setObjectName("splitter_25")
        self.label_27 = QtWidgets.QLabel(self.splitter_25)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(18)
        self.label_27.setFont(font)
        self.label_27.setObjectName("label_27")
        self.doubleSpinBox_6 = QtWidgets.QDoubleSpinBox(self.splitter_25)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.doubleSpinBox_6.sizePolicy().hasHeightForWidth())
        self.doubleSpinBox_6.setSizePolicy(sizePolicy)
        self.doubleSpinBox_6.setMinimumSize(QtCore.QSize(0, 30))
        self.doubleSpinBox_6.setMaximumSize(QtCore.QSize(267, 16664))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.doubleSpinBox_6.setFont(font)
        self.doubleSpinBox_6.setObjectName("doubleSpinBox_6")
        self.label_10 = QtWidgets.QLabel(self.splitter_26)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(18)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.splitter_9 = QtWidgets.QSplitter(self.splitter_26)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.splitter_9.setFont(font)
        self.splitter_9.setOrientation(QtCore.Qt.Vertical)
        self.splitter_9.setObjectName("splitter_9")
        self.splitter_6 = QtWidgets.QSplitter(self.splitter_9)
        self.splitter_6.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_6.setObjectName("splitter_6")
        self.label_11 = QtWidgets.QLabel(self.splitter_6)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(12)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.comboBox_2 = QtWidgets.QComboBox(self.splitter_6)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(14)
        self.comboBox_2.setFont(font)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.splitter_7 = QtWidgets.QSplitter(self.splitter_9)
        self.splitter_7.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_7.setObjectName("splitter_7")
        self.label_12 = QtWidgets.QLabel(self.splitter_7)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(12)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.doubleSpinBox_3 = QtWidgets.QDoubleSpinBox(self.splitter_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.doubleSpinBox_3.sizePolicy().hasHeightForWidth())
        self.doubleSpinBox_3.setSizePolicy(sizePolicy)
        self.doubleSpinBox_3.setMinimumSize(QtCore.QSize(0, 30))
        self.doubleSpinBox_3.setMaximumSize(QtCore.QSize(267, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.doubleSpinBox_3.setFont(font)
        self.doubleSpinBox_3.setObjectName("doubleSpinBox_3")
        self.splitter_8 = QtWidgets.QSplitter(self.splitter_9)
        self.splitter_8.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_8.setObjectName("splitter_8")
        self.label_13 = QtWidgets.QLabel(self.splitter_8)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(12)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(self.splitter_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.doubleSpinBox_2.sizePolicy().hasHeightForWidth())
        self.doubleSpinBox_2.setSizePolicy(sizePolicy)
        self.doubleSpinBox_2.setMinimumSize(QtCore.QSize(0, 30))
        self.doubleSpinBox_2.setMaximumSize(QtCore.QSize(213, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.doubleSpinBox_2.setFont(font)
        self.doubleSpinBox_2.setObjectName("doubleSpinBox_2")
        self.label_14 = QtWidgets.QLabel(self.splitter_26)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(18)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.splitter_10 = QtWidgets.QSplitter(self.splitter_26)
        self.splitter_10.setOrientation(QtCore.Qt.Vertical)
        self.splitter_10.setObjectName("splitter_10")
        self.splitter_11 = QtWidgets.QSplitter(self.splitter_10)
        self.splitter_11.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_11.setObjectName("splitter_11")
        self.label_15 = QtWidgets.QLabel(self.splitter_11)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(12)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.comboBox_3 = QtWidgets.QComboBox(self.splitter_11)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(14)
        self.comboBox_3.setFont(font)
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.splitter_12 = QtWidgets.QSplitter(self.splitter_10)
        self.splitter_12.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_12.setObjectName("splitter_12")
        self.label_16 = QtWidgets.QLabel(self.splitter_12)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(12)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.doubleSpinBox_4 = QtWidgets.QDoubleSpinBox(self.splitter_12)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.doubleSpinBox_4.sizePolicy().hasHeightForWidth())
        self.doubleSpinBox_4.setSizePolicy(sizePolicy)
        self.doubleSpinBox_4.setMinimumSize(QtCore.QSize(0, 30))
        self.doubleSpinBox_4.setMaximumSize(QtCore.QSize(267, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.doubleSpinBox_4.setFont(font)
        self.doubleSpinBox_4.setObjectName("doubleSpinBox_4")
        self.splitter_13 = QtWidgets.QSplitter(self.splitter_10)
        self.splitter_13.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_13.setObjectName("splitter_13")
        self.label_17 = QtWidgets.QLabel(self.splitter_13)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(12)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.doubleSpinBox_5 = QtWidgets.QDoubleSpinBox(self.splitter_13)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.doubleSpinBox_5.sizePolicy().hasHeightForWidth())
        self.doubleSpinBox_5.setSizePolicy(sizePolicy)
        self.doubleSpinBox_5.setMinimumSize(QtCore.QSize(0, 30))
        self.doubleSpinBox_5.setMaximumSize(QtCore.QSize(213, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.doubleSpinBox_5.setFont(font)
        self.doubleSpinBox_5.setObjectName("doubleSpinBox_5")
        self.label_3 = QtWidgets.QLabel(self.splitter_26)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(36)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.splitter_5 = QtWidgets.QSplitter(self.splitter_26)
        self.splitter_5.setOrientation(QtCore.Qt.Vertical)
        self.splitter_5.setObjectName("splitter_5")
        self.splitter = QtWidgets.QSplitter(self.splitter_5)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.label_4 = QtWidgets.QLabel(self.splitter)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.lineEdit = QtWidgets.QLineEdit(self.splitter)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(14)
        self.lineEdit.setFont(font)
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.splitter_2 = QtWidgets.QSplitter(self.splitter_5)
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName("splitter_2")
        self.label_5 = QtWidgets.QLabel(self.splitter_2)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName("label_5")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.splitter_2)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(14)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.splitter_3 = QtWidgets.QSplitter(self.splitter_5)
        self.splitter_3.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_3.setObjectName("splitter_3")
        self.label_6 = QtWidgets.QLabel(self.splitter_3)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_6.setObjectName("label_6")
        self.fontComboBox = QtWidgets.QFontComboBox(self.splitter_3)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(14)
        self.fontComboBox.setFont(font)
        self.fontComboBox.setObjectName("fontComboBox")
        self.splitter_4 = QtWidgets.QSplitter(self.splitter_5)
        self.splitter_4.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_4.setObjectName("splitter_4")
        self.label_7 = QtWidgets.QLabel(self.splitter_4)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(12)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.spinBox = QtWidgets.QSpinBox(self.splitter_4)
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(14)
        self.spinBox.setFont(font)
        self.spinBox.setObjectName("spinBox")
        self.buttonBox_2 = QtWidgets.QDialogButtonBox(self.splitter_26)
        self.buttonBox_2.setMinimumSize(QtCore.QSize(0, 80))
        self.buttonBox_2.setMaximumSize(QtCore.QSize(16777215, 100))
        font = QtGui.QFont()
        font.setFamily("苹方 粗体")
        font.setPointSize(20)
        self.buttonBox_2.setFont(font)
        self.buttonBox_2.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox_2.setObjectName("buttonBox_2")
        self.verticalLayout.addWidget(self.splitter_26)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Scale"))
        self.label_22.setText(_translate("Form", "choose_direction"))
        self.comboBox_4.setItemText(0, _translate("Form", "Veritcal"))
        self.comboBox_4.setItemText(1, _translate("Form", "Horizontal"))
        self.label_18.setText(_translate("Form", "from"))
        self.label_19.setText(_translate("Form", "to"))
        self.label_20.setText(_translate("Form", "Major Ticks"))
        self.label_21.setText(_translate("Form", "Minor Ticks"))
        self.label_23.setText(_translate("Form", "style"))
        self.comboBox_5.setItemText(0, _translate("Form", "By Increment"))
        self.comboBox_5.setItemText(1, _translate("Form", "By Count"))
        self.label_24.setText(_translate("Form", "style"))
        self.comboBox_6.setItemText(0, _translate("Form", "By Counts"))
        self.label_25.setText(_translate("Form", "value"))
        self.label_26.setText(_translate("Form", "count"))
        self.label_2.setText(_translate("Form", "line and ticks"))
        self.label_8.setText(_translate("Form", "choose_direction"))
        self.comboBox.setItemText(0, _translate("Form", "Bottom"))
        self.comboBox.setItemText(1, _translate("Form", "Right"))
        self.comboBox.setItemText(2, _translate("Form", "Topo"))
        self.comboBox.setItemText(3, _translate("Form", "Left"))
        self.label_9.setText(_translate("Form", "show line and ticks"))
        self.pushButton.setText(_translate("Form", "Yes"))
        self.pushButton_2.setText(_translate("Form", "No"))
        self.label_27.setText(_translate("Form", "LineWidth"))
        self.label_10.setText(_translate("Form", "Major Ticks"))
        self.label_11.setText(_translate("Form", "style"))
        self.comboBox_2.setItemText(0, _translate("Form", "In "))
        self.comboBox_2.setItemText(1, _translate("Form", "Out"))
        self.comboBox_2.setItemText(2, _translate("Form", "None"))
        self.label_12.setText(_translate("Form", "length"))
        self.label_13.setText(_translate("Form", "thickness"))
        self.label_14.setText(_translate("Form", "Minor Ticks"))
        self.label_15.setText(_translate("Form", "style"))
        self.comboBox_3.setItemText(0, _translate("Form", "In"))
        self.comboBox_3.setItemText(1, _translate("Form", "None"))
        self.comboBox_3.setItemText(2, _translate("Form", "Out"))
        self.label_16.setText(_translate("Form", "length"))
        self.label_17.setText(_translate("Form", "thickness"))
        self.label_3.setText(_translate("Form", "title"))
        self.label_4.setText(_translate("Form", "x_label"))
        self.label_5.setText(_translate("Form", "y_label"))
        self.label_6.setText(_translate("Form", "font"))
        self.label_7.setText(_translate("Form", "size"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
