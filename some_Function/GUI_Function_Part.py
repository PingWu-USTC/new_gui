from PyQt5 import QtCore,QtWidgets
from PyQt5.QtCore import pyqtSignal

"""
对比度调节
"""
class Ui_Form_contrast(QtWidgets.QWidget):
    slider_emit = pyqtSignal(int,int)
    def __init__(self):
        super(Ui_Form_contrast, self).__init__()
        self.setObjectName("Form")
        self.resize(373, 85)
        self.horizontalLayout = QtWidgets.QHBoxLayout(self)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.splitter = QtWidgets.QSplitter(self)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.label = QtWidgets.QLabel(self.splitter)
        self.label.setObjectName("label")
        self.horizontalSlider_2 = QtWidgets.QSlider(self.splitter)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.label_2 = QtWidgets.QLabel(self.splitter)
        self.label_2.setObjectName("label_2")
        self.horizontalSlider = QtWidgets.QSlider(self.splitter)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalLayout.addWidget(self.splitter)

        self.horizontalSlider_2.setMaximum(100)
        self.horizontalSlider_2.setMinimum(0)
        self.horizontalSlider.setMaximum(200)
        self.horizontalSlider.setMinimum(100)
        self.horizontalSlider.setValue(200)
        self.horizontalSlider_2.setValue(0)
        self.retranslateUi(self)

        self.horizontalSlider.valueChanged.connect(self.slider1)
        self.horizontalSlider_2.valueChanged.connect(self.slider1)
        QtCore.QMetaObject.connectSlotsByName(self)


    def slider1(self):
        value = self.horizontalSlider.value()
        value2 = self.horizontalSlider_2.value()
        self.slider_emit.emit(value2,value)


    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "百分之0到百分之50"))
        self.label_2.setText(_translate("Form", "百分之50到百分之100"))
