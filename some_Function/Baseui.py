# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Baseclass.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow
from matplotlib.backends.backend_qt5agg import  FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QKeySequence
from layers import  draw_information_line_topo
class PlotCanvas(FigureCanvas):
    def __init__(self,parent=None,width=8,height=8,dpi=100):
        fig = Figure(figsize=(width,height),dpi=dpi)
        FigureCanvas.__init__(self,fig)
        self.setParent(parent)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(True)
        FigureCanvas.setSizePolicy(self,sizePolicy)
        FigureCanvas.updateGeometry(self)
        self.setMinimumSize(QtCore.QSize(800, 800))
        self.setMaximumSize(QtCore.QSize(1800, 1800))
class Ui_Form(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setObjectName('Baseui')
        self.resize(800,800)
        self.centralWidget = QtWidgets.QWidget(self)
        self.centralWidget.setObjectName('centerwidget')
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_Form()
    ui.show()
    sys.exit(app.exec_())