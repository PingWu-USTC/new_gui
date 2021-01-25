from PyQt5 import QtCore, QtGui, QtWidgets

"""
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.central_widget = QtWidgets.QWidget()
        self.central_layout = QtWidgets.QVBoxLayout()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(self.central_layout)
        # Lets create some widgets inside
        self.label = QtWidgets.QLabel()
        self.list_view = QtWidgets.QListView()
        self.push_button = QtWidgets.QPushButton()
        self.label.setText('Hi, this is a label. And the next one is a List View :')
        self.push_button.setText('Push Button Here')
        # Lets add the widgets
        self.central_layout.addWidget(self.label)
        self.central_layout.addWidget(self.list_view)
        self.central_layout.addWidget(self.push_button)

        self.resize(200, 400)
        self.sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.sizePolicy.setHeightForWidth(True)
        self.setSizePolicy(self.sizePolicy)

    def heightForWidth(self, width):
        return width * 2


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
"""
"""
import sys
from PyQt5 import QtCore, QtGui, QtWidgets

class MyForm(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.tabWidget.mouseMoveEvent = self.mouseMoveEvent
        self.show()

    def mouseMoveEvent(self, event):
        pos = event.windowPos().toPoint()
        x = pos.x()
        y = pos.y()
        text = "x: {0}, y: {1}".format(x, y)
        self.ui.labelTracking.setText(text)
class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(653, 450)
        Dialog.setMouseTracking(True)
        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setGeometry(QtCore.QRect(160, 0, 481, 451))
        self.tabWidget.setMouseTracking(True)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.tabWidget.addTab(self.tab, "")
        self.labelTracking = QtWidgets.QLabel(Dialog)
        self.labelTracking.setGeometry(QtCore.QRect(10, 80, 131, 61))
        self.labelTracking.setMouseTracking(True)
        self.labelTracking.setText("")
        self.labelTracking.setObjectName("labelTracking")

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Dialog", "Test Tab"))


import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtWidgets import QDialog
class Demo(QDialog):
    def __init__(self):
        super(Demo, self).__init__()
        self.button_label = QLabel('No Button Pressed', self)              # 1
        self.xy_label = QLabel('x:0, y:0', self)                           # 2
        self.global_xy_label = QLabel('global x:0, global y:0', self)      # 3

        self.button_label.setAlignment(Qt.AlignCenter)
        self.xy_label.setAlignment(Qt.AlignCenter)
        self.global_xy_label.setAlignment(Qt.AlignCenter)

        self.v_layout = QVBoxLayout()
        self.v_layout.addWidget(self.button_label)
        self.v_layout.addWidget(self.xy_label)
        self.v_layout.addWidget(self.global_xy_label)
        self.setLayout(self.v_layout)

        self.resize(300, 300)
        self.setMouseTracking(True)                                        # 4

    def mouseMoveEvent(self, QMouseEvent):                                 # 5
        x = QMouseEvent.x()
        y = QMouseEvent.y()
        global_x = QMouseEvent.globalX()
        global_y = QMouseEvent.globalY()

        self.xy_label.setText('x:{}, y:{}'.format(x, y))
        self.global_xy_label.setText('global x:{}, global y:{}'.format(global_x, global_y))

    def mousePressEvent(self, QMouseEvent):                                # 6
        if QMouseEvent.button() == Qt.LeftButton:
            self.button_label.setText('Left Button Pressed')
        elif QMouseEvent.button() == Qt.MidButton:
            self.button_label.setText('Middle Button Pressed')
        elif QMouseEvent.button() == Qt.RightButton:
            self.button_label.setText('Right Button Pressed')

    def mouseReleaseEvent(self, QMouseEvent):                              # 7
        if QMouseEvent.button() == Qt.LeftButton:
            self.button_label.setText('Left Button Released')
        elif QMouseEvent.button() == Qt.MidButton:
            self.button_label.setText('Middle Button Released')
        elif QMouseEvent.button() == Qt.RightButton:
            self.button_label.setText('Right Button Released')

    def mouseDoubleClickEvent(self, QMouseEvent):                          # 8
        if QMouseEvent.button() == Qt.LeftButton:
            self.button_label.setText('Left Button Double Clikced')
        elif QMouseEvent.button() == Qt.MidButton:
            self.button_label.setText('Middle Button Double Clicked')
        elif QMouseEvent.button() == Qt.RightButton:
            self.button_label.setText('Right Button Double Clikced')


from PyQt5 import QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplWidget(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(MplWidget, self).__init__(parent)
        self.canvas = FigureCanvas(Figure())

        vertical_layout = QtWidgets.QVBoxLayout(self)
        vertical_layout.addWidget(self.canvas)

        self.canvas.axes = self.canvas.figure.add_subplot(111)

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_move)

    def on_press(self, event):
        print("press")
        print("event.xdata", event.xdata)
        print("event.ydata", event.ydata)
        print("event.inaxes", event.inaxes)
        print("x", event.x)
        print("y", event.y)

    def on_release(self, event):
        print("release:")
        print("event.xdata", event.xdata)
        print("event.ydata", event.ydata)
        print("event.inaxes", event.inaxes)
        print("x", event.x)
        print("y", event.y)

    def on_move(self, event):
        print("move")
        print("event.xdata", event.xdata)
        print("event.ydata", event.ydata)
        print("event.inaxes", event.inaxes)
        print("x", event.x)
        print("y", event.y)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = MplWidget()
    w.show()
    sys.exit(app.exec_())

import sys

from PyQt5.QtWidgets import (QApplication, QLabel, QLineEdit,
                             QVBoxLayout, QWidget)


class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        hbox = QVBoxLayout(self)

        self.lbl = QLabel(self)
        qle = QLineEdit(self)

        qle.textChanged[str].connect(self.onChanged)

        hbox.addWidget(self.lbl)
        hbox.addSpacing(20)
        hbox.addWidget(qle)

        self.resize(250, 200)
        self.setWindowTitle('QLineEdit')
        self.show()

    def onChanged(self, text):
        self.lbl.setText(text)
        self.lbl.adjustSize()


def main():
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

from PyQt5 import QtWidgets
import os
import numpy as np
from numpy import cos
from mayavi.mlab import contour3d

os.environ['ETS_TOOLKIT'] = 'qt4'
from pyface.qt import QtGui, QtCore
from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor


## create Mayavi Widget and show

class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())

    @on_trait_change('scene.activated')
    def update_plot(self):
        ## PLot to Show
        x, y, z = np.ogrid[-3:3:60j, -3:3:60j, -3:3:60j]
        t = 0
        Pf = 0.45 + ((x * cos(t)) * (x * cos(t)) + (y * cos(t)) * (y * cos(t)) - (z * cos(t)) * (z * cos(t)))
        obj = contour3d(Pf, contours=[0], transparent=False)

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False),
                resizable=True)


class MayaviQWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.visualization = Visualization()

        self.ui = self.visualization.edit_traits(parent=self,
                                                 kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)


#### PyQt5 GUI ####
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        ## MAIN WINDOW
        MainWindow.setObjectName("MainWindow")
        MainWindow.setGeometry(200, 200, 1100, 700)

        ## CENTRAL WIDGET
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

        ## GRID LAYOUT
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")

        ## BUTTONS
        self.button_default = QtWidgets.QPushButton(self.centralwidget)
        self.button_default.setObjectName("button_default")
        self.gridLayout.addWidget(self.button_default, 0, 0, 1, 1)

        self.button_previous_data = QtWidgets.QPushButton(self.centralwidget)
        self.button_previous_data.setObjectName("button_previous_data")
        self.gridLayout.addWidget(self.button_previous_data, 1, 1, 1, 1)
        ## Mayavi Widget 1
        container = QtGui.QWidget()
        mayavi_widget = MayaviQWidget(container)
        self.gridLayout.addWidget(mayavi_widget, 1, 0, 1, 1)
        ## Mayavi Widget 2
        container1 = QtGui.QWidget()
        mayavi_widget = MayaviQWidget(container1)
        self.gridLayout.addWidget(mayavi_widget, 0, 1, 1, 1)

        ## SET TEXT
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Simulator"))
        self.button_default.setText(_translate("MainWindow", "Default Values"))
        self.button_previous_data.setText(_translate("MainWindow", "Previous Values"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    MainWindow = QtWidgets.QMainWindow()

    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)

fig, ax = plt.subplots()

# animated=True tells matplotlib to only draw the artist when we
# explicitly request it
(ln,) = ax.plot(x, np.sin(x), animated=True)

# make sure the window is raised, but the script keeps going
plt.show(block=False)

# stop to admire our empty window axes and ensure it is rendered at
# least once.
#
# We need to fully draw the figure at its final size on the screen
# before we continue on so that :
#  a) we have the correctly sized and drawn background to grab
#  b) we have a cached renderer so that ``ax.draw_artist`` works
# so we spin the event loop to let the backend process any pending operations
plt.pause(0.1)

# get copy of entire figure (everything inside fig.bbox) sans animated artist
bg = fig.canvas.copy_from_bbox(fig.bbox)
# draw the animated artist, this uses a cached renderer
ax.draw_artist(ln)
# show the result to the screen, this pushes the updated RGBA buffer from the
# renderer to the GUI framework so you can see it
fig.canvas.blit(fig.bbox)

for j in range(100):
    # reset the background back in the canvas state, screen unchanged
    fig.canvas.restore_region(bg)
    # update the artist, neither the canvas state nor the screen have changed
    ln.set_ydata(np.sin(x + (j / 100) * np.pi))
    # re-render the artist, updating the canvas state, but not the screen
    ax.draw_artist(ln)
    # copy the image to the GUI state, but screen might not changed yet
    fig.canvas.blit(fig.bbox)
    # flush any pending GUI events, re-painting the screen if needed
    fig.canvas.flush_events()
    # you can put a pause in if you want to slow things down
    # plt.pause(.1)
"""
from PyQt5 import QtCore, QtGui, QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.central_widget = QtWidgets.QWidget()
        self.central_layout = QtWidgets.QVBoxLayout()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(self.central_layout)
        # Lets create some widgets inside
        self.label = QtWidgets.QLabel()
        self.list_view = QtWidgets.QListView()
        self.push_button = QtWidgets.QPushButton()
        self.label.setText('Hi, this is a label. And the next one is a List View :')
        self.push_button.setText('Push Button Here')
        # Lets add the widgets
        self.central_layout.addWidget(self.label)
        self.central_layout.addWidget(self.list_view)
        self.central_layout.addWidget(self.push_button)

        self.resize(200, 400)
        self.sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.sizePolicy.setHeightForWidth(True)
        self.sizePolicy.setWidthForHeight(True)
        self.sizePolicy.hasHeightForWidth()

        self.sizePolicy.hasWidthForHeight()
        self.setSizePolicy(self.sizePolicy)

    def widthForHeight (self, width):
        return width * 2
    """
    def heightForWidth (self, height):
        return height/2
    """
    """
    def resizeEvent(self, a0: QtGui.QResizeEvent):
        a0.accept()
        if a0.size().height()>a0.size().width():
            self.resize(a0.size().width()*2,a0.size().width())
        else:
            self.resize(a0.size().height()*2,a0.size().height())
    """
if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    w = MainWindow()
    w.show()
    sys.exit(app.exec_())