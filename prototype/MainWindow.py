# Form implementation generated from reading ui file 'trial1.ui'
#
# Created by: PyQt6 UI code generator 6.2.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(795, 627)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 20, 761, 551))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.layoutWidget = QtWidgets.QWidget(self.tab_5)
        self.layoutWidget.setGeometry(QtCore.QRect(11, 12, 351, 103))
        self.layoutWidget.setObjectName("layoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.layoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label)
        self.spinBox = QtWidgets.QSpinBox(self.layoutWidget)
        self.spinBox.setObjectName("spinBox")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.spinBox)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_2)
        self.spinBox_2 = QtWidgets.QSpinBox(self.layoutWidget)
        self.spinBox_2.setObjectName("spinBox_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.spinBox_2)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_3)
        self.spinBox_3 = QtWidgets.QSpinBox(self.layoutWidget)
        self.spinBox_3.setObjectName("spinBox_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.spinBox_3)
        self.layoutWidget1 = QtWidgets.QWidget(self.tab_5)
        self.layoutWidget1.setGeometry(QtCore.QRect(420, 10, 131, 101))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.pushButton = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_2.addWidget(self.pushButton, 0, 0, 1, 1)
        self.pushButton_6 = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_6.setObjectName("pushButton_6")
        self.gridLayout_2.addWidget(self.pushButton_6, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab_5, "")
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")
        self.layoutWidget2 = QtWidgets.QWidget(self.tab_6)
        self.layoutWidget2.setGeometry(QtCore.QRect(11, 11, 711, 121))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget2)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_2 = QtWidgets.QPushButton(self.layoutWidget2)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 0, 0, 1, 1)
        self.pushButton_5 = QtWidgets.QPushButton(self.layoutWidget2)
        self.pushButton_5.setObjectName("pushButton_5")
        self.gridLayout.addWidget(self.pushButton_5, 0, 1, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.layoutWidget2)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout.addWidget(self.pushButton_3, 1, 0, 1, 1)
        self.pushButton_4 = QtWidgets.QPushButton(self.layoutWidget2)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout.addWidget(self.pushButton_4, 1, 1, 1, 1)
        self.widget = QtWidgets.QWidget(self.tab_6)
        self.widget.setGeometry(QtCore.QRect(20, 161, 231, 221))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.FC6 = QtWidgets.QCheckBox(self.widget)
        self.FC6.setObjectName("FC6")
        self.gridLayout_3.addWidget(self.FC6, 3, 1, 1, 1)
        self.F3 = QtWidgets.QCheckBox(self.widget)
        self.F3.setObjectName("F3")
        self.gridLayout_3.addWidget(self.F3, 2, 0, 1, 1)
        self.AF4 = QtWidgets.QCheckBox(self.widget)
        self.AF4.setObjectName("AF4")
        self.gridLayout_3.addWidget(self.AF4, 6, 1, 1, 1)
        self.F8 = QtWidgets.QCheckBox(self.widget)
        self.F8.setObjectName("F8")
        self.gridLayout_3.addWidget(self.F8, 5, 1, 1, 1)
        self.P7 = QtWidgets.QCheckBox(self.widget)
        self.P7.setObjectName("P7")
        self.gridLayout_3.addWidget(self.P7, 5, 0, 1, 1)
        self.AF3 = QtWidgets.QCheckBox(self.widget)
        self.AF3.setObjectName("AF3")
        self.gridLayout_3.addWidget(self.AF3, 0, 0, 1, 1)
        self.FC5 = QtWidgets.QCheckBox(self.widget)
        self.FC5.setObjectName("FC5")
        self.gridLayout_3.addWidget(self.FC5, 3, 0, 1, 1)
        self.F7 = QtWidgets.QCheckBox(self.widget)
        self.F7.setObjectName("F7")
        self.gridLayout_3.addWidget(self.F7, 1, 0, 1, 1)
        self.T7 = QtWidgets.QCheckBox(self.widget)
        self.T7.setObjectName("T7")
        self.gridLayout_3.addWidget(self.T7, 4, 0, 1, 1)
        self.F4 = QtWidgets.QCheckBox(self.widget)
        self.F4.setObjectName("F4")
        self.gridLayout_3.addWidget(self.F4, 4, 1, 1, 1)
        self.O1 = QtWidgets.QCheckBox(self.widget)
        self.O1.setObjectName("O1")
        self.gridLayout_3.addWidget(self.O1, 6, 0, 1, 1)
        self.P8 = QtWidgets.QCheckBox(self.widget)
        self.P8.setObjectName("P8")
        self.gridLayout_3.addWidget(self.P8, 1, 1, 1, 1)
        self.T8 = QtWidgets.QCheckBox(self.widget)
        self.T8.setObjectName("T8")
        self.gridLayout_3.addWidget(self.T8, 2, 1, 1, 1)
        self.O2 = QtWidgets.QCheckBox(self.widget)
        self.O2.setObjectName("O2")
        self.gridLayout_3.addWidget(self.O2, 0, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_3)
        self.pushButton_7 = QtWidgets.QPushButton(self.widget)
        self.pushButton_7.setObjectName("pushButton_7")
        self.verticalLayout.addWidget(self.pushButton_7)
        self.tabWidget.addTab(self.tab_6, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 795, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Time period for each cue"))
        self.label_2.setText(_translate("MainWindow", "Time between two cues"))
        self.label_3.setText(_translate("MainWindow", "Number of trials"))
        self.pushButton.setText(_translate("MainWindow", "Start Recording"))
        self.pushButton_6.setText(_translate("MainWindow", "Stop Recording"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("MainWindow", "Record EEG"))
        self.pushButton_2.setText(_translate("MainWindow", "Power Spectral Density"))
        self.pushButton_5.setText(_translate("MainWindow", "Fourier Transform"))
        self.pushButton_3.setText(_translate("MainWindow", "Common Spatial Pattern"))
        self.pushButton_4.setText(_translate("MainWindow", "Wavelet Transform"))
        self.FC6.setText(_translate("MainWindow", "FC6"))
        self.F3.setText(_translate("MainWindow", "F3"))
        self.AF4.setText(_translate("MainWindow", "AF4"))
        self.F8.setText(_translate("MainWindow", "F8"))
        self.P7.setText(_translate("MainWindow", "P7"))
        self.AF3.setText(_translate("MainWindow", "AF3"))
        self.FC5.setText(_translate("MainWindow", "FC5"))
        self.F7.setText(_translate("MainWindow", "F7"))
        self.T7.setText(_translate("MainWindow", "T7"))
        self.F4.setText(_translate("MainWindow", "F4"))
        self.O1.setText(_translate("MainWindow", "O1"))
        self.P8.setText(_translate("MainWindow", "P8"))
        self.T8.setText(_translate("MainWindow", " T8"))
        self.O2.setText(_translate("MainWindow", "O2"))
        self.pushButton_7.setText(_translate("MainWindow", "Update Channel Selection"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_6), _translate("MainWindow", "Feature Extraction"))
