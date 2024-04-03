
from assetbackup import *
from PySide2 import QtWidgets
def main():
    app = QtWidgets.QApplication(sys.argv)
    Window = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(Window)
    Window.show()
    sys.exit(app.exec_())



if __name__ == "__main__":
    main()
