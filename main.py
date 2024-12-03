import sys
from PySide6.QtWidgets import QApplication
from interface import MainWindow


def main():
    app = QApplication(sys.argv)
 #  app.exec()
    w = MainWindow()
    w.show() #rend la fenetre visible à l'ecran
    sys.exit(app.exec()) #sortie propre de l'application 
    
if __name__ == "__main__":
    main()
