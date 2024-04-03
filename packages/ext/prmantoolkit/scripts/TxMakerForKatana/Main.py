# -*- coding: utf-8 -*-
import os
import sys
import MainTxForKat

ScriptRoot = os.path.dirname(os.path.abspath(__file__))

def main():
    nautilusFile = os.environ.get('NAUTILUS_SCRIPT_SELECTED_FILE_PATHS')
    setFile = list()
    if nautilusFile:
        for file in nautilusFile.split('\n'):
            if file:
                if not os.path.isdir(file):
                    fileExtension = os.path.splitext(file)[1]
                    if fileExtension == '.hdr' or fileExtension == '.hdri' or fileExtension == '.exr':
                        MainTxForKat.MainTxForKat(file)


if __name__ == "__main__":
    main()
