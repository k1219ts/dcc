#!/usr/bin/env python
import os
import Main2
import sys

def main(files):
    for filePath in files:
        try:
            Main2.ExrToJpg(filePath)
            sys.stdout.write("Successed!")

        except Exception as e:
            pass

if __name__ == '__main__':

    if len(sys.argv) == 1:
        selected = os.getenv('NAUTILUS_SCRIPT_SELECTED_FILE_PATHS')
        files = []
        for file in selected.split('\n'):
            if not file == "":
                files.append(file)

        if not selected:
            sys.stdout.write( "[ ERROR ] - wrong argument: Please select asset folder.")
            sys.exit(0)

        else:
            sys.stdout.write("selected files>>>")
            sys.stdout.write(selected)
            sys.exit(main(files))

    else:
        sys.stdout.write("[Debug mode]:  %s" %sys.argv[1:])
        sys.exit(main(sys.argv[1:]))
