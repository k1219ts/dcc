import os
import Database
import sys

def alert(msg):
    '''Show a dialog with a simple message.'''
    dialog = gtk.MessageDialog()
    dialog.set_markup(msg)
    dialog.run()


def main(files):
    for filePath in files:
        try:
            if '/assetlib/Texture/' in filePath:
                Database.AddSCItem(filePath)
            elif '/assetlib/_3d' in filePath:
                Database.AddItem(filePath)
            else:
                pass
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
            sys.exit(main(files))

    else:
        sys.stdout.write("[Debug mode]:  %s" %sys.argv[1:])
        sys.exit(main(sys.argv[1:]))

