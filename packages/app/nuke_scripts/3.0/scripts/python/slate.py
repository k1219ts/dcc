import nuke
import getpass


def slate_4th_vendor_yys():
    snode = nuke.createNode('slate_vendor')

    fullPath = nuke.root().name()
    if fullPath.startswith('/netapp/dexter/show'):
        fullPath = fullPath.replace('/netapp/dexter', '')

    seq = fullPath.split('/')[4]
    shot = fullPath.split('/')[5]

    snode['seq'].setValue(seq)
    snode['shot'].setValue(shot.split('_')[-1])
    verScript = "[python {'v' + os.path.basename(nuke.root().name()).split('_v')[-1][:3]}]"

    snode['version'].setValue(verScript)
    snode['artist'].setValue(getpass.getuser().split('.')[0])


    start = int(nuke.knob("first_frame"))
    end = int(nuke.knob("last_frame"))

    snode['input.first_1'].setValue(start)
    snode['input.last_1'].setValue(end)


def slate_4th_vendor():
    snode = nuke.createNode('slate')

    fullPath = nuke.root().name()
    if fullPath.startswith('/netapp/dexter/show'):
        fullPath = fullPath.replace('/netapp/dexter', '')

    seq = fullPath.split('/')[4]
    shot = fullPath.split('/')[5]

    snode['seq'].setValue(seq)
    snode['shot'].setValue(shot.split('_')[-1])
    verScript = "[python {'v' + os.path.basename(nuke.root().name()).split('_v')[-1][:3]}]"

    snode['version'].setValue(verScript)

    start = int(nuke.knob("first_frame"))
    end = int(nuke.knob("last_frame"))

    snode['input.first_1'].setValue(start)
    snode['input.last_1'].setValue(end)
