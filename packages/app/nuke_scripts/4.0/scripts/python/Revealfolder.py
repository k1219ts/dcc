import nuke, os, subprocess

def Revealfolder():
    sel = nuke.selectedNode()
    if sel.knob('file'):
        f = sel['file'].getEvaluatedValue()
    elif sel.knob('vfield_file'):
        f = sel['vfield_file'].value()
    elif sel.knob('out_type'):
        outType = sel.knob('out_type').value()
        f = sel[outType + '_path'].getEvaluatedValue()

    norm = os.path.split(f)[0]
    norm = os.path.normpath(norm)
    # print(norm)
    cmd = ['nautilus', '--no-desktop', '--browser', norm]
    if '8' == os.environ['REZ_CENTOS_MAJOR_VERSION']:
        cmd = ['nautilus', '--browser', norm]
    # print(cmd)
    subprocess.Popen(cmd)
