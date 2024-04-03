import os,hou,subprocess

def convert(file,mode):
    if not mode:
        try:
            hou.hipFile.load(file,1,1)
        except: pass

    ## Code

    path = os.path.dirname(file)

    code = """import os
def save():

    lic = hou.licenseCategory().name()
    
    if lic == 'Commercial':
        ext = 'hip'
    elif lic in ['Education','Indie']:
        ext = 'hiplc'
    else:
        ext = 'hipnc'
    
    filepath = '%s/%s.'+ext
    if os.path.isfile(filepath):
        print "convertedfilepath:File '"+filepath+"' already exists!"
    else:
        hou.hipFile.save(filepath, 0)
    
        print 'convertedfilepath:'+filepath

def build():
    """%(path,hou.expandString('$HIPNAME'))

    nodecode = hou.node('/').asCode(brief=False, recurse=1, save_channels_only=0, save_creation_commands=1, save_keys_in_frames=1, save_outgoing_wires=1, save_parm_values_only=False, save_spare_parms=1, function_name=None)
    nodecode = fixerrors(nodecode)
    nodecode = addexceptions(nodecode)

    code += '\n    '.join(nodecode.split('\n'))
    code += '\nbuild()\nsave()'
    
    ## Save py file
    
    py = '%s/scene_build.py'%path
    node_file = open(py, "w")
    node_file.write(code)
    node_file.close()
    #return
    ## rebuild

    cmd = '"{0}" -c "execfile(\""{1}\"")"'.format(gethython(),os.path.abspath(py).replace('\\','\\\\'))
    process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    res = process.communicate()
    
    os.remove(py)
    newhip = readfromconsole(res)
    
    if newhip:
        if mode:
            return newhip
        else:
            print 'convertedfilepath:%s'%newhip
    else:
        return None

def fixerrors(nodecode):

    ## Fixing known syntax errors

    err0 = "'bool(__import__('htoa.ocio').ocio.has_config)'"
    nodecode=nodecode.replace(err0,'"bool(__import__(\'htoa.ocio\').ocio.has_config)"')
    return nodecode

def addexceptions(nodecode):
    string = ''
    tab = '    '
    curnode = None
    count = 0
    lines = nodecode.split('\n')

    for ind,l in enumerate(lines):
        if ind == len(lines)-1:# if at end of code
            if count > 0:
                if l:
                    string+='\n%s%s\nexcept: pass'%(l,tab)
                else:
                    string+='\nexcept: pass'

            else:
                string+='\n%s'%l

            continue

        if l.startswith(('# Code for /', '# Code to establish connections for /')):
            if l.count('/') > 1:
                splt = l.split('/',1)[1]

                if splt.find(' ') != -1:
                    string+='\n%s%s'%(tab,l)

                else:
                    if count == 0:
                        string+='\ntry:\n%s%s'%(tab,l)
                    else:
                        string+='\nexcept: pass\ntry:\n%s%s'%(tab,l)
                    curnode = splt

                    count+=1

                continue

        if l in ['# Update the parent node.','# Restore the parent and current nodes.']:
            if count == 0:
                string+='\ntry:\n%s%s'%(tab,l)
            else:
                string+='\nexcept: pass\ntry:\n%s%s'%(tab,l)

            count += 1
            curnode = None
        elif count > 0:
            string+='\n%s%s'%(tab,l)
        else:
            string += '\n%s'%l

    return string

def external(file):
    cmd = '"{0}" -c "from utils import promoteHip;promoteHip.convert(""{1}"",0)"'.format(gethython(),file)
    process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    res = process.communicate()

    return readfromconsole(res)

def readfromconsole(res):
    newhip = [l for l in res[0].split('\n') if l.startswith('convertedfilepath:')]
    if newhip:
        return newhip[0].split(':',1)[1].split('\r')[0]
    return None

def gethython():
    bin = hou.expandString('$HB')
    hython = ['%s/%s'%(bin,name) for name in os.listdir(bin) if name.startswith('hython.')][0]
    return os.path.abspath(hython)