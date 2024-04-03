import hou,re

def version(path):
    hipName = path.rsplit('/',1)[1]
    
    verSplit = None
    if len(re.findall('_v(?=\d+)', hipName)) > 0:
        verSplit = re.split('_v(?=\d+)', hipName)
    elif len(re.findall('_V(?=\d+)', hipName)) > 0:
        verSplit = re.split('_V(?=\d+)', hipName)
    
    if verSplit:
        version = re.split(r'(^[^\D]+)', verSplit[-1])[1:][0]

        if version:
            hou.putenv( 'VERSION', str(int(version)) )