import os
import re
import DXRulebook.Interface as rb


IsVer = lambda *args: rb.MatchFlag('ver', args[0])

def Ver(*args):
    ver = 'v%s'%(str(args[0]).zfill(VerDigit()))
    if not IsVer(ver):
        ver = ver.upper()
    return ver

def VerAsInt(*args):
    ver = re.search('\d{%s}' % VerDigit(), args[0])
    if ver:
        return int(ver.group())
    else:
        return int(re.search('\d{3}', args[0]).group())

def VerDigit():
    coder = rb.Coder()
    pattern = coder.Rulebook().flag['ver'].pattern
    return int(re.search(r'(?<={)\d+(?=})', pattern).group())

def setShowConfig(showName):
    showRbPath = '/show/{SHOW}/_config/DXRulebook.yaml'.format(SHOW=showName)

    if os.path.exists(showRbPath):
        print '>> showRbPath:', showRbPath
        os.environ['DXRULEBOOKFILE'] = showRbPath
    else:
        if os.environ.has_key('DXRULEBOOKFILE'):
            del os.environ['DXRULEBOOKFILE']

    rb.Reload()
