import os, sys
import string
import getpass
import optparse

Applications = [
    'Katana', 'Maya', 'RenderMan', 'Mari', 'RezDCC',
    #'Miarmy'
]

VersionMap = {
    'Katana': ['3.2'],
    'RenderMan': ['22.6', '23.0'],
    'Maya': ['2017', '2018', '2019'],
    #'Miarmy': ['2017'],
    'Mari': ['4.5v2', '4.6v1'],
    'RezDCC': []
}

def DebugPrint(status, message):
    '''
    Args:
        status - error, waring, info
    '''
    if status == 'error':
        sys.stdout.write('\033[1;31m # ERROR : ')
    elif status == 'warning':
        sys.stdout.write('\033[1;32m # WARNING : ')
    else:
        sys.stdout.write('\033[1;34m # %s : ' % status)

    sys.stdout.write('\033[0;0m')
    print message

def subPrint(source):
    '''
    Args:
        source (list)
    '''
    for i in range(len(source)):
        msg = source[i]
        if msg:
            print '\t[%s] %s' % (i, msg)

def GetVersions(input):
    '''
    Args:
        - input (str)  : Application name
        - input (list) : versions
    '''
    if type(input).__name__ == 'list':
        versions = input
    else:
        versions = VersionMap[input]
    if versions:
        DebugPrint('Select Version', '')
        subPrint(versions)
        ver = raw_input('>> ')
        if not ver:
            return versions
        return [versions[int(ver)]]


#-------------------------------------------------------------------------------
#
#   PULL
#
#-------------------------------------------------------------------------------
class GitPull:
    def __init__(self):
        self.user = getpass.getuser()
        if self.user == 'plumber':
            self.rootDir = '/backstage'
        else:
            self.rootDir = os.getenv('DEVELOPER_LOCATION')
        assert self.rootDir, '# Developer location not found'

        DebugPrint('Select Application', '')
        subPrint(['all'] + Applications)
        ap = raw_input('>> ')
        assert ap, '# Select Application'

        appList = list()
        versions= list()

        ap = int(ap)
        if ap:
            appName = Applications[ap-1]
            appList.append(appName)
            versions = GetVersions(appName)
        else:
            appList = Applications

        # print appList
        # print versions
        for name in appList:
            print '## Debug :',name
            eval('self.%s(%s)' % (name, versions))


    def getPull(self, dir):
        if not os.path.exists(dir + '/.git'):
            return
        DebugPrint('Git Pull', dir)
        cmd = 'cd %s &&' % dir
        cmd+= ' git pull'
        os.system(cmd)

    def getClone(self, dir, reponame):
        cmd = 'git clone git@10.0.0.13:plumber/{GITNAME}.git {DIR}'.format(GITNAME=reponame, DIR=dir)
        os.system(cmd)
        return True


    #---------------------------------------------------------------------------
    # MARI
    #---------------------------------------------------------------------------
    def Mari(self, versions):
        # VERSION
        if not versions:
            versions = VersionMap['Mari']

        prefix = 'Mari-'
        if self.user == 'plumber':
            prefix = ''
        for v in versions:
            name = prefix + v
            dir  = '{DIR}/apps/Mari/versions/{NAME}'.format(DIR=self.rootDir, NAME=name)
            if os.path.exists(dir):
                self.getPull(dir)
            else:
                self.getClone(dir, 'Mari-{}-script'.format(v))


    #---------------------------------------------------------------------------
    # RENDERMAN
    #---------------------------------------------------------------------------
    def RenderMan(self, versions):
        # VERSION
        if not versions:
            versions = VersionMap['RenderMan']

        prefix = 'RenderMan-'
        if self.user == 'plumber':
            prefix = ''
        for v in versions:
            name = prefix + v
            dir  = '{DIR}/apps/RenderMan/extensions/{NAME}'.format(DIR=self.rootDir, NAME=name)
            if os.path.exists(dir):
                self.getPull(dir)
            else:
                self.getClone(dir, 'RenderMan-{}-extensions'.format(v))


    #---------------------------------------------------------------------------
    # KATANA
    #---------------------------------------------------------------------------
    def Katana(self, versions):
        # VERSION
        if not versions:
            versions = VersionMap['Katana']

        prefix = 'Katana-'
        if self.user == 'plumber':
            prefix = ''
        for v in versions:
            name = prefix + v
            dir = '{DIR}/apps/Katana/plugins/{NAME}'.format(DIR=self.rootDir, NAME=name)
            if os.path.exists(dir):
                self.getPull(dir)
            else:
                self.getClone(dir, 'Katana-{}-plugins'.format(v))


    #---------------------------------------------------------------------------
    # MAYA
    #---------------------------------------------------------------------------
    def Maya(self, versions):
        # VERSION
        if not versions:
            versions = VersionMap['Maya']

        DebugPrint('Select Team', '')
        teamList = ['toolkits', 'global', 'animation', 'asset', 'layout', 'lighting', 'matchmove', 'rigging']
        subPrint(teamList)
        t = raw_input('>> ')
        if t:
            selection = [teamList[int(t)]]
        else:
            selection = teamList

        _toolkits = False
        for v in versions:
            prefix = 'Maya-%s-' % v
            if self.user == 'plumber':
                prefix = ''
            for team in selection:
                name = prefix + team
                dir  = ''
                if team == 'global':
                    dir = '{DIR}/apps/Maya/versions/{VER}/{NAME}'.format(DIR=self.rootDir, VER=v, NAME=name)
                elif team == 'toolkits':
                    _toolkits = True
                else:
                    dir = '{DIR}/apps/Maya/versions/{VER}/team/{NAME}'.format(DIR=self.rootDir, VER=v, NAME=name)
                # Pull or Clone
                if dir:
                    if os.path.exists(dir):
                        self.getPull(dir)
                    else:
                        self.getClone(dir, 'Maya-{VER}-{NAME}'.format(VER=v, NAME=team))

        if _toolkits:
            dir = '{}/apps/Maya/toolkits'.format(self.rootDir)
            if not os.path.exists(dir):
                os.makedirs(dir)
            for n in os.listdir(dir):
                p = os.path.join(dir, n)
                if os.path.isdir(p):
                    self.getPull(p)


    #---------------------------------------------------------------------------
    # RezDCC
    #---------------------------------------------------------------------------
    def RezDCC(self, versions):
        dir = '{}/apps/rez/RezDCC'.format(self.rootDir)
        if os.path.exists(dir):
            self.getPull(dir)
        else:
            self.getClone(dir, 'RezDCC')


#-------------------------------------------------------------------------------
#
#   MAIN
#
#-------------------------------------------------------------------------------
if __name__ == '__main__':

    DebugPrint('Description', 'select process. \n\tex> pull, push')
    proc = raw_input('>> Process : ')
    if proc == 'pull':
        GitPull()
    elif proc == 'push':
        DebugPrint('warning', 'Not support yet!')
    else:
        DebugPrint('error', 'Not support process.')
