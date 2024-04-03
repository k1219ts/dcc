from pymodule import Qt
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore

import site
TractorRoot = '/netapp/backstage/pub/apps/tractor/linux/Tractor-2.2'
site.addsitedir('%s/lib/python2.7/site-packages' % TractorRoot)

import tractor.api.author as author

from ui_proxyPlateRender import Ui_Form

import nuke

import os, sys, datetime, getpass, shutil
from tactic_client_lib import TacticServerStub

server = TacticServerStub(login='taehyung.lee', password='dlxogud',
                          server='10.0.0.51', project='show53')

USER = getpass.getuser()


class ProxyPlateRender(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(ProxyPlateRender, self).__init__(parent)
        self.setWindowTitle('Proxy 2K Plate Render')
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        print USER
        if USER == "dongho.cha" or USER == "kwantae.kim":
            self.ui.btnDelete.setEnabled(True)
            self.ui.cmbMilestone.currentIndexChanged.connect(self.getShot)

        self.ui.btnDelete.clicked.connect(self.deleteShotDir)
        self.ui.btnPlateLoad.clicked.connect(self.loadPlate)
        self.ui.btnRender.clicked.connect(self.goRender)

    def getMilestone(self):
        # self.ui.cmdMilestone.clear()

        shot_exp = "@SOBJECT(sthpw/milestone['project_code','show53']['code','~','^%s'])" % 'MMV'
        info = server.eval(shot_exp)

        for i in info:
            self.ui.cmbMilestone.addItem(i['code'])

    def getShot(self):
        self.ui.listShot.clear()
        MILESTONE = self.ui.cmbMilestone.currentText()

        shot_exp = "@SOBJECT(sthpw/task['project_code','show53']['milestone_code','%s'])" % MILESTONE
        info = server.eval(shot_exp)
        shots = [i['extra_code'] for i in info]

        tmp = []
        for i in shots:
            # print i
            if not i in tmp:
                self.ui.listShot.addItem(str(i))
                tmp.append(i)

    def deleteShotDir(self):

        shots = []
        items = self.ui.listShot.selectedItems()
        for i in range(len(items)):
            shots.append(str(self.ui.listShot.selectedItems()[i].text()))

        #print shots

        if shots:
            result = QtWidgets.QMessageBox.question(self, "infomation", "delete Shots?",
                                                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                    QtWidgets.QMessageBox.Yes)

            if result == QtWidgets.QMessageBox.Yes:
                for shot in shots:
                    seq = shot.split('_')[0]
                    platePath = '/show/prs/shot/_source/MMV_Cache/%s/%s' % (seq, shot)

                    if os.path.isdir(platePath):
                        shutil.rmtree(platePath)
                        print "delete!", platePath
                    else:
                        print "nothing!", platePath

            else:
                print "Cancel Delete"
        else:
            QtWidgets.QMessageBox.information(self, "error", "select shot first.")

    def loadPlate(self):

        shots = []
        items = self.ui.listShot.selectedItems()
        for i in range(len(items)):
            shots.append(str(self.ui.listShot.selectedItems()[i].text()))

        print shots

        if shots:
            for shot in shots:
                seq = shot.split('_')[0]
                platePath = '/show/prs/shot/%s/%s/plates' % (seq, shot)
                # print platePath

                for type in os.listdir(platePath):
                    typePath = os.path.join(platePath, type)
                    fver = sorted(os.listdir(typePath))[-1]
                    plateAbsPath = os.path.join(typePath, fver)
                    nuke.tcl('drop', str(plateAbsPath))

            for i in nuke.allNodes('Read'):
                if i.error():
                    nuke.delete(i)
                else:
                    re = nuke.nodes.Reformat()
                    re['type'].setValue('scale')
                    re['scale'].setValue(0.5)
                    re.setInput(0, i)

                    elements = i['file'].value().split('/')

                    cache_path = '/show/prs/shot/_source/MMV_Cache/' + '/'.join(elements[4:])

                    # print cache_path

                    w = nuke.nodes.Write()
                    w['file'].setValue(cache_path)
                    w['file_type'].setValue('exr')
                    w.setInput(0, re)

        else:
            QtWidgets.QMessageBox.information(self, "error", "select shot first.")

    def goRender(self):

        shots = nuke.allNodes('Write')

        if shots:

            print shots

            result = QtWidgets.QMessageBox.question(self, "infomation", "send renderFarm?",
                                                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                    QtWidgets.QMessageBox.Yes)

            if result == QtWidgets.QMessageBox.Yes:
                print "Render start"

                MILESTONE = self.ui.cmbMilestone.currentText()

                sTime = str(datetime.datetime.now()).split('.')[0].replace(':', '').replace('.', '').replace('-', '').replace(' ', '.')
                fName = '/netapp/dexter/show/prs/shot/_source/MMV_Cache/nuke/' + MILESTONE + '_' + USER + '_' + sTime + '.nk'
                nuke.scriptSaveAs(fName)

                for i in nuke.allNodes('Write'):

                    nodeName = i['name'].value()
                    shot = i['file'].value().split('/')[-5]
                    mkDir = '/'.join(i['file'].value().split('/')[:-1])

                    minFrame = i.firstFrame()
                    maxFrame = i.lastFrame()

                    print nodeName, minFrame, maxFrame, shot, mkDir

                    if not os.path.isdir(mkDir):
                        os.makedirs(mkDir)

                        job = author.Job()
                        job.title = '(MMV Cache) %s' % (shot)
                        job.envkey = ['nuke']
                        job.service = 'nuke'
                        job.tier = 'cache'
                        job.projects = ['comp']
                        job.priority = 1

                        MainJobTask = author.Task(title='job')
                        # MainJobTask.serialsubtasks = 1

                        for frame in range(minFrame, maxFrame, 3):
                            # print frame
                            frameExportTask = author.Task(title='%s %s' % (frame, frame + 3))
                            command = ["/usr/local/Nuke10.0v4/Nuke10.0", "-t"]  # t : terminal
                            startFrame = frame
                            endFrame = frame + 3
                            if maxFrame < endFrame:
                                endFrame = maxFrame
                            command += ["-F", "%d,%d" % (startFrame, endFrame)]  # f frame
                            command += ["-X", "%s" % (nodeName)]  #
                            command += [fName]

                            # print command

                            frameExportTask.addCommand(
                                author.Command(argv=command, envkey=["nuke"], service="nuke", tags=["nuke10"]))
                            MainJobTask.addChild(frameExportTask)

                            job.addChild(MainJobTask)

                        author.setEngineClientParam(hostname="10.0.0.106", port=80, user=getpass.getuser(), debug=True)

                        job.spool()
                        author.closeEngineClient()
                    else:
                        print shot, "2K plate already!"

            else:
                print "Cancel"
        else:
            QtWidgets.QMessageBox.information(self, "error", "run plateLoad first.")


#if __name__ == '__main__':
    # app = QtWidgets.QApplication(sys.argv)
    #mw = ProxyPlateRender()
    #mw.show()
    #mw.getMilestone()
    #mw.getShot()
    # sys.exit(app.exec_())
