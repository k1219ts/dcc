# -*- coding: utf-8 -*-
####################################################
########## coding by RND youkyoung.kim #############
####################################################
import os, site, dxConfig
import getpass
import LayInventory
# tractor api setup
site.addsitedir(dxConfig.getConf("TRACTOR_API"))
import tractor.api.author as author

class LayInvenSpool():
    def __init__(self):
        self.enableDB = '/netapp/backstage/pub/bin/inventory/enableDBRecord.py'
        self.ffmpeg = '/netapp/backstage/pub/apps/ffmpeg_for_exr/bin/ffmpeg_with_env'
        self.gifCreate = '/netapp/backstage/pub/bin/inventory/gifCreate'

    def spoolSet(self, dbsend = {}, sendok=None):
        print dbsend
        self.result_id = dbsend['result_id']
        self.tractorSet()
        self.makePath(dbsend['org'], dbsend['makepath'], dbsend['makefile'])
        if dbsend['makegif']:
            self.makeMovThumb(dbsend['org'], dbsend['makethumb'])
            self.makeGif(dbsend['org'], dbsend['makegif'])
        else:
            self.makeThumb(dbsend['thumbfile'], dbsend['makethumb'], dbsend['makepreview'])
            self.makeTexture(dbsend['texpath'], dbsend['textureorg'], dbsend['texturetarget'])
        self.spoolJob()
        if sendok:
            LayInventory.messageBox(">> Inventory Upload Success !!",
                                    "Tractor Spool Success!!", 'information', ['OK'])

    def tractorSet(self):
        ## tractor spool send
        self.svc = 'Cache||USER'
        titles = '(PrevInventory)'
        self.job = author.Job(title = str(titles))
        # self.job.envkey = ['maya2017']
        self.job.service = self.svc
        self.job.priority = 1000
        self.job.tier = 'user'
        self.job.projects = ['user']

        author.setEngineClientParam(hostname='10.0.0.25',
                                    port=dxConfig.getConf("TRACTOR_PORT"),
                                    user=getpass.getuser(),
                                    debug=True)

        self.jobtask = author.Task(title = "layout inventory")
        self.jobtask.serialsubtasks = 1
        self.job.addChild(self.jobtask)

    def makePath(self, org=None, makepath=None, makefile=None):
        if not os.path.exists(makepath):
        # original file directory create and file copy
            orgtask = author.Task(title = "directory create command")
            orgtask.addCommand(author.Command(argv = ['install', '-d',
                                                      '-m', '755', makepath],
                                                       service = self.svc))
            orgtask.addCommand(author.Command(argv=["/bin/bash", "-c",
                                                     'cp "%s" "%s"' % (org, makefile)],
                                                      service=self.svc))
            self.jobtask.addChild(orgtask)

    def makeThumb(self, thumbfile=None, makethumb=None, makepreview=None):
        if not os.path.exists(makethumb):
        ## thumnail file copy
            thumtask = author.Task(title = "thumnail file copy command",
                                   argv=["%s -i %s -s 320x240 %s"
                                         % (self.ffmpeg, thumbfile, makethumb)],
                                   service=self.svc)
            self.jobtask.addChild(thumtask)

            previewtask = author.Task(title="preview file copy command",
                                      argv=["/bin/bash", "-c",
                                      'cp "%s" "%s"' % (thumbfile, makepreview)],
                                      service=self.svc)
            self.jobtask.addChild(previewtask)

    def makeTexture(self, texpath=None, textureorg=[], texturetarget=[]):
        if textureorg:
            ## texture directory create and file copy
            textask = author.Task(title="texture create directory command")
            textask.addCommand(author.Command(argv = ['install', '-d', '-m', '755', texpath],
                                              service=self.svc))
            self.jobtask.addChild(textask)

            texfiletask = author.Task(title="texture file copy command")
            texlen = len(textureorg)
            for i in range(texlen):
                texorg = textureorg[i]
                texcopy = texturetarget[i]
                texfiletask.addCommand(author.Command(argv=["/bin/bash", "-c",
                                                 'cp "%s" "%s"' % (texorg, texcopy)],
                                                  service=self.svc))
            self.jobtask.addChild(texfiletask)

    def makeMovThumb(self, org=None, makethumb=None):
        thumtask = author.Task(title = "thumnail file create command",
                               argv=[self.ffmpeg,'-i', org, '-vframes',
                               '1','-vf', 'scale=320x240',
                               '-ss', str(48 / 24.0), '-y', makethumb], service=self.svc)
        self.jobtask.addChild(thumtask)

    def makeGif(self, org=None, makegif=None):
        gifTask = author.Task(title="gif create command",
                              argv=[self.gifCreate, org, makegif], service=self.svc)
        self.jobtask.addChild(gifTask)

    def spoolJob(self):
        dbtask = author.Task(title = "db record command",
                             argv=['python', self.enableDB,
                                   'inventory', 'assets',
                                   self.result_id.inserted_id], service=self.svc)
        self.jobtask.addChild(dbtask)


        self.job.spool()
        author.closeEngineClient()


