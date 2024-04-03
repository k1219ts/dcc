# -*- coding: utf-8 -*-
####################################################
########## coding by RND youkyoung.kim #############
####################################################
import sys, os, site
import getpass
import dxConfig
import unicodedata
from pymongo import MongoClient
import datetime
from dxname import tag_parser
from LaySpool import LayInvenSpool

class LayInventorydb():
    def __init__(self):
        self.orgPath = '/dexter/Cache_DATA/LAY/003_Reference/01_mov'
        self.targetPath = '/assetlib/3D/layout/reference'
        self.movList = ['.mov', '.avi', '.mp4', '.flv', '.mkv', '.gif', '.m4v', '.divx']

    def txtConvert(self, org=None):
        uniconvert = unicode(org, 'utf-8') # str > unicode str
        imsi = unicodedata.normalize('NFC', uniconvert)  # unicode mac kr > unicode linux kr
        return imsi.encode('utf-8')

    def dbImport(self, dbsend = []):#mov, avi, mp4, flv
        total = len(dbsend) - 1
        for i, org in enumerate(dbsend):
            org = self.txtConvert(org.encode('utf-8'))
            client = MongoClient(dxConfig.getConf("DB_IP"))
            db = client["inventory"]
            coll = db["assets"]

            tagpath = []
            tagfile = []
            tags = []
            path = os.path.dirname(org)
            file = os.path.basename(org)
            filesplit = os.path.splitext(file)
            filename = filesplit[0]
            movck = filesplit[1]
            replacepath = path.replace(self.orgPath,'')
            print movck
            if movck in self.movList:
                project = 'reference'
                category = replacepath.split('/')[1]

                mkpath = path.replace(self.orgPath, self.targetPath)
                makepath = '%s/%s' % (mkpath, filename)
                makefile = '%s/%s' % (makepath, file)

                thumbfile = '%s_thumb.jpg' %filename
                makethumb = '%s/%s' % (makepath, thumbfile)

                giffile = '%s_gif.gif' %filename
                makegif = '%s/%s' % (makepath, giffile)
                tagpath = tag_parser.run(replacepath)
                tagfile = tag_parser.run(filename)
                tags = list(set(tagpath + tagfile))

                dbRecord = {'name' : filename, 'project' : project,
                            'category': category, 'tags' : tags,
                            'files' : {}}
                dbRecord['type'] = 'PRV_SRC'
                dbRecord['description'] = ''
                dbRecord['enabled'] = False
                dbRecord['time'] = datetime.datetime.now().isoformat()
                dbRecord['user'] = getpass.getuser()
                dbRecord['files']['org'] = makefile
                dbRecord['files']['preview'] = makefile
                dbRecord['files']['thumbnail'] = makethumb
                dbRecord['files']['gif'] = makegif
                print dbRecord

                checkDB = {}
                checkDB['name'] = filename
                checkDB['project'] = project
                checkDB['category'] = category
                isDBExists = coll.find(checkDB).limit(1)

                if not os.path.exists(makefile):
                    if not isDBExists.count():
                        result_id = coll.insert_one(dbRecord)
                        spooldb = {}
                        spooldb = {'org': org, 'makepath': makepath, 'makefile': makefile, 'makethumb': makethumb,
                                   'makegif' : makegif, 'result_id': result_id}
                        if total == i:
                            sendok = 1
                        else:
                            sendok = 0
                        LaySpool = LayInvenSpool()
                        LaySpool.spoolSet(spooldb, sendok)


