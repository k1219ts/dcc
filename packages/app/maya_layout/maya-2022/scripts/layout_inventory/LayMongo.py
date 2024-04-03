# -*- coding: utf-8 -*-
####################################################
########## coding by RND youkyoung.kim #############
####################################################
import os
import getpass
from pymongo import MongoClient
import dxConfig
import datetime
import LayInventory
from LaySpool import LayInvenSpool

class LayInventorydb():
    def __init__(self):
        self.basePath = '/assetlib/3D/layout'

    def dbImport(self, dbsend = {}):
        client = MongoClient(dxConfig.getConf("DB_IP"))
        db = client["inventory"]
        coll = db["assets"]
        titlename = dbsend['name']
        project = dbsend['project']
        category = dbsend['category']
        tags = dbsend['tags']
        thumbfile = dbsend['thumbnail']

        filename = dbsend['org_file']
        makepath = "%s/%s/%s/%s" % (self.basePath, project,
                                    category, filename)
        makefile = "%s/%s" % (makepath, dbsend['org_name'])

        thumbsplit = thumbfile.split('/')[-1]
        makethumb = "%s/%s" % (makepath, thumbsplit)

        previewre = thumbsplit.replace('_thumb', '_preview')
        makepreview = "%s/%s" % (makepath, previewre)
        texpath = "%s/texture" % makepath
        orgpath = dbsend['org_path']
        orgname = dbsend['org_name']
        org = os.path.join(orgpath, orgname)

        dbRecord = {'name' : titlename, 'project' : project,
                    'category': category, 'tags' : tags,
                    'files' : {}}
        dbRecord['type'] = 'PRV_SRC'
        dbRecord['description'] = ''
        dbRecord['enabled'] = False
        dbRecord['time'] = datetime.datetime.now().isoformat()
        dbRecord['user'] = getpass.getuser()
        dbRecord['files']['org'] = makefile
        dbRecord['files']['preview'] = makepreview
        dbRecord['files']['thumbnail'] = makethumb
        dbRecord['files']['tex'] = []

        if dbsend['texture']:
            for i in dbsend['texture']:
                pathtemp = os.path.dirname(i)
                tex = i.replace(orgpath, texpath)
                dbRecord['files']['tex'].append(tex)
        textureorg = dbsend['texture']
        texturetarget = dbRecord['files']['tex']

        # db check : name, project, category duplicated check
        checkDB = {}
        checkDB['name'] = titlename
        checkDB['project'] = project
        checkDB['category'] = category
        isDBExists = coll.find(checkDB).limit(1)

        if not os.path.exists(makefile):
            if isDBExists.count():
                ## if duplicated only tag editable
                LayInventory.messageBox(">> Warning : category, title, name duplicate!!",
                                        "Current scene exists inventory!!", 'warning', ['OK'])
            else:
                result_id = coll.insert_one(dbRecord)
                # tractor
                spooldb = {}
                spooldb = {'org' : org, 'makepath' : makepath, 'makefile' : makefile, 'makethumb' : makethumb,
                           'thumbfile' : thumbfile, 'makepreview' : makepreview, 'texpath' : texpath,
                           'textureorg' : textureorg, 'texturetarget' : texturetarget, 'result_id' : result_id}
                sendok = 1
                LaySpool = LayInvenSpool()
                LaySpool.spoolSet(spooldb, sendok)

