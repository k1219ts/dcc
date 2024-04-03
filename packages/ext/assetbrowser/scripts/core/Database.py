#coding:utf-8
__author__ = "daeseok.chae @ Dexter Studio"
__date__ = "2019.11.14"
__comment__ = '''
    DB Setup
'''

# base module
import getpass
import datetime
import sys
import os

# sys.path.append("/backstage/libs/python_lib")

# using Mongo DB
from pymongo import MongoClient
import dxConfig
gDBIP = dxConfig.getConf("DB_IP")
client = MongoClient(gDBIP)
gDB = client["ASSETLIB"]

'''
* Requires
    - Category
        1. Add Category(categoryName)
        2. Add SubCategory(parentCategory, subCategory)
        3. Get CategoryList(categoryName, subCategory='')
    - Item
        1. Add Item(filepath)
        2. Move Category(moveCategory, objId)
        3. Get Items(category)
        4. Set Tags(tagList)
        5. Set Comment(commentMsg)
        6. Add Reply(replyMsg)
    - Search
        1. Find Tag
        2. Find Name
'''

def AddCategory(categoryName):
    if not gDB.category.find_one({'category':categoryName}):
        gDB.category.insert_one({'category':categoryName})

def AddSubCategory(mainCategory, subCategoryName):
    # @param mainCategory : _id or string
    parentId = mainCategory
    # print(type(mainCategory))
    if type(mainCategory) == unicode:
        item = gDB.category.find_one({'category':mainCategory})
        parentId = item['_id']
    # print(parentId)
    # gDB.category.insert_one({"subCategory": subCategoryName, "parentId": parentId})
    if not gDB.category.find_one({"subCategory" : subCategoryName}):
        gDB.category.insert_one({"subCategory" : subCategoryName, "parentId" : parentId})

def AddchildCategory(mainCategory, subCategory, childCategory):
    parentId = mainCategory

    if type(mainCategory) == unicode:
        item_category = gDB.category.find_one({'category':mainCategory})
        categoryId = item_category['_id']

        item_sub = gDB.category.find_one({'subCategory':subCategory})
        sub = item_sub['_id']

        item_child={}
        item_child['childCategory'] = childCategory
        item_child['categoryId'] =categoryId
        item_child['subID'] =sub

    if not gDB.category.find_one({'childCategory': childCategory}):
        gDB.category.insert_one(item_child)


def EditCategory(old,new):
    if not gDB.category.find_one({'category':new}):
        gDB.category.update_one({'category':old},{ '$set' : {'category': new}})

    gDB.item.update_one({'category': old}, {"$set": {'category': new}})

def EditSubCategory(currentSub,new,currentParent):


    if not gDB.category.find_one({'subCategory': new}):
        gDB.category.update_one({'subCategory':currentSub}, {"$set":{'subCategory': new}})

    if currentParent == 'Texture':
        item= gDB.source.find({'subCategory': currentSub})
        for i in item:
            if i:
                gDB.source.update_one({'subCategory': currentSub}, {"$set":{'subCategory': new}})

    else:
        item= gDB.item.find({'subCategory': currentSub})
        for i in item:
            if i:
                gDB.item.update_one({'subCategory': currentSub}, {"$set":{'subCategory': new}})


def EditChildCategory(currentSub,new,category):
    child = currentSub
    if not gDB.category.find_one({'childCategory': new}):
        gDB.category.update_one({'childCategory':child}, {"$set":{'childCategory': new}})

    if category == 'Texture':
        item= gDB.source.find({'childCategory': child})
        for i in item:
            if i:
                gDB.source.update_one({'childCategory': child}, {"$set":{'childCategory': new}})

    else:
        item= gDB.item.find({'childCategory': child})
        for i in item:
            if i:
                gDB.item.update_one({'childCategory': child}, {"$set":{'childCategory': new}})


def DeleteSubCategory(currentSub,currentParent):
    if currentParent == 'Texture':
        item = gDB.source.find({'subCategory': currentSub})
        for i in item:
            if i:
                i['category'] = 'Texture'
                i['subCategory'] = 'Updated Source'
                gDB.source.update_one({'subCategory': currentSub}, {"$set": i})
            else:
                pass
    else:
        item = gDB.item.find({'subCategory': currentSub})
        for i in item:
            if i:
                i['category'] = 'Default'
                i['subCategory'] = 'Updated Asset'
                gDB.item.update_one({'subCategory': currentSub}, {"$set": i})
            else:
                pass
    gDB.category.delete_one({'subCategory': currentSub})


def DeleteChildCategory(currentChild,sub,category):
    if category == 'Texture':
        item = gDB.source.find({'subCategory': currentChild})
        for i in item:
            if i:
                i['category'] = 'Default'
                i['subCategory'] = 'Updated Source'
                gDB.source.update_one({'subCategory': currentChild}, {"$set": i})
            else:
                pass
    # else:
    #     item = gDB.item.find({'subCategory': currentChild})
    #     for i in item:
    #         if i:
    #             i['category'] = 'Default'
    #             i['subCategory'] = 'Updated Asset'
    #             gDB.item.update_one({'subCategory': currentChild}, {"$set": i})
    #         else:
    #             pass
    gDB.category.delete_one({'childCategory': currentChild})


def DeleteDocument(itemName, category): #delete Item
    if category == 'Texture':
        gDB.source.delete_one({'name': itemName})
    else:
        gDB.item.delete_one({'name': itemName})


def GetCategoryList():
    cursor = gDB.category.find({'category':{'$exists':True}})
    categoryDict = {}
    for itr in cursor:
        # check overlap Key
        if not categoryDict.get(itr['category']):
            categoryDict[itr['category']] = list()

        # parent object Id based find child object
        subCursor = gDB.category.find({"parentId":itr['_id']})
        for subItr in subCursor:
            categoryDict[itr['category']].append(subItr['subCategory'])
    return categoryDict

def MoveCategory(objId, categoryName, subCategory=''):
    item = gDB.item.find_one({'_id':objId})
    item['category'] = categoryName
    if subCategory:
        item['subCategory'] = subCategory
    gDB.item.update_one({'_id':objId}, {"$set":item})

def GetItems(category, subCategory=''):
    queryDict = {'category':category, 'subCategory':subCategory}
    items = gDB.item.find(queryDict)
    return items

def MoveCategoryTest(objId, categoryName, subCategory=''):
    if categoryName == 'Texture':
        for target in gDB.source.find({'_id': objId}):
            target['category'] = categoryName
            # print(item['category'])
            if subCategory:
                target['subCategory'] = subCategory
            gDB.source.update_one({'_id': objId}, {"$set": target})
    else:
        for target in gDB.item.find({'_id': objId}):
            target['category'] = categoryName
            # print(item['category'])
            if subCategory:
                target['subCategory'] = subCategory
            gDB.item.update_one({'_id': objId}, {"$set": target})

def MoveChildCategory(objId, categoryName, subCategory, child):
    if categoryName == 'Texture':
        for target in gDB.source.find({'_id': objId}):
            target['category'] = categoryName
            # print(item['category'])
            if subCategory:
                target['subCategory'] = subCategory
            if child:
                target['childCategory'] = child
            gDB.source.update_one({'_id': objId}, {"$set": target})
    else:
        for target in gDB.item.find({'_id': objId}):
            target['category'] = categoryName
            # print(item['category'])
            if subCategory:
                target['subCategory'] = subCategory
            if child:
                target['childCategory'] = child
            gDB.item.update_one({'_id': objId}, {"$set": target})

def AddBookmarkItem(userName, dbData,getCategory,itemName):
    data = {'category': getCategory,
              'ID': dbData,
            'name': itemName}
    item = gDB.user_config.find_one({'user': userName})
    if item:
        item['bookmark'].append(data)
        gDB.user_config.update_one({'_id':item['_id']}, {"$set":item})
    else:
        item = {'user':userName, 'bookmark':[data]}
        gDB.user_config.insert_one(item)


def GetBookmarkList(userName):
    queryDict = {'user': userName}
    bookmarkList = []
    cursor = gDB.user_config.find(queryDict)
    for itr in cursor:
        if bookmarkList == 0:
            pass
        else:
            for i in itr['bookmark']:
                bookmarkList.append(i)
    return bookmarkList

def UpdateBookmarkList(userName,deletedNameList):
    newList = []
    item = gDB.user_config.find_one({'user': userName})
    for i in item['bookmark']:
        if i['name'] in deletedNameList:
            pass
        else:
            newList.append(i)
    item = gDB.user_config.find_one({'user': userName})
    gDB.user_config.remove(item)

    for n in newList:
        item = gDB.user_config.find_one({'user': userName})
        if item:
            item['bookmark'].append(n)
            gDB.user_config.update_one({'user': userName}, {"$set": item})
        else:
            item = {'user': userName, 'bookmark': [n]}
            gDB.user_config.insert_one(item)


def AddTag(itemName, tagName,category):
    if category == 'Texture':
        item = gDB.source.find_one({'name': itemName})
        list = tagName.split(',')
        item['tag'] =[]
        for i in list:
            item['tag'].append(i)
        gDB.source.update_one({'_id': item['_id']}, {"$set": item})

    else:
        item = gDB.item.find_one({'name': itemName})
        list = tagName.split(',')
        item['tag'] =[]
        for i in list:
            item['tag'].append(i)
        gDB.item.update_one({'_id': item['_id']}, {"$set": item})


def GetSCItems(category, subCategory=''):
    queryDict = {'category':category, 'subCategory':subCategory}
    items = gDB.source.find(queryDict).sort('name', 1)
    return items


def AddSCItem(filePath):
    if os.path.isfile(filePath):
        from PIL import Image, ImageChops, ImageOps
        sourceName = os.path.basename(filePath).split('.')[0]
        dirPath = os.path.dirname(filePath)
        thumbCacheDir = '/dexter/Cache_DATA/ASSET/trash/thumbnailCache'
        if not os.path.exists(thumbCacheDir):
            os.makedirs(thumbCacheDir)
        thumbCacheFile = '%s/%s' % (thumbCacheDir, sourceName + '_preview.png')
        size = (320, 240)
        image = Image.open(filePath)
        image.thumbnail(size, Image.ANTIALIAS)
        image_size = image.size
        alpha = image.convert('RGBA')
        thumb = alpha.crop((0, 0, size[0], size[1]))
        offset_x = max((size[0] - image_size[0]) / 2, 0)
        offset_y = max((size[1] - image_size[1]) / 2, 0)
        thumb = ImageChops.offset(thumb, offset_x, offset_y)
        thumb = ImageOps.fit(image, size, Image.ANTIALIAS, (0.5, 0.5))
        thumb.save(thumbCacheFile)
        os.system('echo dexter2019 | su render -c "%s"' % "mv %s %s/" % (thumbCacheFile, dirPath))
        previewFile = '%s/%s' % (dirPath, sourceName + '_preview.png')
        filePath = dirPath
        tagName = ''

    elif os.path.isdir(filePath):
        splitDir = filePath.split('/')  #['', 'assetlib', 'Texture', 'RealDisplacement', 'AUTUMN-LEAVES-01']
        sourceName = splitDir[-1]
        tagName = splitDir[3].lower()
        previewFile = os.path.join(filePath, "preview.jpg")
        if not os.path.exists(previewFile):
            previewFile = os.path.join(filePath, "preview.png")

    else:
        pass

    itemDict = {'category':'Texture',
                'subCategory': 'Updated Source',
                'tag': [tagName],
                'comment' : '',
                'reply':[{'user':getpass.getuser(), 'comment':'add item', 'time':datetime.datetime.now().isoformat()}],
                'name' : '',
                'files': {}
                }
    if not '/assetlib/Texture' in filePath:
        return "this location isn't Texture Source"

    itemDict['files']['filePath'] = filePath
    itemDict['files']['preview'] = previewFile
    # set asset name
    itemDict['name'] = sourceName
    if not gDB.source.find_one({'name':sourceName}):
        gDB.source.insert_one(itemDict)

def AddDeleteItem(itemName,userName):
    itemDict = {}
    itemDict['name'] = itemName
    itemDict['reply'] = [{'user': userName, 'comment': 'delete item', 'time': datetime.datetime.now().isoformat()}]
    if not gDB.delete.find_one({'name': itemName}):
        gDB.delete.insert_one(itemDict)


def AddItem(filepath):
    '''
        # special : this function using nautilus script. so try nautilus.
        @param
            - filepath : insert files under this path.
    '''
    itemDict = {'category':'unknown',
                'subCategory': 'unknown',
                'tag': [],
                'comment' : '',
                'reply':[{'user':getpass.getuser(), 'comment':'add item', 'time':datetime.datetime.now().isoformat()}],
                'name' : '',
                'files': {}
    }

    # find files insert 'files' paths.
    if not '/assetlib/_3d/asset' in filepath:
        return "this location isn't assetlib"

    assetNameIndexKey = 'asset'
    if "branch" in filepath:
        itemDict['tag'].append('branch')
        assetNameIndexKey = 'branch'

    splitFilePath = filepath.split('/')
    assetIndex = splitFilePath.index(assetNameIndexKey)
    assetName = splitFilePath[assetIndex + 1]

    dirpath = "/".join(splitFilePath[:assetIndex + 2])

    filepath = os.path.join(dirpath, "%s.usd" % assetName)
    previewFile = os.path.join(dirpath, "preview.jpg")

    itemDict['files']['usdfile'] = filepath
    itemDict['files']['preview'] = previewFile

    renderUsdPath = filepath

    for i in os.listdir(dirpath):
        if 'clip' in i:
            renderUsdPath = dirpath + '/model/model.usd'

    customPreviewFile = previewFile.replace('.jpg', '.####.jpg')
    print(renderUsdPath)

    # command = "/backstage/dcc/DCC rez-env usdtoolkit usd_core-20.08 -- usdrecorder -w 320 -ht 240 --purposes render --renderer Prman --outputImagePath {OUTPUTFILE} {USDFILE}".format(USDFILE=renderUsdPath, OUTPUTFILE=customPreviewFile)
    # command = "/WORK_DATA/Develop/dcc/DCC dev rez-env usdtoolkit usd_core-20.08 -- usdrecorder -w 320 -ht 240 --purposes render --renderer Prman --outputImagePath {OUTPUTFILE} {USDFILE}".format(USDFILE=renderUsdPath, OUTPUTFILE=customPreviewFile)
    command = "/backstage/dcc/DCC rez-env usdtoolkit -- usdrecorder -w 320 -ht 240 --purposes render --renderer Prman --outputImagePath {OUTPUTFILE} {USDFILE}".format(
        USDFILE=renderUsdPath, OUTPUTFILE=customPreviewFile)

    if getpass.getuser() == 'render':
        ret = os.system(command)
    else:
        ret = os.system('echo dexter2019 | su render -c "%s"' % command)

    if ret == 0:
        renameCmd = 'echo dexter2019 | su render -c "mv %s %s"' % (
        customPreviewFile.replace('.####.', '.0000.'), previewFile)
        if getpass.getuser() == 'render':
            renameCmd = 'mv %s %s' % (customPreviewFile.replace('.####.', '.0000.'), previewFile)
        ret = os.system(renameCmd)
        if ret == 0:
            print("# Success Make Preview")
        else:
            return "# Failed Rename :"
    else:
        return "# Failed "

    # set asset name
    itemDict['name'] = assetName
    if not gDB.item.find_one({'name':assetName}):
        gDB.item.insert_one(itemDict)
