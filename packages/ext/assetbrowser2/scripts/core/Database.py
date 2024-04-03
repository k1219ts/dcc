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

sys.path.append("/backstage/libs/python_lib")

# using Mongo DB
import pymongo
from pymongo import MongoClient
from pymongo.collection import ReturnDocument
if sys.platform == "linux2":
    import dxConfig
    gDBIP = dxConfig.getConf("DB_IP") # 10.0.0.12:27017
else:
    gDBIP = "10.0.0.12"

client = MongoClient(gDBIP)

gDBNAME = "ASSETLIB"
gDB = client[gDBNAME]

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

def set_database(database):
    global gDB
    global gDBNAME
    gDBNAME = database
    gDB = client[database]

def get_collection(category):
    if category == "Texture":
        return gDB.source
    else:
        return gDB.item

def AddCategory(categoryName):
    from libs.customError import CustomError
    filterdict = {"category": categoryName}
    if gDB.category.find_one(filterdict):
        raise CustomError("It already exists. reason: {}".format(categoryName))

    gDB.category.insert_one(filterdict)

def AddSubCategory(mainCategory, subCategoryName):
    from libs.customError import CustomError
    from libs.utils import is_unicode
    # @param mainCategory : _id or string
    parentId = mainCategory
    if is_unicode(mainCategory):
        item = gDB.category.find_one({"category": mainCategory})
        parentId = item["_id"]

    filterdict = {"subCategory": subCategoryName, "parentId": parentId}
    if gDB.category.find_one(filterdict):
        raise CustomError("It already exists. reason: {}".format(subCategoryName))

    gDB.category.insert_one(filterdict)

# TODO: 체크
def AddchildCategory(mainCategory, subCategory, childCategory):
    from libs.utils import is_unicode
    parentId = mainCategory

    if is_unicode(mainCategory):
        item_category = gDB.category.find_one({'category':mainCategory})
        categoryId = item_category['_id']

        item_sub = gDB.category.find_one({'subCategory':subCategory})
        sub = item_sub['_id']

        item_child = {}
        item_child['childCategory'] = childCategory
        item_child['categoryId'] = categoryId
        item_child['subID'] = sub

    if not gDB.category.find_one({'childCategory': childCategory}):
        gDB.category.insert_one(item_child)

# TODO: gDB.source 추가할 것
def EditCategory(old, new):
    """카테고리를 수정합니다.

    Args:
        old (str): 카테고리명
        new (str): 변경할 카테고리명

    """
    if not gDB.category.find_one({'category': new}):
        gDB.category.update_one({'category': old}, {'$set' : {'category': new}})

    gDB.item.update_one({'category': old}, {"$set": {'category': new}})

def EditSubCategory(currentSub, new, currentParent):
    """서브 카테고리를 수정합니다.

    Args:
        currentSub (str): 서브 카테고리명
        new (str): 변경할 서브 카테고리명
        currentParent (str): 카테고리명

    """
    categoryItem = gDB.category.find_one({'category': currentParent})
    parentId = categoryItem['_id']

    if not gDB.category.find_one({'subCategory': new, 'parentId': parentId}):
        gDB.category.update_one({'subCategory': currentSub, 'parentId': parentId},
                                {"$set":{'subCategory': new}})

    this_collection = get_collection(currentParent)
    items = this_collection.find({'category': currentParent, 'subCategory': currentSub})
    for item in items:
        this_collection.update_one({'_id': item['_id']},
                                   {"$set":{'subCategory': new}})

# TODO: 체크
def EditChildCategory(currentSub, new, category):
    child = currentSub
    if not gDB.category.find_one({'childCategory': new}):
        gDB.category.update_one({'childCategory':child}, {"$set":{'childCategory': new}})

    this_collection = get_collection(category)
    item = this_collection.find({'childCategory': child})
    for i in item:
        if i:
            this_collection.update_one({'childCategory': child}, {"$set":{'childCategory': new}})

def DeleteSubCategory(currentSub, currentParent):
    categoryItem = gDB.category.find_one({"category": currentParent})
    parentId = categoryItem["_id"]

    if currentParent == "Texture":
        this_collection = gDB.source
        item_category = "Texture"
        item_sub_category = "Updated Source"
    else:
        this_collection = gDB.item
        item_category = "Default"
        item_sub_category = "Updated Asset"

    items = this_collection.find({"category": currentParent, "subCategory": currentSub})
    for item in items:
        item["category"] = item_category
        item["subCategory"] = item_sub_category
        this_collection.update_one({"_id": item["_id"]},
                                   {"$set": item})

    gDB.category.delete_one({"subCategory": currentSub, "parentId": parentId})

def DeleteCategory(category_name):
    categoryItem = gDB.category.find_one({"category": category_name})
    category_id = categoryItem["_id"]

    item_category = "Default"
    item_sub_category = "Updated Asset"

    subCategories = gDB.category.find({"parentId": category_id})
    for subCategory in subCategories:
        sub_category_id = subCategory["_id"]
        sub_category_name = subCategory["subCategory"]
        items = gDB.item.find({"category": category_name, "subCategory": sub_category_name})
        for item in items:
            item["category"] = item_category
            item["subCategory"] = item_sub_category
            gDB.item.update_one({"_id": item["_id"]}, {"$set": item})

        gDB.category.delete_one({"_id": sub_category_id})

    gDB.category.delete_one({"_id": category_id})

# TODO: 체크
def DeleteChildCategory(currentChild, sub, category):
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


def DeleteDocument(objId, category): # delete Item
    this_collection = get_collection(category)
    this_collection.delete_one({'_id': objId})

def GetCategoryList():
    cursor = gDB.category.find({'category':{'$exists':True}})
    categoryDict = {}
    for itr in cursor:
        # check overlap Key
        if itr['category'] not in categoryDict:
            categoryDict[itr['category']] = list()

        # parent object Id based find child object
        subCursor = gDB.category.find({"parentId":itr['_id']})
        for subItr in subCursor:
            categoryDict[itr['category']].append(subItr['subCategory'])
    return categoryDict

# def GetChildList():



def AddItem(filepath):
    '''
        # special : this function using nautilus script. so try nautilus.
        @param
            - filepath : insert files under this path.
    '''
    itemDict = {
        'category':'unknown',
        'subCategory': 'unknown',
        'tag': [],
        'comment' : '',
        'reply':[{'user':getpass.getuser(), 'comment':'add item', 'time':datetime.datetime.now().isoformat()}],
        'name' : '',
        'files': {}
    }

    # find files insert 'files' paths.
    if not '/assetlib/3D/asset' in filepath:
        return "this location isn't assetlib"

    assetNameIndexKey = 'asset'
    if "element" in filepath:
        itemDict['tag'].append('element')
        assetNameIndexKey = 'element'

    splitFilePath = filepath.split('/')
    assetIndex = splitFilePath.index(assetNameIndexKey)
    assetName = splitFilePath[assetIndex + 1]

    dirpath = "/".join(splitFilePath[:assetIndex + 2])
    filepath = os.path.join(dirpath, "%s.usd" % assetName)
    previewFile = os.path.join(dirpath, "preview.jpg")

    itemDict['files']['usdfile'] = filepath
    itemDict['files']['preview'] = previewFile


    customPreviewFile = previewFile.replace('.jpg', '.####.jpg')
    command = "/backstage/bin/DCC rez-env usdtoolkit-19.11 -- usdrecorder -w 320 -ht 240 --purposes render --renderer Prman --outputImagePath {OUTPUTFILE} {USDFILE}".format(USDFILE=filepath, OUTPUTFILE=customPreviewFile)
    fullCmd = 'echo dexter2019 | su render -c "%s"' % command
    ret = os.system('echo dexter2019 | su render -c "%s"' % command)
    if ret == 0:
        # print "# Success make preview"
        renameCmd = 'echo dexter2019 | su render -c "mv %s %s"' % (customPreviewFile.replace('.####.', '.0000.'), previewFile)
        ret = os.system(renameCmd)
        if ret == 0:
            print("# Success Make Preview {}".format(command))
        else:
            return "# Failed Rename :" + renameCmd
    else:
        return "# Failed " + fullCmd

    # set asset name
    itemDict['name'] = assetName
    if not gDB.item.find_one({'name':assetName}):
        gDB.item.insert_one(itemDict)

def AbstractAddItem(category, itemDict):
    from libs.customError import CustomError
    this_collection = get_collection(category)
    if this_collection.find_one({"name": itemDict["name"]}):
        raise CustomError("It already exists. reason: {}".format(itemDict["name"]))
    else:
        this_collection.insert_one(itemDict)

def MoveCategory(objId, categoryName, subCategory=''):
    item = gDB.item.find_one({'_id':objId})
    item['category'] = categoryName
    if subCategory:
        item['subCategory'] = subCategory
    gDB.item.update_one({'_id':objId}, {"$set":item})

def AbstractGetItems(collection, query):
    pipelines = list()
    pipelines.append(
        {"$match": query})
    pipelines.append(
        {"$lookup": {
            "from": "tag", "localField": "tags", "foreignField": "_id", "as": "tagObjects"}})
    pipelines.append(
        {"$sort":
            {"name": 1}})

    items = collection.aggregate(pipelines)
    return items

def GetItems(category, subCategory=''):
    this_collection = get_collection(category)
    query = {"category": category, "subCategory": subCategory}

    return AbstractGetItems(this_collection, query)

def GetItem(category, object_id):
    pipelines = list()
    pipelines.append(
        {"$match":
            {"_id": object_id}})
    pipelines.append(
        {"$lookup":
            {"from": "tag", "localField": "tags", "foreignField": "_id", "as": "tagObjects"}})

    # TODO: 싱글 아이템으로 반환되도록 정리 필요
    this_collection = get_collection(category)
    items = this_collection.aggregate(pipelines)

    results = [doc for doc in items]
    return results[0]

def MoveCategoryTest(objId, categoryName, subCategory=''):
    this_collection = get_collection(categoryName)

    for target in this_collection.find({'_id': objId}):
        target['category'] = categoryName
        if subCategory:
            target['subCategory'] = subCategory
        this_collection.update_one({'_id': objId}, {"$set": target})

def MoveChildCategory(objId, categoryName, subCategory, child):
    this_collection = get_collection(categoryName)

    for target in this_collection.find({'_id': objId}):
        target['category'] = categoryName
        if subCategory:
            target['subCategory'] = subCategory
        if child:
            target['childCategory'] = child
        this_collection.update_one({'_id': objId}, {"$set": target})

def AddBookmarkItem(userName, dbData, getCategory, itemName):
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

def AddUserInBookmark(userName):
    user = gDB.user_config.find_one({"user": userName})
    if not user:
        gDB.user_config.insert_one({"user": userName, "bookmark": []})

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

def UpdateBookmarkList(userName, bookmarks):
    return gDB.user_config.find_one_and_update(
        {"user": userName},
        {"$set":{"bookmark": bookmarks}},
        return_document=ReturnDocument.AFTER)

def UpdateBookmarkListOld(userName, deletedNameList):
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

def AddTag(objId, tagName, category):
    this_collection = get_collection(category)

    item = this_collection.find_one({'_id': objId})
    list = tagName.split(',')
    item['tag'] = []
    for i in list:
        item['tag'].append(i)
    this_collection.update_one({'_id': item['_id']}, {"$set": item})

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

    itemDict = {
        'category':'Texture',
        'subCategory': 'Updated Source',
        'tag': [tagName],
        'comment' : '',
        'reply':[
            {
                'user':getpass.getuser(),
                'comment': 'add item',
                'time':datetime.datetime.now().isoformat()
            }
        ],
        'name' : '',
        'files': {},
    }
    if not '/assetlib/Texture' in filePath:
        return "this location isn't Texture Source"

    itemDict['files']['filePath'] = filePath
    itemDict['files']['preview'] = previewFile
    # set asset name
    itemDict['name'] = sourceName
    if not gDB.source.find_one({'name':sourceName}):
        gDB.source.insert_one(itemDict)

def AddDeleteItem(itemName, userName):
    itemDict = {}
    itemDict['name'] = itemName
    itemDict['reply'] = [
        {
            'user': userName,
            'comment': 'delete item',
            'time': datetime.datetime.now().isoformat()
        }
    ]
    gDB.delete.insert_one(itemDict)

def EditItem(category, object_id, datas, reply):
    this_collection = get_collection(category)

    this_collection.find_one_and_update(
        {"_id": object_id},
        { 
            "$addToSet": { "reply": { "$each": reply } },
            "$set": datas
        },
        return_document=ReturnDocument.AFTER)
    return GetItem(category, object_id)

def AddSCItemTest(filePath):
    itemDict = {
        'category': 'Texture',
        'subCategory': 'Updated Source',
        'tag': [],
        'comment' : '',
        'reply':[
            {
                'user':getpass.getuser(),
                'comment': 'add item',
                'time':datetime.datetime.now().isoformat()
            }
        ],
        'name' : '',
        'files': {},
    }
    # if not '/assetlib/Texture' in filePath:
    #     return "this location isn't Texture Source"

    itemDict['files']['filePath'] = filePath
    itemDict['files']['preview'] = os.path.join(filePath, "preview.jpg")

    # set asset name
    sourceName = os.path.basename(filePath)
    itemDict['name'] = sourceName
    if not gDB.source.find_one({'name': sourceName}):
        gDB.source.insert_one(itemDict)

def Search(text):
    results = []
    for collection in [gDB.item, gDB.source]:
        or_op = list()
        or_op.append({"name": {"$regex": text, "$options": 'i'}})
        or_op.append({"tag": {"$elemMatch": {"$regex": text, "$options": 'i'}}})
        or_op.append({"comment": {"$regex": text, "$options": 'i'}})
        or_op.append({"status": {"$regex": text, "$options": 'i'}})
        or_op.append({"tagResults.name": {"$regex": text, "$options": 'i'}})

        pipelines = list()
        pipelines.append(
            {"$lookup":
                {"from": "tag", "localField": "tags", "foreignField": "_id", "as": "tagResults"}})
        pipelines.append(
            {"$unwind": {
                "path": "$tagResults", "preserveNullAndEmptyArrays": True}})
        pipelines.append(
            {"$match":
                {"$or": or_op}})
        pipelines.append(
            {"$project":
                {"tagResults": 0}})
        pipelines.append(
            {"$lookup":
                {"from": "tag", "localField": "tags", "foreignField": "_id", "as": "tagObjects"}})
        pipelines.append(
            {"$group": 
                {"_id": "$_id", "name": {"$addToSet": "$name"}, "matches": {"$push": "$$ROOT"}}}
        )
        pipelines.append(
            {"$sort": {"name": 1}}
        )

        query = collection.aggregate(pipelines)
        results.append(query)

    return results

def TagSearch(object_id):
    items = []
    for collection in [gDB.item, gDB.source]:
        query = {
            "$expr": {
                "$in": [object_id, "$tags"],
            },
        }

        item = AbstractGetItems(collection, query)
        items.append(item)

    return items

def SimpleSearch(field, text):
    items = []
    for collection in [gDB.item, gDB.source]:
        query = {
            field: {"$regex": text, "$options": 'i'}
        }

        item = AbstractGetItems(collection, query)
        items.append(item)

    return items

#
# ASSETLIB2
#
def AddTagItem(name):
    from libs.customError import CustomError
    db = client["ASSETLIB2"]
    filterdict = {"name": name}
    if not db.tag.find_one(filterdict):
        db.tag.insert_one(filterdict)

def EditTagItem(name, object_id):
    db = client["ASSETLIB2"]
    return db.tag.find_one_and_update(
        {"_id": object_id}, {"$set": {"name": name}}, return_document=ReturnDocument.AFTER)

def DeleteTagItem(object_id):
    db = client["ASSETLIB2"]
    return db.tag.find_one_and_delete(
        {"_id": object_id}, sort=[("_id", pymongo.DESCENDING)])

def GetTagItems():
    db = client["ASSETLIB2"]
    return db.tag.find().sort("name", pymongo.ASCENDING)

if __name__ == "__main__":
    path_list = ["/Users/rndvfx/stuff/Texture/CGAxis/rusty_chains_01"]
    for path in path_list:
        AddSCItemTest(path)
