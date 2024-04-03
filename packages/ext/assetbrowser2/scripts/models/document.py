# -*- coding: utf-8 -*-
import datetime
import os

class Document(object):
    def __init__(self, item):
        self._object_id = ''
        self._comment = ''
        self._reply = []
        self._tag = []
        self._category = ''
        self._status = '' # new
        self._files = {}
        self._storage_file_path = '' # new
        self._storage_file_size = '' # new
        self._sub_category = ''
        self._name = ''
        self._images = [] # new
        self._tags = []

        self._preview_path = '' # files.preview
        self._dir_path = ''

        self._timestamp = ''
        self._user = ''
        # one to many
        self._tag_objects = []

        self._db_item = item

    def set_item(self, item):
        self._object_id = item["_id"]
        self._comment = item["comment"]
        self._reply = item["reply"]
        self._tag = item["tag"]
        self._category = item["category"]
        self._files = item["files"]
        self._sub_category = item["subCategory"]
        self._name = item["name"]

        self._preview_path = '' # files.preview
        self._dir_path = ''

        self._status = ''
        if "status" in item:
            self._status = item["status"]

        self._storage_file_path = ''
        if "storageFilePath" in item:
            self._storage_file_path = item["storageFilePath"]

        self._storage_file_size = ''
        if "storageFileSize" in item:
            self._storage_file_size = item["storageFileSize"]

        self._images = []
        if "images" in item:
            self._images = item["images"]

        self._tags = []
        if "tags" in item:
            self._tags = item["tags"]

        self._tag_objects = []
        if "tagObjects" in item:
            self._tag_objects = item["tagObjects"]
            del item["tagObjects"]

        # special
        if "files" in item:
            self._preview_path = item["files"]["preview"]

        if self._reply:
            self._user = self._reply[-1]["user"]

        self._db_item = item

    # for EditForm
    def add_tag_object(self, document):
        item_exists = False
        for row, tag in enumerate(self._tag_objects):
            if tag["_id"] == document["_id"]:
                item_exists = True

        if not item_exists:
            self._tag_objects.append(document)
            self._tags.append(document["_id"])

    # for EditForm
    def remove_tag_object(self, object_id):
        for row, tag in enumerate(self._tag_objects):
            if tag["_id"] == object_id:
                del self._tag_objects[row]
                break

        for row, tag in enumerate(self._tags):
            if tag == object_id:
                del self._tags[row]
                break

    def get_dir_path(self, asset_type):
        if self._storage_file_path:
            return self._storage_file_path

        if asset_type == "Asset":
            index = "usdfile"
        elif asset_type == "Texture":
            index = "filePath"
        else:
            return ''

        path = self._db_item["files"][index]
        return path

    def get_timestamp(self):
        if os.path.exists(self._dir_path):
            mtime = os.path.getmtime(self._dir_path)
            return str(datetime.datetime.fromtimestamp(mtime)).split('.')[0]

        return ''

    @property
    def object_id(self):
        return self._object_id

    @object_id.setter
    def object_id(self, value):
        self._object_id = value

    @property
    def comment(self):
        return self._comment

    @comment.setter
    def comment(self, value):
        self._comment = value

    @property
    def reply(self):
        return self._reply

    @reply.setter
    def reply(self, value):
        self._reply = value

    @property
    def tag(self):
        return self._tag

    @tag.setter
    def tag(self, value):
        self._tag = value

    @property
    def category(self):
        return self._category

    @category.setter
    def category(self, value):
        self._category = value

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = value

    @property
    def files(self):
        return self._files

    @files.setter
    def files(self, value):
        self._files = value

    @property
    def storage_file_path(self):
        return self._storage_file_path

    @storage_file_path.setter
    def storage_file_path(self, value):
        self._storage_file_path = value

    @property
    def storage_file_size(self):
        return self._storage_file_size

    @storage_file_size.setter
    def storage_file_size(self, value):
        self._storage_file_size = value

    @property
    def sub_category(self):
        return self._sub_category

    @sub_category.setter
    def sub_category(self, value):
        self._sub_category = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, value):
        self._images = value

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, value):
        self._tags = value

    @property
    def tag_objects(self):
        return self._tag_objects

    @tag_objects.setter
    def tag_objects(self, value):
        self._tag_objects = value

    @property
    def db_item(self):
        return self._db_item

    @db_item.setter
    def db_item(self, value):
        self._db_item = value

    @property
    def preview_path(self):
        return self._preview_path

    @preview_path.setter
    def preview_path(self, value):
        self._preview_path = value

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = value

    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, value):
        self._user = value

    @property
    def dir_path(self):
        return self._dir_path

    @dir_path.setter
    def dir_path(self, value):
        self._dir_path = value
