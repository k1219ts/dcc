# -*- coding: utf-8 -*-
from models.document import Document

class Texture(Document):
    def __init__(self, item):
        Document.__init__(self, item)

        self.set_item(item)

    def set_item(self, item):
        Document.set_item(self, item)

        self._file_path = ''
        if "files" in item:
            self._file_path = item["files"]["filePath"]

        self._dir_path = self.get_dir_path('Texture')
        self._timestamp = self.get_timestamp()

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, value):
        self._file_path = value
