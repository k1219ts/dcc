# -*- coding: utf-8 -*-
from models.document import Document

class Asset(Document):
    def __init__(self, item):
        Document.__init__(self, item)

        self.set_item(item)

    def set_item(self, item):
        Document.set_item(self, item)

        self._usdfile = ''
        if "files" in item:
            self._usdfile = item["files"]["usdfile"]

        self._dir_path = self.get_dir_path("Asset")
        self._timestamp = self.get_timestamp()

    @property
    def usdfile(self):
        return self._usdfile

    @usdfile.setter
    def usdfile(self, value):
        self._usdfile = value
