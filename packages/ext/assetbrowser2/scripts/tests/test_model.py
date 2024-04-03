# -*- coding: utf-8 -*-
import sys
import unittest
sys.path.append("/Users/rndvfx/Documents/eclipseProjects/AssetBrowser/src")
from models.document import Document

class TestModel(unittest.TestCase):
    def test_document(self):

        data_v1_asset = {
            "_id" : "6218a0cd1e13095b0cc25c24",
            "comment" : "기타\n개똥벌레 (반딧불)",
            "name" : "firefly",
            "tag" : [
                "1953",
                "서부전선",
                "test"
            ],
            "files" : {
                "preview" : "/Users/rndvfx/stuff/Asset/thumbnail/show1/firefly_firefly_render_01_texture_texture_v001.jpg",
                "usdfile" : ""
            },
            "category" : "Prop",
            "subCategory" : "unknown",
            "reply" : [
                {
                    "comment" : "add item",
                    "user" : "rndvfx",
                    "time" : "2022-02-25T18:26:37.610324"
                }
            ]
        }

        data_v1_texture = {
            "_id" : "6218a0cd1e13095b0cc25c24",
            "comment" : "기타\n개똥벌레 (반딧불)",
            "name" : "firefly",
            "tag" : [
                "1953",
                "서부전선",
                "test"
            ],
            "files" : {
                "preview" : "/Users/rndvfx/stuff/Asset/thumbnail/show1/firefly_firefly_render_01_texture_texture_v001.jpg",
                "filePath" : ""
            },
            "category" : "Prop",
            "subCategory" : "unknown",
            "reply" : [
                {
                    "comment" : "add item",
                    "user" : "rndvfx",
                    "time" : "2022-02-25T18:26:37.610324"
                }
            ]
        }

        data_v2_asset = {
            "_id" : "6218a0cd1e13095b0cc25c24",
            "comment" : "기타\n개똥벌레 (반딧불)",
            "status" : "Omit",
            "name" : "firefly",
            "tag" : [
                "1953",
                "서부전선",
                "test"
            ],
            "files" : {
                "preview" : "/Users/rndvfx/stuff/Asset/thumbnail/show1/firefly_firefly_render_01_texture_texture_v001.jpg",
                "usdfile" : ""
            },
            "storageFileSize" : "4.0 K",
            "category" : "Prop",
            "subCategory" : "unknown",
            "storageFilePath" : "/136arch/1953/asset/prop/firefly",
            "reply" : [
                {
                    "comment" : "add item",
                    "user" : "rndvfx",
                    "time" : "2022-02-25T18:26:37.610324"
                }
            ],
            "images" : [
                "/Users/rndvfx/synctest/synctest_model_model_v001.jpg",
                "/Users/rndvfx/synctest/synctest_model_model_v004.jpg",
                "/Users/rndvfx/synctest/synctest_model.jpg",
                "/Users/rndvfx/synctest/synctest_v01_web_model_model_v001.jpg",
                "/Users/rndvfx/synctest/synctest_web_model_model_v004.jpg"
            ]
        }

        data_v2_asset_no_images = {
            "_id" : "6218a0cd1e13095b0cc25c09",
            "comment" : "머스탱 전투기",
            "status" : "Approved",
            "name" : "mustang_bullet",
            "tag" : [
                "1953",
                "서부전선"
            ],
            "files" : {
                "preview" : "/Users/rndvfx/stuff/Asset/thumbnail/show1/mustang_mustang_150602_texture_texture_v056.jpg",
                "usdfile" : ""
            },
            "storageFileSize" : "4.1M",
            "category" : "Default",
            "subCategory" : "unknown",
            "storageFilePath" : "/136arch/1953/asset/ani/mustang_bullet",
            "reply" : [
                {
                    "comment" : "add item",
                    "user" : "rndvfx",
                    "time" : "2022-02-25T18:26:37.567831"
                }
            ]
        }

        document = Document(data_v1_asset)
        self.assertEqual(data_v1_asset["_id"], document.object_id)
        self.assertEqual(data_v1_asset["comment"], document.comment)
        self.assertEqual(data_v1_asset["name"], document.name)
        self.assertEqual(data_v1_asset["tag"], document.tag)
        self.assertEqual(data_v1_asset["files"], document.files)
        self.assertEqual(data_v1_asset["category"], document.category)
        self.assertEqual(data_v1_asset["subCategory"], document.sub_category)
        self.assertEqual(data_v1_asset["reply"], document.reply)
        self.assertEqual(data_v1_asset["files"]["preview"], document.thumbnail_path)
        self.assertEqual(data_v1_asset, document.db_item)
        self.assertEqual(data_v1_asset["reply"][0]["user"], document.user)

        document = Document(data_v1_texture)
        self.assertEqual(data_v1_texture["_id"], document.object_id)
        self.assertEqual(data_v1_texture["comment"], document.comment)
        self.assertEqual(data_v1_texture["name"], document.name)
        self.assertEqual(data_v1_texture["tag"], document.tag)
        self.assertEqual(data_v1_texture["files"], document.files)
        self.assertEqual(data_v1_texture["category"], document.category)
        self.assertEqual(data_v1_texture["subCategory"], document.sub_category)
        self.assertEqual(data_v1_texture["reply"], document.reply)
        self.assertEqual(data_v1_texture["files"]["preview"], document.thumbnail_path)
        self.assertEqual(data_v1_texture, document.db_item)
        self.assertEqual(data_v1_texture["reply"][0]["user"], document.user)

        document = Document(data_v2_asset)
        self.assertEqual(data_v2_asset["_id"], document.object_id)
        self.assertEqual(data_v2_asset["comment"], document.comment)
        self.assertEqual(data_v2_asset["name"], document.name)
        self.assertEqual(data_v2_asset["tag"], document.tag)
        self.assertEqual(data_v2_asset["files"], document.files)
        self.assertEqual(data_v2_asset["category"], document.category)
        self.assertEqual(data_v2_asset["subCategory"], document.sub_category)
        self.assertEqual(data_v2_asset["reply"], document.reply)
        self.assertEqual(data_v2_asset["files"]["preview"], document.thumbnail_path)
        self.assertEqual(data_v2_asset, document.db_item)
        self.assertEqual(data_v2_asset["status"], document.status)
        self.assertEqual(data_v2_asset["storageFileSize"], document.storage_file_size)
        self.assertEqual(data_v2_asset["storageFilePath"], document.storage_file_path)
        self.assertEqual(data_v2_asset["images"], document.images)
        self.assertEqual(data_v2_asset["reply"][0]["user"], document.user)

        document = Document(data_v2_asset_no_images)
        self.assertEqual(data_v2_asset_no_images["_id"], document.object_id)
        self.assertEqual(data_v2_asset_no_images["comment"], document.comment)
        self.assertEqual(data_v2_asset_no_images["name"], document.name)
        self.assertEqual(data_v2_asset_no_images["tag"], document.tag)
        self.assertEqual(data_v2_asset_no_images["files"], document.files)
        self.assertEqual(data_v2_asset_no_images["category"], document.category)
        self.assertEqual(data_v2_asset_no_images["subCategory"], document.sub_category)
        self.assertEqual(data_v2_asset_no_images["reply"], document.reply)
        self.assertEqual(data_v2_asset_no_images["files"]["preview"], document.thumbnail_path)
        self.assertEqual(data_v2_asset_no_images, document.db_item)
        self.assertEqual(data_v2_asset_no_images["status"], document.status)
        self.assertEqual(data_v2_asset_no_images["storageFileSize"], document.storage_file_size)
        self.assertEqual(data_v2_asset_no_images["storageFilePath"], document.storage_file_path)
        self.assertEqual(data_v2_asset_no_images["reply"][0]["user"], document.user)

if __name__ == "__main__":
    unittest.main()
