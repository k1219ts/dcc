import os, shutil
import compileall
import getpass

def linux_pub():
    #compileall.compile_path('/home/taehyung.lee/PycharmProjects/TA/asset_browser')
    compileall.compile_dir('/home/daeseok.chae/PycharmProjects/inventory/inventory',
                           maxlevels=0)

    #target = '/dexter/Cache_DATA/RND/taehyung/work_dev/inven_mac'
    target = '/netapp/backstage/pub/apps/inventory/src/dev'

    shutil.copy('./db_query.pyc', target)
    shutil.copy('./icon_rc.pyc', target)
    shutil.copy('./inventory.pyc', target)
    shutil.copy('./items.pyc', target)
    shutil.copy('./tagEdit.pyc', target)
    shutil.copy('./ui_inventory.pyc', target)
    shutil.copy('./user_config.pyc', target)
    shutil.copy('./viewer.pyc', target)
    shutil.copy('./detailDialog.pyc', target)


def mac_copy():
    target = '/dexter/Cache_DATA/RND/taehyung/work_dev/inven_mac'
    #target = '/dexter/Cache_DATA/RND/taehyung/work_dev/inven_mac_dev'

    shutil.copy('./db_query.py', target)
    shutil.copy('./icon_rc.py', target)
    shutil.copy('./inventory.py', target)
    shutil.copy('./items.py', target)
    shutil.copy('./tagEdit.py', target)
    shutil.copy('./ui_inventory.py', target)
    shutil.copy('./user_config.py', target)
    shutil.copy('./viewer.py', target)
    shutil.copy('./detailDialog.py', target)
#mac_copy()
#linux_pub()
compileall.compile_dir('/home/taehyung.lee/PycharmProjects/asset_browser/asset_browser/inventory',
                       force=True,
                       maxlevels=0)
