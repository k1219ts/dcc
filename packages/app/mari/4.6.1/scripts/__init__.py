#--------------------------------------------------------------------------------
#
#    RenderMan TD
#
#        Sanghun Kim, rman.td@gmail.com
#
#    rman.td 2016.08.07 $4
#
#--------------------------------------------------------------------------------

import mari
import dxpublish.insertDB as insertDB

# import textureExport
import textureExport
import textureImport
import projectSetup
import sceneExporter
import sceneImporter
import refExport
import dxMari
import dxMtm

#    Create Dexter menu
mari.menus.addSeparator( 'MainWindow/&Dexter' )
mari.menus.addSeparator( 'MainWindow/&Dexter/&Geometry' )
mari.menus.addSeparator( 'MainWindow/&Dexter/&Scene' )

   #Add menu
mari.menus.addAction(
        mari.actions.create('Open FileBrowser', 'dxMari.openNautilus()'),
        "MainWindow/&Dexter"
    )
mari.menus.addAction(
        mari.actions.create('ref Export', 'refExport.showRefExportUI()'),
        "MainWindow/&Dexter"
    )
mari.menus.addAction(
        mari.actions.create('Project Set', 'projectSetup.show_ui()'),
        "MainWindow/&Dexter"
    )
mari.menus.addAction(
        mari.actions.create('Update Texture Layer', 'dxMari.updateTextureLayerInfo()'),
        "MainWindow/&Dexter"
    )
mari.menus.addAction(
        mari.actions.create('Channel USD Export', 'textureExport.show_ui()'),
        "MainWindow/&Dexter"
    )
mari.menus.addAction(
        mari.actions.create('Texture Preview', 'textureImport.show_ui()'),
        "MainWindow/&Dexter"
    )
mari.menus.addAction(
        mari.actions.create('Add Version', 'dxMari.addVersion()'),
        "MainWindow/&Dexter/&Geometry"
    )
mari.menus.addAction(
        mari.actions.create('Add Object', 'dxMari.addObject()'),
        "MainWindow/&Dexter/&Geometry"
    )
mari.menus.addAction(
        mari.actions.create('Scene Export', 'sceneExporter.show_ui()'),
        "MainWindow/&Dexter/&Scene"
    )
mari.menus.addAction(
        mari.actions.create('Scene Import', 'sceneImporter.show_ui()'),
        "MainWindow/&Dexter/&Scene"
    )


def openPrjCallback(newPrj, newlyCreated):
    print "openPrjCallback", newPrj.name(), newlyCreated
    print mari.resources.path('MARI_DEFAULT_CAMERA_PATH')
    insertDB.recordWork("mari", "open", mari.resources.path('MARI_DEFAULT_CAMERA_PATH'))


def savePrjCallback(savedPrj):
    print 'savedPrjCallback', savedPrj.name(), savedPrj.info().projectPath()
    insertDB.recordWork("mari", "save", mari.resources.path('MARI_DEFAULT_CAMERA_PATH'))

mari.projects.opened.connect(openPrjCallback)
mari.projects.saved.connect(savePrjCallback)

try:
    mari.prefs.set('Scripts/Mari Command Port/port', 6100)
    if not mari.app.commandPortEnabled():
        mari.app.enableCommandPort(True)
    mari.prefs.set('Scripts/Mari Command Port/localhostOnly', False)
except Exception as e:
    print e.message
