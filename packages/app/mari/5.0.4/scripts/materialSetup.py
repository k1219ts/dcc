import mari
import os

baseDir = "/stdrepo/AST/1.asset/3.Megascan/Megascan_Studio/MegascanStudio_Construction/Asphalt_Coarse_sfonaboa_2K_surface_ms"

def setup(materialName):
    categoryList = mari.images.categories()

    if materialName in categoryList:
        return

    mari.images.addCategory(materialName)

    materialPath = os.path.join(baseDir, materialName)
    for filename in os.listdir(materialPath):
        filePath = os.path.join(materialPath, filename)
        if not filename.startswith(".") or not filePath:
            channel = os.path.splitext(filename)[0].split("_")[-1]
            mari.images.open(filePath, Name = channel)
