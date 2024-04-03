import mari
import os

baseDir = "/dexter/Cache_DATA/ASSET/0.ENV/01.asset_DATA/MegascansStudio/Asphalt_Coarse_sfonaboa_2K_surface_ms"

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