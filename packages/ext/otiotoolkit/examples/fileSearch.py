import scandir

plateName = 'A098C001_200620_R4AK'
ROOT_DIR = 0
DIR_LIST = 1
FILE_LIST = 2
for i in scandir.walk('/stuff/emd/scan'):
    if 'A098C001_200620_R4AK' in i[DIR_LIST]:
        print i
        break