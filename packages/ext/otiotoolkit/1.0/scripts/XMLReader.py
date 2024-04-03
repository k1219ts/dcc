#coding:utf-8
import sys
from adapters.FCP7_XMLParser import FCPXML7Parser

from PySide2 import QtWidgets

# editFileName = '/prod_nas/__DD_PROD/SLC/edit/20201111/S30_MET/S30_MET_v1_CGDI_201111_(Resolve).xml'
# editFileName = '/prod_nas/__DD_PROD/SLC/edit/20201116/S39_DAT_locationEdit/Kang_action_1114_toEditor.xml'

# CDH1
# editFileName = '/prod_nas/__DD_PROD/CDH/edit/during_pre/201111_keyshot/201112_from_edit_opt/1_s041_KEY_IMAGE.xml'
# editFileName = '/prod_nas/__DD_PROD/CDH/edit/during_pre/201112/1_s#041_KEY_IMAGE.xml'

# PRAT2
# editFileName = '/prod_nas/__DD_PROD/PRAT2/edit/201118/haejeok_002_S083_201118.xml'
# editFileName = '/prod_nas/__DD_PROD/PRAT2/edit/201118/haejeok_002_S044_201118.xml'
# editFileName = '/prod_nas/__DD_PROD/PRAT2/edit/201118/haejeok_002_S010_201118.xml'

def main(xmlFileName):
    app = QtWidgets.QApplication(sys.argv)
    # authoringApp = sys.argv[2] # FCP7 or PremierePro
    # if editXMLFile.endswith('.xml') and authoringApp == "FCP7":
    #     parser = FCP7XMLParser(editXMLFile)
    # elif editXMLFile.endswith('.xml') and authoringApp == "PremierePro":
    parser = FCPXML7Parser(xmlFileName)
    # parser = PremiereProXMLParser(xmlFileName)
    # else:
    #     assert False, "Not Supported Editorial File."

    parser.doIt()
    parser.save()
    
if __name__ == '__main__':
    editXMLFile = sys.argv[1]
    main(editXMLFile)
    sys.exit(0)