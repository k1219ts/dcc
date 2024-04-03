import os

os.environ['LD_LIBRARY_PATH'] += ':/netapp/backstage/pub/lib/extern/lib:/netapp/backstage/pub/lib/zelos/lib'
os.environ['PYTHONPATH'] += '/netapp/backstage/pub/lib/zelos/lib:/netapp/backstage/pub/lib/zelos/py'

print os.environ['LD_LIBRARY_PATH']
print os.environ['PYTHONPATH']

import Zelos

# Zelos.AlembicArchive()