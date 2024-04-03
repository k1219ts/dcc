'''
Dexter Tractor Job Main
'''

import site

TractorRoot = '/netapp/backstage/pub/apps/tractor/linux/Tractor-2.2'
site.addsitedir( '%s/lib/python2.7/site-packages' % TractorRoot )
import tractor.api.author as author

from engine import Engine
