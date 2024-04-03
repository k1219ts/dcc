#coding:utf-8
import opentimelineio as otio
from adapters.calculator import *

editFilePath = '/prod_nas/__DD_PROD/SLC/edit/20201111/S30_MET/S30_MET_v1_CGDI_201111_(Resolve).xml'
otioData = otio.adapters.read_from_file(editFilePath, 'fcp_xml')

cleanupTimeline = cleanupTrackData(otioData)