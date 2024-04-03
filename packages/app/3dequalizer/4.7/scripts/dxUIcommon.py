import os
import tde4
import TDE4_common
from imp import reload
reload(TDE4_common)


class setOverscanWidget:
    def __init__(self, requester, camera):
        self.req = requester
        self.overscanList = ['1.08', '1.1', '1.15', '1.2', '1.25', '1.3', 'custom']
        self.overscan = 1.00
        self.plateWidth = tde4.getCameraImageWidth(camera)
        self.plateHeight = tde4.getCameraImageHeight(camera)

    def _isOverscanCallback(self, requester, widget, action):
        mode = tde4.getWidgetValue(requester, 'overscan')
        if mode == 0:
            tde4.setWidgetSensitiveFlag(requester, 'overscan_v', 0)
        if mode == 1:
            tde4.setWidgetSensitiveFlag(requester, 'overscan_v', 1)

    def _overscanValueValidator(self, requester, widget, action):
        overscanIdx = tde4.getWidgetValue(requester, 'overscan_v')
        if self.overscanList[overscanIdx-1] == 'custom':
            tde4.setWidgetSensitiveFlag(requester, 'overscan_custom', 1)
            value = tde4.getWidgetValue(requester, 'overscan_custom')
            if value:   self.overscan = float(value)
            else:       self.overscan = 1.00
            self.computeOverscanResolution(requester)
        else:
            tde4.setWidgetSensitiveFlag(requester, 'overscan_custom', 0)
            tde4.setWidgetValue(requester, 'overscan_custom', '')
            self.overscan = float(self.overscanList[overscanIdx-1])
            self.computeOverscanResolution(requester)

    def _imageReformatScale(self, requester, widget, action):
        mode = tde4.getWidgetValue(requester, 'size')
        path = tde4.getWidgetValue(requester, 'file_path')
        if mode == 1:
            tde4.setWidgetValue(requester, 'file_path', path[:-2] + 'hi')
        else:
            tde4.setWidgetValue(requester, 'file_path', path[:-2] + 'lo')

    def computeOverscanResolution(self, requester):
        ovrWidth = int(self.plateWidth * self.overscan)
        ovrHeight = int(self.plateHeight * self.overscan)
        if tde4.widgetExists(requester, 'os_width'):
            tde4.setWidgetValue(requester, 'os_width', str(ovrWidth))
            tde4.setWidgetValue(requester, 'os_height', str(ovrHeight))

    def computeOverscanValue(self):
        bbox = TDE4_common.bbdld_compute_bounding_box()

        overscanScale = round(bbox[2] / bbox[4], 2)
        if overscanScale < round(bbox[3] / bbox[5], 2):
            overscanScale = round(bbox[3] / bbox[5], 2)

        if bbox[0] < 0.0000 or bbox[1] < 0.0000:
            if overscanScale > float(self.overscanList[-2]):
                self.overscan = overscanScale
            else:
                for i in reversed(self.overscanList):
                    if i != 'custom' and float(i) > overscanScale:
                        self.overscan = float(i)
        return self.overscan

    def getOverscanValue(self):
        overscanIdx = tde4.getWidgetValue(self.req, 'overscan_v')
        if self.overscanList[overscanIdx-1] == 'custom':
            return float(tde4.getWidgetValue(self.req, 'overscan_custom'))
        else:
            return float(self.overscanList[overscanIdx-1])

    def doIt(self):
        tde4.addToggleWidget(self.req, 'overscan','Overscan?', 0)
        tde4.setWidgetCallbackFunction(self.req, 'overscan', 'dxUIovr._isOverscanCallback')
        tde4.addOptionMenuWidget(self.req, 'overscan_v', 'Overscan Value', *self.overscanList)
        tde4.setWidgetCallbackFunction(self.req, 'overscan_v', 'dxUIovr._overscanValueValidator')
        tde4.setWidgetSensitiveFlag(self.req, 'overscan_v', 0)
        tde4.addTextFieldWidget(self.req, 'overscan_custom', 'Overscan Custom', '')
        tde4.setWidgetSensitiveFlag(self.req, 'overscan_custom', 0)
        tde4.setWidgetCallbackFunction(self.req, 'overscan_custom', 'dxUIovr._overscanValueValidator')

        try:
            fileName = tde4.getWidgetValue(self.req, 'file_browser')
            if 'pmodel' in fileName.lower():
                tde4.setWidgetSensitiveFlag(self.req, 'overscan', 0)
                return 1.0
        except:
            pass

        bbox = TDE4_common.bbdld_compute_bounding_box()

        # self.plateWidth = bbox[4]
        # self.plateHeight = bbox[5]

        overscanScale = round(bbox[2] / bbox[4], 2)
        if overscanScale < round(bbox[3] / bbox[5], 2):
            overscanScale = round(bbox[3] / bbox[5], 2)

        if bbox[0] < 0.0000 or bbox[1] < 0.0000:
            tde4.setWidgetValue(self.req, 'overscan', '1')
            tde4.setWidgetSensitiveFlag(self.req, 'overscan_v', 1)

            if overscanScale > float(self.overscanList[-2]):
                self.overscan = overscanScale
                tde4.setWidgetValue(self.req, 'overscan_v', '5')  # custom
                tde4.setWidgetValue(self.req, 'overscan_custom', str(self.overscan))
                tde4.setWidgetSensitiveFlag(self.req, 'overscan_custom', 1)
            else:
                for i in reversed(self.overscanList):
                    if i != 'custom' and float(i) >= overscanScale:
                        self.overscan = float(i)
                        widgetIdx = self.overscanList.index(i) + 1
                        tde4.setWidgetValue(self.req, 'overscan_v', str(widgetIdx))

        return self.overscan
