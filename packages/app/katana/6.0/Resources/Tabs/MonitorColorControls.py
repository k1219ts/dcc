# Python bytecode 2.7 (62211)
# Embedded file name: /data/data/bunsen/dev/bunsenKatana/Plugins/Bunsen/build/Tabs/MonitorColorControls.py
# Compiled at: 2016-10-06 16:48:38
# Decompiled by https://python-decompiler.com
from Katana import os, QtWidgets, QtCore, QT4Widgets, QT4FormWidgets, QT4Color, Utils
import UI4
Common2D = UI4.FormMaster.NodeHints.Common2D
__all__ = ['MonitorCCPanel']

class MonitorColorControl(UI4.Tabs.BaseTab):

    def __init__(self, parent):
        UI4.Tabs.BaseTab.__init__(self, parent)
        QtWidgets.QVBoxLayout(self)
        self.layout().setContentsMargins(4, 4, 4, 4)
        self.__monitorGammaWidget = QT4Color.MonitorGammaWidget(self)
        self.layout().addWidget(self.__monitorGammaWidget, 0)
        # self.connect(self.__monitorGammaWidget, QtCore.SIGNAL('valueChanged'), self.__monitorChanged_CB)
        self.__monitorGammaWidget.valueChanged.connect(self.__monitorChanged_CB)
        self.__colorGradeWidget = QT4Color.ColorDropWidget(self)
        self.layout().addWidget(self.__colorGradeWidget, 10)
        # self.connect(self.__colorGradeWidget, QtCore.SIGNAL('valueChanged'), self.__gradeChanged_CB)
        # self.__colorGradeWidget.valueChanged.connect(self.__gradeChanged_CB)
        self.__updateTimer = QtCore.QTimer(self)
        # self.connect(self.__updateTimer, QtCore.SIGNAL('timeout()'), self.__timer_CB)
        self.__updateTimer.timeout.connect(self.__timer_CB)
        Utils.EventModule.RegisterCollapsedHandler(self.__monitor_drawStateUpdated_CB, 'monitor_drawStateUpdated', None, True)
        return

    def _MonitorColorControl__monitor_drawStateUpdated_CB(self, *args, **kwargs):
        self.__updateTimer.start(100)

    def _MonitorColorControl__timer_CB(self):
        self.__update()

    def _MonitorColorControl__update(self):
        drawState = self.getPrimaryMonitorDrawState()
        if drawState is None:
            return
        # self.__colorGradeWidget.setCC(drawState.getCC())
        monitorValueDict = {}
        monitorValueDict['fstop'] = drawState.getFStopOffset()
        monitorValueDict['gamma'] = max(drawState.getViewGamma())
        monitorValueDict['blackPoint'] = min(drawState.getViewMin())
        monitorValueDict['whitePoint'] = max(drawState.getViewMax())
        monitorValueDict['mute'] = drawState.getViewAdjustmentsMuted()
        self.__monitorGammaWidget.setValueDict(monitorValueDict)
        return

    def _MonitorColorControl__monitorChanged_CB(self, valueDict, isFinal):
        for panel in UI4.App.Tabs.GetTabsByType('Monitor'):
            widget = panel.getMonitorWidget()
            for drawState in widget.getDrawStates():
                if 'fstop' in valueDict:
                    drawState.setFStopOffset(valueDict['fstop'])
                if 'gamma' in valueDict:
                    drawState.setViewGamma(valueDict['gamma'])
                if 'blackPoint' in valueDict:
                    drawState.setViewMin(valueDict['blackPoint'])
                if 'whitePoint' in valueDict:
                    drawState.setViewMax(valueDict['whitePoint'])
                if 'mute' in valueDict:
                    drawState.setViewAdjustmentsMuted(valueDict['mute'])
                    continue

    def _MonitorColorControl__gradeChanged_CB(self, cc, isFinal):
        for panel in UI4.App.Tabs.GetTabsByType('Monitor'):
            widget = panel.getMonitorWidget()
            for drawState in widget.getDrawStates():
                drawState.setCC(cc)

    def getPrimaryMonitorDrawState(self):
        for panel in UI4.App.Tabs.GetTabsByType('Monitor'):
            widget = panel.getMonitorWidget()
            for drawState in widget.getDrawStates():
                return drawState


PluginRegistry = [
 (
  'KatanaPanel', 2.0, 'Monitor Color Control',
  MonitorColorControl)]
