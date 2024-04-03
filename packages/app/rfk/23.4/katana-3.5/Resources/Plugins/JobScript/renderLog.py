from Katana import NodegraphAPI, Nodes3DAPI

import os, sys
import string
import json


class VersionLog:
    def __init__(self, node, outdir=None):
        self.node   = node
        if type(node).__name__ == 'str' or type(node).__name__ == 'unicode':
            self.node = NodegraphAPI.GetNode(node)
        self.outdir = outdir

        self.__TargetVariants = ['aniVer', 'camVer', 'crowdVer', 'groomVer', 'layoutVer', 'simVer']
        self.maxDepth = 9
        self.rootLocation = '/root/world'
        self._Producer = Nodes3DAPI.GetGeometryProducer(self.node)
        self.versions = dict()

    def doIt(self):
        self.GetChildrenProducer(self.rootLocation)
        if self.outdir and self.versions:
            f = open(os.path.join(self.outdir, 'version_log.json'), 'w')
            json.dump(self.versions, f, indent=4)
            f.close()


    def GetChildrenProducer(self, location):
        root = self._Producer.getProducerByPath(location)
        for c in root.iterChildren():
            self.versionVariant(c)
            name = c.getFullName()
            if len(name.split('/')) < self.maxDepth:
                self.GetChildrenProducer(name)


    def versionVariant(self, producer):
        groupAttr = producer.getAttribute('info.usd.selectedVariants')
        if not groupAttr:
            return

        prodName = producer.getFullName()
        for i in range(groupAttr.getNumberOfChildren()):
            name = groupAttr.getChildName(i)
            if name in self.__TargetVariants:
                attr = producer.getAttribute('info.usd.selectedVariants.%s' % name)
                val  = attr.getValue()

                if not self.versions.has_key(prodName):
                    self.versions[prodName] = list()
                self.versions[prodName].append((name, val))
