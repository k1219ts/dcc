#coding:utf-8
from __future__ import print_function

import hou

class HouNodeProc:
    def __init__(self):
        self.undoQueue = []

    def findHouNodes(self, hnode, hnfilter):
        result = []
        if hnode != None:
            for n in hnode.children():
                t = str(n.type())
                if t != None:
                    for hnfi in hnfilter:
                        if hnfi == '*' or t.count(hnfi) > 0:
                            result.append((n.path(), t))
                        result += self.findHouNodes(n, hnfilter)
        return result

    def getSimCacheOpencl(self):
        hnodes = self.findHouNodes(hou.node('/obj'), ['Object dopnet'])
        for (npath, ntype) in hnodes:
            cacheenabledVal = -1
            try: cacheenabledVal = hou.node(npath).parm('cacheenabled').evalAsInt()
            except: pass
            if cacheenabledVal > -1:
                print(npath, ntype, 'cacheenabled', ':', cacheenabledVal)
            
            hnodes1 = self.findHouNodes(hou.node(npath), ['*'])
            for (npath1, ntype1) in hnodes1:
                if ntype1.lower().count('solver') > 0:
                    openclParms = ['opencl', 'useopencl']
                    for ocparm in openclParms:
                        openclVal = -1
                        try: openclVal = hou.node(npath1).parm(ocparm).evalAsInt()
                        except: pass
                        if openclVal > -1:
                            print(npath1, ntype1, ocparm, ':' ,openclVal)

    def disableSimCacheOpencl(self):
        hnodes = self.findHouNodes(hou.node('/obj'), ['Object dopnet'])
        for (npath, ntype) in hnodes:
            cacheenabledVal = -1
            try: cacheenabledVal = hou.node(npath).parm('cacheenabled').evalAsInt()
            except: pass
            if cacheenabledVal > -1:
                self.undoQueue.append({'npath': npath, 'parm': 'cacheenabled', 'val': cacheenabledVal})
                if cacheenabledVal > 0:
                    try:
                        hou.node(npath).parm('cacheenabled').set(0)
                        print(npath, 'cacheenabled', ':', cacheenabledVal, '(set to 0)')
                    except:
                        pass#print(npath, 'cacheenabled', ':', cacheenabledVal, '(set to 0 failed)')
            
            hnodes1 = self.findHouNodes(hou.node(npath), ['*'])
            for (npath1, ntype1) in hnodes1:
                if ntype1.lower().count('solver') > 0:
                    openclParms = ['opencl', 'useopencl']
                    for ocparm in openclParms:
                        openclVal = -1
                        try: openclVal = hou.node(npath1).parm(ocparm).evalAsInt()
                        except: pass
                        if openclVal > -1:
                            self.undoQueue.append({'npath': npath1, 'parm': ocparm, 'val': openclVal})
                            if openclVal > 0:
                                try:
                                    hou.node(npath1).parm(ocparm).set(0)
                                    print(npath1, ocparm, ':', openclVal, '(set to 0)')
                                except:
                                    pass#print(npath1, ocparm, ':', openclVal, '(set to 0 failed)')

    def undoDisableSimCacheOpencl(self):
        for uq in self.undoQueue:
            try: hou.node(uq['npath']).parm(uq['parm']).set(uq['val'])
            except: pass#print('Undo failed', uq['npath'], uq['parm'], ':', uq['val'])
        
