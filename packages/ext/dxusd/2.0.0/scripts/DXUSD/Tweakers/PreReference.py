#coding:utf-8
from __future__ import print_function
import os

from pxr import Sdf, Usd

from DXUSD.Tweakers import Tweaker, ATweaker
import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg


class APreReference(ATweaker):
    def __init__(self, **kwargs):
        # initialize
        ATweaker.__init__(self, **kwargs)

    def Treat(self):
        if not self.inputs:
            msg.errmsg('Treat@%s' % self.__name__, 'No inputs.')
            return var.FAILED
        return var.SUCCESS


class PreReference(Tweaker):
    ARGCLASS = APreReference
    def DoIt(self):
        for path in self.arg.inputs:
            self.editor = Sdf.BatchNamespaceEdit()
            if self.ChangePreReference(utl.AsLayer(path)) == var.FAILED:
                m = 'Cannot find layer - %s'%path
                msg.warning('@%s :'%self.__name__, m)
                continue


    def AddRemover(self, specPath):
        self.editor.Add(specPath, Sdf.Path.emptyPath)


    def ChangePreReference(self, srclyr):
        if not srclyr:
            return var.FAILED

        dspec  = utl.GetDefaultPrim(srclyr)
        changed = False

        with utl.OpenStage(srclyr, loadAll=False) as stage:
            dprim = stage.GetPrimAtPath(dspec.path)
            It = iter(Usd.PrimRange.AllPrims(dprim))

            for p in It:
                if not p.HasProperty('prereferenceLayer') or \
                   not p.HasProperty('prereferencePrim'):
                    continue

                preRefLayerAttr = p.GetAttribute('prereferenceLayer')
                preRefPrimAttr  = p.GetAttribute('prereferencePrim')
                preRefLayer     = preRefLayerAttr.Get()
                preRefPrim      = preRefPrimAttr.Get()

                if not preRefLayer or not preRefPrim:
                    continue

                self.AddRemover(preRefLayerAttr.GetPropertyStack(0)[0].path)
                self.AddRemover(preRefPrimAttr.GetPropertyStack(0)[0].path)

                relpath = utl.GetRelPath(srclyr.realPath, preRefLayer)
                utl.SetReference(p, Sdf.Reference(relpath, preRefPrim))
                changed = True

        if changed:
            srclyr.Apply(self.editor)
            srclyr.Save()

        return var.SUCCESS
