import maya.cmds as cmds
import maya.mel as mel
import string

import rlf

# print StupidModule.rlf.__file__

class DynamicBinding():
    def __init__(self):
        pass
        
    def getCurrentRuleInfo(self):
        currentRule = []
        
        currentScope = mel.eval( 'rman getActiveRlfScope' )
        scopeString = mel.eval( 'rman getRlf("%s")' % currentScope )
        
        readScope = rlf.RLFScope()
        readScope.Deserialize(scopeString)
        
        bindingRuleDict = {"XPath" : "", "shader" : ""}
        
        try:
            for rule in readScope.GetRules():
                bindingRuleDict["XPath"] = rule.GetRuleString()
                bindingRuleDict["shader"] = cmds.listConnections(rule.GetPayloadId(),
                                                                 type = "PxrSurface")[0]
                                                             
                currentRule.append(bindingRuleDict)
        except:
            pass
        
        return currentRule
    
    def setRule(self, assetName, material):
        newRuleDic = {}

        # materialSplit = material.split('_')
        #
        # channel = materialSplit[-2]

        # if materialSplit[-1] != "SHD":
        #     channel = materialSplit[-1]

        newRuleDic["XPath"] = "//*[starts-with(name(), '{0}') and contains(name(), '{1}')]".format(assetName, material.split('_')[-2])
        newRuleDic["shader"] = material
        
        return newRuleDic
            
    def assignRule(self, currentRule):
        scopeRlf = rlf.RLFScope()
        for ruleForBinding in currentRule:
            try:
                scopeRlf.AddRule(self.createScope(xPathScope = ruleForBinding["XPath"],
                                                  shaderScope = ruleForBinding["shader"],
                                                  rlfScope = scopeRlf))
            except:
                continue
            
        serialized = scopeRlf.Serialize()
        newRlf = 'rman setRlf("{0}")'.format(self.serializedReplace(serialized))
        mel.eval(newRlf)
    
    def createScope(self, xPathScope, shaderScope, rlfScope):
        shadingEngineNode = cmds.listConnections(shaderScope, type = 'shadingEngine')[0]
        
        newPayloadvalue = {"Content" : "1", "Label" : shaderScope, "Source" : "1", "Payload" : ""} 
        
        payload = rlfScope.CreatePayload( shadingEngineNode, 'inject', newPayloadvalue)
        return rlf.NewRlfRule("inject", "xpath", xPathScope, "break", payload)
    
    def serializedReplace(self, serialized):
        result = serialized
        result = result.replace( '\\', '\\\\' )
        result = result.replace( '"', '\\"' )
        result = result.replace( '\n', ' ' )
        return result