##-----------------##
## ZNumToString.py ##
##-------------------------------------------------------##
## author: Wanho Choi @ Dexter Digital                   ##
## last update: 2014.10.13                               ##
##-------------------------------------------------------##

import math, sys, array, copy
import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx
import pymel.core as pm

class ZNumToString( OpenMayaMPx.MPxNode ):

	kPluginNodeTypeName = 'ZNumToString'
	kPluginNodeId = OpenMaya.MTypeId( 0x93001 )

	inputObj     = None
	scaleObj     = None
	precisionObj = None
	unitObj      = None
	outputObj    = None

	def __init__( self ):
		OpenMayaMPx.MPxNode.__init__( self )

	@classmethod
	def nodeCreator( self ):
		return OpenMayaMPx.asMPxPtr( self() )

	@classmethod
	def nodeInitializer( self ):

		tAttr = OpenMaya.MFnTypedAttribute()
		nAttr = OpenMaya.MFnNumericAttribute()

		self.inputObj = nAttr.create( 'input', 'input', OpenMaya.MFnNumericData.kDouble,  1.0 )
		self.addAttribute( self.inputObj )

		self.scaleObj = nAttr.create( 'scale', 'scale', OpenMaya.MFnNumericData.kDouble, 1.0 )
		self.addAttribute( self.scaleObj )

		self.precisionObj = nAttr.create( 'precision', 'precision', OpenMaya.MFnNumericData.kInt, 1 )
		nAttr.setMin(0)
		nAttr.setSoftMax(6)
		self.addAttribute( self.precisionObj )

		self.unitObj = tAttr.create( 'unit', 'unit', OpenMaya.MFnData.kString )
		self.addAttribute( self.unitObj )

		self.outputObj = tAttr.create( 'output', 'output', OpenMaya.MFnData.kString )
		self.addAttribute( self.outputObj )

		self.attributeAffects( self.inputObj,     self.outputObj )
		self.attributeAffects( self.scaleObj,     self.outputObj )
		self.attributeAffects( self.precisionObj, self.outputObj )
		self.attributeAffects( self.unitObj,      self.outputObj )

	def compute( self, plug, block ):

		if( plug != ZNumToString.outputObj ):
			return OpenMaya.kUnknownParameter

		input     = block.inputValue( self.inputObj     ).asDouble()
		scale     = block.inputValue( self.scaleObj     ).asDouble()
		precision = block.inputValue( self.precisionObj ).asInt()
		unit      = block.inputValue( self.unitObj      ).asString()

		input *= scale

		output = str( round( input, precision ) ) + ' ' + unit

		outputHnd = OpenMaya.MDataHandle( block.outputValue( self.outputObj ) )
		outputHnd.setString( output )
		block.setClean( plug )

def initializePlugin( mobject ):
	mplugin = OpenMayaMPx.MFnPlugin( mobject, 'Dexter Digital', '1.0', 'Any' )
	try:
		mplugin.registerNode( ZNumToString.kPluginNodeTypeName, ZNumToString.kPluginNodeId, ZNumToString.nodeCreator, ZNumToString.nodeInitializer )
	except:
		raise Exception( 'Failed to register node: %s'%ZNumToString.kPluginNodeTypeName )

def uninitializePlugin( mobject ):
	mplugin = OpenMayaMPx.MFnPlugin( mobject )
	try:
		mplugin.deregisterNode( ZNumToString.kPluginNodeId )
	except:
		raise Exception( 'Failed to unregister node: %s'%ZNumToString.kPluginNodeTypeName )

class AEZNumToStringTemplate(pm.uitypes.AETemplate):
	_nodeType = 'ZNumToString' 
	def __init__(self, nodeName):
		self.beginScrollLayout()
		self.beginLayout("Options",collapse=0)
		self.addControl("input")
		self.addControl("scale")
		self.addControl("precision")
		self.addControl("unit")
		self.addControl("output")
		self.endLayout()
		self.addExtraControls()
		self.endScrollLayout()

