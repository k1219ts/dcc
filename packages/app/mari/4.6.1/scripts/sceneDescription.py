#encoding=utf-8
#--------------------------------------------------------------------------------
#
#	RenderMan TD
#
#		Sanghun Kim, rman.td@gmail.com
#
#	rman.td 2015.06.24 $1
#
#-------------------------------------------------------------------------------

import os, sys
import json
import mari

#-------------------------------------------------------------------------------
#
#	Get Info
#
#-------------------------------------------------------------------------------
def getSceneInfo():
	result = dict()
	geo = mari.geo.current()
	ver = geo.currentVersion()
	result['objects'] = list( ver.meshPaths() )
	channelMap = dict()
	for i in geo.channelList():
		data = dict()
		data['width']  = i.width()
		data['height'] = i.height()
		data['depth']  = i.depth()
		channelMap[ i.name() ] = data
	result['channels'] = channelMap
	return result

def getChannelInfo( channelObject ):
	result = dict()
	result['width']  = channelObject.width()
	result['height'] = channelObject.height()
	result['depth']  = channelObject.depth()
	return result

class ChannelLayers:
	def __init__( self, channelObject ):
		self.m_info = dict()
		self.m_exportPath = None
		self.channelObject = channelObject
		self.channelName   = channelObject.name()
	
	# get layer info
	def getLayerInfo( self, layerObject ):
		result = dict()
		result['name']   = layerObject.name()
		result['type']   = layerObject.className()
		result['mode']   = layerObject.blendMode()
		result['amount'] = layerObject.blendAmount()
		result['vis']    = layerObject.isVisible()
		# Mask
		result['hasMask'] = layerObject.hasMask()
		if result['hasMask']:
			result['isMaskEnabled'] = layerObject.isMaskEnabled()
			if self.m_exportPath:
				imgSet = layerObject.maskImageSet()
				filename = os.path.join( self.m_exportPath, self.channelName, '%s_mask.$UDIM.tif' % result['name'] )
				if not os.path.exists( os.path.dirname(filename) ):
					os.makedirs( os.path.dirname(filename) )
				imgSet.exportImages( filename )
				result['maskImage'] = filename
		
		# ProceduralLayer
		if result['type'] == 'ProceduralLayer':
			result['proceduralType'] = layerObject.proceduralType()
			paramMap = dict()
			for param in layerObject.proceduralParameters():
				value = layerObject.getProceduralParameter( param )
				if type(value).__name__ == 'Color':
					paramMap[param] = 'Color%s' % value.toString()
				else:
					paramMap[param] = value
				#paramMap[param] = value
			result['proceduralParam'] = paramMap
		
		# AdjustmentLayer
		if result['type'] == 'AdjustmentLayer':
			result['adjustmentType'] = layerObject.primaryAdjustmentType()
			paramMap = dict()
			for param in layerObject.primaryAdjustmentParameters():
				value = layerObject.getPrimaryAdjustmentParameter( param )
				paramMap[param] = value
			result['adjustmentParam'] = paramMap
		
		# PaintableLayer
		if result['type'] == 'PaintableLayer' and self.m_exportPath:
			imgSet = layerObject.imageSet()
			filename = os.path.join( self.m_exportPath, self.channelName, '%s.$UDIM.tif' % result['name'] )
			if not os.path.exists( os.path.dirname(filename) ):
				os.makedirs( os.path.dirname(filename) )
			imgSet.exportImages( filename )
			result['image'] = filename
		
		return result

	def getAdjustmentStackInfo( self, layerObject ):
		adjustObject = layerObject.adjustmentStack()
		adjustLayers = adjustObject.layerList()
		adjustInfo   = dict()
		for i in adjustLayers:
			name = i.name()
			info = self.getLayerInfo( i )
			adjustInfo[name] = info
			adjustInfo[name]['index'] = adjustLayers.index(i)
		return adjustInfo

	def getGroupStackInfo( self, layerObject ):
		groupObject = layerObject.groupStack()
		groupLayers = groupObject.layerList()
		groupInfo   = dict()
		for i in groupLayers:
			name = i.name()
			info = self.getLayerInfo( i )
			groupInfo[name] = info
			groupInfo[name]['index'] = groupLayers.index(i)
			# AdjustmentStack
			if info['type'] == 'AdjustmentLayer':
				pass
			else:
				if i.hasAdjustmentStack():
					groupInfo[name]['adjust'] = self.getAdjustmentStackInfo( i )
		return groupInfo
	
	def getLayers( self ):
		layers = self.channelObject.layerList()
		for i in layers:
			name = i.name()
			info = self.getLayerInfo(i)
			self.m_info[name] = info
			self.m_info[name]['index'] = layers.index(i)
			# AdjustmentStack
			if info['type'] == 'AdjustmentLayer':
				pass
			else:
				if i.hasAdjustmentStack():
					self.m_info[name]['adjust'] = self.getAdjustmentStackInfo(i)
			# GroupLayer
			if info['type'] == 'GroupLayer':
				self.m_info[name]['group'] = self.getGroupStackInfo(i)
	
	# export json
	def exportInfo( self ):
		if self.m_exportPath:
			self.getLayers()
			filename = os.path.join( self.m_exportPath, '%s.json' % self.channelName )
			f = open( filename, 'w' )
			json.dump( self.m_info, f, indent=4 )
			f.close()
			print 'export info : %s' % filename

#-------------------------------------------------------------------------------------
#
#	Create
#
#-------------------------------------------------------------------------------------
class CreateChannelLayers:
	def __init__( self, channelObject ):
		self.channelObject = channelObject
	
	def createList( self, layerInfo ):
		indexMap = dict()
		for i in layerInfo:
			indexMap[ layerInfo[i]['index'] ] = i
		indexList = indexMap.keys()
		indexList.sort( reverse=True )
		result = list()
		for i in indexList:
			result.append( indexMap[i] )
		return result

	def createLayer( self, parentObject, layerInfo ):
		# ProceduralLayer
		if layerInfo['type'] == 'ProceduralLayer':
			clayer = eval( 'parentObject.create%s("%s", "%s")' % \
					( layerInfo['type'], layerInfo['name'], layerInfo['proceduralType'] ) )
			paramMap = layerInfo['proceduralParam']
			for param in paramMap:
				value = paramMap[param]
				if type(value).__name__ == 'unicode':
					if value.find('Color') > -1:
						clayer.setProceduralParameter( param, eval('mari.%s' % value) )
					else:
						clayer.setProceduralParameter( param, value )
				else:
					clayer.setProceduralParameter( param, value )
		elif layerInfo['type'] == 'AdjustmentLayer':
			clayer = eval( 'parentObject.create%s("%s", "%s")' % \
					( layerInfo['type'], layerInfo['name'], layerInfo['adjustmentType'] ) )
			paramMap = layerInfo['adjustmentParam']
			for param in paramMap:
				clayer.setPrimaryAdjustmentParameter( param, paramMap[param] )
		elif layerInfo['type'] == 'PaintableLayer':
			clayer = eval( 'parentObject.create%s("%s")' % \
					( layerInfo['type'], layerInfo['name'] ) )
			if layerInfo.has_key('image'):
				imgSet = clayer.imageSet()
				imgSet.importImages( layerInfo['image'], imgSet.SCALE_THE_PATCH )
		else:
			clayer = eval( 'parentObject.create%s("%s")' % \
					( layerInfo['type'], layerInfo['name'] ) )
		
		clayer.setVisibility( layerInfo['vis'] )
		clayer.setBlendMode( layerInfo['mode'] )
		clayer.setBlendAmount( layerInfo['amount'] )

		# Mask
		if layerInfo['hasMask']:
			clayer.makeMask()
			clayer.setMaskEnabled( layerInfo['isMaskEnabled'] )
			if layerInfo.has_key('maskImage'):
				imgSet = clayer.maskImageSet()
				imgSet.importImages( layerInfo['maskImage'], imgSet.SCALE_THE_PATCH )
		return clayer

	def createAdjustmentStack( self, parentObject, adjustInfo ):
		adjustObject = parentObject.makeAdjustmentStack()
		for i in self.createList( adjustInfo ):
			layerObject = self.createLayer( adjustObject, adjustInfo[i] )
	
	def createGroupLayers( self, parentObject, groupInfo ):
		groupObject = parentObject.groupStack()
		for i in self.createList( groupInfo ):
			layerObject = self.createLayer( groupObject, groupInfo[i] )
			# AdjustmentStack
			if groupInfo[i].has_key('adjust'):
				adjustInfo = groupInfo[i]['adjust']
				self.createAdjustmentStack( layerObject, adjustInfo )

	def createLayers( self, channellayersInfo ):
		for i in self.createList( channellayersInfo ):
			layerInfo   = channellayersInfo[i]
			layerObject = self.createLayer( self.channelObject, layerInfo )
			# AdjustmentStack
			if layerInfo.has_key('adjust'):
				adjustInfo = layerInfo['adjust']
				self.createAdjustmentStack( layerObject, adjustInfo )
			# GroupLayer
			if layerInfo['type'] == 'GroupLayer':
				groupInfo = layerInfo['group']
				self.createGroupLayers( layerObject, groupInfo )
	
	# import json
	def importInfo( self, filename ):
		f = open( filename, 'r' )
		data = json.load( f )
		f.close()
		self.createLayers( data )
		print 'import info : %s' % filename


# build scene
def buildScene( importPath, cState ):
	# create layers
	for i in cState:
		if cState[i]:
			infofile = os.path.join( importPath, '%s.json' % i )
			if os.path.exists( infofile ):
				channelObject = mari.geo.current().channel(i)
				createClass   = CreateChannelLayers( channelObject )
				createClass.importInfo( infofile )

#-------------------------------------------------------------------------------
#
#	Export SceneInfo
#
#-------------------------------------------------------------------------------
def exportScene( exportPath, cState ):
	data = dict()
	data = dict()
	geo = mari.geo.current()
	ver = geo.currentVersion()
	data['objects'] = list( ver.meshPaths() )
	channelMap = dict()
	for i in cState:
		if cState[i]:
			channelMap[i] = getChannelInfo( geo.channel(i) )
	if channelMap:
		data['channels'] = channelMap
	
	if exportPath:
		filename = os.path.join( exportPath, '%s_channels.json' % mari.projects.current().name() )
		f = open( filename, 'w' )
		json.dump( data, f, indent=4 )
		f.close()

		# export layers
		for i in data['channels']:
			channelClass = ChannelLayers( geo.channel(i) )
			channelClass.m_exportPath = exportPath
			channelClass.exportInfo()

