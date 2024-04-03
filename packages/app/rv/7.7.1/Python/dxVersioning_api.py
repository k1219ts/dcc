
##
##  This module provides the "front end" of the sample versioning API, including:
##
##  ** the VersionData class
##  ** storage and retrieval of VersionData in the session
##  ** version switching
##

import sys
import os

doDebug = False
if (os.getenv("RV_VERSIONING_DEBUG")) :
    doDebug = True

def deb(s) :
    # print "[api]" + s + "\n"
    if (doDebug) :
        sys.stderr.write ("vapi: " + s + "\n")

from rv.commands import *
import rv.rvtypes
import rv.extra_commands

import dxTacticCommon

def error(s) :
        sys.stderr.write ("ERROR: " + s + "\n")

class VersionData :
    ## XXX todo: stereo media, audio media, audio offset, stereo Offset ?
    ## RV_PATHSWAP processing

    def __init__ (self) :

        self._show        = ""
        self._media       = ()
        self._task        = ()
        self._name        = ()
        self._rangeOffset = ()
        self._cutIn       = ()
        self._cutOut      = ()
        self._audioOffset = ()
        self._color       = ()
        self._crop        = ()
        self._uncrop      = ()
        self._current     = 0
        self._last        = 0
        self._currentTask = 0
        self._source      = ""

    def sanityCheck (self) :

	numVersions = len(self._media)

	if (numVersions == 0) :
	    return False

	for prop in (self._name, self._rangeOffset, self._cutIn, self._cutOut, self._audioOffset, self._color, self._crop, self._uncrop) :
	    if (len(prop) != 0 and len(prop) != numVersions) :
	        return False

        if (self._current < 0 or self._current >= numVersions) :
	    return False

        if (self._last < 0 or self._last >= numVersions) :
	    return False

	return True

    def empty (self) :
        return (len(self._media) == 0)

    def mediaArray (self, index) :
        m = []
    	for p in self._media[index].split("|") :
    	    if (p != "") :
    		m.append(p)
            return m

    def size (self) :
        return len(self._media)

    def _updateSource (self, targetSource, index) :
        deb ("updateSource '%s'\n" % targetSource)

        if (len(self._rangeOffset) == self.size()) :
            prop = targetSource + ".group.rangeOffset"

            setIntProperty (prop, [ self._rangeOffset[index] ], True)

        if (len(self._cutIn) == self.size()) :
            prop = targetSource + ".cut.in"

            setIntProperty (prop, [ self._cutIn[index] ], True)
	    deb ("set cut in to %s" %  self._cutIn[index])

        if (len(self._cutOut) == self.size()) :
            prop = targetSource + ".cut.out"

            setIntProperty (prop, [ self._cutOut[index] ], True)
	    deb ("set cut out to %s" %  self._cutOut[index])

        if (len(self._audioOffset) == self.size()) :
            prop = targetSource + ".group.audioOffset"

            setFloatProperty (prop, [ self._audioOffset[index] ], True)
	    deb ("set audioOffset to %s" %  self._audioOffset[index])

        if (len(self._crop) == self.size()) :
	    c = self._crop[index]
	    formatNode = rv.extra_commands.associatedNode("RVFormat", targetSource)
            prop = formatNode + ".crop."
	    setIntProperty (prop + "xmin", [ c[0] ], True)
	    setIntProperty (prop + "ymin", [ c[1] ], True)
	    setIntProperty (prop + "xmax", [ c[2] ], True)
	    setIntProperty (prop + "ymax", [ c[3] ], True)

	    active = 0
	    if (c[2] > c[0] or c[3] > c[1]) :
	        active = 1

	    setIntProperty (prop + "active", [ active ], True)

        if (len(self._uncrop) == self.size()) :
	    uc = self._uncrop[index]
	    formatNode = rv.extra_commands.associatedNode("RVFormat", targetSource)
            prop = formatNode + ".uncrop."
	    setIntProperty (prop + "width",  [ uc[0] ], True)
	    setIntProperty (prop + "height", [ uc[1] ], True)
	    setIntProperty (prop + "x",      [ uc[2] ], True)
	    setIntProperty (prop + "y",      [ uc[3] ], True)

	    active = 0
	    if (uc[0] > 0 and uc[1] > 0) :
	        active = 1

	    setIntProperty (prop + "active", [ active ], True)

    def setVersion (self, index) :
        print ("setVersion index %s\n" % index)

        if (self.empty()) :
	    return

        if (type(index) == type({})) :
            index = index['_int']

        propPrefix  = self._source + ".versioning."
        lastProp    = propPrefix + "lastIndex"
        currentProp = propPrefix + "currentIndex"
        oldIndex    = 0

        if (not propertyExists (lastProp)) :
            newProperty (lastProp, IntType, 1)
        if (not propertyExists (currentProp)) :
            newProperty (currentProp, IntType, 1)
        else :
            oldIndex = getIntProperty(currentProp)[0]

        setIntProperty (lastProp,    [ oldIndex ], True)
        setIntProperty (currentProp, [ index ],    True)

        self._current = index
        self._last    = oldIndex

        cacheSourceFrame (self._source)
        mode = cacheMode()
        setCacheMode(CacheOff)
        #   In case the new media needs different views, clear out the request.

        setStringProperty (self._source + ".request.stereoViews", [], True)

        deb ("    setSourceMedia source %s media '%s'\n" % (self._source, self.mediaArray(index)))
        setSourceMedia (self._source, self.mediaArray(index), "versioning")

        self._updateSource (self._source, index)

        setCacheMode(mode)


    def setTask (self, index) :
        deb ("setTask index %s\n" % index)

        if (self.empty()) :
	    return

        if (type(index) == type({})) :
            index = index['_int']

        deb ('task idx: %s %s' % (self._task[index], index))

        fileName = self._media[self._current].replace('/Volumes', '')
        tmp = os.path.basename(fileName).split('_')
        shotName = '_'.join(tmp[:2])

        if self._task[index] == 'edit':
            self._media, self._color = dxTacticCommon.getBreakdown(self._show, shotName)
        else:
            self._media, self._color = dxTacticCommon.getSnapshot(self._show, shotName, self._task[index])

        # print '[api] call media', self._media
        # print '[api] call color', self._color

        self._name = []
        for i in self._media:
            self._name.append(os.path.basename(i))

        propPrefix  = self._source + ".versioning."
        currentTask = propPrefix + "currentTask"
        media       = propPrefix + "media"
        name        = propPrefix + "name"
        color       = propPrefix + "color"
        current     = propPrefix + "currentIndex"
        last        = propPrefix + "lastIndex"

        # RELOAD SNAPSHOT LIST
        # if (not propertyExists (currentTask)) :
        #     newProperty (currentTask, IntType, 1)
        setIntProperty(currentTask, [index], True)
        setIntProperty(current, [0], True)
        setIntProperty(last, [0], True)
        setStringProperty(media, list(self._media), True)
        setStringProperty(name, list(self._name), True)
        setFloatProperty(color, self._listFromTupleOfTuples(self._color), True)

        self._currentTask = index
        deb ('currentTask: %s' % self._currentTask)

        # index = len(self._media) - 1

        cacheSourceFrame (self._source)
        mode = cacheMode()
        setCacheMode(CacheOff)

        deb ("    setSourceMedia source %s media '%s'\n" % (self._source, self.mediaArray(0)))
        setSourceMedia (self._source, self.mediaArray(0), "versioning")

        setCacheMode(mode)

    def cycleVersion (self) :
        if (self.empty()) :
	    return

        cur   = self._current
        next  = 0 if (cur == self.size()-1) else cur+1

        self.setVersion (next)

    def _sourceIsThisSingleVersion (self, candidateSource, index) :
        ##
        ##  Check to see if this source is a "single version" source we've created
        ##
        deb ("_sourceIsThisSingleVersion csource %s index %s vsource %s" % (candidateSource, index, self._source))
        prop = candidateSource + ".versioning.singleVersionSource"

        return ((propertyExists(prop) and (getStringProperty(prop)[0] == self._source + ":" + str(index))))

    def _addSingleVersionSource (self, index) :
        ##
        ##  Add a "single version" source (an un-versioned source) for purposes of tiling or
        ##  comparing different versions of same source.
        ##
        deb ("_addSingleVersionSource index %s" % index)

        if (self.empty()) :
           return

        ##
        ##  We squirrel away inputs of default nodes and reset them afterward, since the sources we're
        ##  creating here are "temporary" and we don't want them showing up in the default views.
        ##
        sequenceInputs = nodeConnections ("defaultSequence", False)[0]
        layoutInputs   = nodeConnections ("defaultLayout", False)[0]
        stackInputs    = nodeConnections ("defaultStack", False)[0]
        newSource      = addSourceVerbose (self.mediaArray(index), "explicit")

        setNodeInputs ("defaultSequence", sequenceInputs)
        setNodeInputs ("defaultlLayout",  layoutInputs)
        setNodeInputs ("defaultStack",    stackInputs)

        ##
        ##  Mark new source as being a "single version source" of this source, so we can find it later.
        ##
        prop = newSource + ".versioning.singleVersionSource"
        newProperty (prop, StringType, 1)
        setStringProperty (prop, [ self._source + ":" + str(index) ], True)

        rv.extra_commands.setUIName (nodeGroup (newSource), self._name[index])

        self._updateSource (newSource, index)

        ##
        ##  'Hide' new source in TMP folder
        ##
        if (not nodeExists ("FolderTMP")) :
            newNode ("RVFolderGroup", "FolderTMP")
            rv.extra_commands.setUIName ("FolderTMP", "TMP Folder")

        tmpInputs = nodeConnections ("FolderTMP", False)[0]
        tmpInputs.append (nodeGroup(newSource))
        setNodeInputs ("FolderTMP", tmpInputs)

        return newSource

    def _singleVersionSource (self, index) :
        deb ("singleVersionSource index %s" % index)

        if (self.empty()) :
            return None

        deb ("    getting source nodes")

        sourceNodes = nodesOfType ("RVFileSource")

        singleSource = None

        deb ("    searching source nodes %s" % sourceNodes)
        for n in sourceNodes :
            if (self._sourceIsThisSingleVersion (n, index)) :
                singleSource = n
                break

        if (not singleSource) :
            singleSource = self._addSingleVersionSource (index)

        return singleSource

    def createVersionGroup (self, indices, groupType) :
        deb ("createVersionGroup indices %s" % indices)

        if (self.empty()) :
            return None

        singleSources = []
        for i in indices :
            singleSources.append (nodeGroup (self._singleVersionSource (i)))

        newGroup = newNode ("RV" + groupType + "Group", groupType + "Group000000")

        rv.extra_commands.setUIName (newGroup, "Version " + groupType)

        setNodeInputs (newGroup, singleSources)

        return newGroup


    def _ensurePropExistsAndSet (self, sourceNode, propName, propType, value) :

    	deb ("ensure source %s prop %s type %s val %s" % (sourceNode, propName, str(propType), str(value)))

    	fullPropName = sourceNode + ".versioning." + propName
    	deb ("    fullPropName '%s'" % fullPropName)

    	if (not propertyExists(fullPropName)) :
    	    deb ("    making new prop")
    	    newProperty (fullPropName, propType, 1)

    	if   (propType == IntType) :
    	    setIntProperty (fullPropName, value, True)
    	elif (propType == StringType) :
    	    setStringProperty (fullPropName, value, True)
    	elif (propType == FloatType) :
    	    setFloatProperty (fullPropName, value, True)

    def _listFromTupleOfTuples(self, tup) :
    	l = []
    	for t in tup :
    	    l += list(t)

    	return l

    def setVersionDataOnSource (self, sourceNode) :

    	deb ("setVersionDataOnSource sourceNode %s" % sourceNode)

    	if (not sourceNode) :
    	    error ("can't set VersionData on null source")
    	    return

    	if (not self._media) :
    	    error ("can't set VersionData with empty media")
    	    return

    	if (not self.sanityCheck()) :
    	    error ("VersionData sanity check failed, can't set")
    	    return

    	if (nodeType(sourceNode) != "RVFileSource") :
    	    error ("can't set VersionData on non-FileSource node")
    	    return

    	s = sourceNode

    	self._source = s

        self._ensurePropExistsAndSet(s, "show",   StringType, [self._show])
    	self._ensurePropExistsAndSet(s, "media",  StringType, list(self._media))
        self._ensurePropExistsAndSet(s, "task",   StringType, list(self._task))

    	self._ensurePropExistsAndSet(s, "name",         StringType, list(self._name))
    	self._ensurePropExistsAndSet(s, "rangeOffset",  IntType,    list(self._rangeOffset))
    	self._ensurePropExistsAndSet(s, "cutIn",        IntType,    list(self._cutIn))
    	self._ensurePropExistsAndSet(s, "cutOut",       IntType,    list(self._cutOut))
    	self._ensurePropExistsAndSet(s, "audioOffset",  FloatType,  list(self._audioOffset))

    	self._ensurePropExistsAndSet(s, "currentIndex", IntType,    [self._current])
    	self._ensurePropExistsAndSet(s, "lastIndex",    IntType,    [self._last])
        self._ensurePropExistsAndSet(s, "currentTask",  IntType,    [self._currentTask])

    	#
    	#   These are tuples of tuples:
    	#
    	self._ensurePropExistsAndSet(s, "color",        FloatType,  self._listFromTupleOfTuples(self._color))
    	self._ensurePropExistsAndSet(s, "crop",         IntType,    self._listFromTupleOfTuples(self._crop))
    	self._ensurePropExistsAndSet(s, "uncrop",       IntType,    self._listFromTupleOfTuples(self._uncrop))

##  class VersionData end


def _formSourceName (sourceName) :
    s = sourceName

    if (not s) :
        sourceList = sourcesRendered()

        if (len(sourceList) > 0) :
            s = nodeGroup(sourceList[0]["node"])

    if (not nodeExists(s)) :
        return None

    if (nodeType(s) == "RVSourceGroup") :
        s += "_source"

    if (nodeType(s) != "RVFileSource") :
        return None

    return s

def sourceHasVersionData (sourceName = None) :
    ##  deb ("sourceHasVersioData")

    if (sourceName == "") :
        sourceName = None

    s = _formSourceName (sourceName)

    if (not s) :
        return False

    propPrefix = s + ".versioning."

    if (propertyExists (propPrefix + "media")) :
        return True

    return False


def sourceHasTaskData (sourceName = None) :
    ##  deb ("sourceHasVersioData")

    if (sourceName == "") :
        sourceName = None

    s = _formSourceName (sourceName)

    if (not s) :
        return False

    propPrefix = s + ".versioning."

    if (propertyExists (propPrefix + "task")) :
        return True

    return False

def getVersionDataFromSource (sourceName = None) :

    ##  deb ("getVersionDataFromSource")
    ##  print "getVersionDataFromSource", sourceName
    if (sourceName == "") :
        sourceName = None

    s = _formSourceName (sourceName)

    ## deb ("    s %s" % s)
    vd = VersionData()

    if (not s) :
        return vd

    propPrefix = s + ".versioning."

    if (propertyExists (propPrefix + "show")) :
        vd._show       = tuple (getStringProperty (propPrefix + "show"))

    if (propertyExists (propPrefix + "media")) :
        vd._media       = tuple (getStringProperty (propPrefix + "media"))

    if (len(vd._media) == 0) :
        return vd

    if (propertyExists (propPrefix + "task")) :
        vd._task        = tuple (getStringProperty (propPrefix + "task"))

    if (propertyExists (propPrefix + "name")) :
        vd._name        = tuple (getStringProperty (propPrefix + "name"))

    if (propertyExists (propPrefix + "rangeOffset")) :
        vd._rangeOffset = tuple (getIntProperty    (propPrefix + "rangeOffset"))

    if (propertyExists (propPrefix + "cutIn")) :
        vd._cutIn       = tuple (getIntProperty    (propPrefix + "cutIn"))

    if (propertyExists (propPrefix + "cutOut")) :
        vd._cutOut      = tuple (getIntProperty    (propPrefix + "cutOut"))

    if (propertyExists (propPrefix + "audioOffset")) :
        vd._audioOffset = tuple (getFloatProperty  (propPrefix + "audioOffset"))

    if (len(vd._name) != len(vd._media)) :
        import os.path

        name = []
        for i in range(vd.size()) :
            name.append (os.path.basename (vd.mediaArray(i)[0]))
        vd._name = tuple(name)

    if (propertyExists (propPrefix + "currentIndex")) :
        vd._current     = getIntProperty    (propPrefix + "currentIndex")[0]
    else :
        newProperty    (propPrefix + "currentIndex", IntType, 1)
        setIntProperty (propPrefix + "currentIndex", [ 0 ], True)

    if (propertyExists (propPrefix + "lastIndex")) :
        vd._last        = getIntProperty    (propPrefix + "lastIndex")[0]
    else :
        newProperty    (propPrefix + "lastIndex", IntType, 1)
        setIntProperty (propPrefix + "lastIndex", [ 0 ], True)

    ##  deb ("    color")
    if (propertyExists (propPrefix + "color")) :
        p = getFloatProperty (propPrefix + "color")

        color = []
        count = vd.size()
        if (len(p) == 3 * count) :
            for i in range(count) :
                color.append ((p[3*i], p[3*i+1], p[3*i+2], 1))
        vd._color = tuple(color)

    if (propertyExists (propPrefix + "crop")) :
        p = getIntProperty (propPrefix + "crop")

        crop = []
        count = vd.size()
        if (len(p) == 4 * count) :
            for i in range(count) :
                crop.append ((p[4*i], p[4*i+1], p[4*i+2], p[4*i+3]))
        vd._crop = tuple(crop)

    if (propertyExists (propPrefix + "uncrop")) :
        p = getIntProperty (propPrefix + "uncrop")

        uncrop = []
        count = vd.size()
        if (len(p) == 4 * count) :
            for i in range(count) :
                uncrop.append ((p[4*i], p[4*i+1], p[4*i+2], p[4*i+3]))
        vd._uncrop = tuple(uncrop)

    vd._source = s

    ##  deb ("    vd %s\n" % vd)
    return vd

def cacheSourceFrame (source) :
    prop = source + ".versioning.cachedSourceFrame"

    if (not propertyExists (prop)) :
        newProperty (prop, IntType, 1)

    setIntProperty (prop, [ rv.extra_commands.sourceFrame(frame(), None) ], True)

def resetSourceFrame (source) :
    prop = source + ".versioning.cachedSourceFrame"

    if (propertyExists (prop)) :
        oldF = getIntProperty (prop)[0]
        if (oldF != 1000000) :
            deb ("pyResetSourceFrame oldF %d" % oldF)
            factor = rv.extra_commands.sourceFrame (frame(), None) - oldF
            deb ("pyResetSourceFrame factor %d" % factor)
            deb ("pyResetSourceFrame frame %d" % frame())
            targetFrame = min(frameEnd(), max(frameStart(), frame()-factor))
            deb ("pyResetSourceFrame target %d" % targetFrame)
            setFrame(targetFrame)
            # setFrame(frame()+1)
            deb ("pyResetSourceFrame setFrame OK")
            setIntProperty (prop, [ 1000000], True)

def staticSetVersion (version) :
    vd = getVersionDataFromSource ()

    if (vd) :
        vd.setVersion (version)

def staticSetTask (task) :
    vd = getVersionDataFromSource ()

    if (vd) :
        vd.setTask (task)
