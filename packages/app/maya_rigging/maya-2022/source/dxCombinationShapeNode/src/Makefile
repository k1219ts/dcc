#-
# ==========================================================================
# Copyright (c) 2011 Autodesk, Inc.
# All rights reserved.
# 
# These coded instructions, statements, and computer programs contain
# unpublished proprietary information written by Autodesk, Inc., and are
# protected by Federal copyright law. They may not be disclosed to third
# parties or copied or duplicated in any form, in whole or in part, without
# the prior written consent of Autodesk, Inc.
# ==========================================================================
#+

ifndef INCL_BUILDRULES

#
# Include platform specific build settings
#
TOP := ../..
include $(TOP)/buildrules


#
# Always build the local plug-in when make is invoked from the
# directory.
#
all : plugins

endif

#
# Variable definitions
#

SRCDIR := .
DSTDIR := .

SOURCES  := $(SRCDIR)/dxCombinationShapeNode.cpp
OBJECTS  := $(SRCDIR)/dxCombinationShapeNode.o
PLUGIN   := $(DSTDIR)/dxCombinationShapeNode.$(EXT)
MAKEFILE := $(DSTDIR)/Makefile

#
# Include the optional per-plugin Makefile.inc
#
#    The file can contain macro definitions such as:
#       {pluginName}_EXTRA_CFLAGS
#       {pluginName}_EXTRA_C++FLAGS
#       {pluginName}_EXTRA_INCLUDES
#       {pluginName}_EXTRA_LIBS
-include $(SRCDIR)/Makefile.inc


#
# Set target specific flags.
#

$(OBJECTS): CFLAGS   := $(CFLAGS)   $(EXTRA_CFLAGS)
$(OBJECTS): C++FLAGS := $(C++FLAGS) $(EXTRA_C++FLAGS)
$(OBJECTS): INCLUDES := $(INCLUDES) $(EXTRA_INCLUDES)

depend:			INCLUDES := $(INCLUDES) $(EXTRA_INCLUDES)

$(PLUGIN):  LFLAGS   := $(LFLAGS) $(EXTRA_LFLAGS) 
$(PLUGIN):  LIBS     := $(LIBS)   -lOpenMaya -lOpenMayaAnim -lOpenMayaRender -lFoundation -lclew $(EXTRA_LIBS) 

#
# Rules definitions
#

$(PLUGIN): $(OBJECTS) 
	-rm -f $@
	$(LD) -o $@ $(LFLAGS) $^ $(LIBS)

plugins: $(PLUGIN)

depend:
	makedepend $(INCLUDES) $(MDFLAGS) -f$(DSTDIR)/Makefile $(SOURCES)

clean:
	-rm -f $(OBJECTS)

Clean:
	-rm -f $(MAKEFILE).bak $(OBJECTS) $(PLUGIN)
	
install: $(PLUGIN)
	install -m 0755 $(PLUGIN) $(TOP)/../plug-ins/
