KATANA_HOME = /opt/Katana3.2v3

PRODUCTNAME = USDSkelSkinning

# The output objects dir
OBJDIR = ./out
BUILDDIR = ./build

OUTFILENAME = $(PRODUCTNAME).so
OUTFILEPATH = $(OBJDIR)/$(OUTFILENAME)

RESOURCESOPDIR = $(BUILDDIR)/Ops

# DEFAULT
INCLUDES = -I/usr/include
LIBPATH  = -L/usr/lib -L/usr/lib64

# TBB
INCLUDES += -I/backstage/libs/tbb/2017_U8/gcc-4.8.5/include

# USD
# USD_ROOT = /backstage/libs/USD/pixar-usd/20.02
USD_ROOT = /WORK_DATA/Develop/libs/USD/pixar-usd/20.02

# USD KATANA
USD_KAT_ROOT = $(USD_ROOT)/katana3.2
INCLUDES += -I$(USD_KAT_ROOT)/include
INCLUDES += -I$(USD_KAT_ROOT)/third_party/katana/include
LIBPATH  += -L$(USD_KAT_ROOT)/lib
LIBPATH  += -L$(USD_KAT_ROOT)/third_party/katana/lib

# Plugin sources and includes
PLUGIN_SRC = $(KATANA_HOME)/plugin_apis/src
SOURCES = $(shell find src -name \*.cpp)

# Directly suck in the FnAttribute/FnGeolibOp/FnGeolibUtil/pystring cpp files
SOURCES += $(shell find $(PLUGIN_SRC)/FnAttribute -name \*.cpp)
SOURCES += $(shell find $(PLUGIN_SRC)/FnGeolib/op -name \*.cpp)
SOURCES += $(shell find $(PLUGIN_SRC)/FnGeolib/util -name \*.cpp)
SOURCES += $(shell find $(PLUGIN_SRC)/FnGeolibServices/client -name \*.cpp)
SOURCES += $(shell find $(PLUGIN_SRC)/FnPluginManager -name \*.cpp)
SOURCES += $(shell find $(PLUGIN_SRC)/pystring -name \*.cpp)
SOURCES += $(PLUGIN_SRC)/FnPluginSystem/FnPlugin.cpp
SOURCES += $(PLUGIN_SRC)/FnLogging/client/FnLogging.cpp
SOURCES += $(shell find $(PLUGIN_SRC)/FnAsset -name \*.cpp)

INCLUDES += -I$(KATANA_HOME)/plugin_apis/include -I$(KATANA_HOME)/bin/python2.7/include/python2.7
INCLUDES += -Iinclude
LIBPATH += -L$(KATANA_HOME)/bin
LIBS = -lCEL
LIBS += -ldl -ltf -lgf -lsdf -lusd -lusdGeom -lusdKatana


CFLAGS=-ffloat-store
CXXFLAGS=-Wall -Wextra -Wshadow -Wconversion -Wcast-qual -fPIC -DPIC
CXXFLAGS=-fPIC -DPIC

# Initialize FnAsset API hosts for this Op
CXXFLAGS += -DFNGEOLIBOP_INIT_FNASSET_HOSTS -std=c++11

# Object files and flags
OBJS := $(patsubst %.cpp,$(OBJDIR)/%.o,$(SOURCES))

# Targets:
all: resources

$(OUTFILEPATH): $(OBJS)
	@echo "  Compiling $(PRODUCTNAME)"
	$(CXX) $(CXXFLAGS) $(OBJS) $(LIBPATH) $(LIBS) -shared -o $(OUTFILEPATH) -Wl,-soname,$(OUTFILENAME)

$(OBJDIR)/%.o: %.cpp
	@mkdir -p `dirname $@`
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

resources: $(OUTFILEPATH)
	@mkdir -p $(RESOURCESOPDIR)
	cp $(OUTFILEPATH) $(RESOURCESOPDIR)

clean:
	@echo "  Cleaning $(PRODUCTNAME)"
	@rm -rf $(OBJDIR)
	@rm -f $(BUILDDIR)/Ops/$(PRODUCTNAME).so
