
KATANA_HOME = /opt/Katana3.5v4

PRODUCTNAME = USDNurbsCurves

# The output objects dir
BUILDDIR = ../../Resources
OBJDIR = ./build/out

OUTFILENAME = $(PRODUCTNAME).so
OUTFILEPATH = $(OBJDIR)/$(OUTFILENAME)

RESOURCESOPDIR = $(BUILDDIR)/Ops

# DEFAULT
INCLUDES = -I/usr/include
LIBPATH  = -L/usr/lib -L/usr/lib64

# TBB
INCLUDES += -I$(KATANA_HOME)/external/FnTBB/include

# BOOST
INCLUDES += -I$(KATANA_HOME)/external/FnBoost/include

# KATANA USD
INCLUDES += -I$(KATANA_HOME)/external/FnUSD/include
LIBPATH += -L$(KATANA_HOME)/bin

# BUILD KATANA USD
BUILD_PLUGIN = /backstage/libs/usd_katana/19.11.3/katana-3.5/third_party/katana
INCLUDES += -I$(BUILD_PLUGIN)/lib/usd/include
LIBPATH  += -L$(BUILD_PLUGIN)/lib/usd/libs

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

LIBS += -lusdKatana -lvtKatana
LIBS += -lFntf -lFngf -lFnsdf -lFnusd -lFnusdGeom

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
