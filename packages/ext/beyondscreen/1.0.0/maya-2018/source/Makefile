PROJECT_NAME := BeyondScreenForMaya.so

MAYA_PATH    := $(MAYA_LOCATION)
OPENEXR_PATH := $(REZ_OPENEXR_ROOT)

PLUGINSDIR   := ../plug-ins
HEADER_PATH  := ./include
SOURCE_PATH  := ./src
OBJECT_PATH  := ./object
OUTPUT_PATH  := ./output

CPP_SOURCES  := $(wildcard $(SOURCE_PATH)/*.cpp)
CPP_LIST     := $(notdir $(CPP_SOURCES))
OBJ_LIST     := $(addprefix $(OBJECT_PATH)/,$(CPP_LIST:.cpp=.o))

CC           := g++
CCFLAGS      := -O3 -m64 -fPIC -std=c++11
CCFLAGS      += -pthread -pipe -D_BOOL -DLINUX -DREQUIRE_IOSTREAM -Wno-deprecated -fno-gnu-keywords -fPIC
LDFLAGS      := -shared
INCLUDES     := -I./$(HEADER_PATH)
INCLUDES     += -I$(MAYA_PATH)/include
INCLUDES     += -I/usr/local/include
INCLUDES     += -I$(OPENEXR_PATH)/include
LINKS        := -lglut -lGLU -lGL -lGLEW
LINKS        += -L$(MAYA_PATH)/lib
LINKS        += -L$(OPENEXR_PATH)/lib
LINKS        += -lOpenMaya -lOpenMayaAnim -lFoundation -lOpenMayaUI -lOpenMayaFX
LINKS        += -L/usr/local/lib
LINKS        += -lIlmImf -lIlmThread -lHalf -ljpeg -lpthread -lgomp
SWITCHES     :=

.SILENT:

all: install

$(PROJECT_NAME): $(OBJ_LIST)
	@echo 'Building: $(notdir $@)'
	@mkdir -p $(OUTPUT_PATH)
	$(CC) $(LDFLAGS) $(LINKS) -o $(OUTPUT_PATH)/$@ $(OBJ_LIST)

$(OBJECT_PATH)/%.o: $(SOURCE_PATH)/%.cpp
	@echo 'Compiling: $(notdir $<) -> $(notdir $@)'
	@mkdir -p `dirname $@`
	$(CC) -c $(CCFLAGS) $(INCLUDES) $(SWITCHES) -c $< -o $@

install: $(PROJECT_NAME)
	@mkdir -p $(PLUGINSDIR)
	cp $(OUTPUT_PATH)/$(PROJECT_NAME) $(PLUGINSDIR)

clean:
	rm -f $(OBJECT_PATH)/*.o $(OUTPUT_PATH)/$(PROJECT_NAME)
