include ../../Makefile.base

PROJECT_NAME	:= DxImageFile
TARGET_PATH		:= ../../../lib/plugins

# PRMAN_VER := 22.3
# PRMAN_INSTALL_PATH := /netapp/backstage/pub/apps/renderman2/applications/linux/RenderManProServer-$(PRMAN_VER)
#
# HEADER_PATH := .
# SOURCE_PATH := .
# OBJECT_PATH := .
# TARGET_PATH := .

#-------------------------------------------------------OpenImageIO
INCLUDES	+= -I$(EXTERN_LIBRARY_PATH)/include
LINKS		+= -L$(EXTERN_LIBRARY_PATH)/lib
LINKS		+= -lOpenImageIO

#-------------------------------------------------------Prman
INCLUDES	+= -I$(PRMAN_INSTALL_PATH)/include
LINKS		+= -L$(PRMAN_INSTALL_PATH)/lib

LINKS		+= -lprman-$(PRMAN_VER)

#-------------------------------------------------------------------- library link end
INCLUDES	+= -I$(HEADER_PATH) -I$(SOURCE_PATH)
INCLUDES	+= -I$(PRMAN_EXAMPLE_PATH)/plugins/pattern/texture


CCFLAGS := -O3 -m64 -fpic -fopenmp -std=c++11
LDFLAGS := -shared -fopenmp

DEFINES := -DGLM_ENABLE_EXPERIMENTAL

.SILENT:

all: $(PROJECT_NAME)
	@echo -e '\e[1;33m$(PROJECT_NAME)::build complete.\e[m'

# .o -> .so
$(PROJECT_NAME): $(OBJ_LIST)
	@echo -e '\e[1;35mBuilding : $@\e[m'
	@echo -e '\e[1;32m ----- $$PRMAN_INSTALL_PATH=$(PRMAN_INSTALL_PATH)\e[m'
	@echo -e '\e[1;32m ----- $$EXTERN_LIBRARY_PATH=$(EXTERN_LIBRARY_PATH)\e[m'

	$(CC) $(LDFLAGS) $(LINKS) -o ./$(TARGET_PATH)/$@.so $(OBJ_LIST)
	rm -f ./object/*.o

	mkdir -p -m 755 ./$(TARGET_PATH)/Args
	cp -f $(SOURCE_PATH)/$(PROJECT_NAME).a ./$(TARGET_PATH)/Args/$(PROJECT_NAME).args

# .cpp -> .o
$(OBJECT_PATH)/%.o: $(SOURCE_PATH)/%.cpp
	$(CC) $(CCFLAGS) $(INCLUDES) $(DEFINES) -c $< -o $@

clean:
	rm -f $(OBJECT_PATH)/*.o
	rm -vf $(TARGET_PATH)/$(PROJECT_NAME).so
