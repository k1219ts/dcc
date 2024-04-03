include ../../Makefile.base

PROJECT_NAME 	:= DxHoudiniOcean
TARGET_PATH 	:= ../../../lib/plugins

#-------------------------------------------------------json-c
INCLUDES		+= -I/backstage/libs/jsonc/0.13/include
LINKS			+= -L/backstage/libs/jsonc/0.13/lib
LINKS			+= -ljson-c
LDFLAGS   		+= -Wl,-rpath=/backstage/libs/jsonc/0.13/lib

#-------------------------------------------------------boost/filesystem
INCLUDES 		+= -I/backstage/libs/boost/1.61.0/include
LINKS			+= -L/backstage/libs/boost/1.61.0/lib
LINKS			+= -lboost_filesystem -lboost_system
LDFLAGS   		+= -Wl,-rpath=/backstage/libs/boost/1.61.0/lib

#-------------------------------------------------------Prman
INCLUDES		+= -I$(PRMAN_INSTALL_PATH)/include
LINKS			+= -L$(PRMAN_INSTALL_PATH)/lib
LINKS			+= -lprman-$(PRMAN_VER)
INCLUDES		+= -I$(HEADER_PATH) -I$(SOURCE_PATH)
INCLUDES		+= -I$(PRMAN_EXAMPLE_PATH)/plugins/include


.SILENT:

all: $(PROJECT_NAME)
	@echo -e '\e[1;33m$(PROJECT_NAME)::build complete.\e[m'

# .o -> .so
$(PROJECT_NAME): $(OBJ_LIST)
	@echo -e `gcc --version`
	@echo -e '\e[1;35mBuilding : $@\e[m'
	@echo -e '\e[1;32m ----- $$PRMAN_INSTALL_PATH=$(PRMAN_INSTALL_PATH)\e[m'

	$(CC) $(LDFLAGS) $(LINKS) -o ./$(TARGET_PATH)/$@.so $(OBJ_LIST)

	mkdir -p -m 755 ./$(TARGET_PATH)/Args
	cp -f $(SOURCE_PATH)/$(PROJECT_NAME).a ./$(TARGET_PATH)/Args/$(PROJECT_NAME).args

# .cpp -> .o
$(OBJECT_PATH)/%.o: $(SOURCE_PATH)/%.cpp
	mkdir -p $(OBJECT_PATH)
	$(CC) $(CCFLAGS) $(INCLUDES) $(WARNINGS) -c $< -o $@

clean:
	rm -f $(OBJECT_PATH)/*.o
	rm -vf $(TARGET_PATH)/$(PROJECT_NAME).so
	rm -vf $(TARGET_PATH)/Args/$(PROJECT_NAME).args
