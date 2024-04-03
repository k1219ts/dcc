include ../../Makefile.base

PROJECT_NAME    := DxManifold2D
TARGET_PATH		:= ../../../lib/plugins

#-------------------------------------------------------Prman
INCLUDES		+= -I$(PRMAN_INSTALL_PATH)/include
LINKS			+= -L$(PRMAN_INSTALL_PATH)/lib
LINKS			+= -lprman-$(PRMAN_VER)
INCLUDES		+= -I$(HEADER_PATH) -I$(SOURCE_PATH)

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
	$(CC) $(CCFLAGS) $(INCLUDES) $(WARNINGS) -c $< -o $@

clean:
	rm -f $(OBJECT_PATH)/*.o
	rm -vf $(TARGET_PATH)/$(PROJECT_NAME).so
	rm -vf $(TARGET_PATH)/Args/$(PROJECT_NAME).args
