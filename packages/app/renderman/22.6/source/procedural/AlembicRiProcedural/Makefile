include ../../Makefile.base

PROJECT_NAME      := AlembicRiProcedural.so
TARGET_PATH		  := ../../../lib

#----------------------------------------- boost
INCLUDES           	 	 += -I$(EXTERN_LIBRARY_PATH)/include/boost
LINKS					 += -L$(EXTERN_LIBRARY_PATH)/lib
LINKS                    += -lboost_regex

#-------------------------------------------Alembic
INCLUDES				+= -I$(EXTERN_LIBRARY_PATH)/include
LINKS					+= -L$(EXTERN_LIBRARY_PATH)/lib
LINKS					+= -lAlembic
#-------------------------------------------OpenEXR/IlmBase
INCLUDES				+= -I$(EXTERN_LIBRARY_PATH)/include/OpenEXR
LINKS					+= -L$(EXTERN_LIBRARY_PATH)/lib
LINKS					+= -lHalf -lIex -lIexMath -lIlmThread -lImath
#-------------------------------------------hdf5
INCLUDES				+= -I$(EXTERN_LIBRARY_PATH)/include/hdf5
LINKS					+= -L$(EXTERN_LIBRARY_PATH)/lib
LINKS					+= -lhdf5
#-------------------------------------------json
INCLUDES				+= -I$(EXTERN_LIBRARY_PATH)/include/json-c
LINKS					+= -ljson-c
#-------------------------------------------------------Prman
INCLUDES				+= -I$(PRMAN_INSTALL_PATH)/include
LINKS					+= -L$(PRMAN_INSTALL_PATH)/lib

LINKS					+= -lprman-$(PRMAN_VER)

#-------------------------------------------------------------------- library link end

INCLUDES				+= -I$(HEADER_PATH) -I$(SOURCE_PATH)

.SILENT:

all: $(PROJECT_NAME)
	@echo -e '\e[1;33m$(PROJECT_NAME)::build complete.\e[m'

# .o -> .so
$(PROJECT_NAME): $(OBJ_LIST)
	@echo -e '\e[1;35mBuilding : $@\e[m'
	@echo -e '\e[1;32m ----- $$MAYA_INSTALL_PATH=$(MAYA_INSTALL_PATH)\e[m'
	@echo -e '\e[1;32m ----- $$PRMAN_INSTALL_PATH=$(PRMAN_INSTALL_PATH)\e[m'
	@echo -e '\e[1;32m ----- $$REDSHIFT_SDK_PATH=$(REDSHIFT_SDK_PATH)\e[m'
	@echo -e '\e[1;32m ----- $$EXTERN_LIBRARY_PATH=$(EXTERN_LIBRARY_PATH)\e[m'
	@echo -e '\e[1;32m ----- $$ZELOS_PATH=$(ZELOS_PATH)\e[m'

	$(CC) $(LDFLAGS) $(LINKS) -o ./$(TARGET_PATH)/$@ $(OBJ_LIST)

# .cpp -> .o
$(OBJECT_PATH)/%.o: $(SOURCE_PATH)/%.cpp
	#@echo 'Compiling: $(notdir $<) -> $(notdir $@)'
	$(CC) $(CCFLAGS) $(INCLUDES) $(WARNINGS) -c $< -o $@

# .cu -> .o
$(OBJECT_PATH)/%.o: $(SOURCE_PATH)/%.cu
	#@echo 'Compiling: $(notdir $<) -> $(notdir $@)'
	$(NVCC) $(CUFLAGS) $(INCLUDES) -c $< -o $@


clean:
	rm -f $(OBJECT_PATH)/*.o  $(TARGET_PATH)/$(PROJECT_NAME)
