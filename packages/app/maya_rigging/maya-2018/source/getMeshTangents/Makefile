TOP := ..
include $(TOP)/buildrules

SRCDIR := .
DSTDIR := .
LIBS   := $(LIBS) -lOpenMaya -lOpenMayaAnim -lOpenMayaFX -lOpenMayaRender -lFoundation -lclew $(EXTRA_LIBS)

%.so: %.o
	$(LD) -o $@ $(LFLAGS) $^ $(LIBS)
	install -m 0755 $@ $(TOP)/../plug-ins/
clean:
	rm -f *.o *.so
