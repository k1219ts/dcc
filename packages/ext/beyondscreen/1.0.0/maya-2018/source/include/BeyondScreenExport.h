#ifndef _BeyondScreenExport_h_
#define _BeyondScreenExport_h_

#include <Common.h>

class BeyondScreenExport : public MPxCommand
{
	public:

		static MString name;

	public:

		virtual MStatus doIt( const MArgList& );
		virtual bool isUndoable() const { return false; }

		static void *creator() { return new BeyondScreenExport; }
		static MSyntax newSyntax();

	private:

        int     getStartFrame    ( const MArgDatabase& argData );
        int     getEndFrame      ( const MArgDatabase& argData );
		MString getScreenName    ( const MArgDatabase& argData );
		MString getAimName       ( const MArgDatabase& argData );
		MString getCameraName    ( const MArgDatabase& argData );
		MString getFilePath      ( const MArgDatabase& argData );
		MString getFileName      ( const MArgDatabase& argData );

        void execute
        (
            const int      startFrame,
            const int      endFrame,
            const MString& screenName,
            const MString& aimName,
            const MString& cameraName,
            const MString& filePath,
            const MString& fileName
        );
};

#endif

