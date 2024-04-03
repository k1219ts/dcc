@echo off

set BASEDIR=%~dp0

echo ------ Copy ffmpeg Codec ------
rem xcopy %BASEDIR%\mio_ffmpeg\win10\*.* "C:\Program Files\Shotgun\RV-7.9.2\plugins\MovieFormats\" /y
xcopy %BASEDIR%\mio_ffmpeg\win10\*.* "C:%HOMEPATH%\AppData\Roaming\RV\MovieFormats\" /y

echo ------ Copy python module ------
xcopy %BASEDIR%\pylibs\dxstats\*.* "C:%HOMEPATH%\AppData\Roaming\RV\Python\dxstats\" /e /h /k /y
xcopy %BASEDIR%\pylibs\requests\*.* "C:%HOMEPATH%\AppData\Roaming\RV\Python\requests\" /e /h /k /y
xcopy %BASEDIR%\pylibs\pymongo\*.* "C:%HOMEPATH%\AppData\Roaming\RV\Python\pymongo\" /e /h /k /y
xcopy %BASEDIR%\pylibs\xlwt\*.* "C:%HOMEPATH%\AppData\Roaming\RV\Python\xlwt\" /e /h /k /y
xcopy %BASEDIR%\pylibs\bson\*.* "C:%HOMEPATH%\AppData\Roaming\RV\Python\bson\" /e /h /k /y
xcopy %BASEDIR%\pylibs\tactic_client_lib\*.* "C:%HOMEPATH%\AppData\Roaming\RV\Python\tactic_client_lib\" /e /h /k /y
xcopy %BASEDIR%\pylibs\scandir.py "C:%HOMEPATH%\AppData\Roaming\RV\Python\" /e /h /k /y

echo ----- install RV packages -----

"C:\Program Files\Shotgun\RV-7.9.2\bin\rvpkg" -force -remove C:%HOMEPATH%\AppData\Roaming\RV\Packages\dxSeqLatest-2.5.rvpkg
"C:\Program Files\Shotgun\RV-7.9.2\bin\rvpkg" -force -remove C:%HOMEPATH%\AppData\Roaming\RV\Packages\dxSeqLatest-3.0.rvpkg
"C:\Program Files\Shotgun\RV-7.9.2\bin\rvpkg" -force -remove C:%HOMEPATH%\AppData\Roaming\RV\Packages\dxSeqLatest-3.1.rvpkg
"C:\Program Files\Shotgun\RV-7.9.2\bin\rvpkg" -force -install -add C:%HOMEPATH%\AppData\Roaming\RV\ %BASEDIR%\rvpkg\dxSeqLatest-3.1.rvpkg

"C:\Program Files\Shotgun\RV-7.9.2\bin\rvpkg" -force -remove C:%HOMEPATH%\AppData\Roaming\RV\Packages\dxTacticReview-1.0.rvpkg
"C:\Program Files\Shotgun\RV-7.9.2\bin\rvpkg" -force -install -add C:%HOMEPATH%\AppData\Roaming\RV\ %BASEDIR%\rvpkg\dxTacticReview-1.1.rvpkg

"C:\Program Files\Shotgun\RV-7.9.2\bin\rvpkg" -force -remove C:%HOMEPATH%\AppData\Roaming\RV\Packages\dxOCIO-2.3.rvpkg
"C:\Program Files\Shotgun\RV-7.9.2\bin\rvpkg" -force -install -add C:%HOMEPATH%\AppData\Roaming\RV\ %BASEDIR%\rvpkg\dxOCIO-2.3.rvpkg

"C:\Program Files\Shotgun\RV-7.9.2\bin\rvpkg" -force -remove C:%HOMEPATH%\AppData\Roaming\RV\Packages\dxRenameEditOrder-1.0.rvpkg
"C:\Program Files\Shotgun\RV-7.9.2\bin\rvpkg" -force -install -add C:%HOMEPATH%\AppData\Roaming\RV\ %BASEDIR%\rvpkg\dxRenameEditOrder-1.0.rvpkg


echo ----- install Open RV packages -----

"C:\Program Files\OpenRV\bin\rvpkg" -force -remove C:%HOMEPATH%\AppData\Roaming\RV\Packages\dxSeqLatest-2.5.rvpkg
"C:\Program Files\OpenRV\bin\rvpkg" -force -remove C:%HOMEPATH%\AppData\Roaming\RV\Packages\dxSeqLatest-3.0.rvpkg
"C:\Program Files\OpenRV\bin\rvpkg" -force -remove C:%HOMEPATH%\AppData\Roaming\RV\Packages\dxSeqLatest-3.1.rvpkg
"C:\Program Files\OpenRV\bin\rvpkg" -force -install -add C:%HOMEPATH%\AppData\Roaming\RV\ %BASEDIR%\rvpkg\dxSeqLatest-3.1.rvpkg

"C:\Program Files\OpenRV\bin\rvpkg" -force -remove C:%HOMEPATH%\AppData\Roaming\RV\Packages\dxTacticReview-1.0.rvpkg
"C:\Program Files\OpenRV\bin\rvpkg" -force -install -add C:%HOMEPATH%\AppData\Roaming\RV\ %BASEDIR%\rvpkg\dxTacticReview-1.1.rvpkg

"C:\Program Files\OpenRV\bin\rvpkg" -force -remove C:%HOMEPATH%\AppData\Roaming\RV\Packages\dxOCIO-2.3.rvpkg
"C:\Program Files\OpenRV\bin\rvpkg" -force -install -add C:%HOMEPATH%\AppData\Roaming\RV\ %BASEDIR%\rvpkg\dxOCIO-2.3.rvpkg

"C:\Program Files\OpenRV\bin\rvpkg" -force -remove C:%HOMEPATH%\AppData\Roaming\RV\Packages\dxRenameEditOrder-1.0.rvpkg
"C:\Program Files\OpenRV\bin\rvpkg" -force -install -add C:%HOMEPATH%\AppData\Roaming\RV\ %BASEDIR%\rvpkg\dxRenameEditOrder-1.0.rvpkg

pause
