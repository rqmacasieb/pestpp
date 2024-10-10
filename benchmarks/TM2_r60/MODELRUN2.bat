@ECHO OFF
C:\Users\rqmac\Downloads\TM2_r60\mf2005.exe<startit > NUL
C:\Users\rqmac\Downloads\TM2_r60\MODBORE.EXE < modbore.dat >NUL
C:\Users\rqmac\Downloads\TM2_r60\TM2.exe > NUL
if exist C:\Users\rqmac\Downloads\TM2_r60\MODBORE.EXE goto Check2
@ECHO.
@ECHO Check Batch File
@PAUSE
goto end
:Check2
if not errorlevel 100 goto end
@ECHO.
@ECHO Failed PEST execution
@PAUSE
:end
