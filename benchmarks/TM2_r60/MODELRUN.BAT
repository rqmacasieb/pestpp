@ECHO OFF
c:\simcore\pm8\MODFLO~1\mf2005.exe<startit > NUL
c:\simcore\pm8\pest\MODBORE.EXE < modbore.dat >NUL
if exist c:\simcore\pm8\pest\MODBORE.EXE goto Check2
@ECHO.
@ECHO ------------ PMWIN Message -------------
@ECHO Error: MODBORE.EXE is not located in the directory
@ECHO    c:\simcore\pm8\pest\
@ECHO.
@ECHO Suggestion: 
@ECHO    Copy MODBORE.EXE to the directory given above.
@ECHO --------- End of PMWIN Message ---------
@PAUSE
goto end
:Check2
if not errorlevel 100 goto end
@ECHO.
@ECHO ------------ PMWIN Message -------------
@ECHO Error condition prevents continued PEST execution:
@ECHO Possible reason: 
@ECHO    Solution convergence is not achieved and the model output
@ECHO    files have not been written.
@ECHO.
@ECHO Suggestion: 
@ECHO    1. Press [ESC] to terminate the Pest run.
@ECHO    2. Set a higher convergence criterion or 
@ECHO       increase the allowed iteration numbers.
@ECHO    3. Restart PEST.
@ECHO --------- End of PMWIN Message ---------
@PAUSE
:end
