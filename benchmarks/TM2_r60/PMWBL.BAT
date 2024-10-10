@COPY c:\users\rqmac\DOWNLO~1\ex10\ex10.pm5 flagfile>NUL
C:\Simcore\PM8\PMWBLF.EXE < WBLFILE
@ECHO.
@ECHO ------------ PMWIN Message -------------
@ECHO Budget calculation is complete. See the following file for full run details
@ECHO C:\USERS\RQMAC\DOWNLOADS\EX10\WATERBDG.DAT
@ECHO.
@ECHO In case of difficulties:
@ECHO 1. Make sure that the output term 'Cell-by-cell flow terms' in the Modflow
@ECHO    Output Control dialog box is checked, then run Modflow again.
@ECHO 2. Check the specification of the number of zones. Remember that the valid
@ECHO    range of the zone number is 1 to 50.
@ECHO --------- End of PMWIN Message ---------
@DEL flagfile>NUL
@ECHO.
@ECHO Press any key to view the file
@ECHO WATERBDG.DAT
@PAUSE>NUL
c:\windows\notepad.exe WATERBDG.DAT
