@echo off
rem B.S. Detector - double-click to run on Windows.
cd /d "%~dp0"
where py >nul 2>nul && (py -3 run.py %* & goto :end)
where python >nul 2>nul && (python run.py %* & goto :end)
echo Python 3.9 or newer is required. Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
:end
pause
