@echo off

python setup.py install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit /b 1
