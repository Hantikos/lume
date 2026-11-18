@echo off
REM Run the Lume command line tool from a source checkout (Windows).
setlocal
set ROOT=%~dp0..
python "%ROOT%\lume\__main__.py" %*
