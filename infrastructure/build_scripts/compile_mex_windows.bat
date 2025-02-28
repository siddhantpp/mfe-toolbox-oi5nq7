@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM compile_mex_windows.bat - MFE Toolbox MEX Compilation Script for Windows
REM
REM This script automates the compilation of C source files into Windows-compatible 
REM MEX binaries with optimization flags for the MFE Toolbox.
REM
REM Compiles core computational components for high-performance financial analysis:
REM   - matrix_operations.c: Optimized matrix operations
REM   - mex_utils.c: MEX interface utilities
REM   - agarch_core.c: Asymmetric GARCH model
REM   - armaxerrors.c: ARMAX model residuals
REM   - composite_likelihood.c: Composite likelihood estimation
REM   - egarch_core.c: Exponential GARCH model
REM   - igarch_core.c: Integrated GARCH model
REM   - tarch_core.c: Threshold ARCH model
REM
REM Version: 4.0 (28-Oct-2009)
REM ============================================================================

REM ============================================================================
REM Check environment and set configuration
REM ============================================================================

REM Check for MATLAB_ROOT environment variable
if "%MATLAB_ROOT%"=="" (
    echo Error: MATLAB_ROOT environment variable not set.
    echo Please set MATLAB_ROOT to your MATLAB installation directory.
    echo Example: set MATLAB_ROOT=C:\Program Files\MATLAB\R2009b
    exit /b 1
)

REM Set paths and configuration
set "MATLAB_BIN=%MATLAB_ROOT%\bin"
set "MEX_SOURCE_DIR=..\..\src\backend\mex_source"
set "MEX_OUTPUT_DIR=..\..\src\backend\dlls"
set "MEX_FLAGS=-largeArrayDims -O"
set "INCLUDE_FLAGS=-I"%MEX_SOURCE_DIR%""

REM ============================================================================
REM Environment validation
REM ============================================================================

REM Verify MEX compiler exists
if not exist "%MATLAB_BIN%\mex.exe" (
    echo Error: MEX compiler not found at %MATLAB_BIN%\mex.exe
    echo Please check your MATLAB_ROOT path.
    exit /b 1
)

REM Verify source directory exists
if not exist "%MEX_SOURCE_DIR%" (
    echo Error: Source directory not found at %MEX_SOURCE_DIR%
    exit /b 1
)

REM Check for required source files
set "MISSING_FILES="
set "REQUIRED_HEADERS=matrix_operations.h mex_utils.h"
for %%H in (%REQUIRED_HEADERS%) do (
    if not exist "%MEX_SOURCE_DIR%\%%H" (
        set "MISSING_FILES=!MISSING_FILES! %%H"
    )
)

if not "%MISSING_FILES%"=="" (
    echo Error: Required header files missing:%MISSING_FILES%
    echo Please ensure all required header files are present in %MEX_SOURCE_DIR%
    exit /b 1
)

REM Create output directory if it doesn't exist
if not exist "%MEX_OUTPUT_DIR%" (
    echo Creating output directory %MEX_OUTPUT_DIR%...
    md "%MEX_OUTPUT_DIR%"
    if errorlevel 1 (
        echo Error: Failed to create output directory %MEX_OUTPUT_DIR%
        exit /b 1
    )
)

REM ============================================================================
REM Display compilation information
REM ============================================================================
echo.
echo ============================================================================
echo MFE Toolbox MEX Compilation for Windows
echo ============================================================================
echo Source Directory: %MEX_SOURCE_DIR%
echo Output Directory: %MEX_OUTPUT_DIR%
echo Optimization Flags: %MEX_FLAGS%
echo MATLAB Directory: %MATLAB_ROOT%
echo.
echo This script will compile the MEX files for the MFE Toolbox with performance
echo optimizations for Windows platforms.
echo.

REM ============================================================================
REM Compile utility files first (dependencies for other files)
REM ============================================================================
echo Compiling utility modules...

set "UTIL_FILES=matrix_operations.c mex_utils.c"

for %%F in (%UTIL_FILES%) do (
    call :compile_file "%%F" "utility"
    if errorlevel 1 (
        echo Error: Failed to compile utility module %%F
        exit /b 1
    )
)

REM ============================================================================
REM Compile core MEX files
REM ============================================================================
echo.
echo Compiling core MEX files...

set "CORE_FILES=agarch_core.c armaxerrors.c composite_likelihood.c egarch_core.c igarch_core.c tarch_core.c"

for %%F in (%CORE_FILES%) do (
    call :compile_file "%%F" "core"
    if errorlevel 1 (
        echo Error: Failed to compile core module %%F
        exit /b 1
    )
)

REM ============================================================================
REM Verify all MEX files were created
REM ============================================================================
echo.
echo Verifying compiled MEX files...

set "ALL_FILES=%UTIL_FILES% %CORE_FILES%"
set "COMPILED_COUNT=0"
set "MISSING_FILES="
set "EXPECTED_COUNT=0"

for %%F in (%ALL_FILES%) do (
    set /a EXPECTED_COUNT+=1
    set "FILE_BASE=%%~nF"
    if exist "%MEX_OUTPUT_DIR%\!FILE_BASE!.mexw64" (
        set /a COMPILED_COUNT+=1
    ) else (
        set "MISSING_FILES=!MISSING_FILES! !FILE_BASE!"
    )
)

if %COMPILED_COUNT% EQU %EXPECTED_COUNT% (
    echo [SUCCESS] All %EXPECTED_COUNT% MEX files compiled successfully.
) else (
    echo Error: Expected %EXPECTED_COUNT% MEX files, but found %COMPILED_COUNT%.
    echo Missing files:%MISSING_FILES%
    exit /b 1
)

REM ============================================================================
REM Final status report
REM ============================================================================
echo.
echo ============================================================================
echo MEX compilation completed successfully.
echo All binaries were compiled with the -largeArrayDims flag for large data support.
echo The output files (.mexw64) are located in: %MEX_OUTPUT_DIR%
echo ============================================================================

exit /b 0

REM ============================================================================
REM Function to compile a single file
REM ============================================================================
:compile_file
setlocal
set "FILE_NAME=%~1"
set "FILE_TYPE=%~2"
set "FILE_BASE=%FILE_NAME:.c=%"

echo   - %FILE_NAME%
if not exist "%MEX_SOURCE_DIR%\%FILE_NAME%" (
    echo     [ERROR] Source file not found
    exit /b 1
)

if "%FILE_TYPE%"=="utility" (
    "%MATLAB_BIN%\mex.exe" %MEX_FLAGS% %INCLUDE_FLAGS% -outdir "%MEX_OUTPUT_DIR%" "%MEX_SOURCE_DIR%\%FILE_NAME%"
) else (
    "%MATLAB_BIN%\mex.exe" %MEX_FLAGS% %INCLUDE_FLAGS% -outdir "%MEX_OUTPUT_DIR%" "%MEX_SOURCE_DIR%\%FILE_NAME%" "%MEX_OUTPUT_DIR%\matrix_operations.mexw64" "%MEX_OUTPUT_DIR%\mex_utils.mexw64"
)

if errorlevel 1 (
    echo     [ERROR] Compilation failed
    exit /b 1
) else (
    echo     [SUCCESS] Created %FILE_BASE%.mexw64
)

exit /b 0
endlocal