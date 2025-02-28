# MFE Toolbox Installation Guide

This guide provides detailed instructions for installing and configuring the MATLAB Financial Econometrics (MFE) Toolbox version 4.0 (released October 28, 2009). Follow these steps carefully to ensure proper installation and operation of the toolbox.

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Download Instructions](#download-instructions)
4. [Windows Installation](#windows-installation)
5. [Unix Installation](#unix-installation)
6. [Path Configuration](#path-configuration)
7. [Verification Steps](#verification-steps)
8. [Common Issues](#common-issues)
9. [Custom Compilation](#custom-compilation)
10. [Next Steps](#next-steps)

## Introduction

The MFE Toolbox is a comprehensive MATLAB-based software suite designed for financial time series modeling, econometric analysis, and risk assessment. It provides tools for:

- Financial time series modeling with ARMA/ARMAX
- Volatility modeling using various GARCH implementations
- Bootstrap methods for statistical inference
- Cross-sectional analysis tools
- Statistical testing and distribution analysis
- High-frequency financial data analysis

This toolbox leverages MEX optimization for high-performance computing, allowing sophisticated analyses with minimal computation time. Version 4.0 includes significant performance improvements, enhanced functionality, and improved cross-platform compatibility.

## System Requirements

The MFE Toolbox requires the following system components for proper operation:

### Hardware Requirements

- **Processor**: 1.5 GHz or faster (multi-core recommended)
- **Memory**: 4GB minimum, 8GB+ recommended for large datasets
- **Disk Space**: ~100MB for toolbox installation
- **Display**: 1024x768 or higher resolution

### Software Requirements

- **MATLAB**: Version R2009b (7.9) or compatible
- **Required Toolboxes**: MATLAB Statistics Toolbox
- **Recommended Toolboxes**: MATLAB Optimization Toolbox
- **C Runtime**: Microsoft Visual C++ Redistributable (Windows) or standard C libraries (Unix)

### Platform Support

- **Windows**: Windows 7, Windows 10, Windows Server 2012 or newer (64-bit/PCWIN64)
- **Unix/Linux**: Red Hat Enterprise Linux 7+, SUSE Linux Enterprise Desktop 12+, Ubuntu 16.04 LTS+ (64-bit)

For more platform-specific information, see [Cross-Platform Notes](cross_platform_notes.md).

## Download Instructions

### Obtaining the MFE Toolbox

1. Download the MFEToolbox.zip file (approximately 75MB)
2. Verify the package integrity using the provided checksum
3. Choose an installation location with:
   - Sufficient disk space (~100MB)
   - Appropriate file permissions
   - Short path name on Windows to avoid path length limitations

### Package Contents

The MFEToolbox.zip file contains:

- Core statistical modules (distributions, bootstrap, tests)
- Time series components (ARMA/ARMAX, volatility models)
- Support infrastructure (GUI, utilities)
- MEX source files and pre-compiled binaries
- Documentation and examples

Ensure you have extract/unzip utilities available on your platform before proceeding with installation.

## Windows Installation

Follow these steps to install the MFE Toolbox on Windows systems:

1. **Extract the package**:
   - Right-click MFEToolbox.zip and select "Extract All..."
   - Choose a destination folder with a short path (to avoid MAX_PATH limitations)
   - Ensure you have write permissions to the destination folder

2. **Launch MATLAB**:
   - Start MATLAB with appropriate permissions (administrator rights may be required for permanent path configuration)
   - Ensure you're using a 64-bit MATLAB version compatible with the toolbox

3. **Navigate to the toolbox directory**:
   ```matlab
   cd('C:/path/to/MFEToolbox')
   ```

4. **Run the path configuration utility**:
   ```matlab
   % For temporary installation (session only)
   addToPath(false, true);
   
   % For permanent installation (recommended)
   addToPath(true, true);
   ```

5. **Verify MEX binaries**:
   - Windows-specific MEX binaries (.mexw64) should be automatically detected
   - If MEX acceleration is not working, you may need to install the Microsoft Visual C++ Redistributable Package

Notes for Windows users:
- Path names with spaces should be avoided
- Network installations may require UNC path considerations
- Windows Security or antivirus software may require configuring exceptions for MEX binaries

## Unix Installation

Follow these steps to install the MFE Toolbox on Unix/Linux systems:

1. **Extract the package**:
   ```bash
   unzip MFEToolbox.zip -d /path/to/destination
   ```

2. **Set appropriate permissions**:
   ```bash
   # Navigate to the installation directory
   cd /path/to/MFEToolbox
   
   # Make MEX binaries executable
   chmod +x src/backend/dlls/*.mexa64
   ```

3. **Launch MATLAB**:
   - Start MATLAB from a terminal or application launcher
   - Ensure you're using a 64-bit MATLAB version compatible with the toolbox

4. **Navigate to the toolbox directory**:
   ```matlab
   cd('/path/to/MFEToolbox')
   ```

5. **Run the path configuration utility**:
   ```matlab
   % For temporary installation (session only)
   addToPath(false, true);
   
   % For permanent installation (recommended)
   addToPath(true, true);
   ```

6. **Verify MEX binaries**:
   - Unix-specific MEX binaries (.mexa64) should be automatically detected
   - If MEX acceleration is not working, check for missing shared libraries using `ldd`

Notes for Unix users:
- Case sensitivity matters for file and directory names
- Environment variables may need configuration for some systems
- Shared library dependencies must be satisfied for MEX functionality

## Path Configuration

The MFE Toolbox uses the `addToPath.m` script to configure the MATLAB path. This script:

1. Identifies the current platform (Windows or Unix)
2. Adds all mandatory directories to the MATLAB path
3. Adds platform-specific MEX binary directories
4. Adds optional directories if requested
5. Optionally saves the path configuration permanently

### Function Syntax

```matlab
addToPath(savePath, addOptionalDirs)
```

Parameters:
- `savePath` (logical): When `true`, saves the path permanently using MATLAB's `savepath` function
- `addOptionalDirs` (logical): When `true`, adds optional 'duplication' directory with work-alike functions

### Mandatory Directories

The following directories are always added to the path:
- bootstrap: Bootstrap implementation
- crosssection: Cross-sectional analysis tools
- distributions: Statistical distribution functions
- GUI: ARMAX modeling interface
- multivariate: Multivariate analysis tools
- tests: Statistical testing suite
- timeseries: Time series analysis
- univariate: Univariate analysis tools
- utility: Helper functions
- realized: High-frequency analysis
- mex_source: C source files
- dlls: Platform-specific MEX binaries

### Optional Directories

- duplication: Contains work-alike functions that may be useful in certain environments

### Permanent Path Configuration

For permanent installation (recommended), use:
```matlab
addToPath(true, true);
```

This requires write permissions to MATLAB's path configuration. On some systems, you may need to run MATLAB with administrator/root privileges to save the path permanently.

## Verification Steps

After installation, verify that the MFE Toolbox is correctly installed and functioning using the following steps:

### Manual Verification

Run the following commands to check for the presence of key components:

```matlab
% Check for core distribution functions
exist('gedpdf', 'file')
exist('stdtpdf', 'file')

% Check for time series functions
exist('armaxfilter', 'file')
exist('sacf', 'file')

% Check for volatility functions
exist('tarchfit', 'file')
exist('garchfor', 'file')

% Check for MEX acceleration
exist('agarch_core', 'file')  % Should return 3 for MEX files
exist('tarch_core', 'file')   % Should return 3 for MEX files
```

All existence checks should return values greater than 0, with MEX files returning 3 specifically.

### Automated Verification

Use the provided verification script for comprehensive testing:

```matlab
% Navigate to the verification script location
cd('infrastructure/deployment')

% Run verification
results = installation_verification();

% Review results
disp(['Installation status: ' num2str(results.overallStatus)]);
disp('Component verification results:');
disp(results.mandatoryComponents);
```

The verification script checks:
- Presence of all mandatory components
- Platform-specific MEX binary compatibility
- Basic functionality of core components
- Path configuration correctness

### Basic Functionality Test

Try running a simple example to verify functionality:

```matlab
% Generate random data
data = randn(1000, 1);

% Fit a GARCH model
[parameters, loglikelihood, Ht, residuals, summary] = tarchfit(data, 1, 0, 1);

% Display results
disp('GARCH(1,1) Parameters:');
disp(parameters);
disp(['Log-likelihood: ' num2str(loglikelihood)]);

% Plot conditional variance
figure;
plot(Ht);
title('GARCH(1,1) Conditional Variance');
xlabel('Time');
ylabel('Variance');
```

If this example runs without errors and produces reasonable output, your installation is functioning correctly.

## Common Issues

Here are solutions to common installation issues:

### Path Configuration Problems

**Issue**: Functions not found after installation ('Undefined function or variable' errors)

**Solutions**:
- Ensure addToPath.m was executed successfully
- Check MATLAB path with `path` command to verify directories were added
- Run addToPath with savePath=true to make changes permanent
- Check write permissions if permanent path saving fails
- Use absolute paths if relative paths cause issues

### MEX Binary Issues

**Issue**: MEX files not loading or 'Invalid MEX-file' errors

**Solutions**:
- Verify platform compatibility (use .mexw64 for Windows and .mexa64 for Unix)
- Check if MEX file exists with `exist('agarch_core', 'file')`
- Windows: Install appropriate Microsoft Visual C++ Redistributable
- Unix: Check shared library dependencies with `ldd *.mexa64`
- Set appropriate file permissions on Unix (chmod +x)

### Platform-Specific Issues

**Windows-Specific**:
- Path length limitations: Keep installation path short
- Use administrator privileges for permanent path configuration
- Check Windows security/antivirus blocking MEX execution

**Unix-Specific**:
- Case sensitivity in file paths
- Executable permissions for MEX binaries
- Library path configuration (LD_LIBRARY_PATH)

For more detailed troubleshooting, see the [Troubleshooting Guide](troubleshooting.md).

## Custom Compilation

Pre-compiled MEX binaries are included for both Windows (.mexw64) and Unix (.mexa64) platforms. Custom compilation is only necessary if:

- You need to modify the MEX source code
- The pre-compiled binaries have compatibility issues with your environment
- You want to apply platform-specific optimizations

### Basic Compilation Steps

1. Ensure a compatible C compiler is installed and configured with MATLAB:
   ```matlab
   mex -setup
   ```

2. Navigate to the mex_source directory:
   ```matlab
   cd('src/backend/mex_source')
   ```

3. Compile a specific MEX file:
   ```matlab
   mex -largeArrayDims agarch_core.c
   ```

4. Move the compiled binary to the dlls directory:
   ```matlab
   movefile('agarch_core.mexw64', '../dlls/', 'f');  % For Windows
   movefile('agarch_core.mexa64', '../dlls/', 'f');   % For Unix
   ```

For comprehensive compilation instructions, compiler requirements, and optimization guidance, see the [MEX Compilation Guide](mex_compilation.md).

## Next Steps

After successfully installing the MFE Toolbox, you can:

1. Read the [Getting Started Guide](getting_started.md) for an introduction to basic functionality
2. Explore the example files in the `examples/` directory
3. Review the API documentation in the `docs/` directory
4. Try the ARMAX GUI interface:
   ```matlab
   ARMAX
   ```

5. Experiment with basic functionality as described in the Verification Steps section

The MFE Toolbox is now ready for use in your financial econometrics and time series analysis projects.