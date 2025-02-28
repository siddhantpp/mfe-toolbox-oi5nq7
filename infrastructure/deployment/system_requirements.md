# MFE Toolbox System Requirements

This document outlines the detailed system requirements for installing, compiling, and running the MATLAB Financial Econometrics (MFE) Toolbox version 4.0 (released on October 28, 2009). Meeting these requirements is essential for proper functionality and optimal performance of the toolbox.

## Table of Contents

1. [Hardware Requirements](#1-hardware-requirements)
2. [Software Requirements](#2-software-requirements)
3. [Operating System Requirements](#3-operating-system-requirements)
4. [MATLAB Requirements](#4-matlab-requirements)
5. [Compiler Requirements](#5-compiler-requirements)
6. [Storage Requirements](#6-storage-requirements)
7. [Network Requirements](#7-network-requirements)
8. [Platform-Specific Requirements](#8-platform-specific-requirements)
9. [Verification and Validation](#9-verification-and-validation)
10. [Additional Resources](#10-additional-resources)

## 1. Hardware Requirements

The following hardware specifications are required for running the MFE Toolbox efficiently:

### 1.1 Minimum Requirements
- **Processor**: 1.5 GHz or faster processor
- **Memory (RAM)**: 4GB minimum
- **Disk Space**: 100MB for the toolbox installation
- **Display**: 1024x768 or higher resolution

### 1.2 Recommended Requirements
- **Processor**: Modern multi-core processor (2.5 GHz or faster)
- **Memory (RAM)**: 8GB or more for processing large datasets
- **Disk Space**: 500MB for toolbox and working data
- **Display**: 1920x1080 or higher resolution

### 1.3 Performance Considerations
- Processing large financial datasets may require additional memory
- MEX-optimized computations benefit significantly from multi-core processors
- High-frequency data analysis may require 16GB+ RAM for optimal performance
- Time series analysis with large datasets will benefit from faster storage access

## 2. Software Requirements

The MFE Toolbox has the following software dependencies:

### 2.1 Required Software
- **MATLAB**: Compatible version (see Section 4)
- **MATLAB Statistics Toolbox**: Required for core statistical functions
- **C Runtime Libraries**: Platform-specific (see Section 8)

### 2.2 Optional Software
- **MATLAB Optimization Toolbox**: Recommended for improved optimization routines
- **C Compiler**: Required only for custom MEX compilation (see Section 5)
- **Version Control System**: Recommended for development environments

## 3. Operating System Requirements

The MFE Toolbox supports the following operating systems:

### 3.1 Windows
- **Supported Versions**: Windows 7, Windows 10, Windows Server 2012 or newer
- **Architecture**: 64-bit (PCWIN64)
- **System Type**: Desktop or server installation
- **Additional Requirements**: Microsoft Visual C++ Redistributable Package (version depends on MATLAB release)

### 3.2 Unix/Linux
- **Supported Distributions**: Red Hat Enterprise Linux 7 or newer, SUSE Linux Enterprise Desktop 12 or newer, Ubuntu 16.04 LTS or newer
- **Architecture**: 64-bit
- **Kernel Version**: 2.6.32 or newer
- **Additional Libraries**: Standard C libraries, GLIBC 2.12 or newer

### 3.3 Other Unix-like Systems
- **macOS**: Support depends on MATLAB compatibility
- **Other Unix variants**: Functional but not officially supported

## 4. MATLAB Requirements

The MFE Toolbox requires a compatible MATLAB installation:

### 4.1 MATLAB Version
- **Minimum Version**: MATLAB 7.7 (R2008b)
- **Recommended Version**: MATLAB 7.9 (R2009b) or newer
- **Architecture**: 64-bit installation

### 4.2 Required MATLAB Toolboxes
- **Statistics Toolbox**: Required for probability distributions and statistical functions
  - Core statistical distribution functions
  - Hypothesis testing capabilities
  - Random number generation

### 4.3 Recommended MATLAB Toolboxes
- **Optimization Toolbox**: Enhances parameter estimation performance
  - Improved optimization algorithms
  - Constrained optimization routines
- **Parallel Computing Toolbox**: For multi-core processing (optional)
  - Accelerates bootstrap methods
  - Enables parallel estimation of multiple models

### 4.4 MATLAB Path Configuration
- Administrative or write access to the MATLAB path is required for permanent installation
- Using `addToPath.m` with `savePath=true` requires write permission to MATLAB settings
- Network installations may require special path configuration

## 5. Compiler Requirements

A C compiler is required only if recompiling MEX files from source:

### 5.1 Windows Compiler Requirements
- **Supported Compiler**: Microsoft Visual C++
- **Required Version**: Compatible with your MATLAB version
- **Typical Requirements**:
  - For MATLAB 7.7-7.9: Microsoft Visual C++ 2008 SP1
  - See MathWorks documentation for your specific MATLAB version
- **Additional Components**: Windows SDK matching compiler version

### 5.2 Unix Compiler Requirements
- **Supported Compiler**: GCC (GNU Compiler Collection)
- **Minimum Version**: GCC 4.4.7
- **Recommended Version**: GCC 4.8 or newer
- **Required Packages**: gcc, gcc-c++, development headers (devel packages)
- **Development Tools**: make, binutils

### 5.3 Compiler Configuration
- MATLAB must be able to locate and use the C compiler
- Configure compiler using MATLAB's `mex -setup` command
- Environment variables may need adjustment for compiler paths

### 5.4 Pre-compiled MEX Binaries
- Pre-compiled MEX binaries are included for both platforms (Windows and Unix)
- No compiler is required if using pre-compiled binaries
- Custom compilation is only necessary for:
  - Applying platform-specific optimizations
  - Modifying MEX source code
  - Addressing compatibility issues with specific environments

## 6. Storage Requirements

The following storage specifications apply to the MFE Toolbox:

### 6.1 Installation Size
- **Basic Installation**: ~50MB
- **Full Installation with Documentation**: ~75MB
- **Installation with Source Code and Tests**: ~100MB

### 6.2 Working Storage
- **Temporary Files**: 10-50MB depending on dataset size
- **Analysis Results**: Varies based on usage
- **Large Dataset Analysis**: May require several GB

### 6.3 Performance Considerations
- SSD storage is recommended for improved performance when processing large datasets
- Network storage may introduce latency for shared installations
- Local cache directories may grow with extensive use

## 7. Network Requirements

Network requirements apply primarily to multi-user or shared installations:

### 7.1 Multi-user Deployment
- **File Sharing Protocol**: SMB (Windows) or NFS (Unix)
- **Permissions**: Read access to all toolbox files for all users
- **Path Configuration**: Network path must be accessible to all users

### 7.2 Performance Considerations
- Network latency may affect performance for shared installations
- Consider local installation for performance-critical applications
- Ensure sufficient network bandwidth for multi-user access

## 8. Platform-Specific Requirements

Each supported platform has specific requirements and considerations:

### 8.1 Windows (PCWIN64)
- **MEX Binaries**: .mexw64 format
- **C Runtime**: Microsoft Visual C++ Redistributable Package matching MATLAB version
- **Path Handling**: Managed automatically by addToPath.m
- **Special Considerations**:
  - Path length limitations (MAX_PATH)
  - UNC path handling for network installations
  - Registry access for permanent path configuration

### 8.2 Unix Systems
- **MEX Binaries**: .mexa64 format
- **Library Dependencies**: Standard C libraries (libc, libm, etc.)
- **File Permissions**: Executable permissions for scripts (chmod +x)
- **Special Considerations**:
  - Symbolic link support
  - Case-sensitive filesystem handling
  - Environment variables for MATLAB configuration

## 9. Verification and Validation

To verify that your system meets the requirements and the toolbox is correctly installed:

### 9.1 System Compatibility Check
Before installation, verify:

- MATLAB version compatibility
- Required toolboxes are installed
- Operating system compatibility
- Sufficient disk space and memory

### 9.2 Installation Verification
After installation, run the verification script to validate the installation:

```matlab
% Navigate to the verification script location
cd('infrastructure/deployment')

% Run verification
results = installation_verification();

% Review results
disp(['Installation status: ' num2str(results.overall_success)]);
disp('Component verification results:');
disp(results.details);
```

The verification script checks:
- Presence of all mandatory components
- Platform-specific MEX binary compatibility
- Basic functionality of core components
- MATLAB toolbox dependencies
- Version consistency

### 9.3 Performance Validation
To verify performance optimization:

```matlab
% Generate test data
data = randn(5000, 1);

% Test MEX-optimized GARCH performance
tic;
result = tarch(data, 1, 1, 1);
t1 = toc;
disp(['TARCH execution time: ' num2str(t1) ' seconds']);
```

Performance should be significantly better with MEX optimization compared to pure MATLAB implementations.

## 10. Additional Resources

For more detailed information, refer to the following resources:

### 10.1 Related Documentation
- [Installation Guide](../../docs/installation.md): Detailed installation and deployment instructions
- [Cross-Platform Notes](../../docs/cross_platform_notes.md): Platform-specific considerations
- [MEX Compilation Guide](../../docs/mex_compilation.md): Instructions for recompiling MEX files
- [Performance Optimization](../../docs/performance_optimization.md): Optimizing MFE Toolbox performance

### 10.2 Troubleshooting
- [Troubleshooting Guide](../../docs/troubleshooting.md): Common issues and solutions
- [Installation Verification](infrastructure/deployment/installation_verification.m): Automated verification script
- [Path Configuration](../../src/backend/addToPath.m): Path configuration utility

### 10.3 Support Resources
- System administrators should review the installation guide for enterprise deployments
- Developers should refer to MEX compilation guide for custom optimization
- Use included example files to validate functionality after installation
- For platform-specific deployment considerations, consult the cross-platform notes