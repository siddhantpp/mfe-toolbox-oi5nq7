# Cross-Platform Notes

The MFE Toolbox v4.0 is designed for cross-platform compatibility, supporting both Windows and Unix environments. This document outlines important platform-specific considerations to ensure consistent behavior across different operating systems.

## Supported Platforms

- **Windows**: PCWIN64 architecture with Windows-specific MEX binaries (.mexw64)
- **Unix Systems**: Linux-based systems with Unix-specific MEX binaries (.mexa64)

Both platforms require a compatible MATLAB installation and appropriate C compiler for MEX functionality.

## MEX Binary Differences

The toolbox uses platform-specific binary formats for optimal performance:

- Windows: `.mexw64` binaries in the `src/backend/dlls` directory
- Unix: `.mexa64` binaries in the same directory

The initialization process automatically detects your platform and loads the appropriate binaries.

## Platform Detection and Initialization

The toolbox uses MATLAB's `ispc()` function to detect the current platform and automatically configures the environment accordingly. This happens within the `addToPath.m` script, which is the primary initialization mechanism for the toolbox.

```matlab
% Example of how platform detection works in addToPath.m
if ispc()
    % Windows-specific configuration
    % Load .mexw64 binaries
else
    % Unix-specific configuration
    % Load .mexa64 binaries
end
```

This automatic detection ensures that the appropriate MEX binaries are loaded without user intervention.

## Ensuring Numerical Consistency

Slight differences in floating-point behavior may occur between platforms. Critical financial calculations are validated to maintain consistency within a tolerance of 1e-9, ensuring reliable results regardless of platform.

## Cross-Platform Compilation

To compile MEX files on different platforms:

- Windows: Use `infrastructure/build_scripts/compile_mex_windows.bat`
- Unix: Use `infrastructure/build_scripts/compile_mex_unix.sh`

Both scripts implement the same optimization flags (`-largeArrayDims -O`) for consistent performance.

## Cross-Platform Development Practices

When developing with the MFE Toolbox across multiple platforms, follow these best practices:

1. **Use Platform-Agnostic Paths**: Use MATLAB's `fullfile()` function to construct file paths to ensure compatibility
2. **Test on Multiple Platforms**: Validate functionality on both Windows and Unix when developing custom extensions
3. **Numerical Consistency Checks**: Use the `PlatformCompatibilityTest` class in `src/test/cross_platform/` to validate computational consistency
4. **Validate MEX Compilation**: Compile MEX files on all target platforms to ensure compatibility
5. **Manage Binary Versions**: Keep track of which MEX binary versions correspond to which source versions

Following these practices helps maintain the cross-platform integrity of the toolbox and any custom extensions.

## Platform-Specific Performance Considerations

Performance characteristics may vary slightly between platforms:

### Windows Optimization
- Intel MKL integration often provides better performance for matrix operations on Windows
- Visual Studio compiler optimizations may benefit certain numerical algorithms
- MEX acceleration typically shows 50-70% improvement over pure MATLAB code

### Unix Optimization
- GCC compiler with appropriate flags (-O3) can provide excellent performance
- Memory management is sometimes more efficient on Unix systems
- MEX acceleration typically shows 45-65% improvement over pure MATLAB code

For detailed performance optimization strategies for each platform, consult the Performance Optimization guide.

## Cross-Platform Troubleshooting

Common cross-platform issues include:

### MEX Loading Issues
- **Problem**: MEX files not loading
- **Solution**: Verify binary format matches your platform (.mexw64 for Windows, .mexa64 for Unix)
- **Verification**: Check `exist('agarch_core', 'file')` returns 3 (indicates MEX file found)

### Compiler Compatibility
- **Problem**: MEX compilation errors
- **Solution**: Ensure compiler is compatible with your MATLAB version
- **Windows**: Microsoft Visual C++ compatible with MATLAB version
- **Unix**: GCC 4.4.7 or newer

### Path Configuration
- **Problem**: Functions not found despite toolbox installation
- **Solution**: Run `addToPath` again and verify success message
- **Check**: Use `which functionname -all` to verify correct function is in path

### Performance Differences
- **Problem**: Performance variations between platforms
- **Solution**: Ensure MEX binaries are properly loaded on both platforms
- **Fallback**: If MEX unavailable, toolbox uses slower MATLAB implementations

## Platform-Specific Installation Notes

### Windows Requirements
- MATLAB for Windows (64-bit)
- Microsoft Visual C++ Redistributable for Visual Studio (compatible with MATLAB version)
- Administrator privileges may be required for permanent path configuration
- MEX binaries (.mexw64) must be accessible in the dlls directory

### Unix Requirements
- MATLAB for Unix (64-bit)
- GCC compiler (version 4.4.7 or newer recommended)
- Appropriate file permissions for toolbox directories and MEX binaries
- MEX binaries (.mexa64) must be accessible in the dlls directory

### Installation Verification
Verify your installation with:

```matlab
% Basic verification
isOnPath = exist('gedpdf', 'file');
disp(['MFE Toolbox functions found: ' num2str(isOnPath > 0)]);

% MEX verification (platform-specific)
mexAvailable = exist('agarch_core', 'file');
disp(['MEX optimization available: ' num2str(mexAvailable == 3)]);

% For comprehensive verification, use:
run('infrastructure/deployment/installation_verification.m');
```