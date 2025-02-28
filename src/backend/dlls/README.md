# MFE Toolbox MEX Binaries

This directory contains compiled MEX (MATLAB EXecutable) binary files that provide high-performance computation capabilities for the MFE Toolbox. These optimized C implementations significantly accelerate critical numerical operations in financial econometric analysis, particularly in volatility modeling and time series analysis.

## Binary Files

This directory includes the following MEX binary files:

| Binary File | Source | Description |
|-------------|--------|-------------|
| agarch_core.mexw64 / agarch_core.mexa64 | agarch_core.c | High-performance implementation of Asymmetric GARCH (AGARCH) model core computations. Accelerates parameter estimation and forecasting for AGARCH volatility models through optimized variance computation and likelihood evaluation. |
| armaxerrors.mexw64 / armaxerrors.mexa64 | armaxerrors.c | Fast computation of ARMAX model residual errors. Optimizes ARMA/ARMAX model estimation by accelerating error computation, significantly improving performance for parameter estimation and forecasting. |
| composite_likelihood.mexw64 / composite_likelihood.mexa64 | composite_likelihood.c | Efficient implementation of composite likelihood estimation. Provides accelerated likelihood computation for complex models, enabling faster convergence in parameter estimation. |
| egarch_core.mexw64 / egarch_core.mexa64 | egarch_core.c | High-performance implementation of Exponential GARCH (EGARCH) model. Accelerates parameter estimation and forecasting for EGARCH volatility models through optimized log-variance computation. |
| igarch_core.mexw64 / igarch_core.mexa64 | igarch_core.c | Optimized implementation of Integrated GARCH (IGARCH) model. Accelerates parameter estimation and forecasting for IGARCH volatility models with efficient handling of parameter constraints. |
| tarch_core.mexw64 / tarch_core.mexa64 | tarch_core.c | Efficient implementation of Threshold ARCH/GARCH model. Accelerates parameter estimation and forecasting for TARCH/GARCH volatility models through optimized threshold-based variance computation. |

## Platform Compatibility

The MEX binaries are provided in platform-specific formats to ensure compatibility across different operating systems:

| Platform | File Extension | Compatibility |
|----------|---------------|---------------|
| Windows (PCWIN64) | .mexw64 | 64-bit Windows systems with MATLAB 4.0 or compatible |
| Unix | .mexa64 | 64-bit Unix systems (Linux, macOS) with MATLAB 4.0 or compatible |

The appropriate binary is automatically selected during toolbox initialization based on your operating system.

## Integration with MATLAB

These MEX binaries are seamlessly integrated with the MFE Toolbox's MATLAB functions:

- MEX binaries are automatically loaded by `addToPath.m` during initialization
- Platform-specific binaries are selected based on the operating system detection
- MATLAB functions check for MEX availability and use the accelerated implementation when available
- When MEX binaries are not available or not compatible, functions fall back to pure MATLAB implementations

This dual-implementation approach ensures both performance and compatibility across different environments.

## Performance Benefits

The MEX binaries provide significant performance advantages:

- **Speed**: Achieve >50% performance improvement for computationally intensive operations
- **Memory Efficiency**: Optimized memory usage for large-scale matrix operations
- **Algorithm Optimization**: Efficient implementation of critical numerical algorithms
- **Vectorization**: Leverages vectorized computation for maximum performance

These optimizations are particularly valuable for:
- Large dataset processing
- Computationally intensive volatility modeling
- Repetitive operations in simulation and bootstrap methods
- Parameter estimation involving numerous likelihood evaluations

## Compilation

If you need to recompile the MEX binaries from source:

1. Source files are located in `src/backend/mex_source/`
2. Compilation can be performed using `buildZipFile.m` with the `-largeArrayDims` flag
3. Platform-specific compilation scripts are available in `infrastructure/build_scripts/`
4. A C compiler is required for custom compilation (see MATLAB documentation for compatible compilers)

Compilation command example:
```
mex -largeArrayDims src/backend/mex_source/agarch_core.c
```

## Troubleshooting

Common issues and their solutions:

| Problem | Solution |
|---------|----------|
| MEX files not loading | Ensure binary format matches your platform (.mexw64 for Windows, .mexa64 for Unix). Check MATLAB console for error messages. |
| Missing MEX files | Run `buildZipFile.m` to recompile MEX binaries or download pre-compiled binaries from the repository. |
| Compatibility errors | Verify MATLAB version and C compiler compatibility. See MATLAB documentation for compatible compiler versions. |
| Performance not improved | Confirm MEX files are being correctly loaded by checking function execution path using `which -all function_name`. |

## References

- [MEX Source Code](../mex_source/README.md): Documentation of the C source code used to build these MEX binaries
- [Build System](../buildZipFile.m): Script for compiling and packaging the MEX binaries
- [Path Configuration](../addToPath.m): Script that handles loading the appropriate MEX binaries
- [MATLAB MEX Documentation](https://www.mathworks.com/help/matlab/matlab_external/introducing-mex-files.html): Official MATLAB documentation on MEX files

## See Also

- `src/backend/univariate/` - Functions that utilize the MEX binaries for volatility modeling
- `src/backend/timeseries/` - Time series functions that may use MEX acceleration
- `infrastructure/build_scripts/` - Platform-specific build scripts for MEX compilation