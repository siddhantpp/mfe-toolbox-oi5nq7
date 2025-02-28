# MFE Toolbox MEX Source

## Overview

The MEX (MATLAB EXecutable) source files in the MFE Toolbox provide high-performance C-based implementations of computationally intensive financial econometric operations. These optimized implementations deliver significant performance advantages over native MATLAB code, particularly for large datasets and complex time series models.

Key benefits:
- Substantial performance improvements (>50%) for critical computational paths
- Support for large-scale data processing via `-largeArrayDims` flag
- Cross-platform compatibility (Windows and Unix)
- Memory-efficient matrix operations
- Numerical stability for complex econometric computations

## Source Files

### Core Utilities

#### matrix_operations.h/c
**Purpose**: Core optimized matrix operations for MEX implementations

**Key Functions**:
- `copy_matrix`: Efficient matrix copying with dimensional validation
- `matrix_multiply`: Optimized matrix multiplication
- `transpose_matrix`: Fast matrix transposition
- `matrix_element_multiply`: Element-wise matrix multiplication
- `matrix_scalar_multiply`: Scalar multiplication across matrix elements

**Description**: Provides high-performance matrix manipulation functions optimized for financial time series processing.

#### mex_utils.h/c
**Purpose**: Utility functions for MEX interface and error handling

**Key Functions**:
- `check_inputs`: Validates input parameters from MATLAB
- `check_outputs`: Ensures output parameters are correctly sized
- `mex_error`: Standardized error reporting back to MATLAB
- `matlab_to_c_double`: Safe conversion from MATLAB arrays to C doubles
- `safe_malloc`: Memory allocation with error checking
- `safe_free`: Memory deallocation with validation

**Description**: Common utilities used across all MEX implementations for memory management, input validation, and error handling.

### Model Implementations

#### agarch_core.c
**Purpose**: Asymmetric GARCH model implementation

**Key Functions**:
- `mexFunction`: MEX entry point for MATLAB integration
- `compute_agarch_variance`: Core variance computation for AGARCH models
- `compute_agarch_likelihood`: Likelihood calculation for parameter estimation

**Description**: High-performance implementation of the Asymmetric GARCH model for volatility modeling.

#### armaxerrors.c
**Purpose**: ARMAX model residual calculations

**Key Functions**:
- `mexFunction`: MEX entry point for MATLAB integration
- `compute_armax_errors`: Calculation of residuals for ARMAX models

**Description**: Optimized computation of residuals for ARMAX time series models.

#### composite_likelihood.c
**Purpose**: Composite likelihood estimation

**Key Functions**:
- `mexFunction`: MEX entry point for MATLAB integration
- `compute_composite_likelihood`: Implementation of composite likelihood calculations

**Description**: Fast implementation of composite likelihood calculations for complex models.

#### egarch_core.c
**Purpose**: Exponential GARCH implementation

**Key Functions**:
- `mexFunction`: MEX entry point for MATLAB integration
- `compute_egarch_variance`: Core variance computation for EGARCH models
- `compute_egarch_likelihood`: Likelihood calculation for parameter estimation

**Description**: Accelerated implementation of the Exponential GARCH model.

#### igarch_core.c
**Purpose**: Integrated GARCH implementation

**Key Functions**:
- `mexFunction`: MEX entry point for MATLAB integration
- `compute_igarch_variance`: Core variance computation for IGARCH models
- `compute_igarch_likelihood`: Likelihood calculation for parameter estimation

**Description**: Optimized implementation of the Integrated GARCH model.

#### tarch_core.c
**Purpose**: Threshold ARCH/GARCH implementation

**Key Functions**:
- `mexFunction`: MEX entry point for MATLAB integration
- `compute_tarch_variance`: Variance computation for TARCH/GARCH models
- `compute_tarch_likelihood`: Likelihood calculation for parameter estimation

**Description**: Accelerated implementation of the Threshold ARCH/GARCH model.

## Compilation Instructions

To compile the MEX binaries from the source files:

1. Ensure a compatible C compiler is installed and configured for MEX compilation
   - On Windows: Microsoft Visual C++ or MinGW
   - On Unix: GCC

2. Configure MATLAB to use the compiler:
   ```matlab
   mex -setup
   ```

3. Compile individual source files with the `-largeArrayDims` flag:
   ```matlab
   mex -largeArrayDims agarch_core.c matrix_operations.c mex_utils.c
   ```

4. Alternatively, use the provided `buildZipFile.m` script to automate compilation:
   ```matlab
   buildZipFile
   ```

5. Platform-specific binaries will be created:
   - Windows: `.mexw64` extension
   - Unix: `.mexa64` extension

6. Compiled binaries should be placed in the `src/backend/dlls` directory

### Important Compilation Flags

| Flag | Purpose |
|------|---------|
| `-largeArrayDims` | Enable support for large matrices exceeding 2GB size limit |
| `-g` | Include debugging information for development |
| `-O` | Enable compiler optimizations for performance |

## Integration Architecture

The MEX source code integrates with the MATLAB codebase through a carefully designed architecture:

### MATLAB Interface
- MATLAB functions (e.g., `agarchfit.m`) call MEX implementations when available
- Dynamic detection with platform-specific loading of MEX binaries
- Fallback to pure MATLAB implementation when MEX binaries are unavailable

### Parameter Passing
- Data is passed between MATLAB and C via the MEX API
- Uses `mxArray` structures with utility functions for conversion
- Careful handling of matrix dimensions and data types

### Error Handling
- Comprehensive validation and error propagation
- `mex_error` utility for formatted error messages back to MATLAB
- Detailed error information for troubleshooting

### Memory Management
- Careful allocation and deallocation of memory
- `safe_malloc` and `safe_free` utilities for memory management
- Prevention of memory leaks and buffer overflows

## Performance Considerations

To maximize the performance benefits of the MEX implementations:

1. **Minimize data copying** between MATLAB and C environments
   - Use in-place operations where possible
   - Pass pointers rather than copying data when appropriate

2. **Use optimized matrix operations** for computational efficiency
   - Leverage cache coherence for large matrix operations
   - Consider numerical stability alongside performance

3. **Implement careful memory management** with preallocation
   - Preallocate arrays to avoid reallocation overhead
   - Release memory as soon as it's no longer needed

4. **Consider numerical stability** alongside performance optimizations
   - Implement checks against division by zero
   - Handle potential overflow and underflow conditions

5. **Balance vectorization with cache coherence** for large matrices
   - Vectorized operations are generally faster but may have cache implications
   - Experiment with block sizes for large matrix operations

## Troubleshooting

### Common Issues and Solutions

| Problem | Solution |
|---------|----------|
| Compilation errors | Verify C compiler compatibility and installation |
| Runtime crashes | Check for memory leaks or out-of-bounds access in C code |
| Performance degradation | Profile code and optimize critical paths |
| Numerical instability | Implement safeguards against division by zero and overflow |

## References

- [MEX Binary Files](src/backend/dlls/README.md): Information about the compiled MEX binaries and their usage
- [MATLAB MEX API Documentation](https://www.mathworks.com/help/matlab/matlab_external/introducing-mex-files.html): Official MATLAB documentation on MEX file development
- [Build Automation](src/backend/buildZipFile.m): Script for automated compilation and packaging of the toolbox
- [MATLAB Integration](src/backend/univariate/agarchfit.m): Example of MATLAB function that integrates with MEX implementation

## See Also

- `src/backend/dlls/` - Directory containing compiled MEX binaries
- `src/backend/univariate/` - MATLAB functions using MEX acceleration
- `src/backend/addToPath.m` - Path configuration utility with MEX support