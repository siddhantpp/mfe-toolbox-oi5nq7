# MEX Compilation Guide for MFE Toolbox

## 1. Introduction

### 1.1 Overview of MEX in MFE Toolbox
The MFE Toolbox leverages MEX (MATLAB Executable) files to achieve significant performance improvements for computationally intensive operations. MEX files are dynamically linked subroutines produced from C, C++, or Fortran source code that, when compiled, can be called from MATLAB as if they were built-in functions. In the MFE Toolbox, critical computational paths are implemented in C and exposed to MATLAB through the MEX interface.

### 1.2 Performance Benefits
The integration of MEX-based components in the MFE Toolbox provides several crucial advantages:
- **Execution Speed**: MEX files execute significantly faster than equivalent MATLAB code for computationally intensive operations.
- **Memory Efficiency**: Direct memory management in C allows for more efficient handling of large datasets.
- **Specialized Algorithms**: Implementation of optimized numerical algorithms that leverage low-level programming capabilities.

The MFE Toolbox achieves >50% performance improvement in critical computational paths through MEX optimization.

### 1.3 Core MEX Components
The following critical components are implemented as MEX files in the MFE Toolbox:
- `agarch_core`: Core computations for Asymmetric GARCH models
- `armaxerrors`: ARMAX residual error computation
- `composite_likelihood`: Composite likelihood computation
- `egarch_core`: EGARCH algorithm implementation
- `igarch_core`: IGARCH model computations
- `tarch_core`: TARCH/GARCH variance computations

## 2. Compilation Requirements

### 2.1 Software Requirements

#### 2.1.1 MATLAB Environment
- MATLAB (R2007a or later recommended)
- MATLAB Statistics Toolbox
- MATLAB Optimization Toolbox

#### 2.1.2 C Compiler Requirements

**Windows (PCWIN64):**
- Microsoft Visual C++ Compiler (The specific version depends on your MATLAB version)
- Windows SDK matching your compiler version

**Unix Systems:**
- GCC 4.4.x or later
- GNU Make

### 2.2 Environment Setup

#### 2.2.1 Verifying Compiler Setup in MATLAB
To verify that your compiler is properly set up for MEX compilation in MATLAB, run:

```matlab
mex -setup
```

This command displays the available compilers and allows you to select the appropriate one for your system.

#### 2.2.2 Checking MEX Configuration
To verify your MEX configuration:

```matlab
mexext
```

This returns the extension for MEX files on your platform:
- Windows 64-bit: `mexw64`
- Linux 64-bit: `mexa64`
- macOS 64-bit: `mexmaci64`

## 3. Cross-Platform Compilation Instructions

### 3.1 Windows (PCWIN64) Compilation

#### 3.1.1 Using MATLAB Command Window
1. Navigate to the `mex_source` directory in the MFE Toolbox:
   ```matlab
   cd('path/to/MFEToolbox/mex_source')
   ```

2. Compile a specific MEX file with the large array dimensions flag:
   ```matlab
   mex -largeArrayDims agarch_core.c
   ```

3. Verify compilation success by checking for the `.mexw64` file in the directory.

#### 3.1.2 Using the Build Script
The MFE Toolbox includes `buildZipFile.m` which automates MEX compilation:

1. Navigate to the MFE Toolbox root directory:
   ```matlab
   cd('path/to/MFEToolbox')
   ```

2. Run the build script:
   ```matlab
   buildZipFile
   ```

3. This script handles compilation for all MEX files with appropriate flags.

### 3.2 Unix Systems Compilation

#### 3.2.1 Using MATLAB Command Window
1. Navigate to the `mex_source` directory:
   ```matlab
   cd('path/to/MFEToolbox/mex_source')
   ```

2. Compile a specific MEX file:
   ```matlab
   mex -largeArrayDims agarch_core.c
   ```

3. Verify compilation success by checking for the `.mexa64` file in the directory.

#### 3.2.2 Using the Build Script
Similar to Windows compilation, use:

1. Navigate to the MFE Toolbox root directory:
   ```matlab
   cd('path/to/MFEToolbox')
   ```

2. Run the build script:
   ```matlab
   buildZipFile
   ```

## 4. Performance Optimization

### 4.1 Compilation Flags

#### 4.1.1 Large Array Dimensions
The `-largeArrayDims` flag is crucial for the MFE Toolbox as it enables:
- Support for matrices with more than 2^31-1 elements
- 64-bit indexing for large data processing
- Compatibility with modern MATLAB versions

#### 4.1.2 Optimization Flags
For additional performance gains, consider these platform-specific optimization flags:

**Windows (MSVC):**
```matlab
mex -largeArrayDims -O agarch_core.c
```

**Unix (GCC):**
```matlab
mex -largeArrayDims CFLAGS='-O3 -ffast-math' agarch_core.c
```

### 4.2 Memory Management Best Practices

#### 4.2.1 Pre-allocation
MEX files in the MFE Toolbox use pre-allocation strategies to minimize memory fragmentation:
- Input arrays are validated for appropriate dimensions before computation
- Output arrays are allocated once with the final size
- Temporary arrays are managed carefully to minimize overhead

#### 4.2.2 Memory Access Patterns
The MEX implementations utilize optimized memory access patterns:
- Column-major order (matching MATLAB's internal representation)
- Cache-aware algorithm designs
- Minimal data copying between operations

## 5. Troubleshooting Common Issues

### 5.1 Compilation Errors

#### 5.1.1 Missing Compiler
**Error:**
```
Error using mex
No supported compiler was found.
```

**Solution:**
1. Install a supported compiler for your MATLAB version
2. Run `mex -setup` to configure MATLAB to use the compiler

#### 5.1.2 Header File Errors
**Error:**
```
agarch_core.c(10): fatal error C1083: Cannot open include file: 'mex.h': No such file or directory
```

**Solution:**
1. Ensure MATLAB's include directory is in the compilation path
2. Use the full MEX command with include path:
   ```matlab
   mex -I"matlabroot/extern/include" -largeArrayDims agarch_core.c
   ```

### 5.2 Runtime Errors

#### 5.2.1 Invalid MEX-File
**Error:**
```
Invalid MEX-file 'path/to/agarch_core.mexw64': The specified module could not be found.
```

**Solution:**
1. Recompile the MEX file for your specific platform
2. Ensure all required DLLs are in the MATLAB path or system path
3. Check for compatibility between MATLAB version and MEX file

#### 5.2.2 Memory Allocation Failures
**Error:**
```
Not enough input arguments.
```
or
```
Error using agarch_core
Out of memory.
```

**Solution:**
1. Verify input arguments match the expected format
2. Check available system memory
3. Consider breaking large computations into smaller chunks

## 6. Best Practices for MEX Development in MFE Toolbox

### 6.1 Code Organization

#### 6.1.1 Source Structure
Maintain a clear structure for MEX source files:
- Function headers with detailed documentation
- Input validation section
- Memory allocation section
- Computation section
- Memory cleanup section

#### 6.1.2 Error Handling
Implement robust error handling in MEX files:
```c
/* Example error handling pattern */
if (problem_detected) {
    mexErrMsgIdAndTxt("MFEToolbox:agarch_core:invalidInput",
                      "Detailed error message here.");
}
```

### 6.2 Testing and Validation

#### 6.2.1 Validation Framework
Before distributing compiled MEX files:
1. Validate results against pure MATLAB implementations
2. Test with edge cases and boundary conditions
3. Verify memory management with MATLAB's memory profiler

#### 6.2.2 Cross-Platform Testing
Test compiled MEX files on all target platforms:
- Windows 64-bit environment
- Unix-based systems
- Various MATLAB versions

### 6.3 Distribution Strategy

#### 6.3.1 Pre-compiled Binaries
The MFE Toolbox distribution includes pre-compiled MEX binaries for common platforms:
- Windows 64-bit (`.mexw64`) in the `dlls` directory
- Unix 64-bit (`.mexa64`) in the appropriate location

#### 6.3.2 Source Distribution
Always include the C source files to allow users to compile on unsupported platforms or troubleshoot platform-specific issues.

## 7. MEX API Reference for MFE Toolbox

### 7.1 Common MEX API Functions

#### 7.1.1 Input Argument Handling
```c
/* Get number of input arguments */
nrhs  /* Number of right-hand side (input) arguments */

/* Access input arguments */
prhs[0]  /* First input argument */
mxGetPr(prhs[0])  /* Get pointer to real data */
mxGetM(prhs[0])  /* Get number of rows */
mxGetN(prhs[0])  /* Get number of columns */
```

#### 7.1.2 Output Argument Creation
```c
/* Create output argument */
plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
double *output = mxGetPr(plhs[0]);

/* Alternative for custom dimensions */
mwSize dims[2] = {m, n};
plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
```

### 7.2 MFE Toolbox-Specific Patterns

#### 7.2.1 Standard MEX Entry Point
```c
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Validate number of arguments */
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("MFEToolbox:functionName:invalidNumInputs",
                          "Four inputs required.");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt("MFEToolbox:functionName:maxLHS",
                          "Too many output arguments.");
    }
    
    /* Validate input types and dimensions */
    /* Allocate output memory */
    /* Perform computation */
    /* Clean up temporary memory if necessary */
}
```

## 8. References and Additional Resources

### 8.1 MATLAB Documentation
- [MATLAB MEX Files Guide](https://www.mathworks.com/help/matlab/matlab_external/introducing-mex-files.html)
- [C Matrix API](https://www.mathworks.com/help/matlab/matlab_external/c-matrix-api.html)

### 8.2 MFE Toolbox Documentation
- MFE Toolbox Technical Specification
- Individual MEX source files in the `mex_source` directory

### 8.3 Compiler-Specific Resources
- [Microsoft Visual C++ Compiler](https://visualstudio.microsoft.com/vs/features/cplusplus/)
- [GCC Documentation](https://gcc.gnu.org/onlinedocs/)

---

## Appendix A: MEX File Structure Example

Below is a simplified example of the structure used in MFE Toolbox MEX files:

```c
/* MEX implementation template for MFE Toolbox
 * 
 * Function: agarch_core
 * Description: Core implementation of AGARCH model calculations
 * 
 * Inputs:
 *   prhs[0] - data: T x 1 vector of observations
 *   prhs[1] - parameters: K x 1 vector of model parameters
 *   prhs[2] - p: GARCH lag order
 *   prhs[3] - q: ARCH lag order
 *
 * Outputs:
 *   plhs[0] - ht: T x 1 vector of conditional variances
 */

#include "mex.h"
#include "matrix.h"
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variable declarations */
    double *data, *parameters, *p_ptr, *q_ptr, *ht;
    mwSize T, K, p, q;
    
    /* Step 1: Validate inputs */
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("MFEToolbox:agarch_core:invalidNumInputs",
                          "Four inputs required.");
    }
    
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MFEToolbox:agarch_core:invalidInput",
                          "Data must be a real double array.");
    }
    
    /* Step 2: Extract input data */
    data = mxGetPr(prhs[0]);
    parameters = mxGetPr(prhs[1]);
    p_ptr = mxGetPr(prhs[2]);
    q_ptr = mxGetPr(prhs[3]);
    
    T = mxGetM(prhs[0]);
    K = mxGetM(prhs[1]);
    p = (mwSize)*p_ptr;
    q = (mwSize)*q_ptr;
    
    /* Step 3: Validate dimensions */
    if (T < p || T < q) {
        mexErrMsgIdAndTxt("MFEToolbox:agarch_core:invalidData",
                          "Data length must exceed model order.");
    }
    
    /* Step 4: Allocate output */
    plhs[0] = mxCreateDoubleMatrix(T, 1, mxREAL);
    ht = mxGetPr(plhs[0]);
    
    /* Step 5: Perform computation */
    /* ... Core computation code here ... */
    
    /* No explicit cleanup needed as MATLAB handles memory */
}
```

## Appendix B: Platform-Specific MEX Extensions

| Platform | MEX Extension | Example Filename |
|----------|--------------|-----------------|
| Windows 64-bit | .mexw64 | agarch_core.mexw64 |
| Linux 64-bit | .mexa64 | agarch_core.mexa64 |
| macOS 64-bit | .mexmaci64 | agarch_core.mexmaci64 |
| Windows 32-bit | .mexw32 | agarch_core.mexw32 |
| Linux 32-bit | .mexglx | agarch_core.mexglx |
| macOS 32-bit | .mexmaci | agarch_core.mexmaci |