# Troubleshooting Guide

Comprehensive guide to resolving common issues with the MFE Toolbox v4.0

## Installation Issues

### Path Configuration Problems

#### Problem: Toolbox not found after installation
* Symptoms:
    * "'Undefined function or variable' errors when calling MFE functions"
    * "MATLAB cannot locate MFE Toolbox components"
* Causes:
    * Path not properly configured
    * addToPath.m not executed
    * Path not saved permanently
* Solutions:
    * Run addToPath.m from the toolbox root directory
    * Set savePath parameter to true when calling addToPath
    * Manually add toolbox directories to MATLAB path using pathtool
    * Check MATLAB startup.m for conflicting path configurations
* Code Example:
```matlab
% Execute from the MFE Toolbox root directory
addToPath(true, true);  % Parameters: savePath, addOptionalDirs
```

#### Problem: Path conflicts with other toolboxes
* Symptoms:
    * Wrong function version being called
    * Unexpected behavior from MFE functions
* Causes:
    * Path precedence issues
    * Function name conflicts
* Solutions:
    * Use the 'which -all functionname' command to identify conflicts
    * Reorder MATLAB path using pathtool to prioritize correctly
    * Disambiguate function calls using full package paths
* Code Example:
```matlab
which -all backcast  % Check which backcast function is being called
```

### Directory Structure Issues

#### Problem: Missing mandatory directories
* Symptoms:
    * Error messages about missing directories during addToPath execution
    * Functions from specific components not available
* Causes:
    * Incomplete installation
    * Missing components in ZIP archive
    * Files manually deleted after installation
* Solutions:
    * Reinstall the complete MFE Toolbox from the official ZIP archive
    * Verify all required directories exist: bootstrap, crosssection, distributions, GUI, multivariate, tests, timeseries, univariate, utility, realized, mex_source, dlls
    * Check directory permissions

### Installation Verification Failures

#### Problem: Unable to verify successful installation
* Symptoms:
    * Functions not recognized after installation
    * MEX binaries not detected
    * Path configuration appears incomplete
* Causes:
    * Installation process incomplete
    * Path not correctly saved
    * Missing components or permissions issues
* Solutions:
    * Verify installation using the following code to check core functionality
    * Confirm path configuration with path command
    * Check for existence of key functions and MEX files
    * Review MATLAB command window for any error messages during installation
* Code Example:
```matlab
% Basic installation verification
disp(['gedpdf found: ' num2str(exist('gedpdf', 'file') > 0)]);
disp(['tarchfit found: ' num2str(exist('tarchfit', 'file') > 0)]);
disp(['MEX acceleration available: ' num2str(exist('agarch_core', 'file') == 3)]);

% Check path configuration
disp(path);
```

### Platform-Specific Installation Issues

#### Problem: Windows-specific installation issues
* Symptoms:
    * Path not saving permanently
    * MEX binaries not loading
    * Permission errors during installation
* Causes:
    * Insufficient permissions
    * Path length limitations
    * Missing Visual C++ Redistributable
* Solutions:
    * Run MATLAB as administrator for installation
    * Install in a location with a shorter path
    * Install required Visual C++ Redistributable packages
    * Use UNC paths if network installation is required

#### Problem: Unix-specific installation issues
* Symptoms:
    * MEX binaries not executable
    * Permission denied errors
    * Library dependencies missing
* Causes:
    * File permissions not set correctly
    * Missing shared libraries
    * Incompatible GCC version
* Solutions:
    * Set appropriate file permissions with chmod
    * Install required shared libraries
    * Configure LD_LIBRARY_PATH environment variable
    * Use a compatible GCC version (4.4.7 or newer recommended)
* Code Example:
```bash
# Add appropriate permissions to MEX files
chmod +x /path/to/MFEToolbox/src/backend/dlls/*.mexa64

# Check for missing shared libraries
ldd /path/to/MFEToolbox/src/backend/dlls/agarch_core.mexa64
```

## MEX-Related Issues

### MEX Binary Loading Problems

#### Problem: MEX files not loading
* Symptoms:
    * Performance significantly slower than expected
    * Warnings about missing MEX files
    * "'Invalid MEX-file' errors"
* Causes:
    * Platform mismatch (using .mexw64 on Unix or .mexa64 on Windows)
    * Corrupted MEX binaries
    * Incompatible MATLAB version
    * Missing C runtime libraries
* Solutions:
    * Verify platform compatibility (use .mexw64 for Windows and .mexa64 for Unix)
    * Recompile MEX files using the appropriate platform-specific scripts
    * Install compatible C runtime libraries
    * Check MATLAB version compatibility (v4.0 requires specific MATLAB version)
* Code Example:
```matlab
% Check if MEX file exists and can be called
exist('agarch_core', 'file')
```

#### Problem: MEX compilation failures
* Symptoms:
    * Errors during MEX compilation
    * Missing MEX binaries after compilation attempt
* Causes:
    * Missing C compiler
    * Incompatible compiler version
    * Incorrect compilation flags
    * Source code errors
* Solutions:
    * Install a compatible C compiler (see MATLAB documentation for supported compilers)
    * Run 'mex -setup' to configure the MATLAB-compiler interface
    * Use the appropriate compilation script: infrastructure/build_scripts/compile_mex_windows.bat or infrastructure/build_scripts/compile_mex_unix.sh
    * Ensure the -largeArrayDims flag is used for large data support
* Code Example:
```matlab
% Check configured compiler
mex -setup

% Compile a specific MEX file with detailed output
mex -v -largeArrayDims src/backend/mex_source/agarch_core.c
```

## Statistical Computation Issues

### Numerical Stability Problems

#### Problem: Non-convergence in optimization
* Symptoms:
    * Warnings about maximum iterations reached
    * NaN or Inf results in parameter estimates
    * Unreasonable parameter values
* Causes:
    * Poor initial parameter values
    * Ill-conditioned data
    * Model misspecification
    * Insufficient iterations
* Solutions:
    * Provide better initial parameter values
    * Preprocess data to remove outliers or scale appropriately
    * Try a different model specification
    * Increase maximum iterations or adjust optimization settings
    * Use robust optimization methods
* Code Example:
```matlab
% Example with custom optimization settings for GARCH estimation
options = optimset('MaxIter', 1000, 'TolFun', 1e-8, 'TolX', 1e-8);
parameters = tarchfit(data, p, o, q, [], [], [], options);
```

#### Problem: Matrix singularity issues
* Symptoms:
    * Warnings about singular or nearly singular matrices
    * Errors in matrix inversion operations
    * Unstable estimation results
* Causes:
    * Multicollinearity in regression variables
    * Zero or near-zero variance in data series
    * Insufficient data for parameter estimation
* Solutions:
    * Check input data for multicollinearity
    * Use principal component analysis to reduce dimensionality
    * Add regularization methods to estimation
    * Increase sample size if possible
* Code Example:
```matlab
% Check correlation matrix for multicollinearity
corr_matrix = corrcoef(data);
figure; imagesc(corr_matrix); colorbar; title('Correlation Matrix');
```

### Distribution and Random Number Issues

#### Problem: Distribution parameter estimation failures
* Symptoms:
    * Errors in distribution fitting functions
    * Unreasonable parameter estimates
    * Failed convergence messages
* Causes:
    * Data not compatible with distribution assumptions
    * Extreme outliers in the data
    * Insufficient sample size
    * Poor initial parameter values
* Solutions:
    * Visualize data with histograms to check compatibility with distribution
    * Remove outliers or use robust estimation methods
    * Increase sample size if possible
    * Provide better initial parameter values
    * Try a different distribution family
* Code Example:
```matlab
% Visualize data distribution before fitting
figure; histogram(data, 50); hold on;

% Try different initial parameters
params = gedfit(data, [0.1, 1.5]);
```

## Time Series Model Issues

### ARMA/ARMAX Model Problems

#### Problem: Model instability or non-stationarity
* Symptoms:
    * Unit root or near unit root warnings
    * Explosive forecasts
    * Poor model fit metrics
* Causes:
    * Non-stationary data
    * Misspecified model order
    * Missing important exogenous variables
    * Structural breaks in the data
* Solutions:
    * Test for stationarity using adf_test or pp_test
    * Difference the data if necessary
    * Try different model orders using information criteria (AIC/BIC)
    * Include relevant exogenous variables
    * Test for and account for structural breaks
* Code Example:
```matlab
% Test for stationarity
p_value = adf_test(data);
if p_value > 0.05
    % Data may be non-stationary, consider differencing
    diff_data = diff(data);
end

% Find optimal model order using AIC/BIC
[aic, bic] = aicsbic(data, max_p, max_q);
```

### Volatility Model Issues

#### Problem: GARCH model estimation problems
* Symptoms:
    * Non-convergence warnings
    * Parameter constraint violations
    * Unrealistic persistence values
* Causes:
    * Inappropriate model for the data
    * Poor initial parameter values
    * Insufficient data for volatility dynamics
    * Model overparameterization
* Solutions:
    * Try simpler models first (GARCH(1,1) before more complex specifications)
    * Test for ARCH effects before fitting models
    * Experiment with different distributions for innovations
    * Use good initial values from simpler models
    * Ensure sufficient data for reliable estimation
* Code Example:
```matlab
% Test for ARCH effects
p_value = arch_test(returns, 10);

% Start with simple GARCH model
[parameters, ~, ~, ~, VCV, scores, diagnostics] = tarchfit(returns, 1, 0, 1);

% Use those parameters as starting values for a more complex model
initial_parameters = parameters;
[parameters_complex, ~, ~, ~, VCV, scores, diagnostics] = tarchfit(returns, 2, 1, 2, [], [], initial_parameters);
```

## GUI Issues

### ARMAX GUI Problems

#### Problem: GUI graphical display issues
* Symptoms:
    * Missing plot elements
    * Garbled text or formatting
    * Graphics rendering problems
* Causes:
    * MATLAB version incompatibility
    * Graphics driver issues
    * Java version conflicts
    * Screen resolution limitations
* Solutions:
    * Update graphics drivers
    * Use a compatible MATLAB version
    * Adjust screen DPI settings
    * Try alternative plotting backends if available

#### Problem: GUI unresponsiveness
* Symptoms:
    * Freezing during model estimation
    * Long delays in updating plots
    * Interface stops responding to input
* Causes:
    * Processing large datasets
    * Intensive computation without progress updates
    * Memory limitations
    * Event handler conflicts
* Solutions:
    * Reduce dataset size for GUI operation
    * Use command-line functions for large datasets
    * Increase memory allocation for MATLAB
    * Check for event handler conflicts in custom code
* Code Example:
```matlab
% Launch ARMAX GUI with smaller dataset
subset_data = data(1:1000);
ARMAX(subset_data);
```

## Platform-Specific Issues

### Windows-Specific Problems

#### Problem: DLL loading failures
* Symptoms:
    * Errors about missing DLLs
    * MEX files fail to load
    * Performance degradation
* Causes:
    * Missing Visual C++ redistributable packages
    * Path length limitations in Windows
    * File permission issues
* Solutions:
    * Install appropriate Visual C++ redistributable packages
    * Move toolbox to a shorter path location
    * Check file permissions
    * Run MATLAB as administrator when necessary

### Unix-Specific Problems

#### Problem: Shared library issues
* Symptoms:
    * Errors about missing shared libraries
    * MEX files fail to load
    * Compatibility warnings
* Causes:
    * Missing shared libraries
    * Incompatible library versions
    * Path environment variable issues
* Solutions:
    * Install required shared libraries
    * Update LD_LIBRARY_PATH to include necessary directories
    * Recompile MEX files against available libraries
    * Use compatibility symlinks when needed
* Code Example:
```bash
# Add to .bashrc or equivalent
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libraries
```

## Performance Optimization

### Slow Execution Issues

#### Problem: Functions running slower than expected
* Symptoms:
    * Extended computation time
    * MATLAB becoming unresponsive
    * Memory usage warnings
* Causes:
    * MEX binaries not being used
    * Inefficient algorithm usage
    * Large datasets without optimization
    * Memory limitations
* Solutions:
    * Verify MEX binaries are being loaded correctly
    * Use vectorized operations where possible
    * Preallocate matrices for large computations
    * Break large datasets into manageable chunks
    * Increase MATLAB memory allocation
* Code Example:
```matlab
% Check if MEX acceleration is enabled
isMexEnabled = exist('agarch_core', 'file') == 3;  % Returns 3 for MEX files

% Preallocate matrices for better performance
results = zeros(n, m);
for i = 1:n
    % Computation with preallocation
end
```

## Reporting Issues

How to effectively report issues that cannot be resolved using this guide

* Reporting Guidelines:
    * Check this troubleshooting guide thoroughly before reporting
    * Verify issue is not already documented in the known issues section
    * Collect complete system information (MATLAB version, OS version, toolbox version)
    * Create a minimal reproducible example that demonstrates the issue
    * Include full error messages and stack traces
    * Document steps taken to troubleshoot so far
    * Submit detailed report to the appropriate support channel
* Code Example:
```matlab
% Code to collect system information
ver  % MATLAB version and toolbox info
computer  % Computer architecture
ispc  % Check if Windows platform
display('MFE Toolbox Version: 4.0 (2009-10-28)');
```

## Additional Resources

| Name | Path | Description |
|---|---|---|
| Overview Documentation | docs/README.md | General documentation overview including installation information |
| MEX Compilation Guide | docs/mex_compilation.md | Instructions for compiling MEX binaries |
| Cross-Platform Notes | docs/cross_platform_notes.md | Platform-specific information |
| Performance Optimization Guide | docs/performance_optimization.md | Performance optimization techniques and strategies |
| API Reference | docs/api_reference.md | Complete function reference for all toolbox components |
| MATLAB Documentation | https://www.mathworks.com/help/matlab/ | Official MATLAB documentation |
| MATLAB MEX Documentation | https://www.mathworks.com/help/matlab/matlab_external/introducing-mex-files.html | Official documentation on MEX files |