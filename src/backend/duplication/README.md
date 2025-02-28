# MFE Toolbox Duplication Module

## Overview

The MFE Toolbox Duplication Module provides a set of work-alike functions designed to ensure cross-platform compatibility across different MATLAB installations and versions. This optional module addresses scenarios where specific MATLAB toolboxes (such as Statistics Toolbox or Optimization Toolbox) might not be available in a user's installation, ensuring the MFE Toolbox can function properly in diverse environments.

The duplication module is automatically loaded by the MFE Toolbox initialization process (`addToPath.m`) when optional directories are enabled, providing seamless fallback implementations when required MATLAB functions are not available.

## Work-alike Functions

The duplication module implements the following work-alike functions that mimic the behavior of their MATLAB counterparts:

| Function | Description |
|----------|------------|
| `wa_normpdf` | Alternative implementation of normal probability density function (normally provided by Statistics Toolbox) |
| `wa_normcdf` | Alternative implementation of normal cumulative distribution function (normally provided by Statistics Toolbox) |
| `wa_fmincon` | Simplified implementation of constrained optimization function (normally provided by Optimization Toolbox) |
| `wa_fminsearch` | Alternative implementation of unconstrained optimization function (normally provided by Optimization Toolbox) |
| `wa_cov` | Alternative implementation of covariance matrix calculation (normally provided by Statistics Toolbox) |
| `wa_pcacov` | Alternative implementation of principal component analysis (normally provided by Statistics Toolbox) |
| `wa_qr` | Alternative implementation of QR decomposition for matrices (normally provided by MATLAB base functionality) |
| `wa_erf` | Alternative implementation of error function (normally provided by MATLAB base functionality) |
| `wa_erfinv` | Alternative implementation of inverse error function (normally provided by MATLAB base functionality) |

## Usage

Work-alike functions are automatically integrated into the MFE Toolbox workflow when:

1. The duplication directory is included in the MFE Toolbox installation
2. A required MATLAB function or toolbox is not available in the user's environment

When the MFE Toolbox initializes through `addToPath.m`, it checks for optional directories and integrates the work-alike functions when appropriate. This process is transparent to the user, allowing the toolbox to function without requiring additional configuration.

For example, if the Statistics Toolbox is not available, functions like `wa_normpdf` will be automatically used as replacements for `normpdf` when required by MFE Toolbox components.

## Implementation Notes

### Design Philosophy

The work-alike functions are designed with the following principles:

1. **Numerical Accuracy**: Implementations focus on producing correct results that closely match the original MATLAB functions
2. **Robustness**: Functions include proper error handling and input validation
3. **Compatibility**: APIs mirror the original MATLAB functions to ensure seamless substitution
4. **Documentation**: Each function contains detailed comments describing implementation choices and any limitations

### Limitations

Work-alike functions may have some limitations compared to their MATLAB counterparts:

- Simplified implementations that may not support all optional parameters
- Potentially lower performance for large-scale computations
- Possibly reduced numerical precision in edge cases
- Limited support for specialized input formats

These limitations are documented in each function's header comments.

## Toolbox Detection

The duplication module uses the `isToolboxAvailable` function to detect which MATLAB toolboxes are available in the current installation. This function checks for the presence of key functions that indicate whether a specific toolbox is installed.

During initialization, the `work_alike_init` function:

1. Checks which toolboxes are available
2. Conditionally loads the appropriate work-alike functions based on availability
3. Configures the MATLAB path to prioritize original functions when available

## Performance Considerations

Work-alike functions prioritize correctness and compatibility over maximum performance. While they are designed to be reasonably efficient, they may not match the performance of native MATLAB implementations, especially for:

- Large matrix operations
- Complex numerical integrations
- Highly optimized algorithms
- Hardware-accelerated computations

For performance-critical applications, it is recommended to install the corresponding MATLAB toolboxes when possible.

## Examples

### Checking for Toolbox Availability

```matlab
% Check if Statistics Toolbox is available
hasStats = isToolboxAvailable('statistics');

if ~hasStats
    disp('Using work-alike functions for statistical computations');
    % Your code that relies on statistical functions
else
    disp('Using native MATLAB Statistics Toolbox');
    % Your code that uses native functions
end
```

### Using Work-alike Functions

```matlab
% Example of using work-alike functions as fallbacks
try
    % First try to use the native function
    y = normpdf(x, mu, sigma);
catch
    % Fall back to work-alike implementation if needed
    y = wa_normpdf(x, mu, sigma);
end
```

Note: The above manual fallback approach is typically not needed as the MFE Toolbox handles this automatically when properly initialized with `addToPath.m`.