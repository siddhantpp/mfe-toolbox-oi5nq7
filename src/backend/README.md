# MFE Toolbox Backend

## Overview
The MATLAB Financial Econometrics (MFE) Toolbox is a comprehensive suite of tools for financial time series modeling, econometric analysis, and risk assessment. Version 4.0, released on October 28, 2009, provides a robust framework for quantitative finance and econometric analysis in MATLAB.

## Directory Structure
The MFE Toolbox backend is organized into the following directory structure for modular functionality:

| Directory | Purpose |
|-----------|---------|
| `bootstrap/` | Implementation of block and stationary bootstrap methods for dependent time series |
| `crosssection/` | Tools for cross-sectional analysis of financial data |
| `distributions/` | Implementations of statistical distributions including GED, Hansen's skewed T, and standardized Student's T |
| `GUI/` | Interactive graphical user interface for ARMAX modeling and analysis |
| `multivariate/` | Multivariate time series and volatility models (VAR, VECM, multivariate GARCH variants) |
| `tests/` | Statistical testing suite for time series analysis including unit root, heteroskedasticity, and autocorrelation tests |
| `timeseries/` | Time series modeling tools with ARMA/ARMAX implementations |
| `univariate/` | Univariate volatility models including GARCH, EGARCH, TARCH, IGARCH, and AGARCH |
| `utility/` | Common utility functions for data validation, matrix operations, and parameter checking |
| `realized/` | High-frequency volatility analysis tools for realized volatility measures |
| `mex_source/` | C source code for high-performance MEX implementations |
| `dlls/` | Compiled MEX binaries for Windows (.mexw64) and Unix (.mexa64) platforms |
| `duplication/` | Optional work-alike functions for compatibility (if needed) |

## Installation

Follow these steps to install and initialize the MFE Toolbox in MATLAB:

1. Extract the MFEToolbox.zip archive to your desired location
2. Navigate to the extracted directory in MATLAB
3. Run the `addToPath.m` script to configure the MATLAB path with all required components
4. Optionally, use the 'savePath' parameter to permanently save the path configuration
5. Verify installation by checking that MEX binaries are correctly loaded for your platform

Example code for installation:

```matlab
% To add the MFE Toolbox to the MATLAB path
addToPath;

% To add to path and save permanently
addToPath(true);

% To add optional directories
addToPath(true, true);
```

## Core Components

### Statistical Analysis Core
- Comprehensive statistical distribution functions
- Bootstrap methods for time series data
- Statistical tests for financial time series
- Cross-sectional analysis tools

### Time Series Analysis
- ARMA/ARMAX modeling and forecasting
- Univariate volatility models (GARCH family)
- Multivariate volatility models
- High-frequency data analysis

### Support Infrastructure
- GUI for ARMAX modeling
- Utility functions for data handling
- MEX-based performance optimization
- Cross-platform compatibility layer

## MEX Integration

The MFE Toolbox incorporates high-performance C implementations via the MATLAB MEX interface for computationally intensive operations.

### MEX Files
| File | Purpose |
|------|---------|
| `agarch_core` | Asymmetric GARCH model computations |
| `armaxerrors` | ARMAX residual error calculation |
| `composite_likelihood` | Composite likelihood estimation |
| `egarch_core` | Exponential GARCH model computations |
| `igarch_core` | Integrated GARCH model computations |
| `tarch_core` | Threshold ARCH/GARCH model computations |

### Platform Support
- Windows (PCWIN64): Files with `.mexw64` extension
- Unix: Files with `.mexa64` extension

## Build Process

To build the MFE Toolbox from source code:

1. Ensure MATLAB and a compatible C compiler are installed
2. Run the `buildZipFile.m` script to perform the compilation and packaging
3. The script will compile C source files with the `-largeArrayDims` flag
4. Platform-specific MEX binaries will be generated based on your environment
5. The compiled binaries and MATLAB files will be packaged into MFEToolbox.zip

Example code for building:

```matlab
% To build the MFE Toolbox package
buildZipFile;
```

## Usage Examples

### Time Series Modeling with ARMAX
```matlab
% Load data
data = load('example_data.mat');
y = data.returns;
x = data.factors;

% Estimate ARMAX(1,1) model with exogenous variables
[parameters, errors, LLF, innovations, scores, hessian] = ...
    armaxfilter(y, x, 1, 1);

% Display results
disp('ARMAX(1,1) Model Parameters:');
disp(parameters);
```

### Volatility Modeling with GARCH
```matlab
% Load return data
returns = load('stock_returns.mat');
y = returns.data;

% Estimate GARCH(1,1) model
[parameters, logL, ht] = tarch(y, 1, 0, 1);

% Plot conditional variance
figure;
plot(ht);
title('GARCH(1,1) Conditional Variance');
xlabel('Time');
ylabel('Variance');
```

### Bootstrap Analysis
```matlab
% Generate bootstrap samples for time series
data = randn(500, 1); % Example data
B = 1000; % Number of bootstrap replications
blockSize = 10; % Block size for block bootstrap
[bootData, indices] = block_bootstrap(data, B, blockSize);

% Compute bootstrap statistic (e.g., mean)
bootMeans = mean(bootData);

% Compute confidence intervals
alpha = 0.05;
ci = quantile(bootMeans, [alpha/2, 1-alpha/2]);
fprintf('95%% Confidence Interval: [%.4f, %.4f]\n', ci(1), ci(2));
```

## Performance Considerations

To optimize performance when using the MFE Toolbox:

- Use MEX-accelerated functions for large datasets
- Preallocate matrices for vectorized operations
- Consider memory constraints for high-frequency data analysis
- Leverage multivariate models for related time series

For computationally intensive tasks, the toolbox automatically uses optimized C implementations via MEX when available for your platform.

## System Initialization

The MFE Toolbox follows a systematic initialization process when `addToPath.m` is executed:

1. The system performs a platform check to determine the appropriate MEX binaries
2. Core directories are added to the MATLAB path
3. Optional directories are included if specified
4. The path configuration is saved permanently if requested

This process ensures that all required components are accessible and correctly configured for your specific platform.

## Cross-Platform Compatibility

The MFE Toolbox is designed to work across different operating systems:

- Windows users: MEX binaries with `.mexw64` extension will be loaded
- Unix users: MEX binaries with `.mexa64` extension will be loaded
- Platform detection is automatic during initialization
- Pure MATLAB implementations are available as fallback when MEX binaries cannot be loaded

## License and Attribution

© 2009 All Rights Reserved

This MATLAB Financial Econometrics Toolbox (Version 4.0, released October 28, 2009) is provided as-is. All rights are reserved. Unauthorized copying, modification, or distribution is prohibited.