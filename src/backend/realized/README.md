# Realized Volatility and High-Frequency Analysis

This directory contains MATLAB functions implementing state-of-the-art methods for analyzing high-frequency financial data, with a focus on realized volatility measures, noise-robust estimators, and jump detection techniques.

## Overview

High-frequency financial data provides rich information about intraday price dynamics and volatility patterns. The functions in this directory implement various realized volatility estimators that convert high-frequency return observations into measures of integrated variance and volatility.

## Available Functions

- **rv_compute.m**: Calculates standard realized volatility by summing squared intraday returns
- **bv_compute.m**: Implements bipower variation, a jump-robust estimator of integrated variance
- **rv_kernel.m**: Provides noise-robust kernel-based realized volatility estimators
- **realized_spectrum.m**: Implements spectral analysis of realized volatility
- **jump_test.m**: Implements statistical tests for detecting price jumps based on the ratio of realized volatility to bipower variation

## Implementation Details

These functions implement econometric methods from the high-frequency financial econometrics literature. Key implementation details include:

- Comprehensive input validation for robustness
- Efficient vectorized computation for large datasets
- Support for various sampling frequencies and data formats
- Configurable options for different market microstructure conditions
- Integration with other MFE Toolbox components

## Theoretical Background

The implemented methods are based on the theoretical framework of realized volatility estimation in the presence of microstructure noise and jumps. Key references include:

1. Andersen, T.G., Bollerslev, T., Diebold, F.X., & Labys, P. (2001). The distribution of realized exchange rate volatility.
2. Barndorff-Nielsen, O.E., & Shephard, N. (2004). Power and bipower variation with stochastic volatility and jumps.
3. Zhang, L., Mykland, P.A., & Aït-Sahalia, Y. (2005). A tale of two time scales: Determining integrated volatility with noisy high-frequency data.
4. Barndorff-Nielsen, O.E., & Shephard, N. (2006). Econometrics of testing for jumps in financial economics using bipower variation.

## Usage Examples

For examples demonstrating how to use these functions, refer to the examples/high_frequency_volatility.m script, which provides a comprehensive walkthrough with sample data.

### Basic Usage Pattern

```matlab
% Load high-frequency return data
returns = ...; % Your intraday return data

% Compute standard realized volatility
rv = rv_compute(returns);

% Compute jump-robust bipower variation
bv = bv_compute(returns);

% Test for the presence of jumps
jump_results = jump_test(returns);

% Compute noise-robust kernel estimator
rv_kernel_est = rv_kernel(returns, 'kernel_type', 'Bartlett');
```

## Integration with MFE Toolbox

These high-frequency analysis functions integrate with other components of the MFE Toolbox:

- **Time Series Analysis**: Combining realized measures with ARMA/GARCH modeling
- **Distribution Analysis**: Testing distributional properties of realized measures
- **Bootstrap Methods**: Resampling for robust inference with realized measures
- **Risk Assessment**: Using realized measures for financial risk evaluation

## Performance Considerations

These functions are optimized for efficient computation with large high-frequency datasets. When working with very large datasets (millions of observations), consider:

1. Using the memory-efficient options where available
2. Processing data in smaller time windows when appropriate
3. Utilizing the subsampling options to balance efficiency and accuracy