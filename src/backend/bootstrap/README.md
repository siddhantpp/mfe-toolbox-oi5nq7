# Bootstrap Module - MFE Toolbox

The bootstrap module implements robust resampling techniques for financial time series data with temporal dependence. These methods enable accurate statistical inference when traditional i.i.d. assumptions are violated, as is common in financial time series data.

## Module Components

The bootstrap module consists of the following components:

### block_bootstrap.m
Implements the block bootstrap method for dependent time series. This method resamples blocks of consecutive observations to preserve the dependency structure in the data.

### stationary_bootstrap.m
Implements the stationary bootstrap method, which uses random block sizes with a geometric distribution to ensure stationarity while preserving temporal dependence.

### bootstrap_variance.m
Computes variance and standard error estimates for statistics based on bootstrap resampling, providing robust measures of parameter uncertainty for time series data.

### bootstrap_confidence_intervals.m
Constructs confidence intervals using bootstrap samples, supporting various methods including percentile, basic, studentized, and bias-corrected and accelerated (BCa).

## Key Features

The bootstrap module provides the following key features:

### Time Series Support
Specialized bootstrap methods designed for dependent time series data, preserving the temporal structure of financial data.

### Multiple Bootstrap Methods
Support for both block bootstrap and stationary bootstrap approaches, allowing researchers to select the most appropriate method for their specific data characteristics.

### Robust Inference
Tools for variance estimation and confidence interval construction that remain valid under temporal dependence, heteroskedasticity, and non-normality.

### Flexible Statistics
Works with arbitrary statistic functions, enabling bootstrap-based inference for means, volatilities, correlations, quantiles, and custom metrics.

### Advanced Confidence Intervals
Multiple confidence interval construction methods, including advanced techniques like bias-corrected and accelerated (BCa) intervals for improved coverage accuracy.

## Usage Examples

Basic examples of how to use the bootstrap functions:

### Block Bootstrap Example

```matlab
% Generate bootstrap samples using block bootstrap
data = returns;  % Financial returns time series
B = 1000;       % Number of bootstrap samples
block_size = 20; % Block size for preserving dependence
bootstrap_samples = block_bootstrap(data, B, block_size);

% Compute bootstrap distribution of mean
bootstrap_means = mean(bootstrap_samples);
```

### Stationary Bootstrap Example

```matlab
% Generate bootstrap samples using stationary bootstrap
data = returns;  % Financial returns time series
B = 1000;       % Number of bootstrap samples
p = 0.05;       % Probability parameter (expected block size = 1/p = 20)
bootstrap_samples = stationary_bootstrap(data, B, p);

% Compute bootstrap distribution of volatility
bootstrap_vols = std(bootstrap_samples);
```

### Bootstrap Confidence Intervals Example

```matlab
% Compute 95% confidence interval for mean return
data = returns;
stat_fn = @(x) mean(x);  % Statistic function
alpha = 0.05;           % Significance level (95% CI)
bootstrap_type = 'block';
B = 5000;               % Number of bootstrap replications
bootstrap_params = struct('block_size', 20);
method = 'percentile';  % Confidence interval method

ci_results = bootstrap_confidence_intervals(data, stat_fn, alpha, ...
    bootstrap_type, B, bootstrap_params, method);

fprintf('95%% CI for mean return: [%f, %f]\n', ...
    ci_results.lower_bound, ci_results.upper_bound);
```

## Integration

The bootstrap module integrates with other MFE Toolbox components:

### Time Series Models
Bootstrap methods can be used with ARMA/ARMAX models to assess parameter uncertainty and construct confidence intervals for forecasts.

### Volatility Models
Combined with GARCH-family models to quantify uncertainty in volatility estimates and risk metrics.

### Statistical Tests
Used to improve the robustness of statistical tests when assumptions about error distributions are violated.

### High-Frequency Analysis
Applied to realized volatility measures for improved inference with noisy high-frequency data.

## Documentation

For detailed documentation on bootstrap methods, see docs/bootstrap_methods.md

## Examples

For example applications, see examples/bootstrap_confidence_intervals.m