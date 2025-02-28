# Statistical Tests

The MFE Toolbox provides a comprehensive suite of statistical tests essential for financial econometrics and time series analysis. These tests enable users to evaluate stationarity, normality, autocorrelation, heteroskedasticity, and nonlinear dependence in financial time series, which are critical for proper model specification and validation.

## Table of Contents

- [Unit Root and Stationarity Tests](#unit-root-and-stationarity-tests)
- [Autocorrelation Tests](#autocorrelation-tests)
- [Normality Tests](#normality-tests)
- [Heteroskedasticity Tests](#heteroskedasticity-tests)
- [Nonlinear Dependence Tests](#nonlinear-dependence-tests)
- [Practical Applications](#practical-applications)
- [Integration with Other Toolbox Components](#integration-with-other-toolbox-components)
- [Performance Considerations](#performance-considerations)
- [See Also](#see-also)

## Unit Root and Stationarity Tests

Unit root and stationarity tests are fundamental in time series analysis to determine whether a series is stationary, which is a key assumption for many econometric models. The MFE Toolbox implements three widely-used tests with complementary properties.

### adf_test

Implements the Augmented Dickey-Fuller test for unit roots in time series. The test examines the null hypothesis that a time series has a unit root (is non-stationary) against the alternative that it is stationary.

```matlab
results = adf_test(y, options)
```

**Parameters:**
- `y` - T×1 vector: Time series data to be tested
- `options` - structure: Optional test configuration with fields:
  - `lags`: Integer lag order or 'aic'/'bic' for automatic selection (default: 0)
  - `regression_type`: Model specification: 'n' (none), 'c' (constant), or 'ct' (constant and trend) (default: 'c')
  - `alpha`: Significance level for critical values (default: 0.05)

**Returns:**
- `results` - structure: Test results containing fields: .stat (test statistic), .pval (p-value), .crit_vals (critical values), .lags (lag order used), .regression_type (model specification)

**Example:**
```matlab
% Simple ADF test with default options (constant, no lags)
results = adf_test(returns);
disp(['ADF test statistic: ' num2str(results.stat)]);
disp(['p-value: ' num2str(results.pval)]);

% ADF test with trend and automatic lag selection using AIC
options = struct('regression_type', 'ct', 'lags', 'aic');
results = adf_test(returns, options);
disp(['ADF test statistic: ' num2str(results.stat)]);
disp(['Selected lags: ' num2str(results.lags)]);
```

**Notes:**
The ADF test is sensitive to lag specification. Using information criteria for automatic selection is often preferable to arbitrary selection. The test has low power against near-unit root processes.

### pp_test

Implements the Phillips-Perron test for unit roots in time series. The test uses a non-parametric correction to handle serial correlation, making it more robust than ADF to certain forms of heteroskedasticity and autocorrelation.

```matlab
results = pp_test(y, options)
```

**Parameters:**
- `y` - T×1 vector: Time series data to be tested
- `options` - structure: Optional test configuration with fields:
  - `lags`: Integer lag truncation parameter for spectral density estimation (default: floor(4*(T/100)^0.25))
  - `regression_type`: Model specification: 'n' (none), 'c' (constant), or 'ct' (constant and trend) (default: 'c')
  - `alpha`: Significance level for critical values (default: 0.05)

**Returns:**
- `results` - structure: Test results containing fields: .stat (test statistic), .pval (p-value), .crit_vals (critical values), .lags (lag truncation parameter), .regression_type (model specification)

**Example:**
```matlab
% Simple PP test with default options
results = pp_test(returns);
disp(['PP test statistic: ' num2str(results.stat)]);
disp(['p-value: ' num2str(results.pval)]);

% PP test with trend and custom lag truncation
options = struct('regression_type', 'ct', 'lags', 10);
results = pp_test(returns, options);
disp(['PP test statistic: ' num2str(results.stat)]);
```

**Notes:**
The Phillips-Perron test uses a non-parametric correction for autocorrelation, making it less sensitive to lag specification than the ADF test. However, it may have size distortions in small samples with strong negative moving average components.

### kpss_test

Implements the Kwiatkowski-Phillips-Schmidt-Shin test for stationarity. Unlike ADF and PP tests, KPSS tests the null hypothesis that a series is stationary against the alternative of a unit root.

```matlab
results = kpss_test(y, options)
```

**Parameters:**
- `y` - T×1 vector: Time series data to be tested
- `options` - structure: Optional test configuration with fields:
  - `lags`: Integer lag truncation parameter for spectral density estimation (default: floor(4*(T/100)^0.25))
  - `regression_type`: Model specification: 'c' (level stationarity) or 'ct' (trend stationarity) (default: 'c')
  - `alpha`: Significance level for critical values (default: 0.05)

**Returns:**
- `results` - structure: Test results containing fields: .stat (test statistic), .pval (p-value), .crit_vals (critical values), .lags (lag truncation parameter), .regression_type (model specification)

**Example:**
```matlab
% Simple KPSS test for level stationarity
results = kpss_test(returns);
disp(['KPSS test statistic: ' num2str(results.stat)]);
disp(['p-value: ' num2str(results.pval)]);

% KPSS test for trend stationarity
options = struct('regression_type', 'ct');
results = kpss_test(returns, options);
disp(['KPSS test statistic: ' num2str(results.stat)]);
```

**Notes:**
The KPSS test complements ADF and PP tests by testing the null of stationarity rather than non-stationarity. Joint use of ADF/PP and KPSS tests can provide more robust conclusions about the presence of unit roots.

## Autocorrelation Tests

Autocorrelation tests examine whether observations in a time series are correlated with their own lagged values. These tests are crucial for validating model adequacy and detecting serial dependence in financial time series and model residuals.

### ljungbox

Implements the Ljung-Box Q-test for autocorrelation in time series. The test examines the null hypothesis that a series exhibits no autocorrelation up to a specified lag order.

```matlab
results = ljungbox(data, lags, dofsAdjust)
```

**Parameters:**
- `data` - T×1 vector: Time series data or model residuals
- `lags` - positive integer or vector: Lag order(s) to test (default: min(10, T/5))
- `dofsAdjust` - non-negative integer: Degrees of freedom adjustment for fitted models (default: 0)

**Returns:**
- `results` - structure: Test results containing fields: .q (Q-statistics), .pval (p-values), .cvs (critical values at 1%, 5%, 10%), .h (rejection indicators)

**Example:**
```matlab
% Simple Ljung-Box test on returns
results = ljungbox(returns, 10);
disp('Ljung-Box Q-statistics:');
disp(results.q);
disp('p-values:');
disp(results.pval);

% Test on ARMA model residuals with degrees of freedom adjustment
arma_results = armaxfilter(returns, 1, 1);
residuals = arma_results.residuals;
% Adjust for 2 estimated parameters (AR and MA)
results = ljungbox(residuals, 10, 2);
disp('Adjusted p-values:');
disp(results.pval);
```

**Notes:**
When testing residuals from fitted models, the degrees of freedom should be adjusted by the number of estimated parameters to maintain proper test size. For ARMA(p,q) residuals, set dofsAdjust = p+q.

### lmtest1

Implements the Lagrange Multiplier test for serial correlation in model residuals. The test is based on an auxiliary regression of residuals on lagged residuals and original regressors.

```matlab
results = lmtest1(residuals, regressors, lags)
```

**Parameters:**
- `residuals` - T×1 vector: Model residuals to be tested
- `regressors` - T×k matrix: Original regressors from the model
- `lags` - positive integer: Lag order for serial correlation test (default: 1)

**Returns:**
- `results` - structure: Test results containing fields: .stat (test statistic), .pval (p-value), .dof (degrees of freedom)

**Example:**
```matlab
% Estimate linear regression model
[~, ~, residuals] = regress(y, X);

% Test for first-order serial correlation
results = lmtest1(residuals, X, 1);
disp(['LM test statistic: ' num2str(results.stat)]);
disp(['p-value: ' num2str(results.pval)]);

% Test for higher-order serial correlation
results = lmtest1(residuals, X, 4);
disp(['LM test (4 lags) p-value: ' num2str(results.pval)]);
```

**Notes:**
The LM test is particularly useful for regression models and has good power properties. It's more appropriate than Durbin-Watson for models with lagged dependent variables or higher-order serial correlation.

## Normality Tests

Normality tests evaluate whether a series follows a normal distribution, which is a common assumption in many financial models. These tests examine the skewness and kurtosis of the empirical distribution.

### jarque_bera

Implements the Jarque-Bera test for normality based on sample skewness and kurtosis. The test examines the null hypothesis that the data come from a normal distribution with unknown mean and variance.

```matlab
results = jarque_bera(data)
```

**Parameters:**
- `data` - T×1 vector or T×k matrix: Data to be tested for normality

**Returns:**
- `results` - structure: Test results containing fields: .stat (test statistic), .pval (p-value), .skewness (sample skewness), .kurtosis (sample kurtosis)

**Example:**
```matlab
% Test normality of returns
results = jarque_bera(returns);
disp(['Jarque-Bera statistic: ' num2str(results.stat)]);
disp(['p-value: ' num2str(results.pval)]);
disp(['Skewness: ' num2str(results.skewness)]);
disp(['Excess Kurtosis: ' num2str(results.kurtosis-3)]);

% Test normality of model residuals
arma_results = armaxfilter(returns, 1, 1);
residuals = arma_results.residuals;
results = jarque_bera(residuals);
disp(['Residuals JB p-value: ' num2str(results.pval)]);
```

**Notes:**
Financial returns typically exhibit excess kurtosis (fat tails) and often negative skewness, leading to strong rejections of normality. The test has high power against departures from normality in large samples.

## Heteroskedasticity Tests

Heteroskedasticity tests examine whether the variance of a series is constant over time. These tests are crucial for identifying volatility clustering in financial time series, which motivates the use of GARCH-type models.

### arch_test

Implements Engle's ARCH-LM test for conditional heteroskedasticity (ARCH effects) in time series. The test examines the null hypothesis of no ARCH effects against the alternative of ARCH(q) process.

```matlab
results = arch_test(data, lags)
```

**Parameters:**
- `data` - T×1 vector: Time series data to be tested
- `lags` - positive integer: Lag order for ARCH test (default: 5)

**Returns:**
- `results` - structure: Test results containing fields: .stat (test statistic), .pval (p-value), .lags (lag order used)

**Example:**
```matlab
% Test for ARCH effects with 5 lags
results = arch_test(returns, 5);
disp(['ARCH-LM test statistic: ' num2str(results.stat)]);
disp(['p-value: ' num2str(results.pval)]);

% Test residuals from mean model for remaining ARCH effects
arma_results = armaxfilter(returns, 1, 1);
residuals = arma_results.residuals;
results = arch_test(residuals, 5);
disp(['Residual ARCH test p-value: ' num2str(results.pval)]);
```

**Notes:**
The ARCH test is crucial for identifying the need for volatility models in financial time series. Significant ARCH effects in returns or mean model residuals indicate potential benefits from GARCH-type modeling.

### white_test

Implements White's test for heteroskedasticity in regression residuals. The test examines the null hypothesis of homoskedasticity against the alternative that the variance depends on the regressors and their squares.

```matlab
results = white_test(residuals, regressors)
```

**Parameters:**
- `residuals` - T×1 vector: Regression model residuals
- `regressors` - T×k matrix: Original regressors from the model

**Returns:**
- `results` - structure: Test results containing fields: .stat (test statistic), .pval (p-value), .dof (degrees of freedom)

**Example:**
```matlab
% Estimate linear regression model
[~, ~, residuals] = regress(y, X);

% Test for heteroskedasticity
results = white_test(residuals, X);
disp(['White test statistic: ' num2str(results.stat)]);
disp(['p-value: ' num2str(results.pval)]);
```

**Notes:**
White's test is a general test for heteroskedasticity that doesn't require specifying the form of heteroskedasticity. It's particularly useful in cross-sectional regressions where the error variance might depend on the explanatory variables.

## Nonlinear Dependence Tests

Nonlinear dependence tests examine whether a series exhibits more complex patterns beyond linear autocorrelation. These tests are important for detecting nonlinear structure in financial time series that might not be captured by linear models.

### bds_test

Implements the BDS test for independence and identical distribution in time series. The test examines the null hypothesis that the series is independently and identically distributed (i.i.d.) against an unspecified alternative.

```matlab
results = bds_test(data, options)
```

**Parameters:**
- `data` - T×1 vector: Time series data to be tested
- `options` - structure: Optional test configuration with fields:
  - `max_dim`: Maximum embedding dimension (default: 5)
  - `epsilon`: Distance for proximity determination (default: 0.7*std(data))
  - `bootstrap`: Use bootstrap for p-values (default: false)
  - `nboot`: Number of bootstrap replications if bootstrap=true (default: 1000)

**Returns:**
- `results` - structure: Test results containing fields: .stats (test statistics), .pvals (p-values), .dims (dimensions), .epsilon (distance parameter)

**Example:**
```matlab
% Simple BDS test with default options
results = bds_test(returns);
disp('BDS test statistics:');
disp(results.stats);
disp('p-values:');
disp(results.pvals);

% BDS test with custom options
options = struct('max_dim', 8, 'epsilon', 1.0, 'bootstrap', true);
results = bds_test(returns, options);
disp('Bootstrap p-values:');
disp(results.pvals);
```

**Notes:**
The BDS test has power against a wide range of alternatives to i.i.d., including linear dependence, nonlinear dependence, and chaos. It's often applied to residuals from fitted models to check for remaining structure. The test is computationally intensive, especially with bootstrap p-values.

## Practical Applications

This section demonstrates practical applications of statistical tests in financial time series analysis, showing how to use and interpret multiple tests in combination.

### Model Validation Workflow

A typical workflow for time series model validation involves multiple statistical tests to ensure the model adequately captures the data's features:

```matlab
% 1. Test for stationarity before modeling
adf_results = adf_test(returns);
if adf_results.pval > 0.05
    warning('Series may be non-stationary. Consider differencing.');
end

% 2. Fit an ARMA model
arma_results = armaxfilter(returns, 1, 1);
residuals = arma_results.residuals;

% 3. Test residuals for remaining autocorrelation
ljung_results = ljungbox(residuals, 10, 2);  % Adjust for AR(1) and MA(1)
if any(ljung_results.pval < 0.05)
    warning('Significant autocorrelation remains in residuals.');
end

% 4. Test residuals for normality
jb_results = jarque_bera(residuals);
if jb_results.pval < 0.05
    disp('Residuals are not normally distributed.');
    if jb_results.kurtosis > 3
        disp('Consider using Student''s t or GED distribution.');
    end
end

% 5. Test for ARCH effects in residuals
arch_results = arch_test(residuals, 5);
if arch_results.pval < 0.05
    disp('ARCH effects detected. Consider GARCH modeling.');
end
```

### Complementary Tests for Robust Inference

Using complementary tests with different null hypotheses can strengthen conclusions about data properties:

```matlab
% Test for unit roots using both ADF (null: unit root) and KPSS (null: stationary)
adf_results = adf_test(returns);
kpss_results = kpss_test(returns);

% Interpret combined results
if adf_results.pval < 0.05 && kpss_results.pval > 0.05
    disp('Strong evidence of stationarity (ADF rejects unit root, KPSS does not reject stationarity)');
elseif adf_results.pval >= 0.05 && kpss_results.pval <= 0.05
    disp('Strong evidence of unit root (ADF does not reject unit root, KPSS rejects stationarity)');
elseif adf_results.pval < 0.05 && kpss_results.pval <= 0.05
    disp('Conflicting results - series may be fractionally integrated or have structural breaks');
else
    disp('Inconclusive results - possibly insufficient power');
end
```

## Integration with Other Toolbox Components

Statistical tests in the MFE Toolbox integrate seamlessly with other components for comprehensive financial time series analysis:

### Time Series Models

Statistical tests are essential for proper time series model specification and validation. Use unit root tests to verify stationarity assumptions before applying ARMA models, and diagnostic tests to validate fitted models.

### Volatility Models

ARCH tests detect volatility clustering, informing the need for GARCH-type models. Diagnostic tests validate volatility model adequacy after fitting.

### Bootstrap Methods

Bootstrap techniques can improve the finite-sample properties of statistical tests, particularly in cases where asymptotic distributions may not be reliable.

## Performance Considerations

The statistical tests in the MFE Toolbox are optimized for performance, with efficient implementations that take advantage of MATLAB's vectorized operations. Tests that require intensive computations (e.g., BDS test) offer options to control the computational burden by adjusting parameters such as embedding dimensions and bootstrap replications.

## See Also

- [Time Series Models](time_series_models.md) - ARMA/ARMAX modeling with diagnostic tests
- [Volatility Models](volatility_models.md) - GARCH-type models for conditional variance
- [Distribution Functions](distribution_functions.md) - Statistical distributions for financial modeling
- [Bootstrap Methods](bootstrap_methods.md) - Resampling techniques for improved inference