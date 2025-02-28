# MFE Toolbox - Statistical Tests Module

The Statistical Tests module is a core component of the MFE Toolbox that provides comprehensive tools for hypothesis testing, model validation, and statistical inference in financial econometrics. This module implements a variety of tests commonly used in time series analysis, including unit root tests, autocorrelation tests, normality tests, and heteroskedasticity tests.

## Available Tests

This directory contains the following statistical tests:

1. Unit Root & Stationarity Tests:
   - `adf_test.m` - Augmented Dickey-Fuller test for unit roots
   - `pp_test.m` - Phillips-Perron test for unit roots
   - `kpss_test.m` - Kwiatkowski-Phillips-Schmidt-Shin test for stationarity

2. Independence & Nonlinearity Tests:
   - `bds_test.m` - BDS test for independence and nonlinear structure

3. Autocorrelation Tests:
   - `ljungbox.m` - Ljung-Box Q-test for autocorrelation at multiple lags
   - `lmtest1.m` - Lagrange Multiplier test for serial correlation

4. Heteroskedasticity Tests:
   - `arch_test.m` - Engle's ARCH test for conditional heteroskedasticity
   - `white_test.m` - White's test for heteroskedasticity in residuals

5. Normality Tests:
   - `jarque_bera.m` - Jarque-Bera test for normality based on skewness and kurtosis

## Integration with MFE Toolbox

These tests are designed to work seamlessly with other components of the MFE Toolbox:

- Time Series Models: Use `adf_test`, `pp_test`, and `kpss_test` to verify stationarity assumptions before applying ARMA models
- Volatility Models: Use `arch_test` to detect ARCH effects before fitting GARCH models
- Model Diagnostics: Use `ljungbox`, `jarque_bera`, and `white_test` to validate residuals from fitted models
- Bootstrap Methods: Tests can be combined with bootstrap techniques for improved finite-sample inference

## Usage Examples

Basic usage examples for each test are provided below. For more comprehensive examples, refer to the examples directory.

```matlab
% Unit Root Testing
returns = randn(1000, 1);  % Sample financial returns data
adf_results = adf_test(returns);
disp(['ADF test p-value: ', num2str(adf_results.pval)]);

% Testing for ARCH Effects
arch_results = arch_test(returns, 5);  % Test with 5 lags
disp(['ARCH test p-value: ', num2str(arch_results.pval)]);

% Model Residual Diagnostics
residuals = randn(1000, 1);  % Example model residuals
lb_results = ljungbox(residuals, 10);
disp(['Ljung-Box test p-values: ', num2str(lb_results.pval')]);
```

## Performance Considerations

- All tests are optimized for MATLAB's vectorized operations for efficient computation
- Tests support large datasets through efficient memory management
- Critical test components are implemented through MEX integration where appropriate
- Validation routines ensure robust handling of various input data characteristics

## References

1. Dickey, D.A., and Fuller, W.A. (1979). "Distribution of the estimators for autoregressive time series with a unit root." Journal of the American Statistical Association, 74, 427–431.
2. Phillips, P.C.B., and Perron, P. (1988). "Testing for a unit root in time series regression." Biometrika, 75, 335–346.
3. Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., and Shin, Y. (1992). "Testing the null hypothesis of stationarity against the alternative of a unit root." Journal of Econometrics, 54, 159–178.
4. Brock, W.A., Dechert, W.D., Scheinkman, J.A., and LeBaron, B. (1996). "A test for independence based on the correlation dimension." Econometric Reviews, 15, 197–235.
5. Engle, R.F. (1982). "Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation." Econometrica, 50, 987–1007.
6. Ljung, G.M., and Box, G.E.P. (1978). "On a measure of lack of fit in time series models." Biometrika, 65, 297–303.
7. Jarque, C.M., and Bera, A.K. (1987). "A test for normality of observations and regression residuals." International Statistical Review, 55, 163–172.
8. White, H. (1980). "A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity." Econometrica, 48, 817–838.