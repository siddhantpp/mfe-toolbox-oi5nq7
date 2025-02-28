# MFE Toolbox API Reference

Comprehensive API reference for the MATLAB Financial Econometrics (MFE) Toolbox Version 4.0, released on October 28, 2009. This document provides detailed information about all available functions, their parameters, return values, and usage examples.

## Table of Contents
- [Toolbox Overview](#toolbox-overview)
- [Distribution Functions](#distribution-functions)
- [Bootstrap Functions](#bootstrap-functions)
- [Time Series Analysis](#time-series-analysis)
- [Univariate Volatility Models](#univariate-volatility-models)
- [Multivariate Volatility Models](#multivariate-volatility-models)
- [Realized Volatility](#realized-volatility)
- [Statistical Tests](#statistical-tests)
- [Cross-sectional Analysis](#cross-sectional-analysis)
- [Utility Functions](#utility-functions)
- [GUI Components](#gui-components)

## Toolbox Overview

The MFE Toolbox provides a comprehensive suite of functions for financial econometrics. The functions are organized into several modules, including:

- Distribution Functions: Functions for working with statistical distributions.
- Bootstrap Functions: Functions for performing bootstrap analysis.
- Time Series Analysis: Functions for analyzing time series data.
- Univariate Volatility Models: Functions for modeling univariate volatility.
- Multivariate Volatility Models: Functions for modeling multivariate volatility.
- Realized Volatility: Functions for analyzing realized volatility.
- Statistical Tests: Functions for performing statistical tests.
- Cross-sectional Analysis: Functions for cross-sectional data analysis.
- Utility Functions: Helper functions.
- GUI Components: Graphical User Interface components.

## Distribution Functions

### gedpdf
Brief description of function purpose and capabilities
```matlab
y = gedpdf(x, nu)
```
#### Parameters
- `x` : numeric - Points at which to evaluate the PDF
- `nu` : numeric - Shape parameter (nu > 0)
#### Returns
- `y` : numeric - PDF values corresponding to each element of x
#### Example
```matlab
# Example usage code
```
#### See Also
- [gedcdf](#gedcdf) - Brief description of related function
- [gedinv](#gedinv) - Brief description of another related function

### gedcdf
Brief description of function purpose and capabilities
```matlab
p = gedcdf(x, nu)
```
#### Parameters
- `x` : numeric - Points at which to evaluate the CDF
- `nu` : numeric - Shape parameter (nu > 0)
#### Returns
- `p` : numeric - CDF values corresponding to each element of x
#### Example
```matlab
# Example usage code
```
#### See Also
- [gedpdf](#gedpdf) - Brief description of related function
- [gedinv](#gedinv) - Brief description of another related function

### gedinv
Brief description of function purpose and capabilities
```matlab
x = gedinv(p, nu)
```
#### Parameters
- `p` : numeric - Probabilities at which to evaluate the inverse CDF (0 <= p <= 1)
- `nu` : numeric - Shape parameter (nu > 0)
#### Returns
- `x` : numeric - Quantile values corresponding to each element of p
#### Example
```matlab
# Example usage code
```
#### See Also
- [gedpdf](#gedpdf) - Brief description of related function
- [gedcdf](#gedcdf) - Brief description of another related function

### gedrnd
Brief description of function purpose and capabilities
```matlab
r = gedrnd(nu, m, n)
```
#### Parameters
- `nu` : numeric - Shape parameter (nu > 0)
- `m` : numeric - Number of rows in output array (default: 1)
- `n` : numeric - Number of columns in output array (default: 1)
#### Returns
- `r` : numeric - Matrix of random samples from the GED
#### Example
```matlab
# Example usage code
```
#### See Also
- [gedpdf](#gedpdf) - Brief description of related function
- [gedcdf](#gedcdf) - Brief description of another related function

### gedloglik
Brief description of function purpose and capabilities
```matlab
[logL, logLi] = gedloglik(data, nu)
```
#### Parameters
- `x` : numeric - Data vector
- `nu` : numeric - Shape parameter (nu > 0)
#### Returns
- `logL` : numeric - Total log-likelihood
- `logLi` : numeric - Vector of individual observation log-likelihoods
#### Example
```matlab
# Example usage code
```
#### See Also
- [gedpdf](#gedpdf) - Brief description of related function
- [gedcdf](#gedcdf) - Brief description of another related function

### gedfit
Brief description of function purpose and capabilities
```matlab
[nuhat, muhat, sigmahat, loglikelihood, optim_details] = gedfit(x, options)
```
#### Parameters
- `x` : numeric - Data vector
- `options` : struct - (optional) Optimization options
#### Returns
- `nuhat` : numeric - Estimated shape parameter
- `muhat` : numeric - Estimated location parameter
- `sigmahat` : numeric - Estimated scale parameter
- `loglikelihood` : numeric - Log-likelihood at the estimated parameters
- `optim_details` : struct - Optimization details
#### Example
```matlab
# Example usage code
```
#### See Also
- [gedpdf](#gedpdf) - Brief description of related function
- [gedcdf](#gedcdf) - Brief description of another related function

### skewtpdf
Brief description of function purpose and capabilities
```matlab
y = skewtpdf(x, nu, lambda)
```
#### Parameters
- `x` : numeric - Points at which to evaluate the PDF
- `nu` : numeric - Degrees of freedom parameter (nu > 2)
- `lambda` : numeric - Skewness parameter (-1 < lambda < 1)
#### Returns
- `y` : numeric - PDF values corresponding to each element of x
#### Example
```matlab
# Example usage code
```
#### See Also
- [skewtcdf](#skewtcdf) - Brief description of related function
- [skewtinv](#skewtinv) - Brief description of another related function

### skewtcdf
Brief description of function purpose and capabilities
```matlab
p = skewtcdf(x, nu, lambda)
```
#### Parameters
- `x` : numeric - Points at which to evaluate the CDF
- `nu` : numeric - Degrees of freedom parameter (nu > 2)
- `lambda` : numeric - Skewness parameter (-1 < lambda < 1)
#### Returns
- `p` : numeric - CDF values corresponding to each element of x
#### Example
```matlab
# Example usage code
```
#### See Also
- [skewtpdf](#skewtpdf) - Brief description of related function
- [skewtinv](#skewtinv) - Brief description of another related function

### skewtinv
Brief description of function purpose and capabilities
```matlab
x = skewtinv(p, nu, lambda)
```
#### Parameters
- `p` : numeric - Probabilities at which to evaluate the inverse CDF (0 <= p <= 1)
- `nu` : numeric - Degrees of freedom parameter (nu > 2)
- `lambda` : numeric - Skewness parameter (-1 < lambda < 1)
#### Returns
- `x` : numeric - Quantile values corresponding to each element of p
#### Example
```matlab
# Example usage code
```
#### See Also
- [skewtpdf](#skewtpdf) - Brief description of related function
- [skewtcdf](#skewtcdf) - Brief description of another related function

### skewtrnd
Brief description of function purpose and capabilities
```matlab
r = skewtrnd(nu, lambda, m, n)
```
#### Parameters
- `nu` : numeric - Degrees of freedom parameter (nu > 2)
- `lambda` : numeric - Skewness parameter (-1 < lambda < 1)
- `m` : numeric - Number of rows in output array (default: 1)
- `n` : numeric - Number of columns in output array (default: 1)
#### Returns
- `r` : numeric - Matrix of random samples from the skewed t-distribution
#### Example
```matlab
# Example usage code
```
#### See Also
- [skewtpdf](#skewtpdf) - Brief description of related function
- [skewtcdf](#skewtcdf) - Brief description of another related function

### skewtloglik
Brief description of function purpose and capabilities
```matlab
[logL, logLi] = skewtloglik(x, nu, lambda)
```
#### Parameters
- `x` : numeric - Data vector
- `nu` : numeric - Degrees of freedom parameter (nu > 2)
- `lambda` : numeric - Skewness parameter (-1 < lambda < 1)
#### Returns
- `logL` : numeric - Total log-likelihood
- `logLi` : numeric - Vector of individual observation log-likelihoods
#### Example
```matlab
# Example usage code
```
#### See Also
- [skewtpdf](#skewtpdf) - Brief description of related function
- [skewtcdf](#skewtcdf) - Brief description of another related function

### skewtfit
Brief description of function purpose and capabilities
```matlab
[nuhat, lambdahat, muhat, sigmahat, loglikelihood, optim_details] = skewtfit(x, options)
```
#### Parameters
- `x` : numeric - Data vector
- `options` : struct - (optional) Optimization options
#### Returns
- `nuhat` : numeric - Estimated degrees of freedom
- `lambdahat` : numeric - Estimated skewness parameter
- `muhat` : numeric - Estimated location parameter
- `sigmahat` : numeric - Estimated scale parameter
- `loglikelihood` : numeric - Log-likelihood at the estimated parameters
- `optim_details` : struct - Optimization details
#### Example
```matlab
# Example usage code
```
#### See Also
- [skewtpdf](#skewtpdf) - Brief description of related function
- [skewtcdf](#skewtcdf) - Brief description of another related function

### stdtpdf
Brief description of function purpose and capabilities
```matlab
y = stdtpdf(x, nu)
```
#### Parameters
- `x` : numeric - Points at which to evaluate the PDF
- `nu` : numeric - Degrees of freedom parameter (nu > 2)
#### Returns
- `y` : numeric - PDF values corresponding to each element of x
#### Example
```matlab
# Example usage code
```
#### See Also
- [stdtcdf](#stdtcdf) - Brief description of related function
- [stdtinv](#stdtinv) - Brief description of another related function

### stdtcdf
Brief description of function purpose and capabilities
```matlab
p = stdtcdf(x, nu)
```
#### Parameters
- `x` : numeric - Points at which to evaluate the CDF
- `nu` : numeric - Degrees of freedom parameter (nu > 2)
#### Returns
- `p` : numeric - CDF values corresponding to each element of x
#### Example
```matlab
# Example usage code
```
#### See Also
- [stdtpdf](#stdtpdf) - Brief description of related function
- [stdtinv](#stdtinv) - Brief description of another related function

### stdtinv
Brief description of function purpose and capabilities
```matlab
x = stdtinv(p, nu)
```
#### Parameters
- `p` : numeric - Probabilities at which to evaluate the inverse CDF (0 <= p <= 1)
- `nu` : numeric - Degrees of freedom parameter (nu > 2)
#### Returns
- `x` : numeric - Quantile values corresponding to each element of p
#### Example
```matlab
# Example usage code
```
#### See Also
- [stdtpdf](#stdtpdf) - Brief description of related function
- [stdtcdf](#stdtcdf) - Brief description of another related function

### stdtrnd
Brief description of function purpose and capabilities
```matlab
r = stdtrnd(nu, m, n)
```
#### Parameters
- `nu` : numeric - Degrees of freedom parameter (nu > 2)
- `m` : numeric - Number of rows in output array (default: 1)
- `n` : numeric - Number of columns in output array (default: 1)
#### Returns
- `r` : numeric - Matrix of random samples from the standardized t-distribution
#### Example
```matlab
# Example usage code
```
#### See Also
- [stdtpdf](#stdtpdf) - Brief description of related function
- [stdtcdf](#stdtcdf) - Brief description of another related function

### stdtloglik
Brief description of function purpose and capabilities
```matlab
[logL, logLi] = stdtloglik(x, nu)
```
#### Parameters
- `x` : numeric - Data vector
- `nu` : numeric - Degrees of freedom parameter (nu > 2)
#### Returns
- `logL` : numeric - Total log-likelihood
- `logLi` : numeric - Vector of individual observation log-likelihoods
#### Example
```matlab
# Example usage code
```
#### See Also
- [stdtpdf](#stdtpdf) - Brief description of related function
- [stdtcdf](#stdtcdf) - Brief description of another related function

### stdtfit
Brief description of function purpose and capabilities
```matlab
[nuhat, muhat, sigmahat, loglikelihood, optim_details] = stdtfit(x, options)
```
#### Parameters
- `x` : numeric - Data vector
- `options` : struct - (optional) Optimization options
#### Returns
- `nuhat` : numeric - Estimated degrees of freedom
- `muhat` : numeric - Estimated location parameter
- `sigmahat` : numeric - Estimated scale parameter
- `loglikelihood` : numeric - Log-likelihood at the estimated parameters
- `optim_details` : struct - Optimization details
#### Example
```matlab
# Example usage code
```
#### See Also
- [stdtpdf](#stdtpdf) - Brief description of related function
- [stdtcdf](#stdtcdf) - Brief description of another related function

## Bootstrap Functions

### block_bootstrap
Brief description of function purpose and capabilities
```matlab
output = block_bootstrap(data, block_size, num_replications)
```
#### Parameters
- `data` : type - Description of first parameter
- `block_size` : type - Description of second parameter
- `num_replications` : type - Description of third parameter
#### Returns
- `output` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [stationary_bootstrap](#stationary_bootstrap) - Brief description of related function
- [bootstrap_variance](#bootstrap_variance) - Brief description of another related function

### stationary_bootstrap
Brief description of function purpose and capabilities
```matlab
output = stationary_bootstrap(data, avg_block_size, num_replications)
```
#### Parameters
- `data` : type - Description of first parameter
- `avg_block_size` : type - Description of second parameter
- `num_replications` : type - Description of third parameter
#### Returns
- `output` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [block_bootstrap](#block_bootstrap) - Brief description of related function
- [bootstrap_variance](#bootstrap_variance) - Brief description of another related function

### bootstrap_variance
Brief description of function purpose and capabilities
```matlab
output = bootstrap_variance(data, num_replications)
```
#### Parameters
- `data` : type - Description of first parameter
- `num_replications` : type - Description of second parameter
#### Returns
- `output` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [block_bootstrap](#block_bootstrap) - Brief description of related function
- [stationary_bootstrap](#stationary_bootstrap) - Brief description of another related function

### bootstrap_confidence_intervals
Brief description of function purpose and capabilities
```matlab
output = bootstrap_confidence_intervals(data, alpha)
```
#### Parameters
- `data` : type - Description of first parameter
- `alpha` : type - Description of second parameter
#### Returns
- `output` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [block_bootstrap](#block_bootstrap) - Brief description of related function
- [stationary_bootstrap](#stationary_bootstrap) - Brief description of another related function

## Time Series Analysis

### aicsbic
Brief description of function purpose and capabilities
```matlab
ic = aicsbic(logL, k, T)
```
#### Parameters
- `logL` : type - Description of first parameter
- `k` : type - Description of second parameter
- `T` : type - Description of third parameter
#### Returns
- `ic` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [sacf](#sacf) - Brief description of related function
- [spacf](#spacf) - Brief description of another related function

### sacf
Brief description of function purpose and capabilities
```matlab
[acf, se, ci] = sacf(data, lags, options)
```
#### Parameters
- `data` : type - Description of first parameter
- `lags` : type - Description of second parameter
- `options` : type - Description of third parameter
#### Returns
- `acf` : type - Description of first return value
- `se` : type - Description of second return value
- `ci` : type - Description of third return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [aicsbic](#aicsbic) - Brief description of related function
- [spacf](#spacf) - Brief description of another related function

### spacf
Brief description of function purpose and capabilities
```matlab
[pacf, se, ci] = spacf(data, lags, options)
```
#### Parameters
- `data` : type - Description of first parameter
- `lags` : type - Description of second parameter
- `options` : type - Description of third parameter
#### Returns
- `pacf` : type - Description of first return value
- `se` : type - Description of second return value
- `ci` : type - Description of third return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [aicsbic](#aicsbic) - Brief description of related function
- [sacf](#sacf) - Brief description of another related function

### armaxfilter
Brief description of function purpose and capabilities
```matlab
results = armaxfilter(data, p, q, exogenous, options)
```
#### Parameters
- `data` : type - Description of first parameter
- `p` : type - Description of second parameter
- `q` : type - Description of third parameter
- `exogenous` : type - Description of fourth parameter
- `options` : type - Description of fifth parameter
#### Returns
- `results` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [armafor](#armafor) - Brief description of related function
- [sarima](#sarima) - Brief description of another related function

### armafor
Brief description of function purpose and capabilities
```matlab
[forecasts, variances, paths] = armafor(parameters, data, exogenous, horizon, options)
```
#### Parameters
- `parameters` : type - Description of first parameter
- `data` : type - Description of second parameter
- `exogenous` : type - Description of third parameter
- `horizon` : type - Description of fourth parameter
- `options` : type - Description of fifth parameter
#### Returns
- `forecasts` : type - Description of first return value
- `variances` : type - Description of second return value
- `paths` : type - Description of third return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [armaxfilter](#armaxfilter) - Brief description of related function
- [sarima](#sarima) - Brief description of another related function

### sarima
Brief description of function purpose and capabilities
```matlab
results = sarima(data, p, d, q, P, D, Q, s, options)
```
#### Parameters
- `data` : type - Description of first parameter
- `p` : type - Description of second parameter
- `d` : type - Description of third parameter
- `q` : type - Description of fourth parameter
- `P` : type - Description of fifth parameter
- `D` : type - Description of sixth parameter
- `Q` : type - Description of seventh parameter
- `s` : type - Description of eighth parameter
- `options` : type - Description of ninth parameter
#### Returns
- `results` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [armaxfilter](#armaxfilter) - Brief description of related function
- [armafor](#armafor) - Brief description of another related function

## Univariate Volatility Models

### garchcore
Brief description of function purpose and capabilities
```matlab
ht = garchcore(parameters, data, options)
```
#### Parameters
- `parameters` : type - Description of first parameter
- `data` : type - Description of second parameter
- `options` : type - Description of third parameter
#### Returns
- `ht` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [agarchfit](#agarchfit) - Brief description of related function
- [egarchfit](#egarchfit) - Brief description of another related function

### garchinit
Brief description of function purpose and capabilities
```matlab
parameters = garchinit(data, options)
```
#### Parameters
- `data` : type - Description of first parameter
- `options` : type - Description of second parameter
#### Returns
- `parameters` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [agarchfit](#agarchfit) - Brief description of related function
- [egarchfit](#egarchfit) - Brief description of another related function

### garchlikelihood
Brief description of function purpose and capabilities
```matlab
negLogLik = garchlikelihood(parameters, data, options)
```
#### Parameters
- `parameters` : type - Description of first parameter
- `data` : type - Description of second parameter
- `options` : type - Description of third parameter
#### Returns
- `negLogLik` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [agarchfit](#agarchfit) - Brief description of related function
- [egarchfit](#egarchfit) - Brief description of another related function

### agarchfit
Brief description of function purpose and capabilities
```matlab
model = agarchfit(data, options)
```
#### Parameters
- `data` : type - Description of first parameter
- `options` : type - Description of second parameter
#### Returns
- `model` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [garchcore](#garchcore) - Brief description of related function
- [garchinit](#garchinit) - Brief description of another related function

### egarchfit
Brief description of function purpose and capabilities
```matlab
model = egarchfit(data, options)
```
#### Parameters
- `data` : type - Description of first parameter
- `options` : type - Description of second parameter
#### Returns
- `model` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [garchcore](#garchcore) - Brief description of related function
- [garchinit](#garchinit) - Brief description of another related function

### igarchfit
Brief description of function purpose and capabilities
```matlab
model = igarchfit(data, options)
```
#### Parameters
- `data` : type - Description of first parameter
- `options` : type - Description of second parameter
#### Returns
- `model` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [garchcore](#garchcore) - Brief description of related function
- [garchinit](#garchinit) - Brief description of another related function

### tarchfit
Brief description of function purpose and capabilities
```matlab
model = tarchfit(data, options)
```
#### Parameters
- `data` : type - Description of first parameter
- `options` : type - Description of second parameter
#### Returns
- `model` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [garchcore](#garchcore) - Brief description of related function
- [garchinit](#garchinit) - Brief description of another related function

### nagarchfit
Brief description of function purpose and capabilities
```matlab
model = nagarchfit(data, options)
```
#### Parameters
- `data` : type - Description of first parameter
- `options` : type - Description of second parameter
#### Returns
- `model` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [garchcore](#garchcore) - Brief description of related function
- [garchinit](#garchinit) - Brief description of another related function

### garchfor
Brief description of function purpose and capabilities
```matlab
[forecasts, errors] = garchfor(parameters, data, horizon, model_type, options)
```
#### Parameters
- `parameters` : type - Description of first parameter
- `data` : type - Description of second parameter
- `horizon` : type - Description of third parameter
- `model_type` : type - Description of fourth parameter
- `options` : type - Description of fifth parameter
#### Returns
- `forecasts` : type - Description of first return value
- `errors` : type - Description of second return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [agarchfit](#agarchfit) - Brief description of related function
- [egarchfit](#egarchfit) - Brief description of another related function

## Multivariate Volatility Models

### ccc_mvgarch
Brief description of function purpose and capabilities
```matlab
model = ccc_mvgarch(data, options)
```
#### Parameters
- `data` : type - Description of first parameter
- `options` : type - Description of second parameter
#### Returns
- `model` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [dcc_mvgarch](#dcc_mvgarch) - Brief description of related function
- [bekk_mvgarch](#bekk_mvgarch) - Brief description of another related function

### dcc_mvgarch
Brief description of function purpose and capabilities
```matlab
model = dcc_mvgarch(data, options)
```
#### Parameters
- `data` : type - Description of first parameter
- `options` : type - Description of second parameter
#### Returns
- `model` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [ccc_mvgarch](#ccc_mvgarch) - Brief description of related function
- [bekk_mvgarch](#bekk_mvgarch) - Brief description of another related function

### bekk_mvgarch
Brief description of function purpose and capabilities
```matlab
model = bekk_mvgarch(data, p, q, options)
```
#### Parameters
- `data` : type - Description of first parameter
- `p` : type - Description of second parameter
- `q` : type - Description of third parameter
- `options` : type - Description of fourth parameter
#### Returns
- `model` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [ccc_mvgarch](#ccc_mvgarch) - Brief description of related function
- [dcc_mvgarch](#dcc_mvgarch) - Brief description of another related function

### gogarch
Brief description of function purpose and capabilities
```matlab
model = gogarch(data, options)
```
#### Parameters
- `data` : type - Description of first parameter
- `options` : type - Description of second parameter
#### Returns
- `model` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [ccc_mvgarch](#ccc_mvgarch) - Brief description of related function
- [dcc_mvgarch](#dcc_mvgarch) - Brief description of another related function

## Realized Volatility

### rv_compute
Brief description of function purpose and capabilities
```matlab
RV = rv_compute(data, sampling_frequency)
```
#### Parameters
- `data` : type - Description of first parameter
- `sampling_frequency` : type - Description of second parameter
#### Returns
- `RV` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [bv_compute](#bv_compute) - Brief description of related function
- [rv_kernel](#rv_kernel) - Brief description of another related function

### bv_compute
Brief description of function purpose and capabilities
```matlab
BV = bv_compute(data, sampling_frequency)
```
#### Parameters
- `data` : type - Description of first parameter
- `sampling_frequency` : type - Description of second parameter
#### Returns
- `BV` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [rv_compute](#rv_compute) - Brief description of related function
- [rv_kernel](#rv_kernel) - Brief description of another related function

### rv_kernel
Brief description of function purpose and capabilities
```matlab
RV_Kernel = rv_kernel(data, sampling_frequency, kernel_type)
```
#### Parameters
- `data` : type - Description of first parameter
- `sampling_frequency` : type - Description of second parameter
- `kernel_type` : type - Description of third parameter
#### Returns
- `RV_Kernel` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [rv_compute](#rv_compute) - Brief description of related function
- [bv_compute](#bv_compute) - Brief description of another related function

### realized_spectrum
Brief description of function purpose and capabilities
```matlab
spectrum = realized_spectrum(data, sampling_frequency)
```
#### Parameters
- `data` : type - Description of first parameter
- `sampling_frequency` : type - Description of second parameter
#### Returns
- `spectrum` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [rv_compute](#rv_compute) - Brief description of related function
- [bv_compute](#bv_compute) - Brief description of another related function

### jump_test
Brief description of function purpose and capabilities
```matlab
jumps = jump_test(data, sampling_frequency)
```
#### Parameters
- `data` : type - Description of first parameter
- `sampling_frequency` : type - Description of second parameter
#### Returns
- `jumps` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [rv_compute](#rv_compute) - Brief description of related function
- [bv_compute](#bv_compute) - Brief description of another related function

## Statistical Tests

### adf_test
Brief description of function purpose and capabilities
```matlab
results = adf_test(data, lags, model)
```
#### Parameters
- `data` : type - Description of first parameter
- `lags` : type - Description of second parameter
- `model` : type - Description of third parameter
#### Returns
- `results` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [pp_test](#pp_test) - Brief description of related function
- [kpss_test](#kpss_test) - Brief description of another related function

### pp_test
Brief description of function purpose and capabilities
```matlab
results = pp_test(data, lags, model)
```
#### Parameters
- `data` : type - Description of first parameter
- `lags` : type - Description of second parameter
- `model` : type - Description of third parameter
#### Returns
- `results` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [adf_test](#adf_test) - Brief description of related function
- [kpss_test](#kpss_test) - Brief description of another related function

### kpss_test
Brief description of function purpose and capabilities
```matlab
results = kpss_test(data, lags, model)
```
#### Parameters
- `data` : type - Description of first parameter
- `lags` : type - Description of second parameter
- `model` : type - Description of third parameter
#### Returns
- `results` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [adf_test](#adf_test) - Brief description of related function
- [pp_test](#pp_test) - Brief description of another related function

### bds_test
Brief description of function purpose and capabilities
```matlab
results = bds_test(data, m, epsilon)
```
#### Parameters
- `data` : type - Description of first parameter
- `m` : type - Description of second parameter
- `epsilon` : type - Description of third parameter
#### Returns
- `results` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [arch_test](#arch_test) - Brief description of related function
- [ljungbox](#ljungbox) - Brief description of another related function

### arch_test
Brief description of function purpose and capabilities
```matlab
results = arch_test(data, lags)
```
#### Parameters
- `data` : type - Description of first parameter
- `lags` : type - Description of second parameter
#### Returns
- `results` : type - Description of first return value
#### Example
```matlab
# Example usage code
```
#### See Also
- [bds_test](#bds_test) - Brief description of related function
- [ljungbox](#ljungbox) - Brief description of another related function

### ljungbox
Brief description of function purpose and capabilities
```matlab
results = ljungbox(data, lags, dofsAdjust)