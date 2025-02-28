# Distribution Functions

The MFE Toolbox provides robust implementations of statistical distributions widely used in financial econometrics, focusing on modeling non-normal characteristics of financial returns such as fat tails and asymmetry. This document details the available distribution functions, their parameters, implementation details, and usage examples.

## Table of Contents
- [Overview of Distribution Functions](#overview-of-distribution-functions)
- [Generalized Error Distribution (GED) Functions](#generalized-error-distribution-ged-functions)
- [Hansen's Skewed T-Distribution Functions](#hansens-skewed-t-distribution-functions)
- [Standardized Student's T-Distribution Functions](#standardized-students-t-distribution-functions)
- [Usage Examples](#usage-examples)
- [Integration with Volatility Models](#integration-with-volatility-models)
- [Technical Notes](#technical-notes)
- [See Also](#see-also)

## Overview of Distribution Functions

The distribution module implements three key distributions particularly relevant for financial econometrics:

1. **Generalized Error Distribution (GED)** - Flexible distribution for modeling various kurtosis levels
2. **Hansen's Skewed T-distribution** - Extension of Student's t-distribution that incorporates skewness
3. **Standardized Student's T-distribution** - Variant of t-distribution normalized to have unit variance

Each distribution is implemented with a consistent set of six function types:
- PDF: Probability density function calculation
- CDF: Cumulative distribution function computation
- INV: Inverse CDF/quantile function
- RND: Random number generation
- LOGLIK: Log-likelihood calculation
- FIT: Maximum likelihood parameter estimation

All functions feature comprehensive error checking, robust numerical stability, and optimized performance for both scalar and vectorized operations.

## Generalized Error Distribution (GED) Functions

The GED extends the normal distribution with an additional shape parameter that controls the tail thickness, allowing for both leptokurtic (fat-tailed) and platykurtic (thin-tailed) distributions.

### gedpdf - GED Probability Density Function

Computes the probability density function of the GED for given values.

**Syntax**
```matlab
y = gedpdf(x, nu)
```

**Parameters**
- `x`: numeric - Points at which to evaluate the PDF
- `nu`: numeric - Shape parameter (nu > 0)

**Returns**
- `y`: numeric - PDF values corresponding to each element of x

**Example**
```matlab
x = -4:0.1:4;
nu = 1.5;  % Shape parameter
y = gedpdf(x, nu);
plot(x, y);
title('GED PDF with nu = 1.5');
```

**Mathematical Details**
The PDF of the GED with shape parameter ν is given by:

$$f(x; \nu) = \frac{\nu}{2\lambda\Gamma(1/\nu)}\exp\left(-\left(\frac{|x|}{\lambda}\right)^\nu\right)$$

where $\lambda = \sqrt{\Gamma(1/\nu)/\Gamma(3/\nu)}$ is a scale parameter ensuring unit variance.

### gedcdf - GED Cumulative Distribution Function

Computes the cumulative distribution function of the GED for given values.

**Syntax**
```matlab
p = gedcdf(x, nu)
```

**Parameters**
- `x`: numeric - Points at which to evaluate the CDF
- `nu`: numeric - Shape parameter (nu > 0)

**Returns**
- `p`: numeric - CDF values corresponding to each element of x

### gedinv - GED Inverse Cumulative Distribution Function

Computes the inverse CDF (quantile function) of the GED for given probabilities.

**Syntax**
```matlab
x = gedinv(p, nu)
```

**Parameters**
- `p`: numeric - Probabilities at which to evaluate the inverse CDF (0 ≤ p ≤ 1)
- `nu`: numeric - Shape parameter (nu > 0)

**Returns**
- `x`: numeric - Quantile values corresponding to each element of p

### gedrnd - GED Random Number Generation

Generates random samples from the GED with specified shape parameter.

**Syntax**
```matlab
r = gedrnd(nu, m, n)
```

**Parameters**
- `nu`: numeric - Shape parameter (nu > 0)
- `m`: numeric - Number of rows in output array (default: 1)
- `n`: numeric - Number of columns in output array (default: 1)

**Returns**
- `r`: numeric - Matrix of random samples from the GED

### gedloglik - GED Log-Likelihood Function

Computes the log-likelihood of data under the GED distribution.

**Syntax**
```matlab
[logL, logLi] = gedloglik(data, nu)
```

**Parameters**
- `x`: numeric - Data vector
- `nu`: numeric - Shape parameter (nu > 0)

**Returns**
- `logL`: numeric - Total log-likelihood
- `logLi`: numeric - Vector of individual observation log-likelihoods

### gedfit - GED Parameter Estimation

Estimates the parameters of the GED distribution from data using maximum likelihood.

**Syntax**
```matlab
[nuhat, muhat, sigmahat, loglikelihood, optim_details] = gedfit(x, options)
```

**Parameters**
- `x`: numeric - Data vector
- `options`: struct - (optional) Optimization options

**Returns**
- `nuhat`: numeric - Estimated shape parameter
- `muhat`: numeric - Estimated location parameter
- `sigmahat`: numeric - Estimated scale parameter
- `loglikelihood`: numeric - Log-likelihood at the estimated parameters
- `optim_details`: struct - Optimization details

## Hansen's Skewed T-Distribution Functions

Hansen's skewed t-distribution extends the Student's t-distribution by incorporating a skewness parameter, making it particularly useful for modeling asymmetric financial returns.

### skewtpdf - Skewed T Probability Density Function

Computes the probability density function of Hansen's skewed t-distribution for given values.

**Syntax**
```matlab
y = skewtpdf(x, nu, lambda)
```

**Parameters**
- `x`: numeric - Points at which to evaluate the PDF
- `nu`: numeric - Degrees of freedom parameter (nu > 2)
- `lambda`: numeric - Skewness parameter (-1 < lambda < 1)

**Returns**
- `y`: numeric - PDF values corresponding to each element of x

**Mathematical Details**
Hansen's skewed t-distribution has the PDF:

$$f(x; \nu, \lambda) = \begin{cases} bc\left(1 + \frac{1}{\nu-2}\left(\frac{bx+a}{1-\lambda}\right)^2\right)^{-\frac{\nu+1}{2}} & \text{if } x < -a/b \\ bc\left(1 + \frac{1}{\nu-2}\left(\frac{bx+a}{1+\lambda}\right)^2\right)^{-\frac{\nu+1}{2}} & \text{if } x \geq -a/b \end{cases}$$

where constants a, b, and c are defined to ensure the distribution has the desired properties.

### skewtcdf - Skewed T Cumulative Distribution Function

Computes the cumulative distribution function of Hansen's skewed t-distribution.

**Syntax**
```matlab
p = skewtcdf(x, nu, lambda)
```

**Parameters**
- `x`: numeric - Points at which to evaluate the CDF
- `nu`: numeric - Degrees of freedom parameter (nu > 2)
- `lambda`: numeric - Skewness parameter (-1 < lambda < 1)

**Returns**
- `p`: numeric - CDF values corresponding to each element of x

### skewtinv - Skewed T Inverse Cumulative Distribution Function

Computes the inverse CDF (quantile function) of Hansen's skewed t-distribution.

**Syntax**
```matlab
x = skewtinv(p, nu, lambda)
```

**Parameters**
- `p`: numeric - Probabilities at which to evaluate the inverse CDF (0 ≤ p ≤ 1)
- `nu`: numeric - Degrees of freedom parameter (nu > 2)
- `lambda`: numeric - Skewness parameter (-1 < lambda < 1)

**Returns**
- `x`: numeric - Quantile values corresponding to each element of p

### skewtrnd - Skewed T Random Number Generation

Generates random samples from Hansen's skewed t-distribution.

**Syntax**
```matlab
r = skewtrnd(nu, lambda, m, n)
```

**Parameters**
- `nu`: numeric - Degrees of freedom parameter (nu > 2)
- `lambda`: numeric - Skewness parameter (-1 < lambda < 1)
- `m`: numeric - Number of rows in output array (default: 1)
- `n`: numeric - Number of columns in output array (default: 1)

**Returns**
- `r`: numeric - Matrix of random samples from the skewed t-distribution

### skewtloglik - Skewed T Log-Likelihood Function

Computes the log-likelihood of data under Hansen's skewed t-distribution.

**Syntax**
```matlab
[logL, logLi] = skewtloglik(x, nu, lambda)
```

**Parameters**
- `x`: numeric - Data vector
- `nu`: numeric - Degrees of freedom parameter (nu > 2)
- `lambda`: numeric - Skewness parameter (-1 < lambda < 1)

**Returns**
- `logL`: numeric - Total log-likelihood
- `logLi`: numeric - Vector of individual observation log-likelihoods

### skewtfit - Skewed T Parameter Estimation

Estimates the parameters of Hansen's skewed t-distribution from data using maximum likelihood.

**Syntax**
```matlab
[nuhat, lambdahat, muhat, sigmahat, loglikelihood, optim_details] = skewtfit(x, options)
```

**Parameters**
- `x`: numeric - Data vector
- `options`: struct - (optional) Optimization options

**Returns**
- `nuhat`: numeric - Estimated degrees of freedom
- `lambdahat`: numeric - Estimated skewness parameter
- `muhat`: numeric - Estimated location parameter
- `sigmahat`: numeric - Estimated scale parameter
- `loglikelihood`: numeric - Log-likelihood at the estimated parameters
- `optim_details`: struct - Optimization details

## Standardized Student's T-Distribution Functions

The standardized Student's t-distribution implementation provides a version of the t-distribution normalized to have unit variance regardless of degrees of freedom, which is particularly useful in volatility models.

### stdtpdf - Standardized T Probability Density Function

Computes the probability density function of the standardized Student's t-distribution.

**Syntax**
```matlab
y = stdtpdf(x, nu)
```

**Parameters**
- `x`: numeric - Points at which to evaluate the PDF
- `nu`: numeric - Degrees of freedom parameter (nu > 2)

**Returns**
- `y`: numeric - PDF values corresponding to each element of x

**Mathematical Details**
The PDF of the standardized Student's t-distribution with ν degrees of freedom is:

$$f(x; \nu) = \frac{\Gamma((\nu+1)/2)}{\Gamma(\nu/2)\sqrt{\pi(\nu-2)}}\left(1 + \frac{x^2}{\nu-2}\right)^{-(\nu+1)/2}$$

This distribution has zero mean and unit variance for all valid ν.

### stdtcdf - Standardized T Cumulative Distribution Function

Computes the cumulative distribution function of the standardized Student's t-distribution.

**Syntax**
```matlab
p = stdtcdf(x, nu)
```

**Parameters**
- `x`: numeric - Points at which to evaluate the CDF
- `nu`: numeric - Degrees of freedom parameter (nu > 2)

**Returns**
- `p`: numeric - CDF values corresponding to each element of x

### stdtinv - Standardized T Inverse Cumulative Distribution Function

Computes the inverse CDF (quantile function) of the standardized Student's t-distribution.

**Syntax**
```matlab
x = stdtinv(p, nu)
```

**Parameters**
- `p`: numeric - Probabilities at which to evaluate the inverse CDF (0 ≤ p ≤ 1)
- `nu`: numeric - Degrees of freedom parameter (nu > 2)

**Returns**
- `x`: numeric - Quantile values corresponding to each element of p

### stdtrnd - Standardized T Random Number Generation

Generates random samples from the standardized Student's t-distribution.

**Syntax**
```matlab
r = stdtrnd(nu, m, n)
```

**Parameters**
- `nu`: numeric - Degrees of freedom parameter (nu > 2)
- `m`: numeric - Number of rows in output array (default: 1)
- `n`: numeric - Number of columns in output array (default: 1)

**Returns**
- `r`: numeric - Matrix of random samples from the standardized t-distribution

### stdtloglik - Standardized T Log-Likelihood Function

Computes the log-likelihood of data under the standardized Student's t-distribution.

**Syntax**
```matlab
[logL, logLi] = stdtloglik(x, nu)
```

**Parameters**
- `x`: numeric - Data vector
- `nu`: numeric - Degrees of freedom parameter (nu > 2)

**Returns**
- `logL`: numeric - Total log-likelihood
- `logLi`: numeric - Vector of individual observation log-likelihoods

### stdtfit - Standardized T Parameter Estimation

Estimates the parameters of the standardized Student's t-distribution from data using maximum likelihood.

**Syntax**
```matlab
[nuhat, muhat, sigmahat, loglikelihood, optim_details] = stdtfit(x, options)
```

**Parameters**
- `x`: numeric - Data vector
- `options`: struct - (optional) Optimization options

**Returns**
- `nuhat`: numeric - Estimated degrees of freedom
- `muhat`: numeric - Estimated location parameter
- `sigmahat`: numeric - Estimated scale parameter
- `loglikelihood`: numeric - Log-likelihood at the estimated parameters
- `optim_details`: struct - Optimization details

## Usage Examples

The following examples demonstrate how to use the distribution functions for common financial econometrics tasks.

### Fitting Distributions to Financial Returns

This example shows how to fit different distributions to financial return data and compare them.

```matlab
% Load sample financial returns data
load example_financial_data

% Fit distributions to the data
[nu_ged, mu_ged, sigma_ged] = gedfit(returns);
disp(['GED fit: nu = ' num2str(nu_ged) ', mu = ' num2str(mu_ged) ', sigma = ' num2str(sigma_ged)]);

[nu_skewt, lambda_skewt, mu_skewt, sigma_skewt] = skewtfit(returns);
disp(['Skewed t fit: nu = ' num2str(nu_skewt) ', lambda = ' num2str(lambda_skewt), ...
      ', mu = ' num2str(mu_skewt) ', sigma = ' num2str(sigma_skewt)]);

[nu_stdt, mu_stdt, sigma_stdt] = stdtfit(returns);
disp(['Standardized t fit: nu = ' num2str(nu_stdt) ', mu = ' num2str(mu_stdt), ...
      ', sigma = ' num2str(sigma_stdt)]);

% Plot fitted distributions against histogram
figure;
histfit(returns, 50);
hold on;

x = linspace(min(returns), max(returns), 100);
y_ged = gedpdf((x - mu_ged) / sigma_ged, nu_ged) / sigma_ged;
y_skewt = skewtpdf((x - mu_skewt) / sigma_skewt, nu_skewt, lambda_skewt) / sigma_skewt;
y_stdt = stdtpdf((x - mu_stdt) / sigma_stdt, nu_stdt) / sigma_stdt;

plot(x, y_ged * length(returns) * (max(x) - min(x)) / 50, 'r', 'LineWidth', 2);
plot(x, y_skewt * length(returns) * (max(x) - min(x)) / 50, 'g', 'LineWidth', 2);
plot(x, y_stdt * length(returns) * (max(x) - min(x)) / 50, 'b', 'LineWidth', 2);

legend('Histogram', 'Normal', 'GED', 'Skewed t', 'Standardized t');
title('Fitted Distributions to Financial Returns');
```

### Computing Value-at-Risk (VaR)

This example demonstrates how to compute Value-at-Risk using the inverse CDF functions.

```matlab
% Load financial returns data
load example_financial_data

% Fit standardized t-distribution
[nu, mu, sigma] = stdtfit(returns);

% Compute 1% and 5% VaR
alpha = [0.01, 0.05];
var_levels = mu + sigma * stdtinv(alpha, nu);

disp(['1% VaR: ' num2str(-var_levels(1))]);
disp(['5% VaR: ' num2str(-var_levels(2))]);

% Compare with empirical VaR
empirical_var = -quantile(returns, alpha);
disp(['Empirical 1% VaR: ' num2str(empirical_var(1))]);
disp(['Empirical 5% VaR: ' num2str(empirical_var(2))]);
```

### Monte Carlo Simulation

This example shows how to perform Monte Carlo simulation using the random number generation functions.

```matlab
% Generate 10,000 samples from each distribution
n_samples = 10000;

% GED samples with nu = 1.5 (slightly fat-tailed)
ged_samples = gedrnd(1.5, n_samples, 1);

% Skewed t samples with nu = 5 and lambda = 0.3 (moderate skew)
skewt_samples = skewtrnd(5, 0.3, n_samples, 1);

% Standardized t samples with nu = 4 (moderately fat-tailed)
stdt_samples = stdtrnd(4, n_samples, 1);

% Plot histograms
figure;
subplot(3,1,1);
histogram(ged_samples, 50);
title('GED Samples (\nu = 1.5)');

subplot(3,1,2);
histogram(skewt_samples, 50);
title('Skewed t Samples (\nu = 5, \lambda = 0.3)');

subplot(3,1,3);
histogram(stdt_samples, 50);
title('Standardized t Samples (\nu = 4)');

% Compute sample statistics
disp('Sample Statistics:');
disp(['GED: Mean = ' num2str(mean(ged_samples)) ', Std = ' num2str(std(ged_samples)), ...
      ', Skewness = ' num2str(skewness(ged_samples)) ', Kurtosis = ' num2str(kurtosis(ged_samples))]);
disp(['Skewed t: Mean = ' num2str(mean(skewt_samples)) ', Std = ' num2str(std(skewt_samples)), ...
      ', Skewness = ' num2str(skewness(skewt_samples)) ', Kurtosis = ' num2str(kurtosis(skewt_samples))]);
disp(['Std t: Mean = ' num2str(mean(stdt_samples)) ', Std = ' num2str(std(stdt_samples)), ...
      ', Skewness = ' num2str(skewness(stdt_samples)) ', Kurtosis = ' num2str(kurtosis(stdt_samples))]);
```

## Integration with Volatility Models

The distribution functions are designed to integrate seamlessly with volatility models in the MFE Toolbox. This section demonstrates how to specify custom error distributions in GARCH models.

```matlab
% Load return data
load example_financial_data

% Estimate GARCH(1,1) model with different error distributions

% With normal errors (default)
[parameters, ~, ~, ~, ~, ~, ~] = tarchfit(returns);
disp('GARCH with normal errors:');
disp(parameters);

% With standardized t errors
[parameters_t, ~, ~, ~, ~, ~, ~] = tarchfit(returns, [], 'STUDENTST');
disp('GARCH with Student''s t errors:');
disp(parameters_t);

% With GED errors
[parameters_ged, ~, ~, ~, ~, ~, ~] = tarchfit(returns, [], 'GED');
disp('GARCH with GED errors:');
disp(parameters_ged);

% With skewed t errors
[parameters_skewt, ~, ~, ~, ~, ~, ~] = tarchfit(returns, [], 'SKEWT');
disp('GARCH with skewed t errors:');
disp(parameters_skewt);
```

The volatility models in the MFE Toolbox (`tarchfit`, `egarchfit`, `agarchfit`, etc.) accept the following distribution options:
- 'NORMAL': Standard normal distribution (default)
- 'STUDENTST': Standardized Student's t-distribution
- 'GED': Generalized Error Distribution
- 'SKEWT': Hansen's skewed t-distribution

## Technical Notes

### Implementation Details

All distribution functions in the MFE Toolbox share the following implementation characteristics:

1. **Input Validation**: Thorough parameter validation ensures numerical stability:
   - GED: Shape parameter nu > 0
   - Skewed t: Degrees of freedom nu > 2, skewness parameter -1 < lambda < 1
   - Standardized t: Degrees of freedom nu > 2

2. **Vectorized Operations**: All functions support vectorized inputs for efficient computation with large datasets.

3. **Numerical Stability**: Special care is taken to ensure numerical stability, particularly for extreme parameter values and in distribution tails.

4. **Performance Optimization**: Critical computations are optimized for performance, with careful attention to algorithm selection.

### Precision Considerations

When working with extreme values or parameter settings, consider the following:

1. For very small probabilities (< 1e-10) or very large values, numerical precision may be affected.

2. For the GED with very small shape parameter (nu < 0.1), numerical instabilities may occur in the tails.

3. The skewed t-distribution implementation may face challenges with extreme skewness (lambda → ±1) combined with small degrees of freedom.

4. For fitting functions, using appropriate starting values can significantly improve convergence, especially for complex distributions like the skewed t-distribution.

### MEX Acceleration

Some distribution functions may leverage MEX acceleration for performance-critical operations. The implementation automatically selects the most efficient computation method based on the available platform and compilation options.

## See Also
- [Volatility Models](volatility_models.md) - Documentation for volatility models that use these distributions
- [Examples](../examples/distribution_analysis.m) - Comprehensive examples of distribution analysis