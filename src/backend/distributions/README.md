# Statistical Distributions Module

The distributions module implements robust, high-performance statistical distributions commonly used in financial econometrics. This module provides comprehensive support for the Generalized Error Distribution (GED), Hansen's skewed T-distribution, and the standardized Student's T-distribution, each with consistent function sets for probability calculations, parameter estimation, and random number generation.

## Module Overview

This module is a core component of the MFE Toolbox providing statistical distributions particularly relevant for modeling financial returns that often exhibit non-normal characteristics such as fat tails and asymmetry. Each distribution is implemented with a consistent set of six function types:

- **PDF**: Probability density function calculation
- **CDF**: Cumulative distribution function computation
- **INV**: Inverse CDF/quantile function
- **RND**: Random number generation
- **LOGLIK**: Log-likelihood calculation
- **FIT**: Maximum likelihood parameter estimation

All implementations feature comprehensive error checking, robust numerical stability, and optimized performance for both scalar and vectorized operations.

## Generalized Error Distribution (GED)

The GED implementation provides a flexible distribution that generalizes the normal distribution with an additional parameter controlling tail thickness. Files in this group include:

- `gedpdf.m` - PDF computation
- `gedcdf.m` - CDF computation
- `gedinv.m` - Quantile function
- `gedrnd.m` - Random number generation
- `gedloglik.m` - Log-likelihood calculation
- `gedfit.m` - Parameter estimation

The GED is particularly useful for modeling financial time series with varying degrees of kurtosis.

## Hansen's Skewed T-distribution

Hansen's skewed T-distribution extends the Student's t-distribution by incorporating skewness, offering greater flexibility for modeling asymmetric financial returns. Files in this group include:

- `skewtpdf.m` - PDF computation
- `skewtcdf.m` - CDF computation
- `skewtinv.m` - Quantile function
- `skewtrnd.m` - Random number generation
- `skewtloglik.m` - Log-likelihood calculation
- `skewtfit.m` - Parameter estimation

This distribution is well-suited for capturing both the heavy tails and asymmetry often observed in financial return distributions.

## Standardized Student's T-distribution

The standardized Student's t-distribution implementation provides a version of the t-distribution normalized to have unit variance regardless of degrees of freedom. Files in this group include:

- `stdtpdf.m` - PDF computation
- `stdtcdf.m` - CDF computation
- `stdtinv.m` - Quantile function
- `stdtrnd.m` - Random number generation
- `stdtloglik.m` - Log-likelihood calculation
- `stdtfit.m` - Parameter estimation

This standardized version is particularly useful in financial models where variance is parameterized separately, such as GARCH models.

## Integration with Volatility Models

The distribution functions in this module are designed to integrate seamlessly with the univariate and multivariate volatility models in the MFE Toolbox. Volatility model estimation functions (e.g., `agarchfit`, `egarchfit`, `tarchfit`) accept these distributions as options for the error distribution, enabling more accurate modeling of financial returns.

## Technical Implementation

All distribution functions feature:

- Robust parameter validation via `parametercheck.m` utility
- Data validation via `datacheck.m` and `columncheck.m` utilities
- Vectorized computation for performance optimization
- Comprehensive error handling with informative messages
- Numerical stability for extreme parameter values
- Consistent interface across all distributions

## Usage Examples

For detailed usage examples, please refer to the following documentation:

- Documentation: `docs/distribution_functions.md`
- Examples: `examples/distribution_analysis.m`
- Validation: `src/test/validation/DistributionValidation.m`