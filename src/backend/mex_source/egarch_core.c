/**
 * @file egarch_core.c
 * @brief High-performance C implementation of Exponential GARCH (EGARCH) model
 *
 * This file implements the core computational algorithms for EGARCH volatility models
 * with optimizations for high-performance and numerical stability. The EGARCH model
 * captures asymmetric effects in volatility and guarantees positive variance through
 * log-variance modeling.
 *
 * Implementation of the model:
 * log(σ²ₜ) = ω + Σᵢ₌₁ᵖ[αᵢ(|zₜ₋ᵢ| - E[|z|]) + γᵢzₜ₋ᵢ] + Σⱼ₌₁ᵍβⱼlog(σ²ₜ₋ⱼ)
 *
 * @version 4.0 (28-Oct-2009)
 */

/* External dependencies */
#include "mex.h"      /* MATLAB MEX API - MATLAB 4.0 compatible */
#include "matrix.h"   /* MATLAB MEX API - MATLAB 4.0 compatible */
#include <stdlib.h>   /* C Standard Library - C89/C90 */
#include <math.h>     /* C Standard Library - C89/C90 */
#include <float.h>    /* C Standard Library - C89/C90 */

/* Internal dependencies */
#include "matrix_operations.h"  /* Optimized matrix operations for efficient numerical computation */
#include "mex_utils.h"          /* Provides utility functions for MEX interface and error handling */

/* Constants for numerical stability and model implementation */
#define MIN_LOGVARIANCE -30.0   /* Lower bound for log-variance to prevent underflow */
#define MAX_LOGVARIANCE 30.0    /* Upper bound for log-variance to prevent overflow */
#define LOG_2PI 1.83787706640934534  /* Natural logarithm of 2π for likelihood calculations */
#define E_ABS_Z 0.7978845608028654   /* E[|z|] for standard normal = sqrt(2/π) */

/* Forward declarations for helper functions */
void extract_egarch_parameters(double* parameters, int p, int q, 
                              double* omega, double** alpha, double** gamma, double** beta);
int validate_egarch_parameters(double omega, double* alpha, double* gamma, double* beta, 
                              int p, int q);
void compute_standardized_residuals(double* data, double* variance, 
                                   double* standardized_residuals, mwSize T);
double compute_normal_loglikelihood(double* data, double* variance, mwSize T);
double compute_t_loglikelihood(double* data, double* variance, double nu, mwSize T);
double compute_ged_loglikelihood(double* data, double* variance, double nu, mwSize T);
double compute_skewt_loglikelihood(double* data, double* variance, double nu, double lambda, mwSize T);
double kahan_sum(double* values, mwSize count);

/**
 * @brief Core computation function for EGARCH model
 *
 * Implements the main EGARCH recursion for conditional variance and calculates
 * log-likelihood if requested using the specified error distribution.
 *
 * @param data Time series data (residuals/returns)
 * @param parameters Model parameters (omega, alpha, gamma, beta)
 * @param T Length of the time series
 * @param p Order of ARCH component
 * @param q Order of GARCH component
 * @param backcast Value for initializing variance
 * @param compute_likelihood Flag to compute log-likelihood (0=no, 1=yes)
 * @param distribution_type Distribution for likelihood (1=normal, 2=t, 3=ged, 4=skewed-t)
 * @param distribution_parameters Parameters for the distribution (nu, lambda)
 * @param variance Output array for conditional variance
 * @param logvariance Output array for log-variance
 * @param likelihood Pointer to store log-likelihood value
 * @return Status code (0 for success, non-zero for error)
 */
int egarch_core_compute(
    double* data,
    double* parameters,
    mwSize T,
    int p,
    int q,
    double backcast,
    int compute_likelihood,
    int distribution_type,
    double* distribution_parameters,
    double* variance,
    double* logvariance,
    double* likelihood
) {
    int i, j, t, max_lag;
    double omega, *alpha, *gamma, *beta;
    double z_t, abs_z_t, log_backcast;
    double nu, lambda = 0.0;
    double sum_alpha_z, sum_beta_logvar;
    
    /* Extract EGARCH parameters */
    extract_egarch_parameters(parameters, p, q, &omega, &alpha, &gamma, &beta);
    
    /* Verify parameter constraints */
    if (validate_egarch_parameters(omega, alpha, gamma, beta, p, q) != 0) {
        mex_error("Invalid EGARCH parameters: Model not stationary (sum of beta >= 1)");
        return 1;
    }
    
    /* Extract distribution parameters if needed */
    if (compute_likelihood) {
        if (distribution_type == 2 || distribution_type == 3) {
            /* t or GED distribution: one parameter (nu) */
            nu = distribution_parameters[0];
            if (nu <= 2.0 && distribution_type == 2) {
                mex_error("t distribution requires nu > 2 for finite variance");
                return 1;
            }
            if (nu <= 1.0 && distribution_type == 3) {
                mex_error("GED distribution requires nu > 1");
                return 1;
            }
        } else if (distribution_type == 4) {
            /* Skewed t distribution: two parameters (nu, lambda) */
            nu = distribution_parameters[0];
            lambda = distribution_parameters[1];
            if (nu <= 2.0) {
                mex_error("Skewed t distribution requires nu > 2 for finite variance");
                return 1;
            }
            if (lambda <= -1.0 || lambda >= 1.0) {
                mex_error("Skewed t distribution requires -1 < lambda < 1");
                return 1;
            }
        }
    }
    
    /* Initialize with log of backcast value */
    log_backcast = log(backcast);
    if (log_backcast < MIN_LOGVARIANCE) log_backcast = MIN_LOGVARIANCE;
    if (log_backcast > MAX_LOGVARIANCE) log_backcast = MAX_LOGVARIANCE;
    
    max_lag = (p > q) ? p : q;
    
    /* Initialize variance and log-variance for pre-sample observations */
    for (t = 0; t < max_lag; t++) {
        logvariance[t] = log_backcast;
        variance[t] = exp(log_backcast);
    }
    
    /* Calculate conditional variance for each time point */
    for (t = max_lag; t < T; t++) {
        /* Initialize with omega constant */
        logvariance[t] = omega;
        
        /* Add ARCH and asymmetry terms */
        sum_alpha_z = 0.0;
        for (i = 0; i < p; i++) {
            if (t - i - 1 >= 0) {
                z_t = data[t - i - 1] / sqrt(variance[t - i - 1]);
                abs_z_t = fabs(z_t);
                sum_alpha_z += alpha[i] * (abs_z_t - E_ABS_Z) + gamma[i] * z_t;
            }
        }
        logvariance[t] += sum_alpha_z;
        
        /* Add GARCH terms */
        sum_beta_logvar = 0.0;
        for (j = 0; j < q; j++) {
            if (t - j - 1 >= 0) {
                sum_beta_logvar += beta[j] * logvariance[t - j - 1];
            } else {
                sum_beta_logvar += beta[j] * log_backcast;
            }
        }
        logvariance[t] += sum_beta_logvar;
        
        /* Bound log-variance for numerical stability */
        if (logvariance[t] < MIN_LOGVARIANCE) logvariance[t] = MIN_LOGVARIANCE;
        if (logvariance[t] > MAX_LOGVARIANCE) logvariance[t] = MAX_LOGVARIANCE;
        
        /* Calculate variance from log-variance */
        variance[t] = exp(logvariance[t]);
    }
    
    /* Compute log-likelihood if requested */
    if (compute_likelihood) {
        switch (distribution_type) {
            case 1: /* Normal */
                *likelihood = compute_normal_loglikelihood(data, variance, T);
                break;
            case 2: /* Student's t */
                *likelihood = compute_t_loglikelihood(data, variance, nu, T);
                break;
            case 3: /* GED */
                *likelihood = compute_ged_loglikelihood(data, variance, nu, T);
                break;
            case 4: /* Skewed t */
                *likelihood = compute_skewt_loglikelihood(data, variance, nu, lambda, T);
                break;
            default:
                mex_error("Unknown distribution type: %d", distribution_type);
                return 1;
        }
    }
    
    /* Free allocated memory */
    safe_free((void**)&alpha);
    safe_free((void**)&gamma);
    safe_free((void**)&beta);
    
    return 0;
}

/**
 * @brief Main entry point for the MEX function
 *
 * Processes inputs from MATLAB, validates parameters, calls core computation function,
 * and returns results to MATLAB environment.
 *
 * @param nlhs Number of left-hand side (output) arguments
 * @param plhs Array of pointers to left-hand side arguments
 * @param nrhs Number of right-hand side (input) arguments
 * @param prhs Array of pointers to right-hand side arguments
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    double *data, *parameters, *variance, *logvariance, *likelihood;
    double backcast, *distribution_parameters = NULL;
    mwSize T;
    int p, q, compute_likelihood, distribution_type = 1;
    mwSize data_rows, data_cols, param_rows, param_cols;
    
    /* Validate input and output arguments */
    if (check_inputs(nrhs, prhs, 5) != 0) {
        mex_error("Required inputs: data, parameters, p, q, backcast, [compute_likelihood, distribution_type, distribution_parameters]");
        return;
    }
    
    if (check_outputs(nlhs, 3) != 0) {
        mex_error("Required outputs: variance, logvariance, likelihood");
        return;
    }
    
    /* Extract data */
    if (matlab_to_c_double(prhs[0], &data, &data_rows, &data_cols) != 0) {
        mex_error("Failed to extract data array");
        return;
    }
    T = data_rows;
    
    /* Extract parameters */
    if (matlab_to_c_double(prhs[1], &parameters, &param_rows, &param_cols) != 0) {
        safe_free((void**)&data);
        mex_error("Failed to extract parameters array");
        return;
    }
    
    /* Extract p and q */
    p = (int)mxGetScalar(prhs[2]);
    q = (int)mxGetScalar(prhs[3]);
    
    /* Validate parameter dimensions: omega + p*alpha + p*gamma + q*beta */
    if (param_rows != (1 + 2 * p + q)) {
        safe_free((void**)&data);
        safe_free((void**)&parameters);
        mex_error("Parameters dimension mismatch: Expected %d (1 + 2*p + q), got %d", 
                 1 + 2 * p + q, param_rows);
        return;
    }
    
    /* Extract backcast value */
    if (matlab_to_c_double_scalar(prhs[4], &backcast) != 0) {
        safe_free((void**)&data);
        safe_free((void**)&parameters);
        mex_error("Failed to extract backcast value");
        return;
    }
    
    /* Optional parameters */
    compute_likelihood = 0;
    if (nrhs > 5) {
        compute_likelihood = (int)mxGetScalar(prhs[5]);
    }
    
    if (nrhs > 6) {
        distribution_type = (int)mxGetScalar(prhs[6]);
        if (distribution_type < 1 || distribution_type > 4) {
            safe_free((void**)&data);
            safe_free((void**)&parameters);
            mex_error("Invalid distribution type: must be 1 (normal), 2 (t), 3 (GED), or 4 (skewed t)");
            return;
        }
    }
    
    if (nrhs > 7) {
        mwSize dist_rows, dist_cols;
        if (matlab_to_c_double(prhs[7], &distribution_parameters, &dist_rows, &dist_cols) != 0) {
            safe_free((void**)&data);
            safe_free((void**)&parameters);
            mex_error("Failed to extract distribution parameters");
            return;
        }
        
        /* Validate distribution parameter dimensions */
        if ((distribution_type == 2 || distribution_type == 3) && dist_rows != 1) {
            safe_free((void**)&data);
            safe_free((void**)&parameters);
            safe_free((void**)&distribution_parameters);
            mex_error("t/GED distribution requires 1 parameter (nu)");
            return;
        }
        if (distribution_type == 4 && dist_rows != 2) {
            safe_free((void**)&data);
            safe_free((void**)&parameters);
            safe_free((void**)&distribution_parameters);
            mex_error("Skewed t distribution requires 2 parameters (nu, lambda)");
            return;
        }
    } else if (compute_likelihood && distribution_type != 1) {
        safe_free((void**)&data);
        safe_free((void**)&parameters);
        mex_error("Distribution parameters required for non-normal distributions");
        return;
    }
    
    /* Create output arrays */
    plhs[0] = mxCreateDoubleMatrix(T, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(T, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
    
    variance = mxGetPr(plhs[0]);
    logvariance = mxGetPr(plhs[1]);
    likelihood = mxGetPr(plhs[2]);
    
    /* Initialize likelihood to 0 in case we don't compute it */
    *likelihood = 0.0;
    
    /* Call the core computation function */
    if (egarch_core_compute(data, parameters, T, p, q, backcast, compute_likelihood,
                          distribution_type, distribution_parameters, 
                          variance, logvariance, likelihood) != 0) {
        /* Error occurred, but message already provided by the function */
        /* Just clean up here */
    }
    
    /* Free allocated memory */
    safe_free((void**)&data);
    safe_free((void**)&parameters);
    if (distribution_parameters != NULL) {
        safe_free((void**)&distribution_parameters);
    }
}

/**
 * @brief Extracts individual EGARCH model parameters from the parameter vector
 *
 * @param parameters Array of model parameters [omega; alpha; gamma; beta]
 * @param p Order of ARCH terms
 * @param q Order of GARCH terms
 * @param omega Pointer to store omega (constant) parameter
 * @param alpha Pointer to store allocated alpha coefficients array
 * @param gamma Pointer to store allocated gamma coefficients array
 * @param beta Pointer to store allocated beta coefficients array
 */
void extract_egarch_parameters(
    double* parameters,
    int p,
    int q,
    double* omega,
    double** alpha,
    double** gamma,
    double** beta
) {
    int i;
    
    /* Extract omega (constant term) */
    *omega = parameters[0];
    
    /* Extract alpha (ARCH terms) */
    *alpha = (double*)safe_malloc(p * sizeof(double));
    for (i = 0; i < p; i++) {
        (*alpha)[i] = parameters[i + 1];
    }
    
    /* Extract gamma (asymmetry terms) */
    *gamma = (double*)safe_malloc(p * sizeof(double));
    for (i = 0; i < p; i++) {
        (*gamma)[i] = parameters[i + p + 1];
    }
    
    /* Extract beta (GARCH terms) */
    *beta = (double*)safe_malloc(q * sizeof(double));
    for (i = 0; i < q; i++) {
        (*beta)[i] = parameters[i + 2*p + 1];
    }
}

/**
 * @brief Validates parameter constraints for the EGARCH model
 *
 * Ensures the model is stationary (sum of beta < 1)
 *
 * @param omega Constant term
 * @param alpha ARCH coefficients
 * @param gamma Asymmetry coefficients 
 * @param beta GARCH coefficients
 * @param p Order of ARCH terms
 * @param q Order of GARCH terms
 * @return 0 if parameters are valid, non-zero otherwise
 */
int validate_egarch_parameters(
    double omega,
    double* alpha,
    double* gamma,
    double* beta,
    int p,
    int q
) {
    int j;
    double persistence = 0.0;
    
    /* Calculate persistence (sum of beta) */
    for (j = 0; j < q; j++) {
        persistence += beta[j];
    }
    
    /* Check stationarity constraint */
    if (persistence >= 1.0) {
        return 1;  /* Invalid: non-stationary */
    }
    
    return 0;  /* Valid */
}

/**
 * @brief Computes standardized residuals from data and variance
 *
 * @param data Time series data (residuals/returns)
 * @param variance Conditional variance series
 * @param standardized_residuals Output array for standardized residuals
 * @param T Length of the time series
 */
void compute_standardized_residuals(
    double* data,
    double* variance,
    double* standardized_residuals,
    mwSize T
) {
    mwSize t;
    
    for (t = 0; t < T; t++) {
        standardized_residuals[t] = data[t] / sqrt(variance[t]);
    }
}

/**
 * @brief Computes log-likelihood for normal distribution
 *
 * @param data Time series data (residuals/returns)
 * @param variance Conditional variance series
 * @param T Length of the time series
 * @return Log-likelihood value
 */
double compute_normal_loglikelihood(double* data, double* variance, mwSize T) {
    mwSize t;
    double sum = 0.0, compensation = 0.0;
    double term, y, temp;
    
    /* Use Kahan summation for improved numerical accuracy */
    for (t = 0; t < T; t++) {
        /* log(f(x)) = -0.5*log(2π) - 0.5*log(σ²) - 0.5*(x²/σ²) */
        term = -0.5 * LOG_2PI - 0.5 * log(variance[t]) - 0.5 * (data[t] * data[t]) / variance[t];
        
        /* Kahan summation algorithm */
        y = term - compensation;
        temp = sum + y;
        compensation = (temp - sum) - y;
        sum = temp;
    }
    
    return sum;
}

/**
 * @brief Computes log-likelihood for Student's t distribution
 *
 * @param data Time series data (residuals/returns)
 * @param variance Conditional variance series
 * @param nu Degrees of freedom parameter
 * @param T Length of the time series
 * @return Log-likelihood value
 */
double compute_t_loglikelihood(double* data, double* variance, double nu, mwSize T) {
    mwSize t;
    double sum = 0.0, compensation = 0.0;
    double term, y, temp, z_t_sq;
    
    /* Precompute constants */
    double const_term = lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * log(M_PI * (nu - 2.0));
    
    /* Use Kahan summation for improved numerical accuracy */
    for (t = 0; t < T; t++) {
        z_t_sq = (data[t] * data[t]) / variance[t];
        
        /* log(f(x)) = const - 0.5*log(σ²) - ((nu+1)/2)*log(1 + z²/(nu-2)) */
        term = const_term - 0.5 * log(variance[t]) 
               - ((nu + 1.0) / 2.0) * log(1.0 + z_t_sq / (nu - 2.0));
        
        /* Kahan summation algorithm */
        y = term - compensation;
        temp = sum + y;
        compensation = (temp - sum) - y;
        sum = temp;
    }
    
    return sum;
}

/**
 * @brief Computes log-likelihood for Generalized Error Distribution (GED)
 *
 * @param data Time series data (residuals/returns)
 * @param variance Conditional variance series
 * @param nu Shape parameter (tail thickness)
 * @param T Length of the time series
 * @return Log-likelihood value
 */
double compute_ged_loglikelihood(double* data, double* variance, double nu, mwSize T) {
    mwSize t;
    double sum = 0.0, compensation = 0.0;
    double term, y, temp, abs_z_t;
    
    /* Precompute constants */
    double lambda = sqrt(pow(2.0, -2.0/nu) * exp(lgamma(1.0/nu)) / exp(lgamma(3.0/nu)));
    double const_term = log(nu) - log(lambda) - (1.0 + 1.0/nu) * log(2.0) - lgamma(1.0/nu);
    
    /* Use Kahan summation for improved numerical accuracy */
    for (t = 0; t < T; t++) {
        abs_z_t = fabs(data[t]) / sqrt(variance[t]);
        
        /* log(f(x)) = const - 0.5*log(σ²) - 0.5*((|x|/(λ*σ))^nu) */
        term = const_term - 0.5 * log(variance[t]) 
               - 0.5 * pow(abs_z_t/lambda, nu);
        
        /* Kahan summation algorithm */
        y = term - compensation;
        temp = sum + y;
        compensation = (temp - sum) - y;
        sum = temp;
    }
    
    return sum;
}

/**
 * @brief Computes log-likelihood for Hansen's Skewed t distribution
 *
 * @param data Time series data (residuals/returns)
 * @param variance Conditional variance series
 * @param nu Degrees of freedom parameter
 * @param lambda Skewness parameter (-1 < lambda < 1)
 * @param T Length of the time series
 * @return Log-likelihood value
 */
double compute_skewt_loglikelihood(double* data, double* variance, double nu, double lambda, mwSize T) {
    mwSize t;
    double sum = 0.0, compensation = 0.0;
    double term, y, temp, z_t, a, b, c;
    
    /* Precompute constants */
    double lambda_sq = lambda * lambda;
    a = 4 * lambda * exp(lgamma((nu + 1.0) / 2.0)) / 
        (exp(lgamma(nu / 2.0)) * sqrt(M_PI * (nu - 2.0)));
    b = sqrt(1.0 + 3.0 * lambda_sq - a * a);
    c = log(b + a) - log(2);
    
    /* Use Kahan summation for improved numerical accuracy */
    for (t = 0; t < T; t++) {
        z_t = data[t] / sqrt(variance[t]);
        
        if (z_t < -a/b) {
            /* Left side of the distribution */
            term = c - 0.5 * log(variance[t]) - ((nu + 1.0) / 2.0) * 
                   log(1.0 + (z_t * b + a) * (z_t * b + a) / (nu - 2.0) * (1.0 + lambda));
        } else {
            /* Right side of the distribution */
            term = c - 0.5 * log(variance[t]) - ((nu + 1.0) / 2.0) * 
                   log(1.0 + (z_t * b - a) * (z_t * b - a) / (nu - 2.0) * (1.0 - lambda));
        }
        
        /* Kahan summation algorithm */
        y = term - compensation;
        temp = sum + y;
        compensation = (temp - sum) - y;
        sum = temp;
    }
    
    return sum;
}

/**
 * @brief Implements Kahan summation algorithm for improved floating-point accuracy
 *
 * Reduces floating-point errors in large summations by tracking the running compensation
 * for low-order bits lost during addition.
 *
 * @param values Array of values to sum
 * @param count Number of elements in the array
 * @return Sum with compensated floating-point error
 */
double kahan_sum(double* values, mwSize count) {
    mwSize i;
    double sum = 0.0;
    double compensation = 0.0;
    double y, temp;
    
    for (i = 0; i < count; i++) {
        y = values[i] - compensation;
        temp = sum + y;
        compensation = (temp - sum) - y;
        sum = temp;
    }
    
    return sum;
}