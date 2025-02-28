/**
 * @file agarch_core.c
 * @brief High-performance C implementation of Asymmetric GARCH (AGARCH) model
 *
 * Provides optimized computation of conditional variance recursion and 
 * likelihood values for AGARCH models, significantly accelerating parameter estimation
 * and forecasting for financial time series volatility modeling.
 *
 * @version 4.0 (28-Oct-2009)
 */

/* Standard includes */
#include "mex.h"      /* MATLAB MEX API - MATLAB 4.0 compatible */
#include "matrix.h"   /* MATLAB MEX API - MATLAB 4.0 compatible */
#include <stdlib.h>   /* C Standard Library - C89/C90 */
#include <math.h>     /* C Standard Library - C89/C90 */
#include <float.h>    /* C Standard Library - C89/C90 */

/* Internal includes */
#include "matrix_operations.h"  /* Optimized matrix operations */
#include "mex_utils.h"          /* MEX utility functions */

/* Constant definitions */
#define MIN_VARIANCE 1e-10      /* Minimum allowed variance to prevent numerical issues */
#define LOG_2PI 1.83787706640934534  /* Log of 2*pi, precomputed for efficiency */

/* Function prototypes */
int agarch_core_compute(double* data, double* parameters, mwSize T, int p, int q, 
                        double backcast, int compute_likelihood, int distribution_type,
                        double* variance, double* likelihood);
void extract_agarch_parameters(double* parameters, int p, int q, 
                              double* omega, double** alpha, double* gamma, double** beta);
int validate_agarch_parameters(double omega, double* alpha, double gamma, double* beta, int p, int q);
double compute_normal_loglikelihood(double* data, double* variance, mwSize T);
double compute_t_loglikelihood(double* data, double* variance, double nu, mwSize T);
double compute_ged_loglikelihood(double* data, double* variance, double nu, mwSize T);
double compute_skewt_loglikelihood(double* data, double* variance, double nu, double lambda, mwSize T);

/**
 * @brief MEX entry point for AGARCH core computation
 *
 * Expected inputs:
 *   prhs[0]: data - time series data (T x 1)
 *   prhs[1]: parameters - model parameters [omega, alpha(1...p), gamma, beta(1...q), (nu), (lambda)]
 *   prhs[2]: p - AR order
 *   prhs[3]: q - MA order
 *   prhs[4]: backcast - initial variance value
 *   prhs[5]: compute_likelihood - boolean flag (0/1)
 *   prhs[6]: distribution_type - 1:Normal, 2:Student's t, 3:GED, 4:Skewed t
 *
 * Expected outputs:
 *   plhs[0]: variance - conditional variance series (T x 1)
 *   plhs[1]: likelihood - log-likelihood value (if requested)
 *
 * @param nlhs Number of left-hand side arguments
 * @param plhs Array of pointers to the left-hand side arguments
 * @param nrhs Number of right-hand side arguments
 * @param prhs Array of pointers to the right-hand side arguments
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Declare variables */
    double *data, *parameters, *variance, *likelihood_ptr = NULL;
    mwSize T;
    mwSize rows, cols;
    int p, q, compute_likelihood, distribution_type;
    double backcast, likelihood_val = 0.0;
    
    /* Check input and output arguments */
    if (nrhs < 7) {
        mex_error("Insufficient inputs: expected 7 inputs [data, parameters, p, q, backcast, compute_likelihood, distribution_type]");
    }
    if (nlhs < 1 || nlhs > 2) {
        mex_error("Invalid outputs: expected 1 or 2 outputs [variance, (likelihood)]");
    }
    
    /* Extract data from MATLAB */
    if (matlab_to_c_double(prhs[0], &data, &rows, &cols) != 0) {
        mex_error("Invalid data input: must be a numeric array");
    }
    T = rows;
    
    if (cols != 1) {
        safe_free((void**)&data);
        mex_error("Invalid data input: must be a column vector");
    }
    
    /* Extract parameters */
    if (matlab_to_c_double(prhs[1], &parameters, &rows, &cols) != 0) {
        safe_free((void**)&data);
        mex_error("Invalid parameters input: must be a numeric array");
    }
    
    if (cols != 1) {
        safe_free((void**)&data);
        safe_free((void**)&parameters);
        mex_error("Invalid parameters input: must be a column vector");
    }
    
    /* Extract scalar parameters */
    p = (int)mxGetScalar(prhs[2]);
    q = (int)mxGetScalar(prhs[3]);
    backcast = mxGetScalar(prhs[4]);
    compute_likelihood = (int)mxGetScalar(prhs[5]);
    distribution_type = (int)mxGetScalar(prhs[6]);
    
    /* Validate inputs */
    if (p < 0 || q < 0) {
        safe_free((void**)&data);
        safe_free((void**)&parameters);
        mex_error("Invalid model orders: p and q must be non-negative");
    }
    
    if (distribution_type < 1 || distribution_type > 4) {
        safe_free((void**)&data);
        safe_free((void**)&parameters);
        mex_error("Invalid distribution type: must be 1 (Normal), 2 (Student's t), 3 (GED), or 4 (Skewed t)");
    }
    
    /* Check parameter vector length */
    int min_param_length = p + q + 2; /* omega, alpha(1...p), gamma, beta(1...q) */
    if (distribution_type == 2 || distribution_type == 3) {
        min_param_length += 1; /* Add nu parameter */
    } else if (distribution_type == 4) {
        min_param_length += 2; /* Add nu and lambda parameters */
    }
    
    if (rows < min_param_length) {
        safe_free((void**)&data);
        safe_free((void**)&parameters);
        mex_error("Insufficient parameters: expected at least %d parameters for this model", min_param_length);
    }
    
    /* Create output arrays */
    plhs[0] = create_matlab_matrix(T, 1);
    variance = mxGetPr(plhs[0]);
    
    if (nlhs > 1) {
        plhs[1] = create_matlab_matrix(1, 1);
        likelihood_ptr = mxGetPr(plhs[1]);
    }
    
    /* Call computational function */
    int status = agarch_core_compute(data, parameters, T, p, q, backcast, 
                                    compute_likelihood, distribution_type,
                                    variance, &likelihood_val);
    
    /* Set likelihood output if requested */
    if (nlhs > 1 && likelihood_ptr != NULL) {
        *likelihood_ptr = likelihood_val;
    }
    
    /* Clean up */
    safe_free((void**)&data);
    safe_free((void**)&parameters);
    
    /* Check for computation errors */
    if (status != 0) {
        mex_error("AGARCH computation failed with status code %d", status);
    }
}

/**
 * @brief Main computational function for AGARCH model
 *
 * Implements the AGARCH conditional variance recursion:
 * h_t = omega + sum(alpha_i * (e_{t-i} - gamma * sqrt(h_{t-i}))^2) + sum(beta_j * h_{t-j})
 *
 * @param data Input time series data
 * @param parameters Model parameters [omega, alpha(1...p), gamma, beta(1...q), (nu), (lambda)]
 * @param T Length of time series
 * @param p AR order (number of alpha terms)
 * @param q MA order (number of beta terms)
 * @param backcast Initial variance value for pre-sample period
 * @param compute_likelihood Flag to compute log-likelihood
 * @param distribution_type Distribution type (1:Normal, 2:Student's t, 3:GED, 4:Skewed t)
 * @param variance Output array for conditional variance
 * @param likelihood Pointer to store log-likelihood value
 * @return Status code (0 for success, non-zero for error)
 */
int agarch_core_compute(double* data, double* parameters, mwSize T, int p, int q, 
                       double backcast, int compute_likelihood, int distribution_type,
                       double* variance, double* likelihood)
{
    int i, j, max_order;
    double omega, gamma;
    double *alpha = NULL, *beta = NULL;
    double shock, weighted_lag;
    double nu, lambda;
    
    /* Extract AGARCH parameters */
    extract_agarch_parameters(parameters, p, q, &omega, &alpha, &gamma, &beta);
    
    /* Validate AGARCH parameters */
    if (validate_agarch_parameters(omega, alpha, gamma, beta, p, q) != 0) {
        safe_free((void**)&alpha);
        safe_free((void**)&beta);
        return 1; /* Parameter validation failed */
    }
    
    /* Initialize all variance values to backcast */
    for (i = 0; i < T; i++) {
        variance[i] = backcast;
    }
    
    /* Maximum order for model */
    max_order = (p > q) ? p : q;
    
    /* Main recursion loop */
    for (i = max_order; i < T; i++) {
        /* Start with omega term */
        variance[i] = omega;
        
        /* Add ARCH (alpha) terms */
        for (j = 0; j < p; j++) {
            if (i-1-j >= 0) {
                /* AGARCH shock specification: (e_{t-j} - gamma * sqrt(h_{t-j}))^2 */
                shock = data[i-1-j] - gamma * sqrt(variance[i-1-j]);
                shock *= shock; /* Square it */
                variance[i] += alpha[j] * shock;
            } else {
                /* Use backcast for pre-sample values */
                shock = 0.0 - gamma * sqrt(backcast); /* Assume zero residual in pre-sample */
                shock *= shock;
                variance[i] += alpha[j] * shock;
            }
        }
        
        /* Add GARCH (beta) terms */
        for (j = 0; j < q; j++) {
            if (i-1-j >= 0) {
                variance[i] += beta[j] * variance[i-1-j];
            } else {
                variance[i] += beta[j] * backcast;
            }
        }
        
        /* Ensure variance is positive */
        if (variance[i] < MIN_VARIANCE) {
            variance[i] = MIN_VARIANCE;
        }
    }
    
    /* Compute log-likelihood if requested */
    if (compute_likelihood) {
        switch (distribution_type) {
            case 1: /* Normal */
                *likelihood = compute_normal_loglikelihood(data, variance, T);
                break;
                
            case 2: /* Student's t */
                nu = parameters[p + q + 2]; /* Get degrees of freedom parameter */
                if (nu <= 2.0) {
                    nu = 2.0001; /* Ensure degrees of freedom > 2 for finite variance */
                }
                *likelihood = compute_t_loglikelihood(data, variance, nu, T);
                break;
                
            case 3: /* GED */
                nu = parameters[p + q + 2]; /* Get shape parameter */
                if (nu <= 0.0) {
                    nu = 0.1; /* Ensure positive shape parameter */
                }
                *likelihood = compute_ged_loglikelihood(data, variance, nu, T);
                break;
                
            case 4: /* Skewed t */
                nu = parameters[p + q + 2]; /* Get degrees of freedom parameter */
                if (nu <= 2.0) {
                    nu = 2.0001; /* Ensure degrees of freedom > 2 for finite variance */
                }
                lambda = parameters[p + q + 3]; /* Get skewness parameter */
                if (lambda < -0.99) {
                    lambda = -0.99;
                } else if (lambda > 0.99) {
                    lambda = 0.99;
                }
                *likelihood = compute_skewt_loglikelihood(data, variance, nu, lambda, T);
                break;
                
            default:
                safe_free((void**)&alpha);
                safe_free((void**)&beta);
                return 2; /* Invalid distribution type */
        }
    }
    
    /* Clean up */
    safe_free((void**)&alpha);
    safe_free((void**)&beta);
    
    return 0; /* Success */
}

/**
 * @brief Extracts individual AGARCH parameters from parameter vector
 *
 * @param parameters Full parameter vector
 * @param p AR order (number of alpha terms)
 * @param q MA order (number of beta terms)
 * @param omega Pointer to store omega (constant) parameter
 * @param alpha Pointer to store pointer to alpha coefficients (will be allocated)
 * @param gamma Pointer to store gamma (asymmetry) parameter
 * @param beta Pointer to store pointer to beta coefficients (will be allocated)
 */
void extract_agarch_parameters(double* parameters, int p, int q, 
                              double* omega, double** alpha, double* gamma, double** beta)
{
    int i;
    
    /* Extract omega (constant term) */
    *omega = parameters[0];
    
    /* Extract alpha coefficients */
    *alpha = (double*)safe_malloc(p * sizeof(double));
    for (i = 0; i < p; i++) {
        (*alpha)[i] = parameters[i + 1];
    }
    
    /* Extract gamma (asymmetry parameter) */
    *gamma = parameters[p + 1];
    
    /* Extract beta coefficients */
    *beta = (double*)safe_malloc(q * sizeof(double));
    for (i = 0; i < q; i++) {
        (*beta)[i] = parameters[p + 2 + i];
    }
}

/**
 * @brief Validates AGARCH model parameters for theoretical constraints
 *
 * @param omega Constant term
 * @param alpha Pointer to alpha coefficients
 * @param gamma Asymmetry parameter
 * @param beta Pointer to beta coefficients
 * @param p AR order (number of alpha terms)
 * @param q MA order (number of beta terms)
 * @return 0 if parameters are valid, non-zero otherwise
 */
int validate_agarch_parameters(double omega, double* alpha, double gamma, double* beta, int p, int q)
{
    int i;
    double persistence = 0.0;
    
    /* Check positivity constraint on omega */
    if (omega <= 0.0) {
        return 1; /* Invalid omega parameter */
    }
    
    /* Check non-negativity constraints on alpha */
    for (i = 0; i < p; i++) {
        if (alpha[i] < 0.0) {
            return 2; /* Invalid alpha parameter */
        }
        persistence += alpha[i];
    }
    
    /* Check non-negativity constraints on beta */
    for (i = 0; i < q; i++) {
        if (beta[i] < 0.0) {
            return 3; /* Invalid beta parameter */
        }
        persistence += beta[i];
    }
    
    /* Check covariance stationarity constraint */
    if (persistence >= 1.0) {
        return 4; /* Model not covariance stationary */
    }
    
    return 0; /* Parameters are valid */
}

/**
 * @brief Computes log-likelihood for AGARCH model with normal distribution
 *
 * Implements numerically stable computation of log-likelihood using Kahan summation.
 *
 * @param data Input time series data
 * @param variance Conditional variance series
 * @param T Length of time series
 * @return Log-likelihood value
 */
double compute_normal_loglikelihood(double* data, double* variance, mwSize T)
{
    mwSize i;
    double ll = 0.0;
    double comp = 0.0; /* Compensation term for Kahan summation */
    double term, y, t;
    
    /* Loop through time series */
    for (i = 0; i < T; i++) {
        /* Skip if variance is not positive */
        if (variance[i] <= 0.0) {
            continue;
        }
        
        /* Compute log-likelihood component for normal distribution */
        term = -0.5 * LOG_2PI - 0.5 * log(variance[i]) - 0.5 * data[i] * data[i] / variance[i];
        
        /* Use Kahan summation algorithm for better numerical precision */
        y = term - comp;
        t = ll + y;
        comp = (t - ll) - y;
        ll = t;
    }
    
    return ll;
}

/**
 * @brief Computes log-likelihood for AGARCH model with Student's t distribution
 *
 * Implements numerically stable computation of log-likelihood using Kahan summation.
 *
 * @param data Input time series data
 * @param variance Conditional variance series
 * @param nu Degrees of freedom parameter
 * @param T Length of time series
 * @return Log-likelihood value
 */
double compute_t_loglikelihood(double* data, double* variance, double nu, mwSize T)
{
    mwSize i;
    double ll = 0.0;
    double comp = 0.0; /* Compensation term for Kahan summation */
    double term, y, t;
    double log_term1, log_term2, z_squared;
    
    /* Precompute constant terms */
    log_term1 = lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * log(M_PI * (nu - 2.0));
    
    /* Loop through time series */
    for (i = 0; i < T; i++) {
        /* Skip if variance is not positive */
        if (variance[i] <= 0.0) {
            continue;
        }
        
        /* Compute standardized squared residual */
        z_squared = data[i] * data[i] / variance[i];
        
        /* Compute log-likelihood component for Student's t distribution */
        log_term2 = -0.5 * log(variance[i]) - ((nu + 1.0) / 2.0) * log(1.0 + z_squared / (nu - 2.0));
        term = log_term1 + log_term2;
        
        /* Use Kahan summation algorithm for better numerical precision */
        y = term - comp;
        t = ll + y;
        comp = (t - ll) - y;
        ll = t;
    }
    
    return ll;
}

/**
 * @brief Computes log-likelihood for AGARCH model with GED distribution
 *
 * Implements numerically stable computation of log-likelihood using Kahan summation.
 *
 * @param data Input time series data
 * @param variance Conditional variance series
 * @param nu Shape parameter
 * @param T Length of time series
 * @return Log-likelihood value
 */
double compute_ged_loglikelihood(double* data, double* variance, double nu, mwSize T)
{
    mwSize i;
    double ll = 0.0;
    double comp = 0.0; /* Compensation term for Kahan summation */
    double term, y, t;
    double log_term1, lambda, abs_z;
    
    /* Precompute constant terms */
    lambda = sqrt(pow(2.0, -2.0/nu) * exp(lgamma(1.0/nu)) / exp(lgamma(3.0/nu)));
    log_term1 = log(nu / (2.0 * lambda * exp(lgamma(1.0/nu))));
    
    /* Loop through time series */
    for (i = 0; i < T; i++) {
        /* Skip if variance is not positive */
        if (variance[i] <= 0.0) {
            continue;
        }
        
        /* Compute standardized absolute residual */
        abs_z = fabs(data[i]) / sqrt(variance[i]);
        
        /* Compute log-likelihood component for GED distribution */
        term = log_term1 - 0.5 * log(variance[i]) - 0.5 * pow(abs_z / lambda, nu);
        
        /* Use Kahan summation algorithm for better numerical precision */
        y = term - comp;
        t = ll + y;
        comp = (t - ll) - y;
        ll = t;
    }
    
    return ll;
}

/**
 * @brief Computes log-likelihood for AGARCH model with Hansen's Skewed t distribution
 *
 * Implements numerically stable computation of log-likelihood using Kahan summation.
 *
 * @param data Input time series data
 * @param variance Conditional variance series
 * @param nu Degrees of freedom parameter
 * @param lambda Skewness parameter
 * @param T Length of time series
 * @return Log-likelihood value
 */
double compute_skewt_loglikelihood(double* data, double* variance, double nu, double lambda, mwSize T)
{
    mwSize i;
    double ll = 0.0;
    double comp = 0.0; /* Compensation term for Kahan summation */
    double term, y, t;
    double log_term1, a, b, c, z, g_z;
    
    /* Precompute constant terms */
    a = 4 * lambda * (nu - 2) / ((1 - lambda * lambda) * (nu - 1));
    b = sqrt(1 + 3 * lambda * lambda - a * a);
    c = exp(lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0)) / (sqrt(M_PI * (nu - 2.0)));
    
    /* Loop through time series */
    for (i = 0; i < T; i++) {
        /* Skip if variance is not positive */
        if (variance[i] <= 0.0) {
            continue;
        }
        
        /* Compute standardized residual */
        z = data[i] / sqrt(variance[i]);
        
        /* Apply skewed t distribution */
        if (z < -a/b) {
            g_z = c * pow(b * (-z) - a, -(nu + 1.0) / 2.0) * (1.0 - lambda);
        } else {
            g_z = c * pow(b * z + a, -(nu + 1.0) / 2.0) * (1.0 + lambda);
        }
        
        /* Compute log-likelihood component */
        term = log(g_z) - 0.5 * log(variance[i]);
        
        /* Use Kahan summation algorithm for better numerical precision */
        y = term - comp;
        t = ll + y;
        comp = (t - ll) - y;
        ll = t;
    }
    
    return ll;
}