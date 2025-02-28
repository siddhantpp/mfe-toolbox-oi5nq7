/**
 * @file tarch_core.c
 * @brief High-performance C implementation of the Threshold ARCH (TARCH) volatility model
 *
 * This MEX file provides optimized computation of conditional variances and 
 * log-likelihood for financial time series exhibiting asymmetric volatility 
 * responses to positive and negative shocks (leverage effect).
 *
 * The TARCH model is defined as:
 * h_t = ω + α·r²_{t-1} + γ·r²_{t-1}·I[r_{t-1}<0] + β·h_{t-1}
 *
 * where I[r_{t-1}<0] is an indicator function equal to 1 if the previous
 * return was negative, capturing the leverage effect in financial markets.
 *
 * @version 4.0 (28-Oct-2009)
 */

/* Include required headers */
#include "mex.h"      /* MATLAB MEX API - MATLAB 4.0 compatible */
#include "matrix.h"   /* MATLAB MEX API - MATLAB 4.0 compatible */
#include <stdlib.h>   /* C Standard Library - C89/C90 */
#include <math.h>     /* C Standard Library - C89/C90 */
#include <string.h>   /* C Standard Library - C89/C90 */

/* Internal utility headers */
#include "matrix_operations.h"  /* Optimized matrix operations */
#include "mex_utils.h"          /* MEX interface utilities */

/* Global constants */
#define TARCH_PARAM_COUNT 4 /* Number of parameters in the TARCH model: omega, alpha, gamma, beta */
#define MIN_VARIANCE 1e-12  /* Minimum allowed variance value to ensure numerical stability */

/* Function declarations */
void compute_tarch_variances(const double* data, mwSize T, const double* parameters, 
                            double backcast, double* variances);
double compute_tarch_likelihood(const double* data, const double* variances, mwSize T);
int check_tarch_constraints(const double* parameters);
void cleanup_tarch_memory(double** variances, double** squared_returns, double** negative_indicators);

/**
 * @brief MEX entry point function that interfaces between MATLAB and C implementation
 *        of TARCH model
 *
 * Expected inputs:
 *   - prhs[0]: Time series data (T×1 vector)
 *   - prhs[1]: TARCH parameters [omega, alpha, gamma, beta] (4×1 vector)
 *   - prhs[2]: Backcast value for variance initialization (scalar)
 *   - prhs[3]: (Optional) Boolean flag to compute likelihood (scalar)
 *
 * Outputs:
 *   - plhs[0]: Conditional variances (T×1 vector)
 *   - plhs[1]: (Optional) Log-likelihood value (scalar) if requested
 *
 * @param nlhs Number of left-hand side arguments (outputs)
 * @param plhs Pointer to output arrays
 * @param nrhs Number of right-hand side arguments (inputs)
 * @param prhs Pointer to input arrays
 */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    /* Variables for data extraction */
    double* data = NULL;
    double* parameters = NULL;
    double backcast;
    double compute_likelihood = 0.0;
    double* variances = NULL;
    double likelihood = 0.0;
    mwSize T, num_params;
    mwSize rows, cols;
    int status;
    
    /* Check inputs and outputs */
    if (nrhs < 3 || nrhs > 4) {
        mex_error("TARCH requires 3-4 inputs: data, parameters, backcast, [compute_likelihood]");
    }
    
    if (nlhs < 1 || nlhs > 2) {
        mex_error("TARCH requires 1-2 outputs: variances, [log_likelihood]");
    }
    
    /* Extract time series data */
    status = matlab_to_c_double(prhs[0], &data, &rows, &cols);
    if (status != 0 || cols != 1) {
        mex_error("Data must be a column vector");
    }
    T = rows;
    
    /* Extract parameters */
    status = matlab_to_c_double(prhs[1], &parameters, &rows, &cols);
    if (status != 0 || cols != 1 || rows != TARCH_PARAM_COUNT) {
        mex_error("Parameters must be a %d×1 vector [omega, alpha, gamma, beta]", TARCH_PARAM_COUNT);
    }
    num_params = rows;
    
    /* Check parameter constraints */
    status = check_tarch_constraints(parameters);
    if (status != 0) {
        switch (status) {
            case 1:
                mex_error("Invalid TARCH parameters: omega must be positive");
                break;
            case 2:
                mex_error("Invalid TARCH parameters: alpha, gamma, and beta must be non-negative");
                break;
            case 3:
                mex_error("Invalid TARCH parameters: alpha + 0.5*gamma + beta must be less than 1 for stability");
                break;
            default:
                mex_error("Invalid TARCH parameters");
        }
    }
    
    /* Extract backcast value */
    status = matlab_to_c_double_scalar(prhs[2], &backcast);
    if (status != 0 || backcast <= 0) {
        mex_error("Backcast value must be a positive scalar");
    }
    
    /* Extract compute_likelihood flag if provided */
    if (nrhs > 3) {
        status = matlab_to_c_double_scalar(prhs[3], &compute_likelihood);
        if (status != 0) {
            mex_error("Compute likelihood flag must be a scalar");
        }
    }
    
    /* Allocate memory for variances */
    variances = (double*)safe_malloc(T * sizeof(double));
    
    /* Compute TARCH variances */
    compute_tarch_variances(data, T, parameters, backcast, variances);
    
    /* Create output array for variances */
    plhs[0] = mxCreateDoubleMatrix((mwSize)T, (mwSize)1, mxREAL);
    memcpy(mxGetPr(plhs[0]), variances, T * sizeof(double));
    
    /* Compute and return log-likelihood if requested */
    if (nlhs > 1 || compute_likelihood > 0) {
        likelihood = compute_tarch_likelihood(data, variances, T);
        
        if (nlhs > 1) {
            plhs[1] = mxCreateDoubleScalar(likelihood);
        }
    }
    
    /* Free allocated memory */
    safe_free((void**)&data);
    safe_free((void**)&parameters);
    safe_free((void**)&variances);
}

/**
 * @brief Computes conditional variances for the TARCH model efficiently
 *
 * Implements the TARCH variance equation:
 * h_t = omega + alpha*r_{t-1}^2 + gamma*r_{t-1}^2*I[r_{t-1}<0] + beta*h_{t-1}
 * 
 * @param data Time series data (returns)
 * @param T Length of the time series
 * @param parameters TARCH model parameters [omega, alpha, gamma, beta]
 * @param backcast Value used for variance initialization
 * @param variances Output array for computed conditional variances (pre-allocated, size T)
 */
void compute_tarch_variances(const double* data, mwSize T, const double* parameters, 
                             double backcast, double* variances) {
    mwSize t;
    double omega, alpha, gamma, beta;
    double* squared_returns = NULL;
    double* negative_indicators = NULL;
    double* asymmetric_term = NULL;
    
    /* Extract parameters */
    omega = parameters[0];
    alpha = parameters[1];
    gamma = parameters[2];
    beta = parameters[3];
    
    /* Allocate memory for intermediate calculations */
    squared_returns = (double*)safe_malloc(T * sizeof(double));
    negative_indicators = (double*)safe_malloc(T * sizeof(double));
    asymmetric_term = (double*)safe_malloc(T * sizeof(double));
    
    /* Precompute squared returns and negative indicators */
    for (t = 0; t < T; t++) {
        squared_returns[t] = data[t] * data[t];
        negative_indicators[t] = (data[t] < 0) ? 1.0 : 0.0;
    }
    
    /* Compute the asymmetric term using element-wise multiplication */
    matrix_element_multiply(squared_returns, negative_indicators, asymmetric_term, T, 1);
    
    /* Initialize first variance with backcast */
    variances[0] = backcast;
    
    /* Compute variances recursively using TARCH formula */
    for (t = 1; t < T; t++) {
        /* h_t = omega + alpha*r_{t-1}^2 + gamma*r_{t-1}^2*I[r_{t-1}<0] + beta*h_{t-1} */
        variances[t] = omega + 
                       alpha * squared_returns[t-1] + 
                       gamma * asymmetric_term[t-1] + 
                       beta * variances[t-1];
        
        /* Ensure variance doesn't go below minimum threshold */
        if (variances[t] < MIN_VARIANCE) {
            variances[t] = MIN_VARIANCE;
        }
    }
    
    /* Clean up temporary arrays */
    safe_free((void**)&squared_returns);
    safe_free((void**)&negative_indicators);
    safe_free((void**)&asymmetric_term);
}

/**
 * @brief Computes log-likelihood for the TARCH model based on data and variances
 *
 * Assumes Gaussian innovations and computes:
 * logL = -0.5*T*log(2π) - 0.5*sum(log(h_t) + ε_t^2/h_t)
 * 
 * @param data Time series data (returns)
 * @param variances Conditional variances computed by TARCH model
 * @param T Length of the time series
 * @return Log-likelihood value (negative for optimization)
 */
double compute_tarch_likelihood(const double* data, const double* variances, mwSize T) {
    mwSize t;
    double likelihood = 0.0;
    double log_sum = 0.0, squared_sum = 0.0;
    double* squared_returns = NULL;
    double* inv_variances = NULL;
    double* standardized_terms = NULL;
    double* log_variances = NULL;
    const double LOG_2PI = log(2.0 * M_PI);
    
    /* Allocate memory for calculations */
    squared_returns = (double*)safe_malloc(T * sizeof(double));
    inv_variances = (double*)safe_malloc(T * sizeof(double));
    standardized_terms = (double*)safe_malloc(T * sizeof(double));
    log_variances = (double*)safe_malloc(T * sizeof(double));
    
    /* Compute intermediate terms for all observations */
    for (t = 0; t < T; t++) {
        squared_returns[t] = data[t] * data[t];
        
        /* Handle numerical stability */
        if (variances[t] <= MIN_VARIANCE) {
            inv_variances[t] = 1.0 / MIN_VARIANCE;
            log_variances[t] = log(MIN_VARIANCE);
        } else {
            inv_variances[t] = 1.0 / variances[t];
            log_variances[t] = log(variances[t]);
        }
    }
    
    /* Use matrix operation for element-wise multiplication */
    matrix_element_multiply(squared_returns, inv_variances, standardized_terms, T, 1);
    
    /* Sum log variances and standardized squared returns */
    for (t = 0; t < T; t++) {
        log_sum += log_variances[t];
        squared_sum += standardized_terms[t];
    }
    
    /* Compute final likelihood */
    likelihood = -0.5 * T * LOG_2PI - 0.5 * log_sum - 0.5 * squared_sum;
    
    /* Clean up */
    safe_free((void**)&squared_returns);
    safe_free((void**)&inv_variances);
    safe_free((void**)&standardized_terms);
    safe_free((void**)&log_variances);
    
    /* Return negative log-likelihood for minimization in optimization */
    return -likelihood;
}

/**
 * @brief Validates that TARCH parameters satisfy model constraints
 *
 * Checks the following constraints:
 * - omega > 0 (positivity of base volatility)
 * - alpha >= 0, gamma >= 0, beta >= 0 (non-negativity of coefficients)
 * - alpha + 0.5*gamma + beta < 1 (stability condition)
 * 
 * @param parameters TARCH model parameters [omega, alpha, gamma, beta]
 * @return 0 if all constraints are satisfied, error code otherwise
 */
int check_tarch_constraints(const double* parameters) {
    double omega, alpha, gamma, beta;
    double persistence;
    
    /* Extract parameters */
    omega = parameters[0];
    alpha = parameters[1];
    gamma = parameters[2];
    beta = parameters[3];
    
    /* Check positivity constraint for omega */
    if (omega <= 0) {
        return 1; /* omega must be positive */
    }
    
    /* Check non-negativity constraints */
    if (alpha < 0 || gamma < 0 || beta < 0) {
        return 2; /* alpha, gamma, beta must be non-negative */
    }
    
    /* Check stability constraint */
    persistence = alpha + 0.5 * gamma + beta;
    if (persistence >= 1.0) {
        return 3; /* alpha + 0.5*gamma + beta must be less than 1 */
    }
    
    /* All constraints satisfied */
    return 0;
}

/**
 * @brief Safely releases memory allocated during TARCH computation
 *
 * @param variances Pointer to variances array pointer (can be NULL)
 * @param squared_returns Pointer to squared returns array pointer (can be NULL)
 * @param negative_indicators Pointer to negative indicators array pointer (can be NULL)
 */
void cleanup_tarch_memory(double** variances, double** squared_returns, double** negative_indicators) {
    /* Free variances if allocated */
    if (variances != NULL && *variances != NULL) {
        safe_free((void**)variances);
    }
    
    /* Free squared returns if allocated */
    if (squared_returns != NULL && *squared_returns != NULL) {
        safe_free((void**)squared_returns);
    }
    
    /* Free negative indicators if allocated */
    if (negative_indicators != NULL && *negative_indicators != NULL) {
        safe_free((void**)negative_indicators);
    }
}