/**
 * @file igarch_core.c
 * @brief High-performance C implementation of Integrated GARCH (IGARCH) model core algorithms
 *
 * This implementation provides optimized computation of conditional variance recursion and
 * likelihood calculation for IGARCH models, which have a unit persistence constraint 
 * (sum of alpha and beta coefficients equals 1). The implementation significantly accelerates
 * parameter estimation and forecasting for financial time series with highly persistent volatility.
 *
 * @version 4.0 (28-Oct-2009)
 */

/* Standard library includes */
#include "mex.h"      /* MATLAB MEX API - MATLAB 4.0 compatible */
#include "matrix.h"   /* MATLAB MEX API - MATLAB 4.0 compatible */
#include <stdlib.h>   /* C Standard Library - C89/C90 */
#include <math.h>     /* C Standard Library - C89/C90 */
#include <float.h>    /* C Standard Library - C89/C90 */

/* Internal includes */
#include "matrix_operations.h" /* Optimized matrix operations */
#include "mex_utils.h"         /* MEX interface utilities */

/* Constants for numerical stability and performance */
#define MIN_VARIANCE 1e-10   /* Minimum threshold for variance to prevent division by zero */
#define LOG_2PI 1.83787706640934534  /* Precomputed ln(2π) for likelihood calculation */

/* Function declarations */
int igarch_core_compute(double* data, double* parameters, mwSize T, int p, int q, 
                       double backcast, int compute_likelihood, int distribution_type,
                       double* variance, double* likelihood);
                       
double compute_normal_loglikelihood(double* data, double* variance, mwSize T);
double compute_t_loglikelihood(double* data, double* variance, double nu, mwSize T);
double compute_ged_loglikelihood(double* data, double* variance, double nu, mwSize T);
double compute_skewt_loglikelihood(double* data, double* variance, double nu, double lambda, mwSize T);

int validate_igarch_parameters(double omega, double* alpha, double* beta, int p, int q);
void extract_igarch_parameters(double* parameters, int p, int q, double* omega, double** alpha, double** beta);

/**
 * MEX entry point function - Interface with MATLAB
 *
 * @param nlhs Number of left-hand side arguments (outputs)
 * @param plhs Array of pointers to output arguments
 * @param nrhs Number of right-hand side arguments (inputs)
 * @param prhs Array of pointers to input arguments
 *
 * Expected inputs:
 *   prhs[0]: data - Vector of squared residuals (T×1)
 *   prhs[1]: parameters - Vector of model parameters [omega, alpha_1,...,alpha_p, beta_1,...,beta_q, (nu), (lambda)]
 *   prhs[2]: p - AR order (scalar)
 *   prhs[3]: q - MA order (scalar)
 *   prhs[4]: backcast - Initial variance value (scalar)
 *   prhs[5]: [optional] compute_likelihood - Flag to compute log-likelihood (scalar)
 *   prhs[6]: [optional] distribution_type - Type of error distribution (scalar):
 *            0=Normal, 1=Student's t, 2=GED, 3=Skewed t
 *
 * Expected outputs:
 *   plhs[0]: variance - Vector of conditional variances (T×1)
 *   plhs[1]: [optional] likelihood - Log-likelihood value (scalar)
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables for inputs */
    double *data, *parameters;
    mwSize T, m_parameters, n_parameters;
    int p, q, compute_likelihood = 0, distribution_type = 0;
    double backcast;
    
    /* Variables for outputs */
    double *variance, *likelihood_out;
    double likelihood_value = 0.0;
    
    /* Check inputs and outputs */
    if (nrhs < 5) {
        mex_error("Not enough input arguments. At least 5 arguments required.");
    }
    
    if (nrhs > 7) {
        mex_error("Too many input arguments. Maximum 7 arguments accepted.");
    }
    
    if (nlhs > 2) {
        mex_error("Too many output arguments. Maximum 2 arguments accepted.");
    }
    
    /* Extract data from input arguments */
    /* Input 1: data (T×1 vector of squared residuals) */
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
        mex_error("Data must be a real double vector.");
    }
    
    T = mxGetM(prhs[0]);
    if (mxGetN(prhs[0]) != 1) {
        mex_error("Data must be a column vector.");
    }
    
    data = mxGetPr(prhs[0]);
    
    /* Input 2: parameters vector */
    if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1])) {
        mex_error("Parameters must be a real double vector.");
    }
    
    m_parameters = mxGetM(prhs[1]);
    n_parameters = mxGetN(prhs[1]);
    if (m_parameters != 1 && n_parameters != 1) {
        mex_error("Parameters must be a vector.");
    }
    
    parameters = mxGetPr(prhs[1]);
    
    /* Input 3: p - AR order */
    if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || mxGetNumberOfElements(prhs[2]) != 1) {
        mex_error("AR order (p) must be a scalar.");
    }
    
    p = (int)*mxGetPr(prhs[2]);
    if (p < 0) {
        mex_error("AR order (p) must be non-negative.");
    }
    
    /* Input 4: q - MA order */
    if (!mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) || mxGetNumberOfElements(prhs[3]) != 1) {
        mex_error("MA order (q) must be a scalar.");
    }
    
    q = (int)*mxGetPr(prhs[3]);
    if (q < 0) {
        mex_error("MA order (q) must be non-negative.");
    }
    
    /* Check parameters vector length */
    if (m_parameters * n_parameters < 1 + p + q) {
        mex_error("Parameters vector must contain at least 1 + p + q elements.");
    }
    
    /* Input 5: backcast - Initial variance value */
    if (!mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) || mxGetNumberOfElements(prhs[4]) != 1) {
        mex_error("Backcast value must be a scalar.");
    }
    
    backcast = *mxGetPr(prhs[4]);
    if (backcast <= 0) {
        mex_error("Backcast value must be positive.");
    }
    
    /* Optional Input 6: compute_likelihood flag */
    if (nrhs >= 6) {
        if (!mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]) || mxGetNumberOfElements(prhs[5]) != 1) {
            mex_error("Compute likelihood flag must be a scalar.");
        }
        compute_likelihood = (int)*mxGetPr(prhs[5]);
    }
    
    /* Optional Input 7: distribution_type */
    if (nrhs >= 7) {
        if (!mxIsDouble(prhs[6]) || mxIsComplex(prhs[6]) || mxGetNumberOfElements(prhs[6]) != 1) {
            mex_error("Distribution type must be a scalar.");
        }
        distribution_type = (int)*mxGetPr(prhs[6]);
        if (distribution_type < 0 || distribution_type > 3) {
            mex_error("Distribution type must be 0 (Normal), 1 (Student's t), 2 (GED), or 3 (Skewed t).");
        }
        
        /* Check if we have enough parameters for the specified distribution */
        if (distribution_type == 1 || distribution_type == 2) {
            /* t or GED needs one additional parameter (nu) */
            if (m_parameters * n_parameters < 1 + p + q + 1) {
                mex_error("Parameters vector must contain at least 1 + p + q + 1 elements for t or GED distribution.");
            }
        } else if (distribution_type == 3) {
            /* Skewed t needs two additional parameters (nu, lambda) */
            if (m_parameters * n_parameters < 1 + p + q + 2) {
                mex_error("Parameters vector must contain at least 1 + p + q + 2 elements for skewed t distribution.");
            }
        }
    }
    
    /* Create output arrays */
    plhs[0] = mxCreateDoubleMatrix(T, 1, mxREAL);
    variance = mxGetPr(plhs[0]);
    
    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
        likelihood_out = mxGetPr(plhs[1]);
    }
    
    /* Call the core computation function */
    int status = igarch_core_compute(data, parameters, T, p, q, backcast, 
                                   compute_likelihood, distribution_type, 
                                   variance, &likelihood_value);
    
    /* Check for computation errors */
    if (status != 0) {
        mex_error("Error in IGARCH computation. Check parameters and inputs.");
    }
    
    /* Set likelihood output if requested */
    if (nlhs > 1) {
        *likelihood_out = likelihood_value;
    }
}

/**
 * Main computational function for IGARCH model
 *
 * Implements the conditional variance recursion for the IGARCH model with
 * unit persistence constraint (sum of alpha and beta coefficients equals 1).
 *
 * @param data Input time series data (squared residuals)
 * @param parameters Model parameters [omega, alpha1,...,alphap, beta1,...,betaq, (nu), (lambda)]
 * @param T Length of the time series
 * @param p Number of ARCH terms
 * @param q Number of GARCH terms
 * @param backcast Initial variance value for recursion
 * @param compute_likelihood Flag to compute log-likelihood
 * @param distribution_type Type of error distribution (0=Normal, 1=t, 2=GED, 3=Skewed t)
 * @param variance Output array for conditional variances
 * @param likelihood Pointer to store log-likelihood value
 * @return Status code (0 for success, non-zero for error)
 */
int igarch_core_compute(double* data, double* parameters, mwSize T, int p, int q, 
                       double backcast, int compute_likelihood, int distribution_type,
                       double* variance, double* likelihood) {
    int i, j, t, max_lag;
    double *alpha = NULL, *beta = NULL;
    double omega, resid_sq, this_var;
    
    /* Extract IGARCH parameters */
    double param_omega = parameters[0];
    
    /* Allocate memory for alpha and beta arrays */
    alpha = (double*)safe_malloc(p * sizeof(double));
    beta = (double*)safe_malloc(q * sizeof(double));
    
    /* Extract alpha and beta parameters */
    for (i = 0; i < p; i++) {
        alpha[i] = parameters[i + 1];
    }
    
    for (i = 0; i < q; i++) {
        beta[i] = parameters[i + p + 1];
    }
    
    /* Validate IGARCH parameters */
    if (validate_igarch_parameters(param_omega, alpha, beta, p, q) != 0) {
        safe_free((void**)&alpha);
        safe_free((void**)&beta);
        return 1;  /* Parameter validation failed */
    }
    
    omega = param_omega;
    
    /* Determine maximum lag for the model */
    max_lag = (p > q) ? p : q;
    
    /* Initialize variance with backcast value */
    for (t = 0; t < max_lag; t++) {
        variance[t] = backcast;
    }
    
    /* Main IGARCH recursion loop */
    for (t = max_lag; t < T; t++) {
        /* Start with constant term */
        this_var = omega;
        
        /* Add ARCH terms: alpha * past squared residuals */
        for (i = 0; i < p; i++) {
            if (t - i - 1 >= 0) {
                resid_sq = data[t - i - 1];
                this_var += alpha[i] * resid_sq;
            }
        }
        
        /* Add GARCH terms: beta * past conditional variances */
        for (j = 0; j < q; j++) {
            if (t - j - 1 >= 0) {
                this_var += beta[j] * variance[t - j - 1];
            }
        }
        
        /* Ensure variance is above minimum threshold */
        if (this_var < MIN_VARIANCE) {
            this_var = MIN_VARIANCE;
        }
        
        /* Store computed variance */
        variance[t] = this_var;
    }
    
    /* Compute log-likelihood if requested */
    if (compute_likelihood) {
        switch (distribution_type) {
            case 0:  /* Normal distribution */
                *likelihood = compute_normal_loglikelihood(data, variance, T);
                break;
                
            case 1:  /* Student's t distribution */
                *likelihood = compute_t_loglikelihood(data, variance, parameters[1 + p + q], T);
                break;
                
            case 2:  /* GED distribution */
                *likelihood = compute_ged_loglikelihood(data, variance, parameters[1 + p + q], T);
                break;
                
            case 3:  /* Skewed t distribution */
                *likelihood = compute_skewt_loglikelihood(data, variance, 
                                                        parameters[1 + p + q],
                                                        parameters[2 + p + q], T);
                break;
                
            default:
                *likelihood = compute_normal_loglikelihood(data, variance, T);
        }
    } else {
        *likelihood = 0.0;
    }
    
    /* Free allocated memory */
    safe_free((void**)&alpha);
    safe_free((void**)&beta);
    
    return 0;  /* Success */
}

/**
 * Validates parameter constraints for the IGARCH model with unit persistence
 *
 * Enforces the constraints:
 * - omega > 0
 * - alpha_i >= 0 for all i
 * - beta_j >= 0 for all j
 * - sum(alpha) + sum(beta) = 1 (within numerical tolerance)
 *
 * @param omega Constant term in the variance equation
 * @param alpha Array of ARCH coefficients
 * @param beta Array of GARCH coefficients
 * @param p Number of ARCH terms
 * @param q Number of GARCH terms
 * @return 0 if valid, non-zero if invalid
 */
int validate_igarch_parameters(double omega, double* alpha, double* beta, int p, int q) {
    int i;
    double sum_alpha = 0.0, sum_beta = 0.0, persistence;
    
    /* Check omega */
    if (omega <= 0.0) {
        return 1;  /* omega must be positive */
    }
    
    /* Check alpha coefficients */
    for (i = 0; i < p; i++) {
        if (alpha[i] < 0.0) {
            return 2;  /* alpha must be non-negative */
        }
        sum_alpha += alpha[i];
    }
    
    /* Check beta coefficients */
    for (i = 0; i < q; i++) {
        if (beta[i] < 0.0) {
            return 3;  /* beta must be non-negative */
        }
        sum_beta += beta[i];
    }
    
    /* Check unit persistence constraint */
    persistence = sum_alpha + sum_beta;
    if (fabs(persistence - 1.0) > 1e-6) {
        return 4;  /* sum of alpha and beta must equal 1 */
    }
    
    return 0;  /* Valid parameters */
}

/**
 * Extracts individual IGARCH model parameters from the parameter vector
 *
 * @param parameters Full parameter vector
 * @param p Number of ARCH terms
 * @param q Number of GARCH terms
 * @param omega Pointer to store the constant term
 * @param alpha Pointer to store array of ARCH coefficients (will be allocated)
 * @param beta Pointer to store array of GARCH coefficients (will be allocated)
 */
void extract_igarch_parameters(double* parameters, int p, int q, double* omega, double** alpha, double** beta) {
    int i;
    
    /* Extract omega */
    *omega = parameters[0];
    
    /* Allocate and extract alpha parameters */
    *alpha = (double*)safe_malloc(p * sizeof(double));
    for (i = 0; i < p; i++) {
        (*alpha)[i] = parameters[i + 1];
    }
    
    /* Allocate and extract beta parameters */
    *beta = (double*)safe_malloc(q * sizeof(double));
    for (i = 0; i < q; i++) {
        (*beta)[i] = parameters[i + p + 1];
    }
}

/**
 * Computes log-likelihood for IGARCH model assuming normal distribution
 *
 * Log-likelihood formula for normal distribution:
 * L = -0.5*T*log(2π) - 0.5*sum(log(h_t) + e_t²/h_t)
 *
 * @param data Input time series data (squared residuals)
 * @param variance Array of conditional variances
 * @param T Length of the time series
 * @return Log-likelihood value
 */
double compute_normal_loglikelihood(double* data, double* variance, mwSize T) {
    double loglik = 0.0, component;
    double correction = 0.0;  /* Kahan summation correction term */
    mwSize t;
    
    for (t = 0; t < T; t++) {
        /* Compute log-likelihood component for this observation */
        component = -0.5 * LOG_2PI - 0.5 * log(variance[t]) - 0.5 * (data[t] / variance[t]);
        
        /* Kahan summation algorithm for numerical stability */
        double y = component - correction;
        double temp = loglik + y;
        correction = (temp - loglik) - y;
        loglik = temp;
    }
    
    return loglik;
}

/**
 * Computes log-likelihood for IGARCH model assuming Student's t distribution
 *
 * @param data Input time series data (squared residuals)
 * @param variance Array of conditional variances
 * @param nu Degrees of freedom parameter for t distribution
 * @param T Length of the time series
 * @return Log-likelihood value
 */
double compute_t_loglikelihood(double* data, double* variance, double nu, mwSize T) {
    double loglik = 0.0, component;
    double correction = 0.0;  /* Kahan summation correction term */
    double logconst;
    mwSize t;
    
    /* Validate degrees of freedom parameter */
    if (nu <= 2.0) {
        nu = 2.01;  /* Ensure finite variance */
    }
    
    /* Compute constant terms for t distribution */
    logconst = lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * log(M_PI * (nu - 2.0));
    
    for (t = 0; t < T; t++) {
        /* Compute standardized residual */
        double z_sq = data[t] / variance[t];
        
        /* Compute log-likelihood component for t distribution */
        component = logconst - 0.5 * log(variance[t]) - 
                    ((nu + 1.0) / 2.0) * log(1.0 + z_sq / (nu - 2.0));
        
        /* Kahan summation algorithm for numerical stability */
        double y = component - correction;
        double temp = loglik + y;
        correction = (temp - loglik) - y;
        loglik = temp;
    }
    
    return loglik;
}

/**
 * Computes log-likelihood for IGARCH model assuming Generalized Error Distribution (GED)
 *
 * @param data Input time series data (squared residuals)
 * @param variance Array of conditional variances
 * @param nu Shape parameter for GED
 * @param T Length of the time series
 * @return Log-likelihood value
 */
double compute_ged_loglikelihood(double* data, double* variance, double nu, mwSize T) {
    double loglik = 0.0, component;
    double correction = 0.0;  /* Kahan summation correction term */
    double lambda, logconst;
    mwSize t;
    
    /* Validate shape parameter */
    if (nu <= 0.0) {
        nu = 1.0;  /* Default to Laplace distribution */
    }
    
    /* Compute normalization factor for GED */
    lambda = sqrt(pow(2.0, -2.0/nu) * exp(lgamma(1.0/nu) - lgamma(3.0/nu)));
    
    /* Compute constant term for log-likelihood */
    logconst = log(nu / (2.0 * lambda * exp(lgamma(1.0/nu))));
    
    for (t = 0; t < T; t++) {
        /* Compute standardized residual */
        double z = sqrt(data[t] / variance[t]) * lambda;
        
        /* Compute log-likelihood component for GED */
        component = logconst - 0.5 * log(variance[t]) - 0.5 * pow(fabs(z), nu);
        
        /* Kahan summation algorithm for numerical stability */
        double y = component - correction;
        double temp = loglik + y;
        correction = (temp - loglik) - y;
        loglik = temp;
    }
    
    return loglik;
}

/**
 * Computes log-likelihood for IGARCH model assuming Hansen's Skewed t distribution
 *
 * @param data Input time series data (squared residuals)
 * @param variance Array of conditional variances
 * @param nu Degrees of freedom parameter for skewed t distribution
 * @param lambda Skewness parameter for skewed t distribution
 * @param T Length of the time series
 * @return Log-likelihood value
 */
double compute_skewt_loglikelihood(double* data, double* variance, double nu, double lambda, mwSize T) {
    double loglik = 0.0, component;
    double correction = 0.0;  /* Kahan summation correction term */
    double c, a, b, logconst1, logconst2;
    mwSize t;
    
    /* Validate parameters */
    if (nu <= 2.0) {
        nu = 2.01;  /* Ensure finite variance */
    }
    
    if (lambda < -1.0 || lambda > 1.0) {
        lambda = (lambda < 0) ? -0.99 : 0.99;  /* Constrain lambda */
    }
    
    /* Compute constants for skewed t distribution */
    c = exp(lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0)) / sqrt(M_PI * (nu - 2.0));
    a = 4.0 * lambda * c * ((nu - 2.0) / (nu - 1.0));
    b = sqrt(1.0 + 3.0 * lambda * lambda - a * a);
    
    /* Constants for log-likelihood computation */
    logconst1 = log(b) + 0.5 * log(nu / (nu - 2.0)) + lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * log(M_PI);
    logconst2 = log(b) + 0.5 * log(nu / (nu - 2.0)) + lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * log(M_PI);
    
    for (t = 0; t < T; t++) {
        /* Compute standardized residual */
        double z = sqrt(data[t] / variance[t]);
        
        /* Apply skewed t formula with appropriate branch */
        if (z < -a/b) {
            component = logconst1 - 0.5 * log(variance[t]) - 
                        ((nu + 1.0) / 2.0) * log(1.0 + (1.0 / (nu - 2.0)) * pow((b * z + a) / (1.0 - lambda), 2.0));
        } else {
            component = logconst2 - 0.5 * log(variance[t]) - 
                        ((nu + 1.0) / 2.0) * log(1.0 + (1.0 / (nu - 2.0)) * pow((b * z + a) / (1.0 + lambda), 2.0));
        }
        
        /* Kahan summation algorithm for numerical stability */
        double y = component - correction;
        double temp = loglik + y;
        correction = (temp - loglik) - y;
        loglik = temp;
    }
    
    return loglik;
}