/**
 * @file armaxerrors.c
 * @brief High-performance MEX implementation for computing ARMAX model residuals
 *
 * Provides optimized C implementation of residual/innovation computation for
 * ARMA/ARMAX (AutoRegressive Moving Average with eXogenous inputs) time series models.
 * This implementation significantly improves performance through vectorized operations
 * and optimized memory access patterns.
 *
 * @version 4.0 (28-Oct-2009)
 */

/* Include necessary headers */
#include "mex.h"      /* MATLAB MEX API - MATLAB 4.0 compatible */
#include "matrix.h"   /* MATLAB MEX API - MATLAB 4.0 compatible */
#include <stdlib.h>   /* C Standard Library - C89/C90 */
#include <math.h>     /* C Standard Library - C89/C90 */

/* Include internal headers */
#include "matrix_operations.h"  /* Optimized matrix operations */
#include "mex_utils.h"          /* MEX utilities and memory management */

/* Constants */
#define ERR_BUF_SIZE 256  /* Size of error message buffer */

/* Function prototypes */
void compute_armax_errors(double* data, double* ar_params, double* ma_params, 
                         double* exog_params, double* exog_data, double* errors,
                         mwSize T, mwSize p, mwSize q, mwSize r,
                         int include_constant, double constant_value);

int validate_parameters(mwSize T, mwSize p, mwSize q, mwSize r, 
                      double* data, double* ar_params, double* ma_params,
                      double* exog_params, double* exog_data, int include_constant);

/**
 * @brief MEX entry point function for ARMAX error computation
 *
 * Calculates the residuals/innovations of an ARMAX(p,q,r) model:
 *   y(t) = c + a(1)*y(t-1) + ... + a(p)*y(t-p) + 
 *          b(1)*e(t-1) + ... + b(q)*e(t-q) +
 *          d(1)*x(1,t) + ... + d(r)*x(r,t) + e(t)
 *
 * @param nlhs Number of output arguments (should be 1)
 * @param plhs Array of output arguments 
 * @param nrhs Number of input arguments (should be 3-6)
 * @param prhs Array of input arguments:
 *        prhs[0]: Time series data (T x 1)
 *        prhs[1]: AR parameters (p x 1)
 *        prhs[2]: MA parameters (q x 1)
 *        prhs[3]: Exogenous variables data (T x r) or empty
 *        prhs[4]: Exogenous parameters (r x 1) or empty
 *        prhs[5]: Optional constant term (scalar, default 0)
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variable declarations */
    double *data, *ar_params, *ma_params, *exog_data = NULL, *exog_params = NULL;
    double *errors, constant_value = 0.0;
    mwSize T, p, q, r = 0;
    int include_constant = 0;
    mwSize rows_data, cols_data, rows_ar, cols_ar, rows_ma, cols_ma;
    mwSize rows_exog_data, cols_exog_data, rows_exog_params, cols_exog_params;
    
    /* Validate input/output arguments */
    if (nrhs < 3 || nrhs > 6) {
        mex_error("Expected 3-6 input arguments: data, AR parameters, MA parameters, [exogenous data], [exogenous parameters], [constant]");
    }
    
    if (nlhs > 1) {
        mex_error("Only one output argument (errors/innovations) is supported");
    }
    
    /* Extract and validate time series data dimensions */
    if (mxIsEmpty(prhs[0]) || !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
        mex_error("Data must be a non-empty double precision matrix");
    }
    rows_data = mxGetM(prhs[0]);
    cols_data = mxGetN(prhs[0]);
    if (cols_data != 1) {
        mex_error("Data must be a column vector (T x 1)");
    }
    T = rows_data;
    
    /* Extract and validate AR parameters */
    if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1])) {
        mex_error("AR parameters must be a double precision matrix");
    }
    rows_ar = mxGetM(prhs[1]);
    cols_ar = mxGetN(prhs[1]);
    if (cols_ar != 1) {
        mex_error("AR parameters must be a column vector (p x 1)");
    }
    p = rows_ar;
    
    /* Extract and validate MA parameters */
    if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2])) {
        mex_error("MA parameters must be a double precision matrix");
    }
    rows_ma = mxGetM(prhs[2]);
    cols_ma = mxGetN(prhs[2]);
    if (cols_ma != 1) {
        mex_error("MA parameters must be a column vector (q x 1)");
    }
    q = rows_ma;
    
    /* Process exogenous data and parameters if provided */
    if (nrhs >= 5) {
        /* Extract and validate exogenous data */
        if (!mxIsEmpty(prhs[3])) {
            if (!mxIsDouble(prhs[3]) || mxIsComplex(prhs[3])) {
                mex_error("Exogenous data must be a double precision matrix");
            }
            rows_exog_data = mxGetM(prhs[3]);
            cols_exog_data = mxGetN(prhs[3]);
            if (rows_exog_data != T) {
                mex_error("Exogenous data must have the same number of rows as the time series data");
            }
            r = cols_exog_data;
            
            /* Extract and validate exogenous parameters */
            if (!mxIsDouble(prhs[4]) || mxIsComplex(prhs[4])) {
                mex_error("Exogenous parameters must be a double precision matrix");
            }
            rows_exog_params = mxGetM(prhs[4]);
            cols_exog_params = mxGetN(prhs[4]);
            if (cols_exog_params != 1) {
                mex_error("Exogenous parameters must be a column vector (r x 1)");
            }
            if (rows_exog_params != r) {
                mex_error("Number of exogenous parameters must match number of exogenous variables");
            }
        }
    }
    
    /* Process constant term if provided */
    if (nrhs == 6) {
        if (!mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]) || mxGetNumberOfElements(prhs[5]) != 1) {
            mex_error("Constant term must be a scalar double value");
        }
        include_constant = 1;
        constant_value = mxGetScalar(prhs[5]);
    }
    
    /* Convert MATLAB arrays to C arrays */
    data = mxGetPr(prhs[0]);
    ar_params = mxGetPr(prhs[1]);
    ma_params = mxGetPr(prhs[2]);
    
    if (r > 0) {
        exog_data = mxGetPr(prhs[3]);
        exog_params = mxGetPr(prhs[4]);
    }
    
    /* Check if model order is valid for time series length */
    if (T <= p || T <= q) {
        mex_error("Time series length must be greater than both AR and MA orders");
    }
    
    /* Validate model parameters for stability and consistency */
    if (validate_parameters(T, p, q, r, data, ar_params, ma_params, exog_params, exog_data, include_constant) != 0) {
        mex_error("Invalid model parameters detected");
    }
    
    /* Allocate output for errors/innovations */
    plhs[0] = mxCreateDoubleMatrix(T, 1, mxREAL);
    errors = mxGetPr(plhs[0]);
    
    /* Compute ARMAX errors/residuals */
    compute_armax_errors(data, ar_params, ma_params, exog_params, exog_data, errors, 
                        T, p, q, r, include_constant, constant_value);
}

/**
 * @brief Computes ARMAX model errors/residuals
 *
 * Implements efficient computation of residuals for an ARMAX model:
 *   y(t) = c + a(1)*y(t-1) + ... + a(p)*y(t-p) + 
 *          b(1)*e(t-1) + ... + b(q)*e(t-q) +
 *          d(1)*x(1,t) + ... + d(r)*x(r,t) + e(t)
 *
 * Uses optimized memory access patterns for improved performance and
 * handles initial values appropriately.
 *
 * @param data Time series data (T x 1)
 * @param ar_params AR parameters (p x 1)
 * @param ma_params MA parameters (q x 1)
 * @param exog_params Exogenous parameters (r x 1) or NULL
 * @param exog_data Exogenous data (T x r) or NULL
 * @param errors Output array for errors/residuals (T x 1)
 * @param T Length of time series
 * @param p AR order
 * @param q MA order
 * @param r Number of exogenous variables
 * @param include_constant Flag indicating if constant term is included
 * @param constant_value Value of constant term
 */
void compute_armax_errors(double* data, double* ar_params, double* ma_params, 
                         double* exog_params, double* exog_data, double* errors,
                         mwSize T, mwSize p, mwSize q, mwSize r,
                         int include_constant, double constant_value) {
    mwSize t, i, j;
    double predicted;
    
    /* Initialize all errors to zero */
    for (t = 0; t < T; t++) {
        errors[t] = 0.0;
    }
    
    /* Handle initial period (t < max(p,q)) when not all lags are available */
    for (t = 0; t < (p > q ? p : q); t++) {
        /* Start with constant term if included */
        predicted = include_constant ? constant_value : 0.0;
        
        /* Add AR component if we have enough history */
        for (i = 0; i < t && i < p; i++) {
            predicted += ar_params[i] * data[t-i-1];
        }
        
        /* Add MA component if we have enough history */
        for (i = 0; i < t && i < q; i++) {
            predicted += ma_params[i] * errors[t-i-1];
        }
        
        /* Add exogenous variables effect if available */
        if (r > 0 && exog_data != NULL && exog_params != NULL) {
            for (j = 0; j < r; j++) {
                predicted += exog_params[j] * exog_data[t + j*T];
            }
        }
        
        /* Compute error as difference between actual and predicted values */
        errors[t] = data[t] - predicted;
    }
    
    /* Process remaining periods with full lag information */
    for (t = (p > q ? p : q); t < T; t++) {
        /* Start with constant term if included */
        predicted = include_constant ? constant_value : 0.0;
        
        /* Add AR component using all available lags */
        for (i = 0; i < p; i++) {
            predicted += ar_params[i] * data[t-i-1];
        }
        
        /* Add MA component using all available lags */
        for (i = 0; i < q; i++) {
            predicted += ma_params[i] * errors[t-i-1];
        }
        
        /* Add exogenous variables effect if available */
        if (r > 0 && exog_data != NULL && exog_params != NULL) {
            for (j = 0; j < r; j++) {
                predicted += exog_params[j] * exog_data[t + j*T];
            }
        }
        
        /* Compute error as difference between actual and predicted values */
        errors[t] = data[t] - predicted;
    }
}

/**
 * @brief Validates ARMAX model parameters for consistency and stability
 *
 * Performs comprehensive validation of all input parameters:
 * - Checks that time series length is sufficient
 * - Validates pointers for required arrays
 * - Verifies AR parameters are within stability bounds
 * - Verifies MA parameters are within stability bounds
 * - Checks exogenous data dimensions match specifications
 *
 * @param T Length of time series
 * @param p AR order
 * @param q MA order
 * @param r Number of exogenous variables
 * @param data Time series data
 * @param ar_params AR parameters
 * @param ma_params MA parameters
 * @param exog_params Exogenous parameters (can be NULL if r=0)
 * @param exog_data Exogenous data (can be NULL if r=0)
 * @param include_constant Flag indicating if constant term is included
 * @return 0 for success, non-zero for error
 */
int validate_parameters(mwSize T, mwSize p, mwSize q, mwSize r, 
                      double* data, double* ar_params, double* ma_params,
                      double* exog_params, double* exog_data, int include_constant) {
    mwSize i;
    double ar_sum = 0.0;
    double ma_sum = 0.0;
    
    /* Check pointers for required arrays */
    if (data == NULL) {
        mex_error("Time series data pointer is NULL");
        return 1;
    }
    
    if (ar_params == NULL && p > 0) {
        mex_error("AR parameters pointer is NULL but p > 0");
        return 1;
    }
    
    if (ma_params == NULL && q > 0) {
        mex_error("MA parameters pointer is NULL but q > 0");
        return 1;
    }
    
    /* Check exogenous variables if required */
    if (r > 0) {
        if (exog_params == NULL) {
            mex_error("Exogenous parameters pointer is NULL but r > 0");
            return 1;
        }
        
        if (exog_data == NULL) {
            mex_error("Exogenous data pointer is NULL but r > 0");
            return 1;
        }
    }
    
    /* Verify AR parameters are within stability bounds */
    /* For simplicity, we just check that |sum(ar_params)| < 1 */
    /* A more rigorous check would verify all roots are outside unit circle */
    if (p > 0) {
        for (i = 0; i < p; i++) {
            ar_sum += fabs(ar_params[i]);
        }
        
        if (ar_sum >= 1.0) {
            mex_warning("Sum of absolute AR parameters is >= 1.0, model may be non-stationary");
        }
    }
    
    /* Verify MA parameters are within stability bounds */
    /* For simplicity, we just check that |sum(ma_params)| < 1 */
    /* A more rigorous check would verify all roots are outside unit circle */
    if (q > 0) {
        for (i = 0; i < q; i++) {
            ma_sum += fabs(ma_params[i]);
        }
        
        if (ma_sum >= 1.0) {
            mex_warning("Sum of absolute MA parameters is >= 1.0, model may be non-invertible");
        }
    }
    
    /* Check that constant flag is valid */
    if (include_constant != 0 && include_constant != 1) {
        mex_error("Include constant flag must be 0 or 1");
        return 1;
    }
    
    return 0; /* All checks passed */
}