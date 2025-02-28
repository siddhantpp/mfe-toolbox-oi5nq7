/**
 * @file mex_utils.h
 * @brief Utility functions for MEX interface integration in the MFE Toolbox
 *
 * This header provides common functionality for memory management, input/output validation,
 * error handling, and data conversion between MATLAB and C to support high-performance
 * numerical computations for financial econometric models.
 *
 * @version 4.0 (28-Oct-2009)
 */

#ifndef MEX_UTILS_H
#define MEX_UTILS_H

#include "mex.h"      /* MATLAB MEX API - MATLAB 4.0 compatible */
#include "matrix.h"   /* MATLAB MEX API - MATLAB 4.0 compatible */
#include <stdlib.h>   /* C Standard Library - C89/C90 */
#include <stdio.h>    /* C Standard Library - C89/C90 */
#include <stdarg.h>   /* For variable arguments in error/warning functions */

/**
 * Maximum length for error and warning messages
 */
#define MAX_ERROR_LENGTH 1024

/**
 * Debug mode flag (0 = disabled, 1 = enabled)
 * Set to 0 by default for production builds
 */
#define DEBUG_MODE 0

/**
 * @brief Validates the number and types of input arguments passed to a MEX function
 * 
 * Ensures that the expected number of input arguments are provided and valid
 *
 * @param nrhs Number of right-hand side arguments (inputs)
 * @param prhs Array of pointers to right-hand side arguments
 * @param expected_args Expected number of input arguments
 * @return 0 for success, non-zero for error
 */
int check_inputs(int nrhs, const mxArray **prhs, int expected_args);

/**
 * @brief Validates the number of output arguments expected from a MEX function
 * 
 * Ensures that the requested number of output arguments is compatible
 *
 * @param nlhs Number of left-hand side arguments (outputs)
 * @param expected_outputs Expected number of output arguments
 * @return 0 for success, non-zero for error
 */
int check_outputs(int nlhs, int expected_outputs);

/**
 * @brief Generates formatted error message and terminates MEX execution
 * 
 * Formats a message and calls mexErrMsgTxt to terminate MEX execution with error
 *
 * @param format Format string, similar to printf
 * @param ... Variable arguments for format string
 */
void mex_error(const char *format, ...);

/**
 * @brief Generates formatted warning message during MEX execution
 * 
 * Formats a message and calls mexWarnMsgTxt to issue a warning without termination
 *
 * @param format Format string, similar to printf
 * @param ... Variable arguments for format string
 */
void mex_warning(const char *format, ...);

/**
 * @brief Allocates memory with error checking and automatic error generation
 * 
 * Wraps malloc with error handling, terminating execution on allocation failure
 *
 * @param size Size of memory to allocate in bytes
 * @return Pointer to allocated memory
 */
void *safe_malloc(size_t size);

/**
 * @brief Safely frees memory and sets pointer to NULL to prevent dangling pointers
 * 
 * Frees memory if pointer is valid and sets the original pointer to NULL
 *
 * @param ptr Pointer to the pointer to be freed
 */
void safe_free(void **ptr);

/**
 * @brief Converts MATLAB matrix to C double array with validation
 * 
 * Extracts data from a MATLAB matrix into a newly allocated C double array
 *
 * @param mx_array Input MATLAB array
 * @param c_array Pointer to output C array (will be allocated)
 * @param rows Pointer to store number of rows
 * @param cols Pointer to store number of columns
 * @return 0 for success, non-zero for error
 */
int matlab_to_c_double(const mxArray *mx_array, double **c_array, mwSize *rows, mwSize *cols);

/**
 * @brief Converts MATLAB matrix to C integer array with validation
 * 
 * Extracts and converts data from a MATLAB matrix into a newly allocated C integer array
 *
 * @param mx_array Input MATLAB array
 * @param c_array Pointer to output C array (will be allocated)
 * @param rows Pointer to store number of rows
 * @param cols Pointer to store number of columns
 * @return 0 for success, non-zero for error
 */
int matlab_to_c_int(const mxArray *mx_array, int **c_array, mwSize *rows, mwSize *cols);

/**
 * @brief Extracts a scalar double value from a MATLAB array
 * 
 * Validates and extracts a single scalar value from a MATLAB array
 *
 * @param mx_array Input MATLAB array
 * @param scalar_value Pointer to store the extracted scalar value
 * @return 0 for success, non-zero for error
 */
int matlab_to_c_double_scalar(const mxArray *mx_array, double *scalar_value);

/**
 * @brief Creates a MATLAB matrix with specified dimensions and initializes to zero
 * 
 * Wrapper for mxCreateDoubleMatrix with error handling
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Pointer to created MATLAB matrix
 */
mxArray *create_matlab_matrix(mwSize rows, mwSize cols);

/**
 * @brief Copies C double array to MATLAB matrix
 * 
 * Creates a MATLAB matrix and copies data from a C array
 *
 * @param c_array Input C double array
 * @param rows Number of rows
 * @param cols Number of columns
 * @param mx_array Pointer to store the created MATLAB matrix
 * @return 0 for success, non-zero for error
 */
int c_to_matlab_double(const double *c_array, mwSize rows, mwSize cols, mxArray **mx_array);

/**
 * @brief Validates that a MATLAB array is numeric with expected dimensions
 * 
 * Checks if array is numeric and has the specified dimensions
 *
 * @param mx_array Input MATLAB array
 * @param expected_rows Expected number of rows (-1 for any)
 * @param expected_cols Expected number of columns (-1 for any)
 * @return 0 for success, non-zero for error
 */
int check_numeric_array(const mxArray *mx_array, mwSize expected_rows, mwSize expected_cols);

/**
 * @brief Conditionally prints debug messages when DEBUG_MODE is enabled
 * 
 * Provides verbose output during development without affecting production code
 *
 * @param format Format string, similar to printf
 * @param ... Variable arguments for format string
 */
void debug_print(const char *format, ...);

#endif /* MEX_UTILS_H */