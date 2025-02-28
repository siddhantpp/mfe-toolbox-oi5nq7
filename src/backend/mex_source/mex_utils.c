/**
 * @file mex_utils.c
 * @brief Implementation of utility functions for MEX interface in the MFE Toolbox
 *
 * This file provides common functionality for memory management, matrix operations,
 * and error handling to support high-performance C-based computations for financial
 * econometric models.
 *
 * @version 4.0 (28-Oct-2009)
 */

#include "mex_utils.h"
#include "matrix_operations.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * @brief Validates input arguments to MEX functions for proper dimensions and types
 *
 * @param nrhs Number of right-hand side arguments (inputs)
 * @param prhs Array of pointers to right-hand side arguments
 * @param expected_args Expected number of input arguments
 * @param dimensions Array of expected dimensions for each input (NULL for any)
 * @param classIDs Array of expected class IDs for each input (NULL for any)
 * @return True if inputs are valid, false otherwise
 */
bool checkInputs(int nrhs, const mxArray** prhs, int expected_args, int* dimensions, mxClassID* classIDs) {
    // Check number of input arguments
    if (nrhs != expected_args) {
        mexPrintError("Expected %d input arguments, but got %d", expected_args, nrhs);
        return false;
    }

    // Check dimensions and class types if specified
    if (dimensions != NULL && classIDs != NULL) {
        for (int i = 0; i < expected_args; i++) {
            // Check class ID if specified (mxUNKNOWN_CLASS means any class is accepted)
            if (classIDs[i] != mxUNKNOWN_CLASS && mxGetClassID(prhs[i]) != classIDs[i]) {
                mexPrintError("Input argument %d has invalid class type", i + 1);
                return false;
            }

            // Check dimensions if specified
            // Each input has two dimension values: rows (even indices) and columns (odd indices)
            int expected_rows = dimensions[i * 2];
            int expected_cols = dimensions[i * 2 + 1];
            
            // -1 indicates any number of rows/columns is acceptable
            if (expected_rows != -1 && (int)mxGetM(prhs[i]) != expected_rows) {
                mexPrintError("Input argument %d has %d rows, expected %d", 
                              i + 1, (int)mxGetM(prhs[i]), expected_rows);
                return false;
            }
            
            if (expected_cols != -1 && (int)mxGetN(prhs[i]) != expected_cols) {
                mexPrintError("Input argument %d has %d columns, expected %d", 
                              i + 1, (int)mxGetN(prhs[i]), expected_cols);
                return false;
            }
        }
    }

    return true;
}

/**
 * @brief Creates an output matrix with specified dimensions and class type
 *
 * @param m Number of rows
 * @param n Number of columns
 * @param classID Class ID for the output matrix
 * @return Pointer to the created matrix
 */
mxArray* createOutputMatrix(int m, int n, mxClassID classID) {
    mxArray* output = NULL;

    // Create matrix based on class ID
    switch (classID) {
        case mxDOUBLE_CLASS:
            output = mxCreateDoubleMatrix((mwSize)m, (mwSize)n, mxREAL);
            break;
        case mxSINGLE_CLASS:
            output = mxCreateNumericMatrix((mwSize)m, (mwSize)n, mxSINGLE_CLASS, mxREAL);
            break;
        case mxINT32_CLASS:
            output = mxCreateNumericMatrix((mwSize)m, (mwSize)n, mxINT32_CLASS, mxREAL);
            break;
        case mxINT16_CLASS:
            output = mxCreateNumericMatrix((mwSize)m, (mwSize)n, mxINT16_CLASS, mxREAL);
            break;
        case mxINT8_CLASS:
            output = mxCreateNumericMatrix((mwSize)m, (mwSize)n, mxINT8_CLASS, mxREAL);
            break;
        case mxUINT32_CLASS:
            output = mxCreateNumericMatrix((mwSize)m, (mwSize)n, mxUINT32_CLASS, mxREAL);
            break;
        case mxUINT16_CLASS:
            output = mxCreateNumericMatrix((mwSize)m, (mwSize)n, mxUINT16_CLASS, mxREAL);
            break;
        case mxUINT8_CLASS:
            output = mxCreateNumericMatrix((mwSize)m, (mwSize)n, mxUINT8_CLASS, mxREAL);
            break;
        case mxLOGICAL_CLASS:
            output = mxCreateLogicalMatrix((mwSize)m, (mwSize)n);
            break;
        default:
            mexPrintError("Unsupported class ID for output matrix");
            return NULL;
    }

    // Check if matrix creation was successful
    validateMemoryAllocation(output, "output matrix");

    return output;
}

/**
 * @brief Copies data from an mxArray to a double array
 *
 * @param input Pointer to input mxArray
 * @param output Pointer to output double array (must be pre-allocated)
 * @param length Number of elements to copy
 */
void copyMxArrayToDouble(const mxArray* input, double* output, int length) {
    // Verify input parameters
    if (input == NULL || output == NULL || length <= 0) {
        mexPrintError("Invalid parameters for copyMxArrayToDouble");
        return;
    }

    // Verify that input is numeric
    if (!mxIsNumeric(input)) {
        mexPrintError("Input array must be numeric for copyMxArrayToDouble");
        return;
    }

    // Get pointer to input data
    double* inputData = mxGetPr(input);
    
    // Check if we got valid data
    if (inputData == NULL) {
        mexPrintError("Failed to access data from input mxArray");
        return;
    }

    // Verify that length is within bounds
    mwSize totalElements = mxGetM(input) * mxGetN(input);
    if ((mwSize)length > totalElements) {
        mexPrintError("Requested copy length (%d) exceeds array size (%d)", 
                     length, (int)totalElements);
        return;
    }

    // Copy data to output array
    memcpy(output, inputData, length * sizeof(double));
}

/**
 * @brief Copies data from a double array to an mxArray
 *
 * @param input Pointer to input double array
 * @param output Pointer to output mxArray (must be pre-allocated)
 * @param length Number of elements to copy
 */
void copyDoubleToMxArray(double* input, mxArray* output, int length) {
    // Verify input parameters
    if (input == NULL || output == NULL || length <= 0) {
        mexPrintError("Invalid parameters for copyDoubleToMxArray");
        return;
    }

    // Verify that output is of double class
    if (mxGetClassID(output) != mxDOUBLE_CLASS) {
        mexPrintError("Output mxArray must be of double type for copyDoubleToMxArray");
        return;
    }

    // Get pointer to output data
    double* outputData = mxGetPr(output);
    
    // Check if we got valid data
    if (outputData == NULL) {
        mexPrintError("Failed to access data from output mxArray");
        return;
    }

    // Verify that length is within bounds
    mwSize totalElements = mxGetM(output) * mxGetN(output);
    if ((mwSize)length > totalElements) {
        mexPrintError("Requested copy length (%d) exceeds array size (%d)", 
                     length, (int)totalElements);
        return;
    }

    // Copy data to output mxArray
    memcpy(outputData, input, length * sizeof(double));
}

/**
 * @brief Formats and prints error messages in MEX functions
 *
 * @param error_message Error message to print
 */
void mexPrintError(const char* error_message, ...) {
    char buffer[MAX_ERROR_LENGTH];
    va_list args;
    
    // Format error message with "Error: " prefix
    va_start(args, error_message);
    vsnprintf(buffer, MAX_ERROR_LENGTH - 8, error_message, args);
    va_end(args);
    
    // Prepend "Error: " to the message
    char final_message[MAX_ERROR_LENGTH];
    snprintf(final_message, MAX_ERROR_LENGTH, "Error: %s", buffer);
    
    // Print error message and terminate MEX function
    mexErrMsgTxt(final_message);
}

/**
 * @brief Formats and prints warning messages in MEX functions
 *
 * @param warning_message Warning message to print
 */
void mexPrintWarning(const char* warning_message, ...) {
    char buffer[MAX_ERROR_LENGTH];
    va_list args;
    
    // Format warning message
    va_start(args, warning_message);
    vsnprintf(buffer, MAX_ERROR_LENGTH - 10, warning_message, args);
    va_end(args);
    
    // Prepend "Warning: " to the message
    char final_message[MAX_ERROR_LENGTH];
    snprintf(final_message, MAX_ERROR_LENGTH, "Warning: %s", buffer);
    
    // Print warning message without terminating MEX function
    mexWarnMsgTxt(final_message);
}

/**
 * @brief Checks if memory allocation was successful and throws error if not
 *
 * @param ptr Pointer to check for NULL
 * @param variable_name Name of the variable for error message
 */
void validateMemoryAllocation(void* ptr, const char* variable_name) {
    if (ptr == NULL) {
        char error_message[MAX_ERROR_LENGTH];
        snprintf(error_message, MAX_ERROR_LENGTH, 
                 "Memory allocation failed for %s", variable_name);
        mexErrMsgTxt(error_message);
    }
}

/**
 * @brief Extracts a scalar value from an mxArray
 *
 * @param array Input mxArray containing scalar value
 * @return Scalar value extracted from the mxArray
 */
double mxArrayToScalar(const mxArray* array) {
    // Check if input is NULL
    if (array == NULL) {
        mexPrintError("NULL pointer passed to mxArrayToScalar");
        return 0.0;
    }

    // Check if input is a scalar (1x1 matrix)
    if (mxGetNumberOfElements(array) != 1) {
        mexPrintError("Input is not a scalar value (found %d elements)", 
                     (int)mxGetNumberOfElements(array));
        return 0.0;
    }

    // Check if input is numeric
    if (!mxIsNumeric(array)) {
        mexPrintError("Input is not a numeric value");
        return 0.0;
    }

    // Get scalar value (handles type conversion automatically)
    return mxGetScalar(array);
}

/**
 * @brief Releases allocated memory for multiple pointers
 *
 * @param num_pointers Number of pointers to clean up
 * @param pointers Array of pointers to free
 */
void cleanupMemory(int num_pointers, void** pointers) {
    if (pointers == NULL) {
        return;
    }

    for (int i = 0; i < num_pointers; i++) {
        if (pointers[i] != NULL) {
            free(pointers[i]);
            pointers[i] = NULL;
        }
    }
}

/* 
 * Implementations of functions declared in the header but not in the JSON spec 
 * These maintain compatibility with the header file
 */

/**
 * @brief Validates the number and types of input arguments
 */
int check_inputs(int nrhs, const mxArray **prhs, int expected_args) {
    if (nrhs != expected_args) {
        mex_error("Expected %d input arguments, but got %d", expected_args, nrhs);
        return 1; // Error
    }
    return 0; // Success
}

/**
 * @brief Validates the number of output arguments
 */
int check_outputs(int nlhs, int expected_outputs) {
    if (nlhs != expected_outputs) {
        mex_error("Expected %d output arguments, but got %d", expected_outputs, nlhs);
        return 1; // Error
    }
    return 0; // Success
}

/**
 * @brief Generates formatted error message
 */
void mex_error(const char *format, ...) {
    char buffer[MAX_ERROR_LENGTH];
    va_list args;
    
    va_start(args, format);
    vsnprintf(buffer, MAX_ERROR_LENGTH, format, args);
    va_end(args);
    
    mexErrMsgTxt(buffer);
}

/**
 * @brief Generates formatted warning message
 */
void mex_warning(const char *format, ...) {
    char buffer[MAX_ERROR_LENGTH];
    va_list args;
    
    va_start(args, format);
    vsnprintf(buffer, MAX_ERROR_LENGTH, format, args);
    va_end(args);
    
    mexWarnMsgTxt(buffer);
}

/**
 * @brief Allocates memory with error checking
 */
void *safe_malloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        mex_error("Memory allocation failed (requested %zu bytes)", size);
    }
    return ptr;
}

/**
 * @brief Safely frees memory and sets pointer to NULL
 */
void safe_free(void **ptr) {
    if (ptr != NULL && *ptr != NULL) {
        free(*ptr);
        *ptr = NULL;
    }
}

/**
 * @brief Converts MATLAB matrix to C double array
 */
int matlab_to_c_double(const mxArray *mx_array, double **c_array, mwSize *rows, mwSize *cols) {
    if (mx_array == NULL || c_array == NULL || rows == NULL || cols == NULL) {
        mex_error("Invalid parameters for matlab_to_c_double");
        return 1;
    }
    
    // Get dimensions
    *rows = mxGetM(mx_array);
    *cols = mxGetN(mx_array);
    mwSize total_elements = (*rows) * (*cols);
    
    // Validate input is numeric
    if (!mxIsNumeric(mx_array)) {
        mex_error("Input must be numeric for matlab_to_c_double");
        return 1;
    }
    
    // Allocate memory
    *c_array = (double*)safe_malloc(total_elements * sizeof(double));
    
    // Copy data
    double *mx_data = mxGetPr(mx_array);
    if (mx_data == NULL) {
        mex_error("Failed to access MATLAB array data");
        safe_free((void**)c_array);
        return 1;
    }
    
    memcpy(*c_array, mx_data, total_elements * sizeof(double));
    return 0;
}

/**
 * @brief Converts MATLAB matrix to C integer array
 */
int matlab_to_c_int(const mxArray *mx_array, int **c_array, mwSize *rows, mwSize *cols) {
    if (mx_array == NULL || c_array == NULL || rows == NULL || cols == NULL) {
        mex_error("Invalid parameters for matlab_to_c_int");
        return 1;
    }
    
    // Get dimensions
    *rows = mxGetM(mx_array);
    *cols = mxGetN(mx_array);
    mwSize total_elements = (*rows) * (*cols);
    
    // Validate input is numeric
    if (!mxIsNumeric(mx_array)) {
        mex_error("Input must be numeric for matlab_to_c_int");
        return 1;
    }
    
    // Allocate memory
    *c_array = (int*)safe_malloc(total_elements * sizeof(int));
    
    // Copy and convert data
    double *mx_data = mxGetPr(mx_array);
    if (mx_data == NULL) {
        mex_error("Failed to access MATLAB array data");
        safe_free((void**)c_array);
        return 1;
    }
    
    for (mwSize i = 0; i < total_elements; i++) {
        (*c_array)[i] = (int)mx_data[i];
    }
    
    return 0;
}

/**
 * @brief Extracts a scalar double value from a MATLAB array
 */
int matlab_to_c_double_scalar(const mxArray *mx_array, double *scalar_value) {
    if (mx_array == NULL || scalar_value == NULL) {
        mex_error("Invalid parameters for matlab_to_c_double_scalar");
        return 1;
    }
    
    // Check if input is a scalar
    if (mxGetNumberOfElements(mx_array) != 1) {
        mex_error("Input is not a scalar value (found %d elements)", 
                 (int)mxGetNumberOfElements(mx_array));
        return 1;
    }
    
    // Check if input is numeric
    if (!mxIsNumeric(mx_array)) {
        mex_error("Input is not numeric");
        return 1;
    }
    
    // Get scalar value
    *scalar_value = mxGetScalar(mx_array);
    return 0;
}

/**
 * @brief Creates a MATLAB matrix with specified dimensions
 */
mxArray *create_matlab_matrix(mwSize rows, mwSize cols) {
    mxArray *result = mxCreateDoubleMatrix(rows, cols, mxREAL);
    if (result == NULL) {
        mex_error("Failed to create MATLAB matrix of size %d x %d", 
                 (int)rows, (int)cols);
    }
    return result;
}

/**
 * @brief Copies C double array to MATLAB matrix
 */
int c_to_matlab_double(const double *c_array, mwSize rows, mwSize cols, mxArray **mx_array) {
    if (c_array == NULL || mx_array == NULL) {
        mex_error("Invalid parameters for c_to_matlab_double");
        return 1;
    }
    
    // Create MATLAB matrix
    *mx_array = create_matlab_matrix(rows, cols);
    if (*mx_array == NULL) {
        return 1;
    }
    
    // Copy data
    double *mx_data = mxGetPr(*mx_array);
    if (mx_data == NULL) {
        mex_error("Failed to access MATLAB array data");
        mxDestroyArray(*mx_array);
        *mx_array = NULL;
        return 1;
    }
    
    memcpy(mx_data, c_array, rows * cols * sizeof(double));
    return 0;
}

/**
 * @brief Validates that a MATLAB array is numeric with expected dimensions
 */
int check_numeric_array(const mxArray *mx_array, mwSize expected_rows, mwSize expected_cols) {
    if (mx_array == NULL) {
        mex_error("NULL pointer passed to check_numeric_array");
        return 1;
    }
    
    // Check if numeric
    if (!mxIsNumeric(mx_array)) {
        mex_error("Input is not numeric");
        return 1;
    }
    
    // Get dimensions
    mwSize rows = mxGetM(mx_array);
    mwSize cols = mxGetN(mx_array);
    
    // Check dimensions if expected_rows/cols != -1
    if (expected_rows != (mwSize)-1 && rows != expected_rows) {
        mex_error("Expected %d rows, got %d", (int)expected_rows, (int)rows);
        return 1;
    }
    
    if (expected_cols != (mwSize)-1 && cols != expected_cols) {
        mex_error("Expected %d columns, got %d", (int)expected_cols, (int)cols);
        return 1;
    }
    
    return 0;
}

/**
 * @brief Conditionally prints debug messages
 */
void debug_print(const char *format, ...) {
    #if DEBUG_MODE
    va_list args;
    va_start(args, format);
    
    char buffer[MAX_ERROR_LENGTH];
    vsnprintf(buffer, MAX_ERROR_LENGTH, format, args);
    
    mexPrintf("DEBUG: %s\n", buffer);
    
    va_end(args);
    #endif
}