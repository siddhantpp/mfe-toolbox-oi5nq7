/**
 * @file matrix_operations.c
 * @brief Implementation of optimized matrix operations for high-performance financial econometric computations
 *
 * This file provides critical matrix manipulation functions that serve as the
 * foundation for volatility models and time series analysis in the MFE Toolbox.
 * Implementations focus on numerical stability, memory efficiency, and cache
 * optimization for high-performance financial econometric applications.
 *
 * @version 4.0 (28-Oct-2009)
 */

/* Include headers */
#include "matrix_operations.h"
#include "mex_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* Define constants for matrix operations */
#define MATRIX_EPSILON (DBL_EPSILON * 100.0)
#define BLOCK_SIZE 64

/**
 * @brief Creates a deep copy of a source matrix into a target matrix with error checking
 *
 * @param source Pointer to source matrix data
 * @param target Pointer to target matrix data (must be pre-allocated)
 * @param rows Number of rows in the matrices
 * @param cols Number of columns in the matrices
 */
void copy_matrix(const double *source, double *target, mwSize rows, mwSize cols) {
    mwSize i;
    
    /* Validate inputs */
    if (source == NULL || target == NULL) {
        mex_error("copy_matrix: NULL pointer provided for source or target");
        return;
    }
    
    if (rows <= 0 || cols <= 0) {
        mex_error("copy_matrix: Invalid dimensions (%d x %d)", rows, cols);
        return;
    }
    
    /* Check if source and target are the same (no copy needed) */
    if (source == target) {
        return;
    }
    
    /* Copy each row for optimal cache performance */
    for (i = 0; i < rows; i++) {
        memcpy(target + i * cols, source + i * cols, cols * sizeof(double));
    }
}

/**
 * @brief Performs optimized matrix multiplication between two matrices (A * B)
 *
 * Uses cache-friendly algorithms and blocking techniques for improved performance.
 * Validates input dimensions for multiplication compatibility.
 *
 * @param matrix_a Pointer to first matrix (A)
 * @param matrix_b Pointer to second matrix (B)
 * @param result Pointer to result matrix (must be pre-allocated with size [rows_a × cols_b])
 * @param rows_a Number of rows in matrix A
 * @param cols_a Number of columns in matrix A (must equal rows of matrix B)
 * @param cols_b Number of columns in matrix B
 */
void matrix_multiply(const double *matrix_a, const double *matrix_b, double *result,
                     mwSize rows_a, mwSize cols_a, mwSize cols_b) {
    mwSize i, j, k;
    mwSize ii, jj, kk;
    double sum;
    
    /* Validate inputs */
    if (matrix_a == NULL || matrix_b == NULL || result == NULL) {
        mex_error("matrix_multiply: NULL pointer provided for input matrices");
        return;
    }
    
    if (rows_a <= 0 || cols_a <= 0 || cols_b <= 0) {
        mex_error("matrix_multiply: Invalid dimensions (%d x %d) * (%d x %d)", 
                  rows_a, cols_a, cols_a, cols_b);
        return;
    }
    
    /* Initialize result matrix to zeros */
    memset(result, 0, rows_a * cols_b * sizeof(double));
    
    /* Blocked matrix multiplication for better cache utilization */
    for (ii = 0; ii < rows_a; ii += BLOCK_SIZE) {
        for (jj = 0; jj < cols_b; jj += BLOCK_SIZE) {
            for (kk = 0; kk < cols_a; kk += BLOCK_SIZE) {
                /* Process blocks of matrices */
                for (i = ii; i < (ii + BLOCK_SIZE < rows_a ? ii + BLOCK_SIZE : rows_a); i++) {
                    for (j = jj; j < (jj + BLOCK_SIZE < cols_b ? jj + BLOCK_SIZE : cols_b); j++) {
                        sum = result[i * cols_b + j];
                        for (k = kk; k < (kk + BLOCK_SIZE < cols_a ? kk + BLOCK_SIZE : cols_a); k++) {
                            sum += matrix_a[i * cols_a + k] * matrix_b[k * cols_b + j];
                        }
                        result[i * cols_b + j] = sum;
                    }
                }
            }
        }
    }
}

/**
 * @brief Transposes a matrix into a new result matrix efficiently
 *
 * Implements a blocked algorithm for better cache utilization, especially for large matrices.
 * Handles both square and non-square matrices appropriately.
 *
 * @param matrix Pointer to source matrix data
 * @param result Pointer to result matrix data (must be pre-allocated with size [cols × rows])
 * @param rows Number of rows in source matrix
 * @param cols Number of columns in source matrix
 */
void transpose_matrix(const double *matrix, double *result, mwSize rows, mwSize cols) {
    mwSize i, j, ii, jj;
    
    /* Validate inputs */
    if (matrix == NULL || result == NULL) {
        mex_error("transpose_matrix: NULL pointer provided");
        return;
    }
    
    if (rows <= 0 || cols <= 0) {
        mex_error("transpose_matrix: Invalid dimensions (%d x %d)", rows, cols);
        return;
    }
    
    /* For small matrices, use simple algorithm */
    if (rows < BLOCK_SIZE && cols < BLOCK_SIZE) {
        for (i = 0; i < rows; i++) {
            for (j = 0; j < cols; j++) {
                result[j * rows + i] = matrix[i * cols + j];
            }
        }
        return;
    }
    
    /* For larger matrices, use blocked algorithm for better cache behavior */
    for (ii = 0; ii < rows; ii += BLOCK_SIZE) {
        for (jj = 0; jj < cols; jj += BLOCK_SIZE) {
            /* Process a block of the matrix */
            for (i = ii; i < (ii + BLOCK_SIZE < rows ? ii + BLOCK_SIZE : rows); i++) {
                for (j = jj; j < (jj + BLOCK_SIZE < cols ? jj + BLOCK_SIZE : cols); j++) {
                    result[j * rows + i] = matrix[i * cols + j];
                }
            }
        }
    }
}

/**
 * @brief Adds two matrices element-wise with bounds checking
 *
 * @param matrix_a Pointer to first matrix (A)
 * @param matrix_b Pointer to second matrix (B)
 * @param result Pointer to result matrix (must be pre-allocated with size [rows × cols])
 * @param rows Number of rows in the matrices
 * @param cols Number of columns in the matrices
 */
void matrix_addition(const double *matrix_a, const double *matrix_b, double *result,
                     mwSize rows, mwSize cols) {
    mwSize i, j, idx;
    mwSize total_elems = rows * cols;
    
    /* Validate inputs */
    if (matrix_a == NULL || matrix_b == NULL || result == NULL) {
        mex_error("matrix_addition: NULL pointer provided");
        return;
    }
    
    if (rows <= 0 || cols <= 0) {
        mex_error("matrix_addition: Invalid dimensions (%d x %d)", rows, cols);
        return;
    }
    
    /* Process elements in cache-friendly order */
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            idx = i * cols + j;
            result[idx] = matrix_a[idx] + matrix_b[idx];
        }
    }
}

/**
 * @brief Subtracts second matrix from first matrix element-wise with validation
 *
 * @param matrix_a Pointer to first matrix (A)
 * @param matrix_b Pointer to second matrix (B)
 * @param result Pointer to result matrix (must be pre-allocated with size [rows × cols])
 * @param rows Number of rows in the matrices
 * @param cols Number of columns in the matrices
 */
void matrix_subtraction(const double *matrix_a, const double *matrix_b, double *result,
                       mwSize rows, mwSize cols) {
    mwSize i, j, idx;
    mwSize total_elems = rows * cols;
    
    /* Validate inputs */
    if (matrix_a == NULL || matrix_b == NULL || result == NULL) {
        mex_error("matrix_subtraction: NULL pointer provided");
        return;
    }
    
    if (rows <= 0 || cols <= 0) {
        mex_error("matrix_subtraction: Invalid dimensions (%d x %d)", rows, cols);
        return;
    }
    
    /* Process elements in cache-friendly order */
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            idx = i * cols + j;
            result[idx] = matrix_a[idx] - matrix_b[idx];
        }
    }
}

/**
 * @brief Performs element-wise (Hadamard) multiplication of two matrices
 *
 * Multiples corresponding elements in both matrices and stores in result matrix.
 * Uses optimized memory access patterns for improved performance.
 *
 * @param matrix_a Pointer to first matrix (A)
 * @param matrix_b Pointer to second matrix (B)
 * @param result Pointer to result matrix (must be pre-allocated with size [rows × cols])
 * @param rows Number of rows in the matrices
 * @param cols Number of columns in the matrices
 */
void matrix_element_multiply(const double *matrix_a, const double *matrix_b, double *result,
                            mwSize rows, mwSize cols) {
    mwSize i, j, idx;
    mwSize total_elems = rows * cols;
    
    /* Validate inputs */
    if (matrix_a == NULL || matrix_b == NULL || result == NULL) {
        mex_error("matrix_element_multiply: NULL pointer provided");
        return;
    }
    
    if (rows <= 0 || cols <= 0) {
        mex_error("matrix_element_multiply: Invalid dimensions (%d x %d)", rows, cols);
        return;
    }
    
    /* Process elements in cache-friendly order */
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            idx = i * cols + j;
            result[idx] = matrix_a[idx] * matrix_b[idx];
        }
    }
}

/**
 * @brief Multiplies each element of a matrix by a scalar value efficiently
 *
 * @param matrix Pointer to source matrix data
 * @param result Pointer to result matrix data (must be pre-allocated with size [rows × cols])
 * @param scalar The scalar value to multiply by
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 */
void matrix_scalar_multiply(const double *matrix, double *result, double scalar,
                           mwSize rows, mwSize cols) {
    mwSize i, j, idx;
    mwSize total_elems = rows * cols;
    
    /* Validate inputs */
    if (matrix == NULL || result == NULL) {
        mex_error("matrix_scalar_multiply: NULL pointer provided");
        return;
    }
    
    if (rows <= 0 || cols <= 0) {
        mex_error("matrix_scalar_multiply: Invalid dimensions (%d x %d)", rows, cols);
        return;
    }
    
    /* Handle special case: scalar = 1.0 (just copy the matrix) */
    if (fabs(scalar - 1.0) < MATRIX_EPSILON) {
        copy_matrix(matrix, result, rows, cols);
        return;
    }
    
    /* Handle special case: scalar = 0.0 (set all elements to zero) */
    if (fabs(scalar) < MATRIX_EPSILON) {
        memset(result, 0, total_elems * sizeof(double));
        return;
    }
    
    /* Process elements in cache-friendly order */
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            idx = i * cols + j;
            result[idx] = matrix[idx] * scalar;
        }
    }
}

/**
 * @brief Computes the sum of all elements in a matrix with numerical stability
 *
 * Uses Kahan summation algorithm for improved numerical accuracy in
 * floating-point operations to minimize accumulated errors.
 *
 * @param matrix Pointer to matrix data
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 * @return Sum of all matrix elements
 */
double matrix_sum(const double *matrix, mwSize rows, mwSize cols) {
    mwSize i, j, idx;
    double sum = 0.0;
    double c = 0.0;       /* Compensation variable for Kahan summation */
    double y, t;          /* Temporary variables for Kahan summation */
    
    /* Validate inputs */
    if (matrix == NULL) {
        mex_error("matrix_sum: NULL pointer provided");
        return 0.0;
    }
    
    if (rows <= 0 || cols <= 0) {
        mex_error("matrix_sum: Invalid dimensions (%d x %d)", rows, cols);
        return 0.0;
    }
    
    /* Use Kahan summation for better numerical stability */
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            idx = i * cols + j;
            y = matrix[idx] - c;
            t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
    }
    
    return sum;
}

/**
 * @brief Computes the mean of each row in a matrix with improved numerical stability
 *
 * Uses Kahan summation for computing accurate row sums, then divides by column count.
 *
 * @param matrix Pointer to matrix data
 * @param result Pointer to result array (must be pre-allocated with size [rows])
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 */
void matrix_row_mean(const double *matrix, double *result, mwSize rows, mwSize cols) {
    mwSize i, j, idx;
    double sum, c;     /* Sum and compensation for Kahan summation */
    double y, t;       /* Temporary variables for Kahan summation */
    
    /* Validate inputs */
    if (matrix == NULL || result == NULL) {
        mex_error("matrix_row_mean: NULL pointer provided");
        return;
    }
    
    if (rows <= 0 || cols <= 0) {
        mex_error("matrix_row_mean: Invalid dimensions (%d x %d)", rows, cols);
        return;
    }
    
    /* Special case: empty matrix or divide by zero */
    if (cols == 0) {
        memset(result, 0, rows * sizeof(double));
        return;
    }
    
    /* Compute mean for each row with Kahan summation */
    for (i = 0; i < rows; i++) {
        sum = 0.0;
        c = 0.0;
        
        /* Compute row sum using Kahan summation for numerical stability */
        for (j = 0; j < cols; j++) {
            idx = i * cols + j;
            y = matrix[idx] - c;
            t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        
        /* Compute mean for this row */
        result[i] = sum / cols;
    }
}

/**
 * @brief Computes the mean of each column in a matrix with improved numerical stability
 *
 * Uses column-major processing for better cache efficiency and numerically
 * stable summation for accurate results.
 *
 * @param matrix Pointer to matrix data
 * @param result Pointer to result array (must be pre-allocated with size [cols])
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 */
void matrix_col_mean(const double *matrix, double *result, mwSize rows, mwSize cols) {
    mwSize i, j, idx;
    double sum, c;     /* Sum and compensation for Kahan summation */
    double y, t;       /* Temporary variables for Kahan summation */
    
    /* Validate inputs */
    if (matrix == NULL || result == NULL) {
        mex_error("matrix_col_mean: NULL pointer provided");
        return;
    }
    
    if (rows <= 0 || cols <= 0) {
        mex_error("matrix_col_mean: Invalid dimensions (%d x %d)", rows, cols);
        return;
    }
    
    /* Special case: empty matrix or divide by zero */
    if (rows == 0) {
        memset(result, 0, cols * sizeof(double));
        return;
    }
    
    /* Initialize result array */
    memset(result, 0, cols * sizeof(double));
    
    /* Process each column separately */
    for (j = 0; j < cols; j++) {
        sum = 0.0;
        c = 0.0;
        
        /* Compute column sum using Kahan summation for numerical stability */
        for (i = 0; i < rows; i++) {
            idx = i * cols + j;
            y = matrix[idx] - c;
            t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        
        /* Compute mean for this column */
        result[j] = sum / rows;
    }
}

/**
 * @brief Creates a diagonal matrix from a vector with zero initialization
 *
 * Initializes all matrix elements to zero, then sets diagonal elements to
 * corresponding values from the input vector.
 *
 * @param vector Pointer to vector data
 * @param matrix Pointer to result matrix data (must be pre-allocated with size [size × size])
 * @param size Size of the vector and dimension of the square matrix
 */
void create_diagonal_matrix(const double *vector, double *matrix, mwSize size) {
    mwSize i, idx;
    
    /* Validate inputs */
    if (vector == NULL || matrix == NULL) {
        mex_error("create_diagonal_matrix: NULL pointer provided");
        return;
    }
    
    if (size <= 0) {
        mex_error("create_diagonal_matrix: Invalid size (%d)", size);
        return;
    }
    
    /* Initialize all elements to zero */
    memset(matrix, 0, size * size * sizeof(double));
    
    /* Set diagonal elements from the vector */
    for (i = 0; i < size; i++) {
        idx = i * size + i;  /* Diagonal index */
        matrix[idx] = vector[i];
    }
}

/**
 * @brief Computes the trace (sum of diagonal elements) of a square matrix
 *
 * @param matrix Pointer to square matrix data
 * @param size Dimension of the square matrix
 * @return Trace value of the matrix
 */
double matrix_trace(const double *matrix, mwSize size) {
    mwSize i;
    double trace = 0.0;
    double c = 0.0;    /* Compensation variable for Kahan summation */
    double y, t;       /* Temporary variables for Kahan summation */
    
    /* Validate inputs */
    if (matrix == NULL) {
        mex_error("matrix_trace: NULL pointer provided");
        return 0.0;
    }
    
    if (size <= 0) {
        mex_error("matrix_trace: Invalid size (%d)", size);
        return 0.0;
    }
    
    /* Sum diagonal elements using Kahan summation for numerical stability */
    for (i = 0; i < size; i++) {
        /* Use stride optimization for accessing diagonal elements */
        y = matrix[i * size + i] - c;
        t = trace + y;
        c = (t - trace) - y;
        trace = t;
    }
    
    return trace;
}

/**
 * @brief Computes the Frobenius norm of a matrix with numerical stability
 *
 * The Frobenius norm is the square root of the sum of the squares of all elements.
 * Uses a numerically stable algorithm to prevent overflow.
 *
 * @param matrix Pointer to matrix data
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 * @return Frobenius norm of the matrix
 */
double matrix_frobenius_norm(const double *matrix, mwSize rows, mwSize cols) {
    mwSize i, j, idx;
    double scale = 0.0;     /* Scale factor for numerical stability */
    double sum_squares = 0.0;
    double abs_val, scaled_val;
    
    /* Validate inputs */
    if (matrix == NULL) {
        mex_error("matrix_frobenius_norm: NULL pointer provided");
        return 0.0;
    }
    
    if (rows <= 0 || cols <= 0) {
        mex_error("matrix_frobenius_norm: Invalid dimensions (%d x %d)", rows, cols);
        return 0.0;
    }
    
    /* First pass: find the largest absolute value for scaling */
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            idx = i * cols + j;
            abs_val = fabs(matrix[idx]);
            if (abs_val > scale) {
                scale = abs_val;
            }
        }
    }
    
    /* Handle the case of zero matrix */
    if (scale <= 0.0) {
        return 0.0;
    }
    
    /* Second pass: compute scaled sum of squares */
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            idx = i * cols + j;
            scaled_val = matrix[idx] / scale;
            sum_squares += scaled_val * scaled_val;
        }
    }
    
    /* Return the properly scaled norm */
    return scale * sqrt(sum_squares);
}