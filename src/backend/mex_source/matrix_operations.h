#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

/**
 * @file matrix_operations.h
 * @brief Optimized matrix operations for high-performance financial econometric computations
 *
 * This header provides essential matrix manipulation functions that serve as the
 * foundation for volatility models and time series analysis in the MFE Toolbox.
 * Optimized for performance and numerical stability in financial applications.
 *
 * @version 4.0 (28-Oct-2009)
 */

/* External dependencies */
#include "mex.h"      /* MATLAB MEX API - MATLAB 4.0 compatible */
#include "matrix.h"   /* MATLAB MEX API - MATLAB 4.0 compatible */
#include <stdlib.h>   /* C Standard Library - C89/C90 */
#include <math.h>     /* C Standard Library - C89/C90 */

/**
 * @brief Creates a deep copy of a source matrix into a target matrix with error checking
 *
 * @param source Pointer to source matrix data
 * @param target Pointer to target matrix data (must be pre-allocated)
 * @param rows Number of rows in the matrices
 * @param cols Number of columns in the matrices
 */
void copy_matrix(double* source, double* target, mwSize rows, mwSize cols);

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
void matrix_multiply(double* matrix_a, double* matrix_b, double* result, 
                     mwSize rows_a, mwSize cols_a, mwSize cols_b);

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
void transpose_matrix(double* matrix, double* result, mwSize rows, mwSize cols);

/**
 * @brief Adds two matrices element-wise with bounds checking
 *
 * @param matrix_a Pointer to first matrix (A)
 * @param matrix_b Pointer to second matrix (B)
 * @param result Pointer to result matrix (must be pre-allocated with size [rows × cols])
 * @param rows Number of rows in the matrices
 * @param cols Number of columns in the matrices
 */
void matrix_addition(double* matrix_a, double* matrix_b, double* result, 
                     mwSize rows, mwSize cols);

/**
 * @brief Subtracts second matrix from first matrix element-wise with validation
 *
 * @param matrix_a Pointer to first matrix (A)
 * @param matrix_b Pointer to second matrix (B)
 * @param result Pointer to result matrix (must be pre-allocated with size [rows × cols])
 * @param rows Number of rows in the matrices
 * @param cols Number of columns in the matrices
 */
void matrix_subtraction(double* matrix_a, double* matrix_b, double* result, 
                       mwSize rows, mwSize cols);

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
void matrix_element_multiply(double* matrix_a, double* matrix_b, double* result, 
                            mwSize rows, mwSize cols);

/**
 * @brief Multiplies each element of a matrix by a scalar value efficiently
 *
 * @param matrix Pointer to source matrix data
 * @param result Pointer to result matrix data (must be pre-allocated with size [rows × cols])
 * @param scalar The scalar value to multiply by
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 */
void matrix_scalar_multiply(double* matrix, double* result, double scalar, 
                           mwSize rows, mwSize cols);

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
double matrix_sum(double* matrix, mwSize rows, mwSize cols);

/**
 * @brief Computes the mean of each row in a matrix with stability considerations
 *
 * Uses numerically stable summation methods to calculate accurate row means.
 *
 * @param matrix Pointer to matrix data
 * @param result Pointer to result array (must be pre-allocated with size [rows])
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 */
void matrix_row_mean(double* matrix, double* result, mwSize rows, mwSize cols);

/**
 * @brief Computes the mean of each column in a matrix with stability considerations
 *
 * Uses column-major processing for better cache efficiency and numerically
 * stable summation for accurate results.
 *
 * @param matrix Pointer to matrix data
 * @param result Pointer to result array (must be pre-allocated with size [cols])
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 */
void matrix_col_mean(double* matrix, double* result, mwSize rows, mwSize cols);

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
void create_diagonal_matrix(double* vector, double* matrix, mwSize size);

#endif /* MATRIX_OPERATIONS_H */