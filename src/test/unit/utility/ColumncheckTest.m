classdef ColumncheckTest < BaseTest
    % COLUMNCHECKTEST Unit test class for the columncheck utility function that validates
    % column vectors and converts row vectors to column format.
    
    properties
        testDataGenerator
        testTolerance
        rowVector
        columnVector
        matrix
    end
    
    methods
        function obj = ColumncheckTest()
            % Constructor initializes a new ColumncheckTest instance with test data for vector format validation
            obj@BaseTest('ColumncheckTest');
            
            % Initialize test data generator with fixed seed for reproducibility
            obj.testDataGenerator = TestDataGenerator();
            
            % Set tolerance for numeric comparisons
            obj.testTolerance = 1e-12;
            
            % Initialize sample test vectors and matrices
            obj.rowVector = [];
            obj.columnVector = [];
            obj.matrix = [];
        end
        
        function setUp(obj)
            % Prepares the test environment before each test method runs
            setUp@BaseTest(obj);
            
            % Reset random number generator for reproducibility
            rng(1);
            
            % Generate fresh test data
            obj.rowVector = [1, 2, 3, 4, 5];
            obj.columnVector = [1; 2; 3; 4; 5];
            obj.matrix = [1, 2; 3, 4; 5, 6];
        end
        
        function tearDown(obj)
            % Cleans up the test environment after each test method completes
            tearDown@BaseTest(obj);
        end
        
        function testEmptyInput(obj)
            % Tests that columncheck correctly rejects empty input arrays
            
            % Test empty vector
            emptyVector = [];
            obj.assertThrows(@() columncheck(emptyVector, 'emptyVector'), 'MATLAB:error');
            
            % Test with name parameter to check error message
            try
                columncheck([], 'testVar');
                obj.assertTrue(false, 'Expected error was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'testVar cannot be empty'), ...
                    'Error message does not contain expected text');
            end
        end
        
        function testNonNumericInput(obj)
            % Tests that columncheck correctly rejects non-numeric input data
            
            % Test with cell array
            cellInput = {1, 2, 3};
            obj.assertThrows(@() columncheck(cellInput, 'cellInput'), 'MATLAB:error');
            
            % Test with string
            stringInput = 'test';
            obj.assertThrows(@() columncheck(stringInput, 'stringInput'), 'MATLAB:error');
            
            % Test with logical array
            logicalInput = logical([1, 0, 1]);
            obj.assertThrows(@() columncheck(logicalInput, 'logicalInput'), 'MATLAB:error');
            
            % Check error message content
            try
                columncheck('test', 'stringVar');
                obj.assertTrue(false, 'Expected error was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'stringVar must be numeric'), ...
                    'Error message does not contain expected text');
            end
        end
        
        function testRowVectorConversion(obj)
            % Tests that columncheck correctly converts row vectors to column vectors
            
            % Test with simple row vector
            inputRow = [1, 2, 3, 4, 5];
            expectedColumn = [1; 2; 3; 4; 5];
            resultColumn = columncheck(inputRow, 'inputRow');
            
            % Check dimensions
            [rows, cols] = size(resultColumn);
            obj.assertEqual(rows, 5, 'Incorrect number of rows after conversion');
            obj.assertEqual(cols, 1, 'Incorrect number of columns after conversion');
            
            % Check values using precise comparison
            obj.assertMatrixEqualsWithTolerance(resultColumn, expectedColumn, obj.testTolerance, ...
                'Row vector not correctly converted to column vector');
            
            % Test with row vector containing floating-point values
            floatRow = [1.1, 2.2, 3.3];
            floatColumn = columncheck(floatRow, 'floatRow');
            obj.assertEqual(size(floatColumn, 1), 3, 'Incorrect number of rows');
            obj.assertEqual(size(floatColumn, 2), 1, 'Incorrect number of columns');
            
            % Test with random row vector
            try
                % First try using the testDataGenerator
                randomRow = obj.testDataGenerator.generateNormalSample(1, 10);
                randomColumn = columncheck(randomRow, 'randomRow');
                obj.assertEqual(size(randomColumn, 1), 10, 'Incorrect number of rows in random vector');
                obj.assertEqual(size(randomColumn, 2), 1, 'Incorrect number of columns in random vector');
            catch
                % Fallback to using randn directly if testDataGenerator doesn't work as expected
                randomRow = randn(1, 10);
                randomColumn = columncheck(randomRow, 'randomRow');
                obj.assertEqual(size(randomColumn, 1), 10, 'Incorrect number of rows in random vector');
                obj.assertEqual(size(randomColumn, 2), 1, 'Incorrect number of columns in random vector');
            end
        end
        
        function testColumnVectorPreservation(obj)
            % Tests that columncheck preserves column vectors without modification
            
            % Test with simple column vector
            inputColumn = [1; 2; 3; 4; 5];
            resultColumn = columncheck(inputColumn, 'inputColumn');
            
            % Check that the result is identical to the input
            obj.assertMatrixEqualsWithTolerance(resultColumn, inputColumn, obj.testTolerance, ...
                'Column vector incorrectly modified');
            
            % Check dimensions are preserved
            [rows, cols] = size(resultColumn);
            obj.assertEqual(rows, 5, 'Number of rows changed unexpectedly');
            obj.assertEqual(cols, 1, 'Number of columns changed unexpectedly');
            
            % Test with column vector containing floating-point values
            floatColumn = [1.1; 2.2; 3.3];
            resultFloatColumn = columncheck(floatColumn, 'floatColumn');
            obj.assertMatrixEqualsWithTolerance(resultFloatColumn, floatColumn, obj.testTolerance, ...
                'Floating-point column vector incorrectly modified');
            
            % Test with larger column vector
            largeColumn = zeros(100, 1);
            resultLargeColumn = columncheck(largeColumn, 'largeColumn');
            obj.assertEqual(size(resultLargeColumn, 1), 100, 'Number of rows changed unexpectedly');
            obj.assertEqual(size(resultLargeColumn, 2), 1, 'Number of columns changed unexpectedly');
        end
        
        function testMatrixRejection(obj)
            % Tests that columncheck correctly rejects multi-column matrices
            
            % Test with 2x2 matrix
            matrix2x2 = [1, 2; 3, 4];
            obj.assertThrows(@() columncheck(matrix2x2, 'matrix2x2'), 'MATLAB:error');
            
            % Test with 3x2 matrix
            matrix3x2 = [1, 2; 3, 4; 5, 6];
            obj.assertThrows(@() columncheck(matrix3x2, 'matrix3x2'), 'MATLAB:error');
            
            % Test with larger matrix
            largeMatrix = ones(10, 3);
            obj.assertThrows(@() columncheck(largeMatrix, 'largeMatrix'), 'MATLAB:error');
            
            % Check error message content
            try
                columncheck([1, 2; 3, 4], 'testMatrix');
                obj.assertTrue(false, 'Expected error was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'testMatrix must be a column vector or a row vector'), ...
                    'Error message does not contain expected text');
            end
        end
        
        function testScalarHandling(obj)
            % Tests that columncheck correctly handles scalar inputs
            
            % Test with integer scalar
            scalarInt = 5;
            resultInt = columncheck(scalarInt, 'scalarInt');
            
            % Scalar should remain unchanged (it's both a row and column vector)
            obj.assertEqual(resultInt, scalarInt, 'Scalar value incorrectly modified');
            
            % Check dimensions
            [rows, cols] = size(resultInt);
            obj.assertEqual(rows, 1, 'Scalar should have 1 row');
            obj.assertEqual(cols, 1, 'Scalar should have 1 column');
            
            % Test with floating-point scalar
            scalarFloat = 3.14159;
            resultFloat = columncheck(scalarFloat, 'scalarFloat');
            obj.assertMatrixEqualsWithTolerance(resultFloat, scalarFloat, obj.testTolerance, ...
                'Floating-point scalar incorrectly modified');
        end
        
        function testNameParameter(obj)
            % Tests that columncheck correctly includes variable name in error messages
            
            % Test with empty input and custom name
            try
                columncheck([], 'customVar');
                obj.assertTrue(false, 'Expected error was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'customVar cannot be empty'), ...
                    'Error message does not contain custom variable name');
            end
            
            % Test with non-numeric input and custom name
            try
                columncheck('string', 'strVar');
                obj.assertTrue(false, 'Expected error was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'strVar must be numeric'), ...
                    'Error message does not contain custom variable name');
            end
            
            % Test with matrix input and custom name
            try
                columncheck([1, 2; 3, 4], 'matrixVar');
                obj.assertTrue(false, 'Expected error was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'matrixVar must be a column vector or a row vector'), ...
                    'Error message does not contain custom variable name');
            end
        end
        
        function testEdgeCases(obj)
            % Tests columncheck with edge cases like very large vectors
            
            % Test with large row vector
            largeRowVector = ones(1, 10000);
            largeColumnVector = columncheck(largeRowVector, 'largeRowVector');
            obj.assertEqual(size(largeColumnVector, 1), 10000, 'Incorrect number of rows after conversion');
            obj.assertEqual(size(largeColumnVector, 2), 1, 'Incorrect number of columns after conversion');
            
            % Test with row vector containing special values
            specialRow = [0, Inf, -Inf, realmax, realmin];
            specialColumn = columncheck(specialRow, 'specialRow');
            obj.assertEqual(size(specialColumn, 1), 5, 'Incorrect number of rows after conversion');
            obj.assertEqual(size(specialColumn, 2), 1, 'Incorrect number of columns after conversion');
            obj.assertMatrixEqualsWithTolerance(specialColumn, specialRow', obj.testTolerance, ...
                'Special values not preserved during conversion');
            
            % Test with eye matrix (identity) - should fail as it's a matrix, not a vector
            identityMatrix = eye(3);
            obj.assertThrows(@() columncheck(identityMatrix, 'identityMatrix'), 'MATLAB:error');
        end
    end
end