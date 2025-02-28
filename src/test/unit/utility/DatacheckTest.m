classdef DatacheckTest < BaseTest
    % DATACHECKTEST Unit tests for the datacheck utility function
    %
    % This test class validates that the datacheck utility function correctly
    % validates numerical data inputs by checking that data is:
    %   1. Not empty
    %   2. Numeric
    %   3. Free of NaN values
    %   4. Free of infinite values
    %
    % The tests confirm datacheck's ability to identify invalid inputs and
    % return appropriate error messages, which is essential for ensuring
    % computational stability in econometric functions.
    %
    % See also: datacheck, BaseTest, NumericalComparator
    
    properties
        testData     % Matrix of test data
        comparator   % NumericalComparator instance
        tolerance    % Default tolerance for numerical comparisons
    end
    
    methods
        function obj = DatacheckTest()
            % Constructor initializes the test class
            obj = obj@BaseTest();
        end
        
        function setUp(obj)
            % Prepare test environment before each test method runs
            setUp@BaseTest(obj);
            
            % Create a random test matrix with known good data
            obj.testData = rand(10, 5);
            
            % Initialize numerical comparator for matrix equality testing
            obj.comparator = NumericalComparator();
            
            % Set default tolerance for numerical comparisons
            obj.tolerance = 1e-10;
        end
        
        function tearDown(obj)
            % Clean up after each test method runs
            tearDown@BaseTest(obj);
            
            % Clear test data and comparator
            obj.testData = [];
            obj.comparator = [];
        end
        
        function testValidInput(obj)
            % Tests that datacheck correctly validates and returns a valid numeric matrix
            
            % Create valid numeric matrix
            validMatrix = rand(5, 5);
            
            % Call datacheck with valid input
            result = datacheck(validMatrix, 'validMatrix');
            
            % Verify that result matches input exactly
            compResult = obj.comparator.compareMatrices(validMatrix, result, obj.tolerance);
            obj.assertTrue(compResult.isEqual, 'datacheck should return valid data unchanged');
        end
        
        function testEmptyInput(obj)
            % Tests that datacheck correctly handles and reports empty matrix inputs
            
            % Create an empty matrix
            emptyMatrix = [];
            dataName = 'emptyMatrix';
            
            % Check for expected error
            errorCaught = false;
            try
                datacheck(emptyMatrix, dataName);
            catch ME
                errorCaught = true;
                % Verify error message content
                obj.assertTrue(contains(ME.message, 'empty'), 'Error message should mention empty data');
                obj.assertTrue(contains(ME.message, dataName), 'Error message should contain data name');
            end
            
            obj.assertTrue(errorCaught, 'datacheck should throw an error for empty input');
        end
        
        function testNonNumericInput(obj)
            % Tests that datacheck correctly handles and reports non-numeric inputs
            
            % Create non-numeric input (cell array)
            cellInput = {1, 2, 3};
            dataName = 'cellInput';
            
            % Check for expected error
            errorCaught = false;
            try
                datacheck(cellInput, dataName);
            catch ME
                errorCaught = true;
                % Verify error message content
                obj.assertTrue(contains(ME.message, 'numeric'), 'Error message should mention numeric requirement');
                obj.assertTrue(contains(ME.message, dataName), 'Error message should contain data name');
            end
            
            obj.assertTrue(errorCaught, 'datacheck should throw an error for non-numeric input');
        end
        
        function testNaNValues(obj)
            % Tests that datacheck correctly identifies and reports NaN values in input data
            
            % Create valid matrix with NaN values inserted
            nanMatrix = obj.testData;
            nanMatrix(2, 3) = NaN;
            dataName = 'nanMatrix';
            
            % Check for expected error
            errorCaught = false;
            try
                datacheck(nanMatrix, dataName);
            catch ME
                errorCaught = true;
                % Verify error message content
                obj.assertTrue(contains(ME.message, 'NaN'), 'Error message should mention NaN values');
                obj.assertTrue(contains(ME.message, dataName), 'Error message should contain data name');
            end
            
            obj.assertTrue(errorCaught, 'datacheck should throw an error for NaN values');
        end
        
        function testInfValues(obj)
            % Tests that datacheck correctly identifies and reports infinite values in input data
            
            % Create valid matrix with infinite values inserted
            infMatrix = obj.testData;
            infMatrix(1, 1) = Inf;
            dataName = 'infMatrix';
            
            % Check for expected error
            errorCaught = false;
            try
                datacheck(infMatrix, dataName);
            catch ME
                errorCaught = true;
                % Verify error message content
                obj.assertTrue(contains(ME.message, 'Inf'), 'Error message should mention Inf values');
                obj.assertTrue(contains(ME.message, dataName), 'Error message should contain data name');
            end
            
            obj.assertTrue(errorCaught, 'datacheck should throw an error for Inf values');
        end
        
        function testDataNameDisplay(obj)
            % Tests that datacheck correctly incorporates the provided data name in error messages
            
            % Create invalid input with custom name
            customNameData = [];
            customName = 'myCustomData';
            
            % Check for expected error with specific data name
            errorCaught = false;
            try
                datacheck(customNameData, customName);
            catch ME
                errorCaught = true;
                % Verify error message content
                obj.assertTrue(contains(ME.message, customName), 'Error message should contain the data name');
            end
            
            obj.assertTrue(errorCaught, 'datacheck should throw an error with appropriate data name');
        end
        
        function testMultipleDimensionalInput(obj)
            % Tests that datacheck correctly handles multi-dimensional arrays
            
            % Create and test valid 3D array
            array3D = rand(3, 4, 2);
            result3D = datacheck(array3D, '3D Array');
            compResult3D = obj.comparator.compareMatrices(reshape(array3D, [], 1), reshape(result3D, [], 1), obj.tolerance);
            obj.assertTrue(compResult3D.isEqual, 'datacheck should handle 3D arrays correctly');
            
            % Create and test valid 4D array
            array4D = rand(2, 3, 2, 2);
            result4D = datacheck(array4D, '4D Array');
            compResult4D = obj.comparator.compareMatrices(reshape(array4D, [], 1), reshape(result4D, [], 1), obj.tolerance);
            obj.assertTrue(compResult4D.isEqual, 'datacheck should handle 4D arrays correctly');
            
            % Test 3D array with NaN
            invalidArray3D = array3D;
            invalidArray3D(1, 1, 1) = NaN;
            
            % Check for expected error
            errorCaught = false;
            try
                datacheck(invalidArray3D, 'invalid3D');
            catch ME
                errorCaught = true;
                % Verify error message content
                obj.assertTrue(contains(ME.message, 'NaN'), 'Error message should mention NaN values');
            end
            
            obj.assertTrue(errorCaught, 'datacheck should throw an error for 3D array with NaN');
        end
    end
end