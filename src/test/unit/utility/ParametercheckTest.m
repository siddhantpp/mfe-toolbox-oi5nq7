classdef ParametercheckTest < BaseTest
    % PARAMETERCHECKTEST Test class for validating the parametercheck utility function
    %
    % This test suite ensures that parametercheck correctly performs its core
    % validation responsibilities: verifying that parameters are non-empty,
    % numeric, scalar when required, and within specified bounds with
    % appropriate error messages.
    %
    % The tests verify that parametercheck correctly:
    %   - Validates and returns valid numeric parameters
    %   - Detects and reports empty parameters
    %   - Enforces numeric type requirements
    %   - Enforces scalar constraints when specified
    %   - Identifies NaN values
    %   - Identifies infinite values
    %   - Enforces lower and upper bounds
    %   - Enforces integer constraints
    %   - Enforces non-negative and positive value constraints
    %   - Includes parameter names in error messages
    %   - Handles multidimensional arrays
    %   - Applies multiple constraints simultaneously
    %
    % See also: parametercheck, BaseTest, NumericalComparator
    
    properties
        testParameter  % Test parameter for validation tests
        testOptions    % Options structure for parametercheck
        comparator     % NumericalComparator instance for comparing values
        tolerance      % Tolerance for numerical comparisons
    end
    
    methods
        function obj = ParametercheckTest()
            % Initialize the ParametercheckTest class instance
            
            % Call parent constructor
            obj@BaseTest();
            
            % Initialize properties with default values
            obj.testParameter = [];
            obj.testOptions = struct();
            obj.comparator = [];
            obj.tolerance = 1e-10;
        end
        
        function setUp(obj)
            % Prepares the test environment before each test method is run
            
            % Call parent setUp method
            setUp@BaseTest(obj);
            
            % Initialize test data
            obj.testParameter = rand(3, 4);  % Create a random test matrix
            obj.testOptions = struct();      % Empty options structure
            obj.comparator = NumericalComparator();  % Create comparator for numeric comparisons
            obj.tolerance = 1e-10;           % Set default tolerance
        end
        
        function tearDown(obj)
            % Cleans up the test environment after each test method is run
            
            % Call parent tearDown method
            tearDown@BaseTest(obj);
            
            % Clear test data
            obj.testParameter = [];
            obj.testOptions = struct();
            obj.comparator = [];
        end
        
        function testValidInput(obj)
            % Tests that parametercheck correctly validates and returns a valid numeric parameter
            
            % Create a valid parameter
            validParam = rand(3, 4);
            
            % Call parametercheck with valid input
            result = parametercheck(validParam, 'testParam', obj.testOptions);
            
            % Verify that the parameter is returned unchanged
            compareResult = obj.comparator.compareMatrices(validParam, result, obj.tolerance);
            obj.assertTrue(compareResult.isEqual, 'Parameter was modified when it should be unchanged');
        end
        
        function testEmptyInput(obj)
            % Tests that parametercheck correctly handles and reports empty parameter inputs
            
            % Create an empty parameter
            emptyParam = [];
            
            % Test that error is thrown for empty input
            emptyCheckFcn = @() parametercheck(emptyParam, 'emptyParam', obj.testOptions);
            
            try
                emptyCheckFcn();
                obj.assertTrue(false, 'Expected error for empty parameter was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'cannot be empty'), ...
                    'Error message should indicate empty parameter');
                obj.assertTrue(contains(ME.message, 'emptyParam'), ...
                    'Error message should include parameter name');
            end
        end
        
        function testNonNumericInput(obj)
            % Tests that parametercheck correctly handles and reports non-numeric parameter inputs
            
            % Create non-numeric inputs
            cellParam = {1, 2, 3};
            charParam = 'string';
            structParam = struct('field', 1);
            
            % Test cell array input
            cellCheckFcn = @() parametercheck(cellParam, 'cellParam', obj.testOptions);
            try
                cellCheckFcn();
                obj.assertTrue(false, 'Expected error for cell array parameter was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'must be numeric'), ...
                    'Error message should indicate non-numeric parameter');
            end
            
            % Test char input
            charCheckFcn = @() parametercheck(charParam, 'charParam', obj.testOptions);
            try
                charCheckFcn();
                obj.assertTrue(false, 'Expected error for char parameter was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'must be numeric'), ...
                    'Error message should indicate non-numeric parameter');
            end
            
            % Test struct input
            structCheckFcn = @() parametercheck(structParam, 'structParam', obj.testOptions);
            try
                structCheckFcn();
                obj.assertTrue(false, 'Expected error for struct parameter was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'must be numeric'), ...
                    'Error message should indicate non-numeric parameter');
            end
        end
        
        function testScalarRequirement(obj)
            % Tests that parametercheck correctly enforces scalar requirements when specified
            
            % Create valid non-scalar numeric matrix
            nonScalarParam = rand(2, 3);
            
            % Create valid scalar parameter
            scalarParam = 42;
            
            % Set isscalar option
            obj.testOptions.isscalar = true;
            
            % Test non-scalar input with scalar requirement
            nonScalarCheckFcn = @() parametercheck(nonScalarParam, 'nonScalarParam', obj.testOptions);
            try
                nonScalarCheckFcn();
                obj.assertTrue(false, 'Expected error for non-scalar parameter was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'must be a scalar'), ...
                    'Error message should indicate scalar requirement');
            end
            
            % Test scalar input with scalar requirement
            result = parametercheck(scalarParam, 'scalarParam', obj.testOptions);
            obj.assertEqual(scalarParam, result, 'Scalar parameter should be validated and returned unchanged');
        end
        
        function testNaNValues(obj)
            % Tests that parametercheck correctly identifies and reports NaN values in parameters
            
            % Create matrix with NaN values
            nanParam = rand(3, 4);
            nanParam(2, 2) = NaN;
            
            % Test that error is thrown for NaN values
            nanCheckFcn = @() parametercheck(nanParam, 'nanParam', obj.testOptions);
            try
                nanCheckFcn();
                obj.assertTrue(false, 'Expected error for NaN values was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'cannot contain NaN'), ...
                    'Error message should reference NaN values');
            end
        end
        
        function testInfValues(obj)
            % Tests that parametercheck correctly identifies and reports infinite values in parameters
            
            % Create matrix with Inf values
            infParam = rand(3, 4);
            infParam(1, 3) = Inf;
            
            % Test that error is thrown for Inf values
            infCheckFcn = @() parametercheck(infParam, 'infParam', obj.testOptions);
            try
                infCheckFcn();
                obj.assertTrue(false, 'Expected error for Inf values was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'cannot contain infinite'), ...
                    'Error message should reference infinite values');
            end
        end
        
        function testLowerBound(obj)
            % Tests that parametercheck correctly enforces lower bound constraints
            
            % Create test parameters
            belowBoundParam = rand(3, 4) - 0.5;  % Some values below 0
            aboveBoundParam = rand(3, 4);        % All values between 0 and 1
            
            % Set lower bound option
            obj.testOptions.lowerBound = 0;
            
            % Test parameter with values below lower bound
            belowBoundCheckFcn = @() parametercheck(belowBoundParam, 'belowBoundParam', obj.testOptions);
            try
                belowBoundCheckFcn();
                obj.assertTrue(false, 'Expected error for values below lower bound was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'must be greater than or equal'), ...
                    'Error message should reference lower bound');
            end
            
            % Test parameter with values at or above lower bound
            result = parametercheck(aboveBoundParam, 'aboveBoundParam', obj.testOptions);
            obj.assertEqual(aboveBoundParam, result, 'Parameter should be validated and returned unchanged');
        end
        
        function testUpperBound(obj)
            % Tests that parametercheck correctly enforces upper bound constraints
            
            % Create test parameters
            belowBoundParam = rand(3, 4) * 0.5;  % All values between 0 and 0.5
            aboveBoundParam = rand(3, 4) + 0.5;  % Some values above 0.5
            
            % Set upper bound option
            obj.testOptions.upperBound = 0.5;
            
            % Test parameter with values above upper bound
            aboveBoundCheckFcn = @() parametercheck(aboveBoundParam, 'aboveBoundParam', obj.testOptions);
            try
                aboveBoundCheckFcn();
                obj.assertTrue(false, 'Expected error for values above upper bound was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'must be less than or equal'), ...
                    'Error message should reference upper bound');
            end
            
            % Test parameter with values at or below upper bound
            result = parametercheck(belowBoundParam, 'belowBoundParam', obj.testOptions);
            obj.assertEqual(belowBoundParam, result, 'Parameter should be validated and returned unchanged');
        end
        
        function testIntegerConstraint(obj)
            % Tests that parametercheck correctly enforces integer constraints when specified
            
            % Create test parameters
            nonIntegerParam = rand(3, 4);         % Random non-integer values
            integerParam = randi([1, 10], 3, 4);  % Random integer values
            
            % Set integer constraint option
            obj.testOptions.isInteger = true;
            
            % Test parameter with non-integer values
            nonIntegerCheckFcn = @() parametercheck(nonIntegerParam, 'nonIntegerParam', obj.testOptions);
            try
                nonIntegerCheckFcn();
                obj.assertTrue(false, 'Expected error for non-integer values was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'must contain only integer'), ...
                    'Error message should reference integer constraint');
            end
            
            % Test parameter with integer values
            result = parametercheck(integerParam, 'integerParam', obj.testOptions);
            obj.assertEqual(integerParam, result, 'Parameter should be validated and returned unchanged');
        end
        
        function testNonNegativeConstraint(obj)
            % Tests that parametercheck correctly enforces non-negative constraints when specified
            
            % Create test parameters
            negativeParam = rand(3, 4) - 1;  % Some negative values
            positiveParam = rand(3, 4);      % All positive values
            
            % Set non-negative constraint option
            obj.testOptions.isNonNegative = true;
            
            % Test parameter with negative values
            negativeCheckFcn = @() parametercheck(negativeParam, 'negativeParam', obj.testOptions);
            try
                negativeCheckFcn();
                obj.assertTrue(false, 'Expected error for negative values was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'must contain only non-negative'), ...
                    'Error message should reference non-negative constraint');
            end
            
            % Test parameter with non-negative values
            result = parametercheck(positiveParam, 'positiveParam', obj.testOptions);
            obj.assertEqual(positiveParam, result, 'Parameter should be validated and returned unchanged');
        end
        
        function testPositiveConstraint(obj)
            % Tests that parametercheck correctly enforces positive constraints when specified
            
            % Create test parameters
            nonPositiveParam = rand(3, 4) - 0.5;  % Some zero or negative values
            positiveParam = rand(3, 4) + 0.1;     % All positive values
            
            % Set positive constraint option
            obj.testOptions.isPositive = true;
            
            % Test parameter with non-positive values
            nonPositiveCheckFcn = @() parametercheck(nonPositiveParam, 'nonPositiveParam', obj.testOptions);
            try
                nonPositiveCheckFcn();
                obj.assertTrue(false, 'Expected error for non-positive values was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'must contain only positive'), ...
                    'Error message should reference positive constraint');
            end
            
            % Test parameter with positive values
            result = parametercheck(positiveParam, 'positiveParam', obj.testOptions);
            obj.assertEqual(positiveParam, result, 'Parameter should be validated and returned unchanged');
        end
        
        function testParameterNameDisplay(obj)
            % Tests that parametercheck correctly incorporates the provided parameter name in error messages
            
            % Create various invalid inputs
            emptyParam = [];
            nonNumericParam = 'string';
            nanParam = [1, NaN, 3];
            infParam = [1, 2, Inf];
            
            % Define custom parameter name
            paramName = 'customParamName';
            
            % Test empty parameter
            emptyCheckFcn = @() parametercheck(emptyParam, paramName, obj.testOptions);
            try
                emptyCheckFcn();
                obj.assertTrue(false, 'Expected error was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, paramName), ...
                    'Error message should include the specified parameter name');
            end
            
            % Test non-numeric parameter
            nonNumericCheckFcn = @() parametercheck(nonNumericParam, paramName, obj.testOptions);
            try
                nonNumericCheckFcn();
                obj.assertTrue(false, 'Expected error was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, paramName), ...
                    'Error message should include the specified parameter name');
            end
            
            % Test NaN parameter
            nanCheckFcn = @() parametercheck(nanParam, paramName, obj.testOptions);
            try
                nanCheckFcn();
                obj.assertTrue(false, 'Expected error was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, paramName), ...
                    'Error message should include the specified parameter name');
            end
            
            % Test Inf parameter
            infCheckFcn = @() parametercheck(infParam, paramName, obj.testOptions);
            try
                infCheckFcn();
                obj.assertTrue(false, 'Expected error was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, paramName), ...
                    'Error message should include the specified parameter name');
            end
        end
        
        function testMultipleDimensionalInput(obj)
            % Tests that parametercheck correctly handles multi-dimensional arrays
            
            % Create valid 3D and 4D arrays
            valid3D = rand(2, 3, 4);
            valid4D = rand(2, 2, 2, 2);
            
            % Create invalid 3D array with NaN
            invalid3D = rand(2, 3, 4);
            invalid3D(1, 2, 3) = NaN;
            
            % Create invalid 4D array with Inf
            invalid4D = rand(2, 2, 2, 2);
            invalid4D(1, 1, 2, 2) = Inf;
            
            % Test valid 3D array
            result3D = parametercheck(valid3D, 'valid3D', obj.testOptions);
            obj.assertEqual(valid3D, result3D, 'Valid 3D array should be returned unchanged');
            
            % Test valid 4D array
            result4D = parametercheck(valid4D, 'valid4D', obj.testOptions);
            obj.assertEqual(valid4D, result4D, 'Valid 4D array should be returned unchanged');
            
            % Test invalid 3D array
            invalid3DCheckFcn = @() parametercheck(invalid3D, 'invalid3D', obj.testOptions);
            try
                invalid3DCheckFcn();
                obj.assertTrue(false, 'Expected error for 3D array with NaN was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'cannot contain NaN'), ...
                    'Error message should reference NaN values');
            end
            
            % Test invalid 4D array
            invalid4DCheckFcn = @() parametercheck(invalid4D, 'invalid4D', obj.testOptions);
            try
                invalid4DCheckFcn();
                obj.assertTrue(false, 'Expected error for 4D array with Inf was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'cannot contain infinite'), ...
                    'Error message should reference infinite values');
            end
        end
        
        function testMultipleConstraints(obj)
            % Tests that parametercheck correctly applies multiple constraints simultaneously
            
            % Create a suitable test parameter that should pass all constraints
            validParam = randi([1, 5], 3, 4);  % Integer values between 1 and 5
            
            % Create parameters that violate specific constraints
            notIntegerParam = validParam + 0.5;                  % Not integers
            notPositiveParam = validParam;
            notPositiveParam(2, 2) = 0;                          % Not all positive
            belowLowerBoundParam = validParam;
            belowLowerBoundParam(1, 1) = 0;                      % Below lower bound of 1
            aboveUpperBoundParam = validParam;
            aboveUpperBoundParam(3, 3) = 6;                      % Above upper bound of 5
            
            % Set multiple constraints
            obj.testOptions.isInteger = true;
            obj.testOptions.isPositive = true;
            obj.testOptions.lowerBound = 1;
            obj.testOptions.upperBound = 5;
            
            % Test valid parameter against all constraints
            result = parametercheck(validParam, 'validParam', obj.testOptions);
            obj.assertEqual(validParam, result, 'Parameter should pass all constraints');
            
            % Test non-integer parameter
            notIntegerCheckFcn = @() parametercheck(notIntegerParam, 'notIntegerParam', obj.testOptions);
            try
                notIntegerCheckFcn();
                obj.assertTrue(false, 'Expected error for non-integer values was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'integer'), ...
                    'Error message should reference integer constraint');
            end
            
            % Test non-positive parameter
            notPositiveCheckFcn = @() parametercheck(notPositiveParam, 'notPositiveParam', obj.testOptions);
            try
                notPositiveCheckFcn();
                obj.assertTrue(false, 'Expected error for non-positive values was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'positive'), ...
                    'Error message should reference positive constraint');
            end
            
            % Test below lower bound parameter
            belowLowerBoundCheckFcn = @() parametercheck(belowLowerBoundParam, 'belowLowerBoundParam', obj.testOptions);
            try
                belowLowerBoundCheckFcn();
                obj.assertTrue(false, 'Expected error for values below lower bound was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'greater than or equal'), ...
                    'Error message should reference lower bound');
            end
            
            % Test above upper bound parameter
            aboveUpperBoundCheckFcn = @() parametercheck(aboveUpperBoundParam, 'aboveUpperBoundParam', obj.testOptions);
            try
                aboveUpperBoundCheckFcn();
                obj.assertTrue(false, 'Expected error for values above upper bound was not thrown');
            catch ME
                obj.assertTrue(contains(ME.message, 'less than or equal'), ...
                    'Error message should reference upper bound');
            end
        end
    end
end