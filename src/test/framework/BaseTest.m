classdef BaseTest
    % BASETEST Base class for all test cases in the MFE Toolbox testing framework
    %
    % This class provides common testing functionality including setup, teardown,
    % assertions with appropriate numerical tolerance handling, and test lifecycle
    % management. It serves as the foundation for all test cases in the MFE Toolbox.
    %
    % The BaseTest class implements robust assertion methods that handle the
    % floating-point precision requirements of financial calculations. It 
    % provides standardized mechanisms for validating test conditions with
    % appropriate error handling and numerical stability.
    %
    % Example:
    %   % Create a test class that inherits from BaseTest
    %   classdef MyTestCase < BaseTest
    %       methods
    %           function testMyFunction(testCase)
    %               % Test implementation
    %               expected = [1.1, 2.2; 3.3, 4.4];
    %               actual = myFunctionUnderTest(input);
    %               testCase.assertMatrixEqualsWithTolerance(expected, actual, 1e-8);
    %           end
    %       end
    %   end
    %
    %   % Run all tests in the class
    %   testObj = MyTestCase();
    %   results = testObj.runAllTests();
    %
    % See also: NumericalComparator, parametercheck
    
    properties
        defaultTolerance     % Default tolerance for numerical comparisons
        verbose              % Flag to control output verbosity
        testResults          % Structure to store test results
        testName             % Name of the test case
        testDataPath         % Path to test data directory
        numericalComparator  % NumericalComparator instance for floating-point comparisons
    end
    
    methods
        function obj = BaseTest(testName)
            % Initialize a new BaseTest instance with optional test name
            %
            % INPUTS:
            %   testName - Optional name for the test case (char)
            %
            % OUTPUTS:
            %   obj - Initialized BaseTest instance
            
            % Set defaultTolerance to 1e-10 for numerical comparisons
            obj.defaultTolerance = 1e-10;
            
            % Set verbose to false by default to control output verbosity
            obj.verbose = false;
            
            % Initialize empty test results structure
            obj.testResults = struct();
            
            % Set test name from parameter or derive from class name
            if nargin < 1 || isempty(testName)
                classInfo = metaclass(obj);
                obj.testName = classInfo.Name;
            else
                obj.testName = testName;
            end
            
            % Set test data path to src/test/data
            obj.testDataPath = 'src/test/data';
            
            % Create a NumericalComparator instance for floating-point comparisons
            obj.numericalComparator = NumericalComparator();
        end
        
        function setUp(obj)
            % Set up test environment before each test method execution
            %
            % This method initializes the test environment before each test method
            % runs. Subclasses can override this method to perform specific
            % initialization logic.
            
            % Initialize test state
            obj.testResults = struct();
            
            % Reset test results structure
            obj.testResults.startTime = tic;
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            %
            % This method cleans up the test environment after each test method
            % runs. Subclasses can override this method to perform specific
            % cleanup logic.
            
            % Clean up any temporary resources
            % (subclasses should implement specific cleanup logic)
            
            % Calculate test execution time
            if isfield(obj.testResults, 'startTime')
                obj.testResults.executionTime = toc(obj.testResults.startTime);
            end
            
            % Record test completion status
            if ~isfield(obj.testResults, 'status')
                obj.testResults.status = 'completed';
            end
        end
        
        function result = runTest(obj, testMethodName)
            % Execute a specific test method with proper setup and teardown
            %
            % INPUTS:
            %   testMethodName - Name of the test method to execute (char)
            %
            % OUTPUTS:
            %   result - Test result information (struct)
            
            % Validate that the test method exists using exist function
            if ~exist([class(obj), '.', testMethodName], 'method')
                error('BaseTest:InvalidTestMethod', 'Test method ''%s'' does not exist', testMethodName);
            end
            
            % Call setUp method to prepare the test environment
            try
                obj.setUp();
                
                % Execute test method in try-catch block
                try
                    % Call the test method
                    obj.(testMethodName)();
                    result.status = 'passed';
                catch ME
                    % Test failed
                    result.status = 'failed';
                    result.error = ME;
                    
                    % Display error information if verbose mode is enabled
                    if obj.verbose
                        warning('Test %s::%s failed: %s', obj.testName, testMethodName, ME.message);
                    end
                end
                
                % Call tearDown method to clean up resources
                obj.tearDown();
                
                % Record test result information (passed/failed, time, errors)
                if isfield(obj.testResults, 'executionTime')
                    result.executionTime = obj.testResults.executionTime;
                else
                    result.executionTime = 0;
                end
                
            catch ME
                % Setup or teardown failed
                result.status = 'error';
                result.error = ME;
                result.executionTime = 0;
                
                if obj.verbose
                    warning('Setup or teardown for %s::%s failed: %s', obj.testName, testMethodName, ME.message);
                end
            end
            
            % Return test result structure
            result.name = testMethodName;
            obj.testResults.(testMethodName) = result;
        end
        
        function results = runAllTests(obj)
            % Execute all test methods in the class with test prefix
            %
            % This method finds and runs all methods in the class with names
            % starting with 'test', collecting and reporting the results.
            %
            % OUTPUTS:
            %   results - Aggregated test results (struct)
            
            % Get all methods in the class with 'test' prefix using methods function
            methodList = methods(obj);
            isTestMethod = strncmp('test', methodList, 4);
            testMethods = methodList(isTestMethod);
            
            % Initialize results structure for storing outcomes
            results = struct('testName', obj.testName, 'methods', struct(), 'summary', []);
            
            % Run each test method using runTest
            for i = 1:length(testMethods)
                methodName = testMethods{i};
                methodResult = obj.runTest(methodName);
                results.methods.(methodName) = methodResult;
            end
            
            % Calculate summary statistics (passed, failed, total)
            numTests = length(testMethods);
            numPassed = 0;
            numFailed = 0;
            totalTime = 0;
            
            methodNames = fieldnames(results.methods);
            for i = 1:length(methodNames)
                methodName = methodNames{i};
                methodResult = results.methods.(methodName);
                
                if strcmp(methodResult.status, 'passed')
                    numPassed = numPassed + 1;
                else
                    numFailed = numFailed + 1;
                end
                
                totalTime = totalTime + methodResult.executionTime;
            end
            
            % Return comprehensive test results
            results.summary = struct(...
                'numTests', numTests, ...
                'numPassed', numPassed, ...
                'numFailed', numFailed, ...
                'totalExecutionTime', totalTime ...
            );
            
            % Display summary if verbose mode is enabled
            if obj.verbose
                fprintf('Test Summary for %s:\n', obj.testName);
                fprintf('  Total Tests: %d\n', numTests);
                fprintf('  Passed: %d\n', numPassed);
                fprintf('  Failed: %d\n', numFailed);
                fprintf('  Total Execution Time: %.4f seconds\n', totalTime);
            end
        end
        
        function assertEqual(obj, expected, actual, message)
            % Assert that two values are equal, with type-appropriate comparison
            %
            % INPUTS:
            %   expected - Expected value
            %   actual - Actual value to compare
            %   message - Optional custom error message
            %
            % OUTPUTS:
            %   No return value; throws error if assertion fails
            
            % Set default message if not provided
            if nargin < 4 || isempty(message)
                message = 'Values are not equal';
            end
            
            % Check types of values using isa function
            if isnumeric(expected) && isnumeric(actual)
                % For numeric values, use tolerance-based comparison
                obj.assertAlmostEqual(expected, actual, message);
            elseif ischar(expected) && ischar(actual)
                % For non-numeric values, use exact comparison
                if ~strcmp(expected, actual)
                    error('BaseTest:AssertionFailed', '%s: Expected ''%s'' but got ''%s''', ...
                        message, expected, actual);
                end
            elseif islogical(expected) && islogical(actual)
                % For logical values, use exact comparison
                if expected ~= actual
                    error('BaseTest:AssertionFailed', '%s: Expected %d but got %d', ...
                        message, expected, actual);
                end
            elseif iscell(expected) && iscell(actual)
                % For cell arrays, check dimensions and elements
                if ~isequal(size(expected), size(actual))
                    error('BaseTest:AssertionFailed', '%s: Cell array dimensions do not match', message);
                end
                
                for i = 1:numel(expected)
                    try
                        obj.assertEqual(expected{i}, actual{i}, sprintf('%s (cell element %d)', message, i));
                    catch ME
                        error('BaseTest:AssertionFailed', '%s: Mismatch at cell element %d: %s', ...
                            message, i, ME.message);
                    end
                end
            elseif isstruct(expected) && isstruct(actual)
                % For structs, check fields and their values
                expectedFields = fieldnames(expected);
                actualFields = fieldnames(actual);
                
                if ~isequal(sort(expectedFields), sort(actualFields))
                    error('BaseTest:AssertionFailed', '%s: Struct fields do not match', message);
                end
                
                for i = 1:length(expectedFields)
                    field = expectedFields{i};
                    try
                        obj.assertEqual(expected.(field), actual.(field), sprintf('%s (struct field ''%s'')', message, field));
                    catch ME
                        error('BaseTest:AssertionFailed', '%s: Mismatch in struct field ''%s'': %s', ...
                            message, field, ME.message);
                    end
                end
            else
                % For other types, use exact comparison
                if ~isequal(expected, actual)
                    error('BaseTest:AssertionFailed', '%s: Values are not equal', message);
                end
            end
        end
        
        function assertEqualsWithTolerance(obj, expected, actual, tolerance, message)
            % Assert that two numeric values are equal within specified tolerance
            %
            % INPUTS:
            %   expected - Expected numeric value
            %   actual - Actual numeric value to compare
            %   tolerance - Tolerance for comparison
            %   message - Optional custom error message
            %
            % OUTPUTS:
            %   No return value; throws error if assertion fails
            
            % Use NumericalComparator.compareScalars for robust comparison
            result = obj.numericalComparator.compareScalars(expected, actual, tolerance);
            
            % Set default message if not provided
            if nargin < 5 || isempty(message)
                message = 'Values differ by more than the specified tolerance';
            end
            
            % Check if absolute difference is within tolerance
            if ~result.isEqual
                % Throw error with diagnostic information if comparison fails
                error('BaseTest:AssertionFailed', '%s: Expected %g but got %g (difference: %g, tolerance: %g)', ...
                    message, expected, actual, result.absoluteDifference, tolerance);
            end
        end
        
        function assertMatrixEqualsWithTolerance(obj, expected, actual, tolerance, message)
            % Assert that two matrices are equal element-wise within specified tolerance
            %
            % INPUTS:
            %   expected - Expected matrix
            %   actual - Actual matrix to compare
            %   tolerance - Tolerance for element-wise comparison
            %   message - Optional custom error message
            %
            % OUTPUTS:
            %   No return value; throws error if assertion fails
            
            % Check matrix dimensions using size function
            if ~isequal(size(expected), size(actual))
                if nargin < 5 || isempty(message)
                    message = 'Matrix dimensions do not match';
                end
                error('BaseTest:AssertionFailed', '%s: Expected size %s but got %s', ...
                    message, mat2str(size(expected)), mat2str(size(actual)));
            end
            
            % Set default message if not provided
            if nargin < 5 || isempty(message)
                message = 'Matrices differ by more than the specified tolerance';
            end
            
            % Use NumericalComparator.compareMatrices for robust comparison
            result = obj.numericalComparator.compareMatrices(expected, actual, tolerance);
            
            % Throw error with diagnostic information if any element differs beyond tolerance
            if ~result.isEqual
                % Include indices of mismatched elements in error message
                error('BaseTest:AssertionFailed', '%s: Matrices differ at %d elements. Max difference: %g, tolerance: %g', ...
                    message, result.mismatchCount, result.maxAbsoluteDifference, tolerance);
            end
        end
        
        function assertAlmostEqual(obj, expected, actual, message)
            % Assert that two values are approximately equal using default tolerance
            %
            % INPUTS:
            %   expected - Expected value
            %   actual - Actual value to compare
            %   message - Optional custom error message
            %
            % OUTPUTS:
            %   No return value; throws error if assertion fails
            
            % Set default message if not provided
            if nargin < 4 || isempty(message)
                message = 'Values are not approximately equal';
            end
            
            % Calculate appropriate tolerance using NumericalComparator.calculateTolerance
            tolerance = obj.numericalComparator.calculateTolerance(expected, actual);
            
            % Call assertEqualsWithTolerance or assertMatrixEqualsWithTolerance based on input type
            if isscalar(expected) && isscalar(actual)
                obj.assertEqualsWithTolerance(expected, actual, tolerance, message);
            elseif isnumeric(expected) && isnumeric(actual)
                obj.assertMatrixEqualsWithTolerance(expected, actual, tolerance, message);
            else
                error('BaseTest:InvalidInput', 'Expected and actual values must be numeric');
            end
        end
        
        function assertTrue(obj, condition, message)
            % Assert that a condition is true
            %
            % INPUTS:
            %   condition - Logical condition to verify
            %   message - Optional custom error message
            %
            % OUTPUTS:
            %   No return value; throws error if assertion fails
            
            % Set default message if not provided
            if nargin < 3 || isempty(message)
                message = 'Condition is not true';
            end
            
            % Check if condition is logical true
            if ~all(condition(:))
                % Throw error with message if condition is false
                error('BaseTest:AssertionFailed', '%s', message);
            end
        end
        
        function assertFalse(obj, condition, message)
            % Assert that a condition is false
            %
            % INPUTS:
            %   condition - Logical condition to verify
            %   message - Optional custom error message
            %
            % OUTPUTS:
            %   No return value; throws error if assertion fails
            
            % Set default message if not provided
            if nargin < 3 || isempty(message)
                message = 'Condition is not false';
            end
            
            % Check if condition is logical false
            if any(condition(:))
                % Throw error with message if condition is true
                error('BaseTest:AssertionFailed', '%s', message);
            end
        end
        
        function assertThrows(obj, func, expectedExceptionID, message)
            % Assert that a function throws an expected exception
            %
            % INPUTS:
            %   func - Function handle to execute
            %   expectedExceptionID - Expected exception ID
            %   message - Optional custom error message
            %
            % OUTPUTS:
            %   No return value; throws error if assertion fails
            
            % Set default message if not provided
            if nargin < 4 || isempty(message)
                message = 'Function did not throw the expected exception';
            end
            
            % Try to execute the function in a try-catch block
            try
                func();
                % If no exception occurs, throw assertion error
                error('BaseTest:AssertionFailed', '%s', message);
            catch ME
                % If exception occurs, check if the exception ID matches expected
                if ~strcmp(ME.identifier, expectedExceptionID)
                    error('BaseTest:AssertionFailed', '%s: Expected exception ''%s'' but got ''%s''', ...
                        message, expectedExceptionID, ME.identifier);
                end
            end
        end
        
        function data = loadTestData(obj, dataFileName)
            % Load test data from a MAT file in the test data directory
            %
            % INPUTS:
            %   dataFileName - Name of the data file
            %
            % OUTPUTS:
            %   data - Test data structure
            
            % Construct full path to data file using testDataPath
            dataFilePath = fullfile(obj.testDataPath, dataFileName);
            
            % Validate file exists using exist function
            if ~exist(dataFilePath, 'file')
                error('BaseTest:FileNotFound', 'Test data file ''%s'' not found', dataFilePath);
            end
            
            % Load MAT file using load function
            data = load(dataFilePath);
        end
        
        function executionTime = measureExecutionTime(obj, func, varargin)
            % Measure execution time of a function
            %
            % INPUTS:
            %   func - Function handle to execute
            %   varargin - Arguments to pass to the function
            %
            % OUTPUTS:
            %   executionTime - Execution time in seconds
            
            % Start timer using tic
            startTime = tic;
            
            % Execute function with provided arguments
            try
                func(varargin{:});
            catch ME
                % Re-throw any errors that occur
                rethrow(ME);
            end
            
            % Stop timer using toc
            executionTime = toc(startTime);
        end
        
        function memoryInfo = checkMemoryUsage(obj, func, varargin)
            % Measure memory usage during function execution
            %
            % INPUTS:
            %   func - Function handle to execute
            %   varargin - Arguments to pass to the function
            %
            % OUTPUTS:
            %   memoryInfo - Memory usage information
            
            % Record baseline memory usage
            baseline = whos();
            baselineBytes = sum([baseline.bytes]);
            
            % Execute function with provided arguments
            try
                func(varargin{:});
            catch ME
                % Re-throw any errors that occur
                rethrow(ME);
            end
            
            % Record final memory usage
            final = whos();
            finalBytes = sum([final.bytes]);
            
            % Calculate memory difference
            memoryDiff = finalBytes - baselineBytes;
            
            % Return structured memory usage information
            memoryInfo = struct(...
                'baselineBytes', baselineBytes, ...
                'finalBytes', finalBytes, ...
                'memoryDifference', memoryDiff, ...
                'memoryDifferenceMB', memoryDiff / (1024^2) ...
            );
        end
        
        function filePath = findTestDataFile(obj, fileName)
            % Find a test data file in the test data directory
            %
            % INPUTS:
            %   fileName - Name of the file to find
            %
            % OUTPUTS:
            %   filePath - Full path to the test data file
            
            % Construct path to file in test data directory
            filePath = fullfile(obj.testDataPath, fileName);
            
            % Check if file exists using exist function
            if ~exist(filePath, 'file')
                error('BaseTest:FileNotFound', 'Test data file ''%s'' not found', filePath);
            end
        end
        
        function summary = getSummary(obj)
            % Get summary of test execution results
            %
            % OUTPUTS:
            %   summary - Test summary information
            
            % Compile information from testResults structure
            summary = struct(...
                'testName', obj.testName, ...
                'numTests', 0, ...
                'numPassed', 0, ...
                'numFailed', 0, ...
                'executionTime', 0 ...
            );
            
            % Check if test results are available
            if ~isstruct(obj.testResults) || isempty(fieldnames(obj.testResults))
                return;
            end
            
            % Get field names (test method names)
            fieldNames = fieldnames(obj.testResults);
            
            % Calculate summary statistics
            for i = 1:length(fieldNames)
                field = fieldNames{i};
                
                % Skip non-test fields
                if ismember(field, {'startTime', 'name', 'executionTime', 'status'})
                    continue;
                end
                
                result = obj.testResults.(field);
                
                % Count test
                summary.numTests = summary.numTests + 1;
                
                % Count passed/failed
                if strcmp(result.status, 'passed')
                    summary.numPassed = summary.numPassed + 1;
                else
                    summary.numFailed = summary.numFailed + 1;
                end
                
                % Include timing information
                if isfield(result, 'executionTime')
                    summary.executionTime = summary.executionTime + result.executionTime;
                end
            end
        end
        
        function setVerbose(obj, verboseFlag)
            % Set verbose mode for test output
            %
            % INPUTS:
            %   verboseFlag - Logical flag to enable/disable verbose mode
            %
            % OUTPUTS:
            %   void - No return value
            
            % Validate verboseFlag is logical
            if ~islogical(verboseFlag) && ~(isnumeric(verboseFlag) && (verboseFlag == 0 || verboseFlag == 1))
                error('BaseTest:InvalidInput', 'verboseFlag must be a logical value');
            end
            
            % Set verbose property to specified value
            obj.verbose = logical(verboseFlag);
        end
    end
end