classdef DocExamplesTest < BaseTest
    % DOCEXAMPLESTEST Test class for validating documentation examples in the MFE Toolbox
    %
    % This test suite executes the examples provided in the documentation and
    % example files to ensure they function correctly, produce expected outputs,
    % and maintain synchronization with the actual implementation.
    %
    % The DocExamplesTest class validates that the examples are accurate,
    % executable, and consistent with the current implementation. It verifies
    % that the API usage demonstrated in the examples aligns with the design
    % principles of the MFE Toolbox.
    %
    % Properties:
    %   exampleModules - Cell array of paths to example files to test
    %   exampleResults - Struct for storing test outcomes
    %   reporter       - TestReporter instance for logging test results
    %   captureOutput  - Logical flag to capture console output during example execution
    %   standardTolerance - Numerical tolerance for comparisons
    
    properties
        exampleModules     % Cell array of example module paths
        exampleResults     % Struct to store test results
        reporter           % TestReporter instance
        captureOutput      % Logical flag to capture console output
        standardTolerance  % Standard numerical tolerance
    end
    
    methods
        function obj = DocExamplesTest()
            % Initializes a new DocExamplesTest instance
            %
            % The constructor initializes the test environment, sets up the example
            % modules to be tested, creates a TestReporter instance, and configures
            % the output capture and numerical tolerance settings.
            
            % Call the parent class constructor (BaseTest)
            obj = obj@BaseTest();
            
            % Initialize exampleModules with paths to example files to test
            obj.exampleModules = {
                'src/test/examples/DistributionsExample.m',
                'src/test/examples/GARCHExample.m'
            };
            
            % Initialize exampleResults structure for storing test outcomes
            obj.exampleResults = struct();
            
            % Create TestReporter instance for logging test results
            obj.reporter = TestReporter('Documentation Examples Test');
            
            % Set captureOutput to true to capture console output during example execution
            obj.captureOutput = true;
            
            % Set standardTolerance to 1e-8 for numerical comparisons
            obj.standardTolerance = 1e-8;
        end
        
        function setUp(obj)
            % Prepares the test environment before each test case
            %
            % This method configures warning settings, verifies the availability of
            % example files, prepares test data directory access, and sets up
            % exception handling for example execution.
            
            % Call parent class setUp method
            setUp@BaseTest(obj);
            
            % Configure warning settings for example execution
            warning('off', 'MATLAB:rmpath:DirNotFound');
            warning('off', 'MATLAB:load:variableNotFound');
            
            % Verify availability of example files
            for i = 1:length(obj.exampleModules)
                modulePath = obj.exampleModules{i};
                if ~exist(modulePath, 'file')
                    error('DocExamplesTest:MissingExample', 'Example file not found: %s', modulePath);
                end
            end
            
            % Prepare test data directory access
            addpath('src/test/data');
            
            % Set up exception handling for example execution
            obj.exampleResults = struct();
        end
        
        function tearDown(obj)
            % Cleans up after test execution
            %
            % This method resets warning states, cleans up temporary variables,
            % releases file handles, and calls the parent class tearDown method.
            
            % Reset warning state to default
            warning('on', 'MATLAB:rmpath:DirNotFound');
            warning('on', 'MATLAB:load:variableNotFound');
            
            % Clean up any temporary variables created during tests
            clearvars -except obj;
            
            % Release file handles if any were left open
            fclose('all');
            
            % Call parent class tearDown method
            tearDown@BaseTest(obj);
        end
        
        function testDistributionExamples(obj)
            % Tests the examples for statistical distribution functions
            %
            % This method verifies the existence and correct execution of the
            % DistributionsExample module, demonstrating GED, skewed T, and
            % standardized Student's T distributions.
            
            % Verify that DistributionsExample module exists
            if ~obj.verifyExampleAvailability('src/test/examples/DistributionsExample.m')
                return;
            end
            
            % Execute demonstrateGED function with error capturing
            gedResult = obj.executeExample(@demonstrateGED, {});
            obj.assertTrue(gedResult.success, 'demonstrateGED failed');
            
            % Execute demonstrateSkewedT function with error capturing
            skewtResult = obj.executeExample(@demonstrateSkewedT, {});
            obj.assertTrue(skewtResult.success, 'demonstrateSkewedT failed');
            
            % Execute demonstrateStandardizedT function with error capturing
            stdtResult = obj.executeExample(@demonstrateStandardizedT, {});
            obj.assertTrue(stdtResult.success, 'demonstrateStandardizedT failed');
            
            % Log successful execution of distribution examples
            obj.reporter.logMessage('Distribution examples executed successfully');
        end
        
        function testVolatilityExamples(obj)
            % Tests the examples for volatility modeling functions
            %
            % This method verifies the existence and correct execution of the
            % GARCHExample module, demonstrating GARCH volatility modeling.
            
            % Verify that GARCHExample module exists
            if ~obj.verifyExampleAvailability('src/test/examples/GARCHExample.m')
                return;
            end
            
            % Execute run_garch_example function with error capturing
            garchResult = obj.executeExample(@run_garch_example, {});
            obj.assertTrue(garchResult.success, 'run_garch_example failed');
            
            % Load test data for validation
            load financial_returns;
            
            % Execute estimate_multiple_garch_variants with test data
            multipleGarchResult = obj.executeExample(@estimate_multiple_garch_variants, {returns});
            obj.assertTrue(multipleGarchResult.success, 'estimate_multiple_garch_variants failed');
            
            % Log successful execution of volatility examples
            obj.reporter.logMessage('Volatility examples executed successfully');
        end
        
        function testTimeSeriesExamples(obj)
            % Tests the examples for time series modeling functions
            %
            % This method verifies the existence and correct execution of the
            % time series modeling examples.
            
            obj.reporter.logMessage('Time Series examples test not yet implemented');
        end
        
        function testBootstrapExamples(obj)
            % Tests the examples for bootstrap methods
            %
            % This method verifies the existence and correct execution of the
            % bootstrap methods examples.
            
            obj.reporter.logMessage('Bootstrap examples test not yet implemented');
        end
        
        function testMultivariateExamples(obj)
            % Tests the examples for multivariate analysis functions
            %
            % This method verifies the existence and correct execution of the
            % multivariate analysis examples.
            
            obj.reporter.logMessage('Multivariate examples test not yet implemented');
        end
        
        function testCrossSectionExamples(obj)
            % Tests the examples for cross-sectional analysis functions
            %
            % This method verifies the existence and correct execution of the
            % cross-sectional analysis examples.
            
            obj.reporter.logMessage('Cross-Section examples test not yet implemented');
        end
        
        function testHighFrequencyExamples(obj)
            % Tests the examples for high-frequency data analysis
            %
            % This method verifies the existence and correct execution of the
            % high-frequency data analysis examples.
            
            obj.reporter.logMessage('High-Frequency examples test not yet implemented');
        end
        
        function testStatisticalTestsExamples(obj)
            % Tests the examples for statistical testing functions
            %
            % This method verifies the existence and correct execution of the
            % statistical testing functions examples.
            
            obj.reporter.logMessage('Statistical Tests examples test not yet implemented');
        end
        
        function testBasicUsageExamples(obj)
            % Tests the basic usage examples demonstrating core functionality
            %
            % This method verifies the existence and correct execution of the
            % basic usage examples.
            
            obj.reporter.logMessage('Basic Usage examples test not yet implemented');
        end
        
        function result = executeExample(obj, exampleFunction, arguments)
            % Helper method to execute an example function with error checking
            %
            % This method executes the specified example function with the provided
            % arguments, capturing any errors or output that occur during execution.
            %
            % INPUTS:
            %   exampleFunction - Function handle to the example function
            %   arguments       - Cell array of arguments to pass to the function
            %
            % OUTPUTS:
            %   result          - Struct with execution results (output, success, error)
            
            % Verify example function exists and is callable
            if ~isa(exampleFunction, 'function_handle')
                error('DocExamplesTest:InvalidInput', 'exampleFunction must be a function handle');
            end
            
            % Prepare error catching mechanism
            result = struct('output', '', 'success', false, 'error', []);
            
            % If captureOutput is true, use evalc to capture console output
            if obj.captureOutput
                try
                    [result.output] = evalc('exampleFunction(arguments{:});');
                    result.success = true;
                catch ME
                    result.success = false;
                    result.error = ME;
                end
            else
                % Execute example function with provided arguments in try-catch block
                try
                    exampleFunction(arguments{:});
                    result.success = true;
                catch ME
                    result.success = false;
                    result.error = ME;
                end
            end
            
            % If failure occurred, log detailed error information
            if ~result.success
                obj.reporter.logError(sprintf('Example %s failed: %s', func2str(exampleFunction), result.error.message));
            end
        end
        
        function isValid = validateExampleOutput(obj, executionResult, expectedPatterns)
            % Validates that example output matches expected patterns or values
            %
            % This method checks if the output from an executed example contains
            % the expected patterns or values, and validates numerical results
            % against expected tolerance if applicable.
            %
            % INPUTS:
            %   executionResult - Struct with execution results (output, success, error)
            %   expectedPatterns - Struct with expected patterns or values
            %
            % OUTPUTS:
            %   isValid         - Validation success flag (true if all validations pass)
            
            isValid = true;
            
            % Check if execution was successful
            if ~executionResult.success
                obj.reporter.logError('Example execution failed, skipping output validation');
                isValid = false;
                return;
            end
            
            % Extract output text or return value
            output = executionResult.output;
            
            % Verify that output contains expected patterns or values
            if isfield(expectedPatterns, 'text')
                expectedText = expectedPatterns.text;
                if ~contains(output, expectedText)
                    obj.reporter.logError(sprintf('Output does not contain expected text: %s', expectedText));
                    isValid = false;
                end
            end
            
            % Check numerical results against expected tolerance if applicable
            if isfield(expectedPatterns, 'value')
                expectedValue = expectedPatterns.value;
                actualValue = executionResult.value;
                if abs(actualValue - expectedValue) > obj.standardTolerance
                    obj.reporter.logError(sprintf('Value mismatch: expected %g, got %g', expectedValue, actualValue));
                    isValid = false;
                end
            end
            
            % Validate figure generation if visualization was expected
            if isfield(expectedPatterns, 'figure')
                expectedFigureTitle = expectedPatterns.figure;
                % Add logic to check if a figure with the expected title exists
                % This might involve iterating through existing figures and checking their titles
                % (Implementation depends on how figure titles are stored/accessed)
            end
        end
        
        function isAvailable = verifyExampleAvailability(obj, moduleName)
            % Verifies that required example modules are available for testing
            %
            % This method checks if the specified module exists in the MATLAB path
            % and logs a warning if it is not found, skipping the test.
            %
            % INPUTS:
            %   moduleName - Name of the module to verify
            %
            % OUTPUTS:
            %   isAvailable - Availability status (true if module is available)
            
            % Check if module exists in MATLAB path using exist function
            if exist(moduleName, 'file') == 2
                isAvailable = true;
            else
                % If not found, log a warning about skipped test
                obj.reporter.logMessage(sprintf('Skipping test: Module %s not found', moduleName));
                isAvailable = false;
            end
        end
    end
end