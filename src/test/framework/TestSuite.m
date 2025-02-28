classdef TestSuite
    % TESTSUITE A class that organizes and executes a collection of related test cases for the MFE Toolbox
    %
    % This class provides functionality for test discovery, organization, execution, and result 
    % aggregation to support comprehensive validation of toolbox components.
    %
    % Properties:
    %   name        - Name of the test suite
    %   testCases   - Cell array containing test case instances
    %   reporter    - TestReporter instance for result reporting
    %   results     - Structure containing test execution results
    %   passCount   - Number of passed tests
    %   failCount   - Number of failed tests
    %   errorCount  - Number of tests with errors
    %   isVerbose   - Flag to enable verbose output
    %   stopOnFail  - Flag to stop execution on first failure
    %   isParallel  - Flag to enable parallel execution if supported
    %
    % Methods:
    %   TestSuite             - Constructor to create a new test suite
    %   addTest               - Adds a test case to the suite
    %   addTestsFromFolder    - Discovers and adds test cases from a folder
    %   execute               - Executes all test cases in the suite
    %   executeTest           - Executes a single test case by index or name
    %   getTestCount          - Returns the number of test cases in the suite
    %   getResults            - Returns the results of the last test execution
    %   getSummary            - Returns a summary of test execution results
    %   reset                 - Resets all test results and counters
    %   filter                - Filters test cases based on name pattern
    %   setVerbose            - Sets the verbose mode for execution output
    %   setStopOnFail         - Sets whether to stop on first failure
    %   setParallel           - Sets whether to use parallel execution
    %   setReporter           - Sets a custom TestReporter instance
    %   displaySummary        - Displays test execution summary
    %
    % Example:
    %   % Create a test suite and add tests
    %   suite = TestSuite('Core Components');
    %   suite.addTest(MatrixTest());
    %   suite.addTestsFromFolder('src/test/statistics');
    %   
    %   % Configure and execute tests
    %   suite.setVerbose(true);
    %   suite.setStopOnFail(true);
    %   results = suite.execute();
    %   
    %   % Display summary
    %   suite.displaySummary();
    %
    % See also: BaseTest, TestReporter, parametercheck
    
    properties
        name        % Name of the test suite
        testCases   % Cell array of test cases
        reporter    % TestReporter instance for result reporting
        results     % Structure to store test results
        passCount   % Number of passed tests
        failCount   % Number of failed tests
        errorCount  % Number of tests with errors
        isVerbose   % Flag to enable verbose output
        stopOnFail  % Flag to stop execution on first failure
        isParallel  % Flag to enable parallel execution
    end
    
    methods
        function obj = TestSuite(name, options)
            % Create a new TestSuite instance with the specified name
            %
            % INPUTS:
            %   name    - Name of the test suite (string)
            %   options - Structure with configuration options (optional)
            %
            % OUTPUTS:
            %   obj - TestSuite instance
            
            % Initialize the test suite with the given name
            if nargin < 1 || isempty(name)
                obj.name = 'Unnamed Test Suite';
            else
                obj.name = name;
            end
            
            % Create empty cell array for storing test cases
            obj.testCases = {};
            
            % Initialize pass, fail, and error counts to zero
            obj.passCount = 0;
            obj.failCount = 0;
            obj.errorCount = 0;
            
            % Create a new TestReporter instance
            obj.reporter = TestReporter();
            
            % Set default options for verbose mode, stopOnFail, and parallel execution
            obj.isVerbose = false;
            obj.stopOnFail = false;
            obj.isParallel = false;
            
            % Apply any provided options from the options struct
            if nargin > 1 && ~isempty(options) && isstruct(options)
                if isfield(options, 'isVerbose') && islogical(options.isVerbose)
                    obj.isVerbose = options.isVerbose;
                end
                
                if isfield(options, 'stopOnFail') && islogical(options.stopOnFail)
                    obj.stopOnFail = options.stopOnFail;
                end
                
                if isfield(options, 'isParallel') && islogical(options.isParallel)
                    obj.isParallel = options.isParallel;
                end
                
                if isfield(options, 'reporter') && isa(options.reporter, 'TestReporter')
                    obj.reporter = options.reporter;
                end
            end
            
            % Initialize empty results structure
            obj.results = struct();
        end
        
        function obj = addTest(obj, testCase)
            % Adds a test case to the suite
            %
            % INPUTS:
            %   testCase - A test case instance that is a subclass of BaseTest
            %
            % OUTPUTS:
            %   obj - Self reference for method chaining
            
            % Validate that testCase is a subclass of BaseTest using isa function
            if ~isa(testCase, 'BaseTest')
                error('TestSuite:InvalidTestCase', 'Test case must be a subclass of BaseTest');
            end
            
            % Add the test case to the testCases cell array
            obj.testCases{end+1} = testCase;
            
            % Log addition of test case if verbose mode is enabled
            if obj.isVerbose
                fprintf('Added test case: %s\n', class(testCase));
            end
            
            % Return this TestSuite instance for method chaining
        end
        
        function numAdded = addTestsFromFolder(obj, folderPath, filePattern)
            % Adds all test cases matching a specified pattern from a folder
            %
            % INPUTS:
            %   folderPath   - Path to the folder containing test files
            %   filePattern  - Pattern to match test files (optional, default: '*Test.m')
            %
            % OUTPUTS:
            %   numAdded - Number of tests added
            
            % Validate folderPath is a string and exists using parametercheck
            if ~ischar(folderPath)
                error('TestSuite:InvalidInput', 'folderPath must be a string');
            end
            
            if ~exist(folderPath, 'dir')
                error('TestSuite:InvalidFolder', 'Folder path does not exist: %s', folderPath);
            end
            
            % Use default filePattern '*Test.m' if not provided
            if nargin < 3 || isempty(filePattern)
                filePattern = '*Test.m';
            end
            
            % Find all MATLAB files matching the pattern using dir and regexp
            files = dir(fullfile(folderPath, filePattern));
            numAdded = 0;
            
            % For each matching file, attempt to load and instantiate the test class
            for i = 1:length(files)
                fileName = files(i).name;
                
                % Extract class name (remove .m extension)
                className = fileName(1:end-2);
                
                try
                    % Check if the class exists
                    if exist(className, 'class')
                        % Instantiate the test class
                        testInstance = eval([className '()']);
                        
                        % Verify it's a BaseTest subclass and add to suite
                        if isa(testInstance, 'BaseTest')
                            obj = obj.addTest(testInstance);
                            numAdded = numAdded + 1;
                            
                            if obj.isVerbose
                                fprintf('Loaded test: %s\n', className);
                            end
                        else
                            warning('TestSuite:NotBaseTest', 'Class %s is not a BaseTest subclass', className);
                        end
                    end
                catch ME
                    warning('TestSuite:LoadError', 'Error loading test %s: %s', className, ME.message);
                end
            end
            
            % Count and return the number of tests added
        end
        
        function results = execute(obj)
            % Executes all test cases in the suite
            %
            % OUTPUTS:
            %   results - Summary of test results
            
            % Initialize result tracking variables and timers
            totalTests = length(obj.testCases);
            obj.passCount = 0;
            obj.failCount = 0;
            obj.errorCount = 0;
            obj.results = struct();
            
            % Start execution timer using tic
            suiteStartTime = tic;
            
            % If isVerbose is true, display execution start message
            if obj.isVerbose
                fprintf('Executing test suite: %s (%d tests)\n', obj.name, totalTests);
                fprintf('----------------------------------------\n');
            end
            
            % For each test case in testCases cell array:
            for i = 1:length(obj.testCases)
                testCase = obj.testCases{i};
                
                if obj.isVerbose
                    fprintf('Running test: %s\n', class(testCase));
                end
                
                % If test is BaseTest instance, execute using runAllTests
                if isa(testCase, 'BaseTest')
                    % Execute the test case
                    testResult = testCase.runAllTests();
                    testName = testResult.testName;
                    
                    % Store the result
                    obj.results.(testName) = testResult;
                    
                    % Record test results and update counts
                    obj.passCount = obj.passCount + testResult.summary.numPassed;
                    obj.failCount = obj.failCount + testResult.summary.numFailed;
                    
                    % Report results
                    if ~isempty(obj.reporter)
                        obj.reporter.addTestResults(testResult, testName);
                    end
                    
                    % If stopOnFail is true and test failed, break execution
                    if obj.stopOnFail && testResult.summary.numFailed > 0
                        if obj.isVerbose
                            fprintf('Stopping execution due to test failure\n');
                        end
                        break;
                    end
                end
            end
            
            % Stop execution timer using toc
            suiteExecutionTime = toc(suiteStartTime);
            
            % Compile execution statistics and result summary
            summary = struct(...
                'testSuiteName', obj.name, ...
                'totalTests', totalTests, ...
                'passCount', obj.passCount, ...
                'failCount', obj.failCount, ...
                'errorCount', obj.errorCount, ...
                'executionTime', suiteExecutionTime, ...
                'passRate', (obj.passCount / max(1, (obj.passCount + obj.failCount + obj.errorCount))) * 100 ...
            );
            
            % If reporter is configured, report suite results
            if ~isempty(obj.reporter)
                obj.reporter.displaySummary();
            end
            
            % Display execution summary if verbose mode is enabled
            if obj.isVerbose
                fprintf('----------------------------------------\n');
                fprintf('Test suite execution completed: %s\n', obj.name);
                fprintf('Passed: %d, Failed: %d, Errors: %d\n', obj.passCount, obj.failCount, obj.errorCount);
                fprintf('Execution time: %.2f seconds\n', suiteExecutionTime);
                fprintf('Pass rate: %.1f%%\n', summary.passRate);
            end
            
            % Return comprehensive result structure
            results = struct(...
                'summary', summary, ...
                'details', obj.results ...
            );
        end
        
        function result = executeTest(obj, testIdentifier)
            % Executes a single test case by index or name
            %
            % INPUTS:
            %   testIdentifier - Index or name of test to execute
            %
            % OUTPUTS:
            %   result - Result of the specific test
            
            % Determine if testIdentifier is an index or name
            if isnumeric(testIdentifier) && isscalar(testIdentifier)
                % Validate index is within bounds
                if testIdentifier < 1 || testIdentifier > length(obj.testCases)
                    error('TestSuite:InvalidIndex', 'Invalid test index: %d', testIdentifier);
                end
                
                % Get test case by index
                testCase = obj.testCases{testIdentifier};
            elseif ischar(testIdentifier)
                % Find test by name
                testIndex = 0;
                for i = 1:length(obj.testCases)
                    if strcmp(class(obj.testCases{i}), testIdentifier)
                        testIndex = i;
                        break;
                    end
                end
                
                if testIndex == 0
                    error('TestSuite:TestNotFound', 'Test not found: %s', testIdentifier);
                end
                
                testCase = obj.testCases{testIndex};
            else
                error('TestSuite:InvalidIdentifier', 'Test identifier must be an index or name');
            end
            
            % Execute the test case
            if obj.isVerbose
                fprintf('Executing single test: %s\n', class(testCase));
            end
            
            % Execute the test case using BaseTest.runTest or runAllTests
            testResult = testCase.runAllTests();
            
            % Record and return test results
            testName = testResult.testName;
            obj.results.(testName) = testResult;
            
            % Update counters
            obj.passCount = obj.passCount + testResult.summary.numPassed;
            obj.failCount = obj.failCount + testResult.summary.numFailed;
            
            % Report results
            if ~isempty(obj.reporter)
                obj.reporter.addTestResults(testResult, testName);
            end
            
            result = testResult;
        end
        
        function count = getTestCount(obj)
            % Returns the total number of test cases in the suite
            %
            % OUTPUTS:
            %   count - Number of test cases
            
            % Return the length of the testCases cell array
            count = length(obj.testCases);
        end
        
        function results = getResults(obj)
            % Returns the results of the last test execution
            %
            % OUTPUTS:
            %   results - Test execution results
            
            % Return the results structure containing pass/fail counts and details
            results = obj.results;
        end
        
        function summary = getSummary(obj)
            % Returns a summary of test execution results
            %
            % OUTPUTS:
            %   summary - Summary statistics of test execution
            
            % Compile pass, fail, and error counts
            totalTests = obj.getTestCount();
            totalExecuted = obj.passCount + obj.failCount + obj.errorCount;
            
            % Calculate pass rate percentage
            if totalExecuted > 0
                passRate = (obj.passCount / totalExecuted) * 100;
            else
                passRate = 0;
            end
            
            % Include execution timing information
            executionTime = 0;
            testNames = fieldnames(obj.results);
            
            for i = 1:length(testNames)
                testName = testNames{i};
                if isfield(obj.results.(testName), 'summary') && ...
                   isfield(obj.results.(testName).summary, 'totalExecutionTime')
                    executionTime = executionTime + obj.results.(testName).summary.totalExecutionTime;
                end
            end
            
            % Create a structured summary with all metrics
            summary = struct(...
                'testSuiteName', obj.name, ...
                'totalTests', totalTests, ...
                'testsExecuted', totalExecuted, ...
                'passCount', obj.passCount, ...
                'failCount', obj.failCount, ...
                'errorCount', obj.errorCount, ...
                'passRate', passRate, ...
                'executionTime', executionTime ...
            );
        end
        
        function obj = reset(obj)
            % Resets all test results and counters
            %
            % OUTPUTS:
            %   obj - Self reference for method chaining
            
            % Clear the results struct
            obj.results = struct();
            
            % Reset pass, fail, and error counts to zero
            obj.passCount = 0;
            obj.failCount = 0;
            obj.errorCount = 0;
            
            % Return this TestSuite instance for method chaining
        end
        
        function obj = filter(obj, pattern)
            % Filters test cases based on name pattern
            %
            % INPUTS:
            %   pattern - Pattern to match test names
            %
            % OUTPUTS:
            %   obj - Self reference for method chaining
            
            % Validate pattern is a string using parametercheck
            if ~ischar(pattern)
                error('TestSuite:InvalidInput', 'pattern must be a string');
            end
            
            % Create a new filtered test case array
            filteredTests = {};
            
            % For each test in testCases, check if name matches the pattern
            for i = 1:length(obj.testCases)
                testCase = obj.testCases{i};
                className = class(testCase);
                
                % If name matches, include in the filtered array
                if ~isempty(regexp(className, pattern, 'once'))
                    filteredTests{end+1} = testCase;
                    
                    if obj.isVerbose
                        fprintf('Test matched filter: %s\n', className);
                    end
                end
            end
            
            % Replace testCases with the filtered array
            obj.testCases = filteredTests;
            
            % Return this TestSuite instance for method chaining
        end
        
        function obj = setVerbose(obj, verboseFlag)
            % Sets the verbose mode for execution output
            %
            % INPUTS:
            %   verboseFlag - Enable/disable verbose output
            %
            % OUTPUTS:
            %   obj - Self reference for method chaining
            
            % Validate verboseFlag is logical
            if ~islogical(verboseFlag) && ~(isnumeric(verboseFlag) && (verboseFlag == 0 || verboseFlag == 1))
                error('TestSuite:InvalidInput', 'verboseFlag must be a logical value');
            end
            
            % Set isVerbose property to verboseFlag
            obj.isVerbose = logical(verboseFlag);
            
            % Return this TestSuite instance for method chaining
        end
        
        function obj = setStopOnFail(obj, stopFlag)
            % Sets whether test execution should stop on first failure
            %
            % INPUTS:
            %   stopFlag - Enable/disable stopping on failure
            %
            % OUTPUTS:
            %   obj - Self reference for method chaining
            
            % Validate stopFlag is logical
            if ~islogical(stopFlag) && ~(isnumeric(stopFlag) && (stopFlag == 0 || stopFlag == 1))
                error('TestSuite:InvalidInput', 'stopFlag must be a logical value');
            end
            
            % Set stopOnFail property to stopFlag
            obj.stopOnFail = logical(stopFlag);
            
            % Return this TestSuite instance for method chaining
        end
        
        function obj = setParallel(obj, parallelFlag)
            % Sets whether tests should execute in parallel if supported
            %
            % INPUTS:
            %   parallelFlag - Enable/disable parallel execution
            %
            % OUTPUTS:
            %   obj - Self reference for method chaining
            
            % Validate parallelFlag is logical
            if ~islogical(parallelFlag) && ~(isnumeric(parallelFlag) && (parallelFlag == 0 || parallelFlag == 1))
                error('TestSuite:InvalidInput', 'parallelFlag must be a logical value');
            end
            
            % Set isParallel property to parallelFlag
            obj.isParallel = logical(parallelFlag);
            
            % Check if Parallel Computing Toolbox is available
            if parallelFlag && exist('parfor', 'file') ~= 5
                warning('TestSuite:ParallelNotAvailable', ...
                    'Parallel Computing Toolbox is not available. Tests will run sequentially.');
            end
            
            % Return this TestSuite instance for method chaining
        end
        
        function obj = setReporter(obj, customReporter)
            % Sets a custom TestReporter instance for result reporting
            %
            % INPUTS:
            %   customReporter - TestReporter instance
            %
            % OUTPUTS:
            %   obj - Self reference for method chaining
            
            % Validate customReporter is a TestReporter instance using isa
            if ~isa(customReporter, 'TestReporter')
                error('TestSuite:InvalidReporter', 'Reporter must be a TestReporter instance');
            end
            
            % Set reporter property to customReporter
            obj.reporter = customReporter;
            
            % Return this TestSuite instance for method chaining
        end
        
        function displaySummary(obj)
            % Displays a summary of test execution results to the console
            %
            % OUTPUTS:
            %   void - No return value
            
            % Get test summary using getSummary method
            summary = obj.getSummary();
            
            % If reporter is configured, use reporter.displaySummary
            if ~isempty(obj.reporter)
                obj.reporter.displaySummary();
                return;
            end
            
            % Otherwise, display formatted summary information using fprintf
            fprintf('==========================================================\n');
            fprintf('TEST SUITE SUMMARY: %s\n', obj.name);
            fprintf('==========================================================\n');
            fprintf('Total Tests: %d\n', summary.totalTests);
            fprintf('Executed: %d\n', summary.testsExecuted);
            fprintf('Passed: %d (%.1f%%)\n', summary.passCount, summary.passRate);
            fprintf('Failed: %d\n', summary.failCount);
            fprintf('Errors: %d\n', summary.errorCount);
            fprintf('Execution Time: %.2f seconds\n', summary.executionTime);
            fprintf('==========================================================\n');
            
            % Include failure details if any
            if summary.failCount > 0 || summary.errorCount > 0
                fprintf('\nFAILURE DETAILS:\n');
                fprintf('------------------\n');
                
                % Get all test names
                testNames = fieldnames(obj.results);
                
                % Display details for each failed test
                for i = 1:length(testNames)
                    testName = testNames{i};
                    testResult = obj.results.(testName);
                    
                    if isfield(testResult, 'methods')
                        methodNames = fieldnames(testResult.methods);
                        
                        for j = 1:length(methodNames)
                            methodName = methodNames{j};
                            methodResult = testResult.methods.(methodName);
                            
                            if ~strcmp(methodResult.status, 'passed')
                                fprintf('%s::%s - %s\n', testName, methodName, methodResult.status);
                                
                                if isfield(methodResult, 'error')
                                    fprintf('  Error: %s\n', methodResult.error.message);
                                    
                                    if isfield(methodResult.error, 'stack') && ~isempty(methodResult.error.stack)
                                        fprintf('  Stack Trace:\n');
                                        stack = methodResult.error.stack;
                                        maxEntries = min(3, length(stack));
                                        
                                        for k = 1:maxEntries
                                            fprintf('    %s (line %d)\n', stack(k).name, stack(k).line);
                                        end
                                    end
                                end
                                
                                fprintf('\n');
                            end
                        end
                    end
                end
            end
        end
    end
end