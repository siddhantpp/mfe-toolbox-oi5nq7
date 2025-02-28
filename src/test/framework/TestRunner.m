classdef TestRunner
    % TESTRUNNER Orchestrates the execution of test suites for the MFE Toolbox, managing test discovery, execution, and results reporting
    %
    % The TestRunner class provides a comprehensive interface for discovering,
    % configuring, and executing test suites for the MFE Toolbox. It manages the
    % aggregation of test results and provides reporting capabilities.
    %
    % Properties:
    %   testSuites          - Cell array containing test suite instances
    %   config              - Configuration options structure
    %   reporter            - TestReporter instance for results reporting
    %   results             - Structure containing test execution results
    %   isVerbose           - Flag to enable verbose output
    %   stopOnFail          - Flag to stop execution on first failure
    %   trackPerformance    - Flag to enable performance tracking
    %   totalExecutionTime  - Total execution time in seconds
    %   reportFormats       - Cell array of report formats
    %   reportTitle         - String title for generated reports
    %   outputDirectory     - Directory for report output
    %
    % Methods:
    %   TestRunner            - Constructor to create a new TestRunner instance
    %   addTestSuite          - Adds a test suite to the runner
    %   createTestSuite       - Creates a new test suite and adds it to the runner
    %   discoverTestsInFolder - Discovers and adds tests from a folder
    %   run                   - Executes all test suites
    %   runTestSuite          - Executes a specific test suite
    %   setConfig             - Sets configuration options
    %   setVerbose            - Sets verbose mode
    %   setStopOnFail         - Sets stop-on-fail behavior
    %   setTrackPerformance   - Sets performance tracking
    %   setReportFormats      - Sets report output formats
    %   setReportTitle        - Sets report title
    %   setOutputDirectory    - Sets output directory for reports
    %   getResults            - Returns test execution results
    %   getStatistics         - Returns test statistics
    %   generateReport        - Generates test reports
    %   displaySummary        - Displays result summary
    %   reset                 - Resets test results
    %
    % Example:
    %   % Create a test runner
    %   runner = TestRunner();
    %
    %   % Configure test execution
    %   runner.setVerbose(true);
    %   runner.setStopOnFail(true);
    %   
    %   % Discover and run tests
    %   runner.discoverTestsInFolder('src/test/distributions', '*Test.m', 'Distribution Tests');
    %   results = runner.run();
    %   
    %   % Generate reports
    %   runner.setReportFormats({'text', 'html'});
    %   runner.generateReport();
    %   
    % See also: TestSuite, TestReporter, BaseTest
    
    properties
        testSuites          % Cell array of test suites
        config              % Configuration structure
        reporter            % TestReporter instance
        results             % Results structure
        isVerbose           % Verbose output flag
        stopOnFail          % Stop on failure flag
        trackPerformance    % Performance tracking flag
        totalExecutionTime  % Total execution time
        reportFormats       % Cell array of report formats
        reportTitle         % Report title
        outputDirectory     % Output directory
    end
    
    methods
        function obj = TestRunner(options)
            % Creates a new TestRunner instance with default configuration
            %
            % INPUTS:
            %   options - Optional structure with configuration settings
            %
            % OUTPUTS:
            %   obj - TestRunner instance
            
            % Initialize empty cell array for storing test suites
            obj.testSuites = {};
            
            % Create default configuration with isVerbose=false, stopOnFail=false, trackPerformance=true
            obj.isVerbose = false;
            obj.stopOnFail = false;
            obj.trackPerformance = true;
            
            % Create a new TestReporter instance for results handling
            obj.reporter = TestReporter();
            
            % Initialize empty results structure
            obj.results = struct();
            
            % Set default reportFormats to {'text'}
            obj.reportFormats = {'text'};
            
            % Set default reportTitle to 'MFE Toolbox Test Report'
            obj.reportTitle = 'MFE Toolbox Test Report';
            
            % Set default outputDirectory to current directory
            obj.outputDirectory = pwd;
            
            % Apply any provided options from the options struct
            if nargin > 0 && ~isempty(options) && isstruct(options)
                obj = obj.setConfig(options);
            end
            
            % Set totalExecutionTime to 0
            obj.totalExecutionTime = 0;
        end
        
        function obj = addTestSuite(obj, testSuite)
            % Adds a test suite to the runner for execution
            %
            % INPUTS:
            %   testSuite - TestSuite instance to add
            %
            % OUTPUTS:
            %   obj - Self reference for method chaining
            
            % Validate that testSuite is a TestSuite instance
            if ~isa(testSuite, 'TestSuite')
                error('TestRunner:InvalidInput', 'testSuite must be a TestSuite instance');
            end
            
            % Add the test suite to the testSuites cell array
            obj.testSuites{end+1} = testSuite;
            
            % Log addition of test suite if verbose mode is enabled
            if obj.isVerbose
                fprintf('Added test suite: %s\n', testSuite.name);
            end
            
            % Return this TestRunner instance for method chaining
        end
        
        function testSuite = createTestSuite(obj, suiteName)
            % Creates a new test suite with specified name and adds it to the runner
            %
            % INPUTS:
            %   suiteName - Name for the new test suite
            %
            % OUTPUTS:
            %   testSuite - The newly created test suite
            
            % Create a new TestSuite instance with the provided name
            testSuite = TestSuite(suiteName);
            
            % Configure the test suite with runner's settings (verbose, stopOnFail)
            testSuite.setVerbose(obj.isVerbose);
            testSuite.setStopOnFail(obj.stopOnFail);
            
            % Add the new suite to the runner using addTestSuite
            obj = obj.addTestSuite(testSuite);
            
            % Return the created test suite for further configuration
        end
        
        function count = discoverTestsInFolder(obj, folderPath, filePattern, suiteName)
            % Discovers and adds tests from a specified directory to a test suite
            %
            % INPUTS:
            %   folderPath  - Path to folder containing test files
            %   filePattern - Optional pattern to match test files (default: '*Test.m')
            %   suiteName   - Optional name for the test suite (default: 'Discovered Tests')
            %
            % OUTPUTS:
            %   count - Number of discovered tests
            
            % Validate folderPath exists using exist function
            if ~exist(folderPath, 'dir')
                error('TestRunner:InvalidFolder', 'Folder path does not exist: %s', folderPath);
            end
            
            % Use default filePattern '*Test.m' if not provided
            if nargin < 3 || isempty(filePattern)
                filePattern = '*Test.m';
            end
            
            % Use default suiteName 'Discovered Tests' if not provided
            if nargin < 4 || isempty(suiteName)
                suiteName = 'Discovered Tests';
            end
            
            % Create or retrieve a test suite with the specified name
            suiteExists = false;
            for i = 1:length(obj.testSuites)
                if strcmp(obj.testSuites{i}.name, suiteName)
                    testSuite = obj.testSuites{i};
                    suiteExists = true;
                    break;
                end
            end
            
            if ~suiteExists
                testSuite = obj.createTestSuite(suiteName);
            end
            
            % Use TestSuite.addTestsFromFolder to add tests matching pattern
            count = testSuite.addTestsFromFolder(folderPath, filePattern);
            
            % Log discovery results if verbose mode is enabled
            if obj.isVerbose
                fprintf('Discovered %d tests in folder: %s\n', count, folderPath);
            end
            
            % Return the count of discovered tests
        end
        
        function results = run(obj)
            % Executes all test suites and collects results
            %
            % OUTPUTS:
            %   results - Aggregated test results
            
            % Check if any test suites have been added
            if isempty(obj.testSuites)
                warning('TestRunner:NoTestSuites', 'No test suites have been added.');
                results = struct('summary', struct('totalTests', 0, 'passCount', 0, 'failCount', 0, 'errorCount', 0));
                return;
            end
            
            % Start execution timer using tic
            startTime = tic;
            
            % Initialize result tracking variables
            totalTests = 0;
            passCount = 0;
            failCount = 0;
            errorCount = 0;
            suiteResults = struct();
            
            % If isVerbose is true, display execution start message
            if obj.isVerbose
                fprintf('=== Starting test execution with %d suites ===\n', length(obj.testSuites));
            end
            
            % For each test suite in testSuites cell array:
            for i = 1:length(obj.testSuites)
                testSuite = obj.testSuites{i};
                
                if obj.isVerbose
                    fprintf('Executing test suite: %s\n', testSuite.name);
                end
                
                % Execute the test suite using TestSuite.execute
                suiteResult = testSuite.execute();
                
                % Record test results and update aggregated statistics
                suiteResults.(genvarname(testSuite.name)) = suiteResult;
                totalTests = totalTests + suiteResult.summary.totalTests;
                passCount = passCount + suiteResult.summary.passCount;
                failCount = failCount + suiteResult.summary.failCount;
                errorCount = errorCount + suiteResult.summary.errorCount;
                
                % If reporter is configured, add all results to reporter
                if ~isempty(obj.reporter)
                    obj.reporter.addTestResults(suiteResult, testSuite.name);
                end
                
                % If stopOnFail is true and suite had failures, break execution
                if obj.stopOnFail && failCount > 0
                    if obj.isVerbose
                        fprintf('Stopping execution due to test failure\n');
                    end
                    break;
                end
            end
            
            % Stop execution timer using toc and store in totalExecutionTime
            obj.totalExecutionTime = toc(startTime);
            
            % Compile execution statistics and result summary
            summary = struct(...
                'totalTests', totalTests, ...
                'passCount', passCount, ...
                'failCount', failCount, ...
                'errorCount', errorCount, ...
                'executionTime', obj.totalExecutionTime, ...
                'passRate', (passCount / max(1, totalTests)) * 100 ...
            );
            
            % Return comprehensive result structure
            obj.results = struct(...
                'summary', summary, ...
                'suiteResults', suiteResults ...
            );
            
            results = obj.results;
        end
        
        function results = runTestSuite(obj, suiteIdentifier)
            % Executes a specific test suite by index or name
            %
            % INPUTS:
            %   suiteIdentifier - Index or name of test suite to execute
            %
            % OUTPUTS:
            %   results - Results of the specific suite
            
            % Determine if suiteIdentifier is an index or name
            if isnumeric(suiteIdentifier) && isscalar(suiteIdentifier)
                % Locate the specified test suite in testSuites array
                if suiteIdentifier < 1 || suiteIdentifier > length(obj.testSuites)
                    error('TestRunner:InvalidIndex', 'Invalid test suite index: %d', suiteIdentifier);
                end
                testSuite = obj.testSuites{suiteIdentifier};
            elseif ischar(suiteIdentifier)
                % Find test suite by name
                suiteIndex = 0;
                for i = 1:length(obj.testSuites)
                    if strcmp(obj.testSuites{i}.name, suiteIdentifier)
                        suiteIndex = i;
                        break;
                    end
                end
                
                if suiteIndex == 0
                    error('TestRunner:SuiteNotFound', 'Test suite not found: %s', suiteIdentifier);
                end
                
                testSuite = obj.testSuites{suiteIndex};
            else
                error('TestRunner:InvalidInput', 'Suite identifier must be an index or name');
            end
            
            % Execute the test suite using TestSuite.execute
            results = testSuite.execute();
            
            % Record and return test results
            obj.results.(genvarname(testSuite.name)) = results;
        end
        
        function obj = setConfig(obj, options)
            % Sets configuration options for test execution
            %
            % INPUTS:
            %   options - Structure with configuration options
            %
            % OUTPUTS:
            %   obj - Self reference for method chaining
            
            % Validate options structure
            if ~isstruct(options)
                error('TestRunner:InvalidInput', 'options must be a structure');
            end
            
            % If options.isVerbose is provided, set isVerbose property
            if isfield(options, 'isVerbose')
                obj = obj.setVerbose(options.isVerbose);
            end
            
            % If options.stopOnFail is provided, set stopOnFail property
            if isfield(options, 'stopOnFail')
                obj = obj.setStopOnFail(options.stopOnFail);
            end
            
            % If options.trackPerformance is provided, set trackPerformance property
            if isfield(options, 'trackPerformance')
                obj = obj.setTrackPerformance(options.trackPerformance);
            end
            
            % If options.reportFormats is provided, set reportFormats and update reporter
            if isfield(options, 'reportFormats')
                obj = obj.setReportFormats(options.reportFormats);
            end
            
            % If options.reportTitle is provided, set reportTitle and update reporter
            if isfield(options, 'reportTitle')
                obj = obj.setReportTitle(options.reportTitle);
            end
            
            % If options.outputDirectory is provided, set outputDirectory and update reporter
            if isfield(options, 'outputDirectory')
                obj = obj.setOutputDirectory(options.outputDirectory);
            end
            
            % Return this TestRunner instance for method chaining
        end
        
        function obj = setVerbose(obj, verboseFlag)
            % Sets the verbose mode for execution output
            %
            % INPUTS:
            %   verboseFlag - Logical flag to enable/disable verbose output
            %
            % OUTPUTS:
            %   obj - Self reference for method chaining
            
            % Validate verboseFlag is logical using islogical
            if ~islogical(verboseFlag)
                error('TestRunner:InvalidInput', 'verboseFlag must be a logical value');
            end
            
            % Set isVerbose property to verboseFlag
            obj.isVerbose = verboseFlag;
            
            % Update verbosity setting on all test suites
            for i = 1:length(obj.testSuites)
                obj.testSuites{i}.setVerbose(verboseFlag);
            end
            
            % Update reporter's verbose setting if reporter is initialized
            if ~isempty(obj.reporter)
                obj.reporter.setVerboseOutput(verboseFlag);
            end
            
            % Return this TestRunner instance for method chaining
        end
        
        function obj = setStopOnFail(obj, stopFlag)
            % Sets whether test execution should stop on first failure
            %
            % INPUTS:
            %   stopFlag - Logical flag to enable/disable stopping on failure
            %
            % OUTPUTS:
            %   obj - Self reference for method chaining
            
            % Validate stopFlag is logical using islogical
            if ~islogical(stopFlag)
                error('TestRunner:InvalidInput', 'stopFlag must be a logical value');
            end
            
            % Set stopOnFail property to stopFlag
            obj.stopOnFail = stopFlag;
            
            % Update stopOnFail setting on all test suites
            for i = 1:length(obj.testSuites)
                obj.testSuites{i}.setStopOnFail(stopFlag);
            end
            
            % Return this TestRunner instance for method chaining
        end
        
        function obj = setTrackPerformance(obj, trackFlag)
            % Sets whether performance metrics should be collected during testing
            %
            % INPUTS:
            %   trackFlag - Logical flag to enable/disable performance tracking
            %
            % OUTPUTS:
            %   obj - Self reference for method chaining
            
            % Validate trackFlag is logical using islogical
            if ~islogical(trackFlag)
                error('TestRunner:InvalidInput', 'trackFlag must be a logical value');
            end
            
            % Set trackPerformance property to trackFlag
            obj.trackPerformance = trackFlag;
            
            % Update reporter's performance tracking setting if reporter is initialized
            if ~isempty(obj.reporter)
                obj.reporter.setIncludePerformanceData(trackFlag);
            end
            
            % Return this TestRunner instance for method chaining
        end
        
        function obj = setReportFormats(obj, formats)
            % Sets the output formats for test reports
            %
            % INPUTS:
            %   formats - Cell array of strings with formats ('text', 'html', 'xml')
            %
            % OUTPUTS:
            %   obj - Self reference for method chaining
            
            % Validate formats is a cell array of strings using iscell and ischar
            if ~iscell(formats)
                error('TestRunner:InvalidInput', 'formats must be a cell array of strings');
            end
            
            for i = 1:length(formats)
                if ~ischar(formats{i})
                    error('TestRunner:InvalidInput', 'Each format must be a string');
                end
            end
            
            % Set reportFormats property to the specified formats
            obj.reportFormats = formats;
            
            % Update reporter's report formats if reporter is initialized
            if ~isempty(obj.reporter)
                obj.reporter.setReportFormats(formats);
            end
            
            % Return this TestRunner instance for method chaining
        end
        
        function obj = setReportTitle(obj, title)
            % Sets the title used in generated reports
            %
            % INPUTS:
            %   title - String title for reports
            %
            % OUTPUTS:
            %   obj - Self reference for method chaining
            
            % Validate title is a non-empty string using parametercheck
            parametercheck(title, 'title', struct('isscalar', false));
            
            % Set reportTitle property to title
            obj.reportTitle = title;
            
            % Update reporter's report title if reporter is initialized
            if ~isempty(obj.reporter)
                obj.reporter.setReportTitle(title);
            end
            
            % Return this TestRunner instance for method chaining
        end
        
        function obj = setOutputDirectory(obj, directory)
            % Sets the output directory for generated reports
            %
            % INPUTS:
            %   directory - String path to output directory
            %
            % OUTPUTS:
            %   obj - Self reference for method chaining
            
            % Validate directory is a string and exists or can be created
            if ~ischar(directory)
                error('TestRunner:InvalidInput', 'directory must be a string');
            end
            
            if ~exist(directory, 'dir')
                % Try to create the directory
                [success, message] = mkdir(directory);
                if ~success
                    error('TestRunner:DirectoryError', 'Failed to create directory: %s. Error: %s', directory, message);
                end
            end
            
            % Set outputDirectory property to directory
            obj.outputDirectory = directory;
            
            % Update reporter's output path if reporter is initialized
            if ~isempty(obj.reporter)
                obj.reporter.setReportOutputPath(directory);
            end
            
            % Return this TestRunner instance for method chaining
        end
        
        function results = getResults(obj)
            % Returns the results of the last test execution
            %
            % OUTPUTS:
            %   results - Test execution results
            
            % Return the results structure containing pass/fail counts and details
            results = obj.results;
        end
        
        function stats = getStatistics(obj)
            % Returns statistical information about test execution results
            %
            % OUTPUTS:
            %   stats - Test statistics including counts and pass rates
            
            % If reporter is initialized, use reporter.getTestStatistics
            if ~isempty(obj.reporter)
                stats = obj.reporter.getTestStatistics();
                return;
            end
            
            % Otherwise, calculate statistics from results structure
            if isempty(fieldnames(obj.results))
                % No results yet, return empty statistics
                stats = struct(...
                    'totalTests', 0, ...
                    'passCount', 0, ...
                    'failCount', 0, ...
                    'errorCount', 0, ...
                    'executionTime', obj.totalExecutionTime, ...
                    'passRate', 0 ...
                );
                return;
            end
            
            % Include execution time, pass/fail counts, and pass rate
            if isfield(obj.results, 'summary')
                stats = obj.results.summary;
            else
                % Aggregate results across all suites
                totalTests = 0;
                passCount = 0;
                failCount = 0;
                errorCount = 0;
                
                suiteNames = fieldnames(obj.results);
                for i = 1:length(suiteNames)
                    suiteName = suiteNames{i};
                    if isstruct(obj.results.(suiteName)) && isfield(obj.results.(suiteName), 'summary')
                        totalTests = totalTests + obj.results.(suiteName).summary.totalTests;
                        passCount = passCount + obj.results.(suiteName).summary.passCount;
                        failCount = failCount + obj.results.(suiteName).summary.failCount;
                        if isfield(obj.results.(suiteName).summary, 'errorCount')
                            errorCount = errorCount + obj.results.(suiteName).summary.errorCount;
                        end
                    end
                end
                
                % Return comprehensive statistics structure
                stats = struct(...
                    'totalTests', totalTests, ...
                    'passCount', passCount, ...
                    'failCount', failCount, ...
                    'errorCount', errorCount, ...
                    'executionTime', obj.totalExecutionTime, ...
                    'passRate', (passCount / max(1, totalTests)) * 100 ...
                );
            end
        end
        
        function reportInfo = generateReport(obj)
            % Generates test reports in all configured formats
            %
            % OUTPUTS:
            %   reportInfo - Report generation status and file paths
            
            % Ensure results structure contains data to report
            if isempty(fieldnames(obj.results))
                error('TestRunner:NoResults', 'No test results available. Run tests before generating reports.');
            end
            
            % If reporter is not initialized, create a new TestReporter instance
            if isempty(obj.reporter)
                obj.reporter = TestReporter();
            end
            
            % Configure reporter with reportFormats, reportTitle, and outputDirectory
            obj.reporter.setReportFormats(obj.reportFormats);
            obj.reporter.setReportTitle(obj.reportTitle);
            obj.reporter.setReportOutputPath(obj.outputDirectory);
            
            % Generate reports using reporter.generateReport
            reportInfo = obj.reporter.generateReport();
            
            % Return struct containing generation status and file paths
        end
        
        function displaySummary(obj)
            % Displays a summary of test execution results to the console
            %
            % OUTPUTS:
            %   void - No return value
            
            % If reporter is initialized, use reporter.displaySummary
            if ~isempty(obj.reporter)
                obj.reporter.displaySummary();
                return;
            end
            
            % Otherwise, print formatted summary using fprintf
            stats = obj.getStatistics();
            
            fprintf('==========================================================\n');
            fprintf('TEST EXECUTION SUMMARY\n');
            fprintf('==========================================================\n');
            fprintf('Total Tests: %d\n', stats.totalTests);
            fprintf('Passed: %d (%.1f%%)\n', stats.passCount, stats.passRate);
            fprintf('Failed: %d\n', stats.failCount);
            fprintf('Errors: %d\n', stats.errorCount);
            fprintf('Execution Time: %.2f seconds\n', stats.executionTime);
            fprintf('==========================================================\n');
        end
        
        function obj = reset(obj)
            % Resets all test results and counters
            %
            % OUTPUTS:
            %   obj - Self reference for method chaining
            
            % Clear the results struct
            obj.results = struct();
            
            % Reset totalExecutionTime to zero
            obj.totalExecutionTime = 0;
            
            % If reporter is initialized, clear reporter results
            if ~isempty(obj.reporter)
                obj.reporter.clearResults();
            end
            
            % Return this TestRunner instance for method chaining
        end
    end
end