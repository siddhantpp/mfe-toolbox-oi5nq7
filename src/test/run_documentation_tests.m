% run_documentation_tests.m
% This script executes documentation validation tests for the MFE Toolbox,
% ensuring help text completeness, format consistency, and example
% functionality. It discovers and runs tests from the
% src/test/documentation directory to validate the quality and accuracy of
% the toolbox's documentation.

% Main script logic that executes documentation tests for the MFE Toolbox
try
    % Step 1: Parse input arguments using parseInputArguments
    config = parseInputArguments(varargin{:});
    
    % Step 2: Ensure toolbox is initialized with ensureToolboxInitialized
    isInitialized = ensureToolboxInitialized();
    if ~isInitialized
        error('MFE:InitializationError', 'Failed to initialize MFE Toolbox for testing.');
    end
    
    % Step 3: Display execution header using displayHeader
    displayHeader();
    
    % Step 4: Create TestRunner instance for managing test execution
    testRunner = TestRunner();
    
    % Step 5: Configure TestRunner with verbose mode and report formats
    testRunner.setVerbose(config.isVerbose);
    testRunner.setReportFormats(config.reportFormats);
    
    % Step 6: Set report title to 'MFE Toolbox Documentation Test Report'
    testRunner.setReportTitle('MFE Toolbox Documentation Test Report');
    
    % Step 7: Set output directory for reports to 'test_reports/documentation'
    testRunner.setOutputDirectory('test_reports/documentation');
    
    % Step 8: Create test suites for each documentation test category using createDocumentationTestSuites
    discoveryStats = createDocumentationTestSuites(testRunner, config.docTestCategories);
    
    % Step 9: Start timer to measure total test execution time
    tic;
    
    % Step 10: Execute all test suites using testRunner.run()
    results = testRunner.run();
    
    % Step 11: Stop timer and calculate total execution time
    executionTime = toc;
    
    % Step 12: Display test execution summary using testRunner.displaySummary()
    testRunner.displaySummary();
    
    % Step 13: Generate test reports using testRunner.generateReport()
    reportInfo = testRunner.generateReport();
    
    % Step 14: If running in external context (not called by run_all_tests.m), return execution results
    if nargin == 0
        % Step 15: Set exit code based on test success if running in standalone mode
        if results.summary.failCount > 0 || results.summary.errorCount > 0
            exit(1); % Indicate failure
        else
            exit(0); % Indicate success
        end
    end
    
catch ME
    fprintf(2, 'An error occurred during documentation tests:\n%s\n', ME.message);
    ME.stack(1);
    exit(1); % Indicate failure due to exception
end

%--------------------------------------------------------------------------
function config = parseInputArguments(varargin)
    % PARSEINPUTARGUMENTS Parses and validates input arguments for the test script
    %
    % INPUTS:
    %   varargin - Variable input arguments
    %
    % OUTPUTS:
    %   config - Parsed configuration options

    % Initialize default configuration (isVerbose=false, reportFormats={'text', 'html'})
    config.isVerbose = false;
    config.reportFormats = {'text', 'html'};
    config.docTestCategories = {'helptext', 'examples'};
    
    % Process input arguments in pairs or as individual flags
    i = 1;
    while i <= nargin
        arg = varargin{i};
        
        if ischar(arg)
            switch lower(arg)
                case {'verbose', '-v'}
                    % Handle 'verbose' or '-v' flag to enable verbose mode
                    config.isVerbose = true;
                    i = i + 1;
                case {'reportformats', '-r'}
                    % Handle 'reportFormats' or '-r' to set report format array
                    if i + 1 <= nargin
                        config.reportFormats = varargin{i + 1};
                        i = i + 2;
                    else
                        error('Missing value for reportFormats');
                    end
                case {'categories', '-c'}
                    % Handle 'categories' or '-c' to specify specific documentation test categories
                    if i + 1 <= nargin
                        config.docTestCategories = varargin{i + 1};
                        i = i + 2;
                    else
                        error('Missing value for categories');
                    end
                otherwise
                    error('Unknown argument: %s', arg);
            end
        else
            error('Invalid argument type: %s', class(arg));
        end
    end
    
    % Validate all argument values are of expected types
    if ~islogical(config.isVerbose)
        error('verbose must be a logical value');
    end
    
    if ~iscell(config.reportFormats)
        error('reportFormats must be a cell array');
    end
    
    if ~iscell(config.docTestCategories)
        error('categories must be a cell array');
    end
    
    % Return parsed configuration as a struct
end

%--------------------------------------------------------------------------
function isInitialized = ensureToolboxInitialized()
    % ENSURETOOLBOXINITIALIZED Ensures the MFE Toolbox is properly initialized for testing
    %
    % OUTPUTS:
    %   isInitialized - Initialization success status

    % Check if toolbox path is already configured
    toolboxPathConfigured = exist('addToPath', 'file') == 2;
    
    % If not properly configured, call addToPath function
    if ~toolboxPathConfigured
        fprintf('MFE Toolbox not initialized. Running addToPath...\n');
        addToPathResult = addToPath(false, true);
        if ~addToPathResult
            warning('addToPath failed. Tests may not run correctly.');
            isInitialized = false;
            return;
        end
    end
    
    % Verify that critical toolbox components are accessible
    isInitialized = exist('sacf', 'file') == 2 && exist('garchfit', 'file') == 2;
    
    % Return boolean indicating initialization success
end

%--------------------------------------------------------------------------
function displayHeader()
    % DISPLAYHEADER Displays a formatted header for the documentation test execution
    %
    % OUTPUTS:
    %   void - No return value

    % Print separator line of equals signs
    fprintf('==========================================================\n');
    
    % Print centered title 'MFE Toolbox Documentation Tests'
    fprintf('%s\n', centerText('MFE Toolbox Documentation Tests', 58));
    
    % Print current date and time
    fprintf('%s\n', centerText(datestr(now), 58));
    
    % Print separator line of equals signs
    fprintf('==========================================================\n');
end

%--------------------------------------------------------------------------
function centeredText = centerText(text, width)
    % CENTERTEXT Centers a text string within a specified width
    %
    % INPUTS:
    %   text - The text string to center
    %   width - The total width to center the text within
    %
    % OUTPUTS:
    %   centeredText - The centered text string

    paddingLength = max(0, floor((width - length(text)) / 2));
    padding = repmat(' ', 1, paddingLength);
    centeredText = [padding, text, padding];
    
    % Ensure the centered text does not exceed the specified width
    centeredText = centeredText(1:min(length(centeredText), width));
end

%--------------------------------------------------------------------------
function discoveryStats = createDocumentationTestSuites(testRunner, docTestCategories)
    % CREATEDOCUMENTATIONTESTSUITES Creates test suites for documentation test categories
    %
    % INPUTS:
    %   testRunner - TestRunner instance
    %   docTestCategories - Cell array of documentation test categories
    %
    % OUTPUTS:
    %   discoveryStats - Test discovery statistics by category

    % Initialize statistics structure for tracking discovery results
    discoveryStats = struct();
    
    % Create a test suite for help text validation tests
    helpTextSuite = testRunner.createTestSuite('Help Text Validation');
    
    % Build path to documentation test directory using fullfile
    docTestDir = fullfile('src', 'test', 'documentation');
    
    % Add HelpTextValidationTest tests to the help text test suite
    helpTextCount = helpTextSuite.addTestsFromFolder(docTestDir, 'HelpTextValidationTest.m');
    
    % Create a test suite for documentation examples validation
    examplesSuite = testRunner.createTestSuite('Documentation Examples');
    
    % Add DocExamplesTest tests to the examples test suite
    examplesCount = examplesSuite.addTestsFromFolder(docTestDir, 'DocExamplesTest.m');
    
    % Store number of discovered tests in statistics structure
    discoveryStats.helpText = helpTextCount;
    discoveryStats.examples = examplesCount;
    
    % Return discovery statistics by category
end