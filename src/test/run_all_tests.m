function results = run_all_tests(varargin)
% RUN_ALL_TESTS Master script that orchestrates the execution of all test categories for the MFE Toolbox.
% This script coordinates unit tests, integration tests, MEX tests, performance tests,
% cross-platform tests, validation tests, system tests, documentation tests, and example tests,
% providing a comprehensive verification of the entire toolbox functionality.
%
% INPUTS:
%   varargin - Optional inputs in parameter/value pairs or standalone flags:
%              'verbose' or '-v'    : Enable verbose output
%              'stopOnFail' or '-s' : Stop execution on first test failure
%              'reportFormats' or '-r' : Cell array of report formats
%
% OUTPUTS:
%   results - Test execution results and statistics
%
% COMMENTS:
%   This script is the primary entry point for running all tests in the MFE Toolbox.
%   It ensures that all components are tested and that the toolbox functions correctly
%   across different environments.
%
% EXAMPLES:
%   % Run all tests with default settings
%   run_all_tests
%
%   % Run all tests with verbose output
%   run_all_tests('verbose', true)
%
%   % Run all tests and generate HTML report
%   run_all_tests('reportFormats', {'text', 'html'})
%
% See also: TestRunner, addToPath

% Define global variables for test configuration
global testCategories isVerbose stopOnFail skipCategories reportFormats reportTitle mainReporter categoryResults;

% Define test categories
testCategories = {'unit', 'integration', 'mex', 'performance', 'cross_platform', 'validation', 'system', 'documentation', 'example'};

% Initialize default configuration
isVerbose = false;
stopOnFail = false;
skipCategories = cell(0);
reportFormats = {'text', 'html'};
reportTitle = 'MFE Toolbox Comprehensive Test Report';
mainReporter = [];
categoryResults = struct();

% Step 1: Parse input arguments using parseInputArguments function
config = parseInputArguments(varargin{:});
isVerbose = config.isVerbose;
stopOnFail = config.stopOnFail;
skipCategories = config.skipCategories;
reportFormats = config.reportFormats;
reportTitle = config.reportTitle;

% Step 2: Ensure toolbox is initialized with ensureToolboxInitialized function
if ~ensureToolboxInitialized()
    error('MFE:InitializationError', 'Failed to initialize MFE Toolbox for testing.');
end

% Step 3: Display execution header using displayHeader function
displayHeader();

% Step 4: Start timer for total test execution time
totalStartTime = tic;

% Step 5: Initialize empty struct for storing category results
categoryResults = struct();

% Step 6: For each test category in testCategories:
for i = 1:length(testCategories)
    categoryName = testCategories{i};
    
    % Step 7: Check if category should be run using shouldRunCategory function
    if shouldRunCategory(categoryName, config)
        % Step 8: If category should run, execute it using runTestCategory function
        categoryResults.(categoryName) = runTestCategory(categoryName, config);
        
        % Step 9: If stopOnFail is enabled and category had failures, break execution
        if stopOnFail && categoryResults.(categoryName).summary.failCount > 0
            fprintf('Test execution stopped due to failure in category: %s\\n', categoryName);
            break;
        end
    else
        fprintf('Skipping test category: %s\\n', categoryName);
    end
end

% Step 10: Calculate total execution time
totalExecutionTime = toc(totalStartTime);

% Step 11: Aggregate results from all categories using aggregateResults function
aggregatedResults = aggregateResults(categoryResults);

% Step 12: Generate comprehensive test report using generateMainReport function
reportGenerationStatus = generateMainReport(aggregatedResults, reportFormats, reportTitle);

% Step 13: Display summary of all test execution
TestRunner.displaySummary(aggregatedResults);

% Step 14: If running in standalone mode, set exit code based on test success
if ~isdeployed && isempty(dbstack)
    if aggregatedResults.summary.failCount > 0 || aggregatedResults.summary.errorCount > 0
        exit(1); % Indicate failure
    else
        exit(0); % Indicate success
    end
end

% Step 15: Return consolidated test results
results = aggregatedResults;

%--------------------------------------------------------------------------
function parsedConfig = parseInputArguments(varargin)
% Parses and validates command line arguments for configuring test execution

% Initialize default configuration
parsedConfig.isVerbose = false;
parsedConfig.stopOnFail = false;
parsedConfig.skipCategories = {};
parsedConfig.onlyCategories = {};
parsedConfig.reportFormats = {'text', 'html'};
parsedConfig.reportTitle = 'MFE Toolbox Comprehensive Test Report';

% Process input arguments in pairs or as individual flags
i = 1;
while i <= nargin
    arg = varargin{i};
    
    if ischar(arg)
        switch lower(arg)
            case {'verbose', '-v'}
                % Handle 'verbose' or '-v' flag to enable verbose mode
                parsedConfig.isVerbose = true;
                i = i + 1;
            case {'stoponfail', '-s'}
                % Handle 'stopOnFail' or '-s' flag to enable stopping on first failure
                parsedConfig.stopOnFail = true;
                i = i + 1;
            case {'skip', '-k'}
                % Handle 'skip' or '-k' to specify test categories to skip
                if i + 1 <= nargin && iscell(varargin{i + 1})
                    parsedConfig.skipCategories = varargin{i + 1};
                    i = i + 2;
                else
                    error('Value for skip argument must be a cell array');
                end
            case {'only', '-o'}
                % Handle 'only' or '-o' to specify only specific test categories to run
                if i + 1 <= nargin && iscell(varargin{i + 1})
                    parsedConfig.onlyCategories = varargin{i + 1};
                    i = i + 2;
                else
                    error('Value for only argument must be a cell array');
                end
            case {'reportformats', '-r'}
                % Handle 'reportFormats' or '-r' to set report format array
                if i + 1 <= nargin && iscell(varargin{i + 1})
                    parsedConfig.reportFormats = varargin{i + 1};
                    i = i + 2;
                else
                    error('Value for reportFormats argument must be a cell array');
                end
            case {'reporttitle'}
                 if i + 1 <= nargin && ischar(varargin{i + 1})
                    parsedConfig.reportTitle = varargin{i + 1};
                    i = i + 2;
                 else
                    error('Value for reportTitle argument must be a string');
                 end
            otherwise
                error('Unknown argument: %s', arg);
        end
    else
        error('Invalid argument type: %s', class(arg));
    end
end

% Validate all argument values are of expected types
if ~islogical(parsedConfig.isVerbose)
    error('verbose must be a logical value');
end

if ~islogical(parsedConfig.stopOnFail)
    error('stopOnFail must be a logical value');
end

if ~iscell(parsedConfig.skipCategories)
    error('skipCategories must be a cell array');
end

if ~iscell(parsedConfig.onlyCategories)
    error('onlyCategories must be a cell array');
end

if ~iscell(parsedConfig.reportFormats)
    error('reportFormats must be a cell array');
end

% Return parsed configuration as a struct
end

%--------------------------------------------------------------------------
function success = ensureToolboxInitialized()
% Ensures the MFE Toolbox is properly initialized for testing

% Check if toolbox path is already configured
toolboxPathConfigured = exist('addToPath', 'file') == 2;

% If not properly configured, call addToPath function
if ~toolboxPathConfigured
    fprintf('MFE Toolbox not initialized. Running addToPath...\\n');
    addToPathResult = addToPath(false, true);
    if ~addToPathResult
        warning('addToPath failed. Tests may not run correctly.');
        success = false;
        return;
    end
end

% Verify that critical toolbox components are accessible
success = exist('TestRunner', 'class') == 8 && exist('addToPath', 'file') == 2;
if ~success
    warning('Critical components not found. MFE Toolbox initialization incomplete.');
end

end

%--------------------------------------------------------------------------
function displayHeader()
% Displays a formatted header for the test execution

% Print separator line
fprintf('==========================================================\\n');

% Print centered title
titleText = 'MFE Toolbox Comprehensive Tests';
paddingLength = floor((58 - length(titleText)) / 2);
padding = repmat(' ', 1, paddingLength);
fprintf('%s%s%s\\n', padding, titleText, padding);

% Print current date and time
currentTime = datestr(now, 'yyyy-mm-dd HH:MM:SS');
fprintf('Timestamp: %s\\n', currentTime);

% Print separator line
fprintf('==========================================================\\n\\n');

% Print configuration information if verbose mode is enabled
global isVerbose testCategories skipCategories;
if isVerbose
    fprintf('Configuration:\\n');
    fprintf('  Verbose Mode: %s\\n', conditional(isVerbose, 'Enabled', 'Disabled'));
    fprintf('  Test Categories: %s\\n', strjoin(testCategories, ', '));
    if ~isempty(skipCategories)
        fprintf('  Skipping Categories: %s\\n', strjoin(skipCategories, ', '));
    end
    fprintf('\\n');
end

end

%--------------------------------------------------------------------------
function result = conditional(condition, trueValue, falseValue)
% Helper function to return one of two values based on a condition
if condition
    result = trueValue;
else
    result = falseValue;
end
end

%--------------------------------------------------------------------------
function shouldRun = shouldRunCategory(categoryName, config)
% Determines if a specific test category should be executed based on configuration

% Check if category is in the skipCategories list
if ~isempty(config.skipCategories) && any(strcmp(config.skipCategories, categoryName))
    shouldRun = false;
    return;
end

% Check if onlyCategories is specified and category is not in that list
if ~isempty(config.onlyCategories) && ~any(strcmp(config.onlyCategories, categoryName))
    shouldRun = false;
    return;
end

% If all checks pass, the category should be executed
shouldRun = true;
end

%--------------------------------------------------------------------------
function categoryResults = runTestCategory(categoryName, config)
% Executes tests for a specific category and captures results

% Print start message if verbose mode is enabled
global isVerbose;
if isVerbose
    fprintf('Starting tests for category: %s\\n', categoryName);
end

% Determine script name based on category
scriptName = sprintf('run_%s_tests.m', categoryName);

% Start timer for category execution
startTime = tic;

% Execute category test script with appropriate configuration options
try
    % Construct command string with verbose and report formats
    commandStr = scriptName;
    if config.isVerbose
        commandStr = [commandStr, ' ''verbose'''];
    end
    if ~isempty(config.reportFormats)
        commandStr = [commandStr, ' ''reportFormats'', {''', strjoin(config.reportFormats, ''','''), '''}'];
    end
    
    % Execute the command and capture the output
    eval(commandStr);
    
    % Capture results from the workspace
    categoryResults = evalin('base', 'results');
    
catch ME
    % Handle any errors during test execution
    fprintf(2, 'An error occurred during %s tests:\\n%s\\n', categoryName, ME.message);
    categoryResults = struct();
    categoryResults.summary.failCount = 1;
    categoryResults.summary.errorCount = 1;
end

% Stop timer and calculate execution time
executionTime = toc(startTime);

% Add execution time to category results
categoryResults.summary.executionTime = executionTime;

% Print completion message with execution time if verbose mode is enabled
if isVerbose
    fprintf('Completed tests for category: %s (%.2f seconds)\\n', categoryName, executionTime);
end

end

%--------------------------------------------------------------------------
function aggregatedResults = aggregateResults(categoryResults)
% Combines results from all test categories into a consolidated report

% Initialize aggregate counters
totalTests = 0;
passCount = 0;
failCount = 0;
errorCount = 0;
aggregatedTime = 0;

% Get category names
categoryNames = fieldnames(categoryResults);

% For each test category in categoryResults:
for i = 1:length(categoryNames)
    categoryName = categoryNames{i};
    
    % Add category pass count to aggregate passed counter
    passCount = passCount + categoryResults.(categoryName).summary.passCount;
    
    % Add category fail count to aggregate failed counter
    failCount = failCount + categoryResults.(categoryName).summary.failCount;
    
    % Add category error count to aggregate error counter
    errorCount = errorCount + categoryResults.(categoryName).summary.errorCount;
    
    % Add category total test count to aggregate total test counter
    totalTests = totalTests + categoryResults.(categoryName).summary.totalTests;
    
    % Add category execution time to aggregated time
    aggregatedTime = aggregatedTime + categoryResults.(categoryName).summary.executionTime;
end

% Calculate overall pass rate
if totalTests > 0
    passRate = (passCount / totalTests) * 100;
else
    passRate = 0;
end

% Return structure with aggregated statistics and detailed category results
aggregatedResults = struct();
aggregatedResults.summary.totalTests = totalTests;
aggregatedResults.summary.passCount = passCount;
aggregatedResults.summary.failCount = failCount;
aggregatedResults.summary.errorCount = errorCount;
aggregatedResults.summary.passRate = passRate;
aggregatedResults.summary.executionTime = aggregatedTime;
aggregatedResults.categoryResults = categoryResults;

end

%--------------------------------------------------------------------------
function reportGenerationStatus = generateMainReport(aggregatedResults, reportFormats, reportTitle)
% Generates comprehensive test report combining all test categories

% Create TestReporter instance for main report
global mainReporter;
mainReporter = TestReporter();

% Configure report formats and title
mainReporter.setReportFormats(reportFormats);
mainReporter.setReportTitle(reportTitle);

% Add all test results to the reporter
categoryNames = fieldnames(aggregatedResults.categoryResults);
for i = 1:length(categoryNames)
    categoryName = categoryNames{i};
    mainReporter.addTestResults(aggregatedResults.categoryResults.(categoryName), categoryName);
end

% Generate comprehensive report in all specified formats
reportGenerationStatus = mainReporter.generateReport();

end