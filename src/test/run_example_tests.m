function varargout = run_example_tests(varargin)
% RUN_EXAMPLE_TESTS Executes all example tests for the MFE Toolbox
%
% This script discovers and runs examples from the src/test/examples directory
% to demonstrate proper usage of the MFE Toolbox through executable tests
% that validate functionality.
%
% USAGE:
%   run_example_tests
%   run_example_tests('verbose', true)
%   run_example_tests('stopOnFail', true)
%   run_example_tests('reportFormats', {'text', 'html'})
%   run_example_tests('groups', {'distributions', 'timeseries'})
%   results = run_example_tests(...)
%
% INPUTS:
%   Optional name-value pairs:
%   'verbose'       - [boolean] Enable verbose output (default: false)
%   'stopOnFail'    - [boolean] Stop on first test failure (default: false)
%   'reportFormats' - [cell array] Report formats to generate (default: {'text'})
%   'groups'        - [cell array] Specific example groups to run (default: all groups)
%                     Valid groups: 'distributions', 'timeseries', 'volatility',
%                     'bootstrap', 'multivariate', 'realized', 'statistical',
%                     'crosssection', 'highfrequency'
%
% OUTPUTS:
%   results - [struct] Test execution results if requested
%
% EXAMPLES:
%   % Run all example tests with default settings
%   run_example_tests
%
%   % Run examples from specific groups with verbose output
%   run_example_tests('verbose', true, 'groups', {'distributions', 'timeseries'})
%
%   % Generate HTML report in addition to text report
%   run_example_tests('reportFormats', {'text', 'html'})
%
% See also: TestRunner, addToPath

% Parse input arguments
config = parseInputArguments(varargin{:});
isVerbose = config.isVerbose;
stopOnFail = config.stopOnFail;
reportFormats = config.reportFormats;
exampleGroups = config.exampleGroups;

% Ensure the MFE Toolbox is properly initialized
if ~ensureToolboxInitialized()
    error('Failed to initialize MFE Toolbox. Please check installation.');
end

% Display execution header
displayHeader(isVerbose, stopOnFail, reportFormats);

% Create and configure TestRunner
testRunner = TestRunner();
testRunner.setVerbose(isVerbose);
testRunner.setStopOnFail(stopOnFail);
testRunner.setReportFormats(reportFormats);

% Create test suites for each example group
discoveryStats = createExampleTestSuites(testRunner, exampleGroups, isVerbose);

% Start timer for execution
tic;

% Execute all test suites with error handling
try
    results = testRunner.run();
    executionSuccess = true;
catch ME
    executionSuccess = false;
    fprintf('\nError during example test execution: %s\n', ME.message);
    if isVerbose && ~isempty(ME.stack)
        fprintf('Stack trace:\n');
        for i = 1:length(ME.stack)
            fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
        end
    end
    
    % Create minimal results structure for output
    results = struct('summary', struct('totalTests', 0, 'passCount', 0, ...
        'failCount', 0, 'errorCount', 1, 'executionTime', toc, 'error', ME));
end

% Calculate total execution time
executionTime = toc;

% Display summary and generate reports only if execution was successful
if executionSuccess
    % Display execution summary
    testRunner.displaySummary();
    
    % Generate test reports
    try
        testRunner.generateReport();
    catch ME
        fprintf('\nWarning: Failed to generate report: %s\n', ME.message);
    end
    
    % Display execution time
    if isVerbose
        fprintf('\nTotal example execution time: %.2f seconds\n', executionTime);
    end
    
    % Set exit code based on success if running in standalone mode
    if isVerbose
        totalFailed = results.summary.failCount + results.summary.errorCount;
        if totalFailed > 0
            fprintf('\nExample tests completed with %d failures and %d errors.\n', ...
                results.summary.failCount, results.summary.errorCount);
        else
            fprintf('\nAll example tests passed successfully!\n');
        end
    end
end

% If running in external context, return results
if nargout > 0
    varargout{1} = results;
end
end

%--------------------------------------------------------------------------
function config = parseInputArguments(varargin)
% Parse and validate input arguments for the test script
%
% INPUTS:
%   varargin - Variable input arguments
%
% OUTPUTS:
%   config - Parsed configuration options

% Initialize default configuration
config = struct();
config.isVerbose = false;
config.stopOnFail = false;
config.reportFormats = {'text'};
config.exampleGroups = {'distributions', 'timeseries', 'volatility', 'bootstrap', ...
                      'multivariate', 'realized', 'statistical', 'crosssection', ...
                      'highfrequency'};

% Process input arguments in pairs or as individual flags
i = 1;
while i <= length(varargin)
    if ischar(varargin{i})
        switch lower(varargin{i})
            case {'verbose', '-v'}
                if i < length(varargin) && (islogical(varargin{i+1}) || isnumeric(varargin{i+1}))
                    config.isVerbose = logical(varargin{i+1});
                    i = i + 2;
                else
                    config.isVerbose = true;
                    i = i + 1;
                end
                
            case {'stoponfail', '-s'}
                if i < length(varargin) && (islogical(varargin{i+1}) || isnumeric(varargin{i+1}))
                    config.stopOnFail = logical(varargin{i+1});
                    i = i + 2;
                else
                    config.stopOnFail = true;
                    i = i + 1;
                end
                
            case {'reportformats', '-r'}
                if i < length(varargin) && iscell(varargin{i+1})
                    config.reportFormats = varargin{i+1};
                    i = i + 2;
                else
                    error('Report formats must be provided as a cell array.');
                end
                
            case {'groups', '-g'}
                if i < length(varargin) && iscell(varargin{i+1})
                    config.exampleGroups = varargin{i+1};
                    i = i + 2;
                else
                    error('Example groups must be provided as a cell array.');
                end
                
            otherwise
                error('Unknown parameter: %s', varargin{i});
        end
    else
        error('Parameters must be specified as name-value pairs.');
    end
end

% Validate configuration values
if ~islogical(config.isVerbose)
    error('verbose parameter must be a logical value.');
end

if ~islogical(config.stopOnFail)
    error('stopOnFail parameter must be a logical value.');
end

if ~iscell(config.reportFormats)
    error('reportFormats parameter must be a cell array.');
end

if ~iscell(config.exampleGroups)
    error('exampleGroups parameter must be a cell array.');
end
end

%--------------------------------------------------------------------------
function success = ensureToolboxInitialized()
% Ensures the MFE Toolbox is properly initialized for running examples
%
% OUTPUTS:
%   success - Initialization success status

% Check if core components are already accessible
if exist('gevpdf', 'file') == 2 && ...
   exist('armaxfilter', 'file') == 2 && ...
   exist('agarch', 'file') == 2
    % Toolbox appears to be already initialized
    success = true;
    return;
end

% Initialize the toolbox if not already available
try
    % Call addToPath to initialize the toolbox (don't save path)
    addToPath(false);
    
    % Verify initialization was successful
    success = exist('gevpdf', 'file') == 2 && ...
              exist('armaxfilter', 'file') == 2 && ...
              exist('agarch', 'file') == 2;
catch
    success = false;
end
end

%--------------------------------------------------------------------------
function displayHeader(isVerbose, stopOnFail, reportFormats)
% Displays a formatted header for the example test execution
%
% INPUTS:
%   isVerbose     - Flag for verbose output
%   stopOnFail    - Flag to stop on first failure
%   reportFormats - Cell array of report formats

% Display separator line and title
fprintf('==========================================================\n');
fprintf('                MFE Toolbox Example Tests                 \n');
fprintf('==========================================================\n');
fprintf('Date: %s\n', datestr(now, 'dd-mmm-yyyy HH:MM:SS'));
fprintf('==========================================================\n\n');

% Display configuration information if verbose mode is enabled
if isVerbose
    fprintf('Configuration:\n');
    fprintf('  Verbose Output: %s\n', conditional(isVerbose, 'Enabled', 'Disabled'));
    fprintf('  Stop on Failure: %s\n', conditional(stopOnFail, 'Enabled', 'Disabled'));
    
    reportFormatStr = '';
    for i = 1:length(reportFormats)
        if i > 1
            reportFormatStr = [reportFormatStr, ', '];
        end
        reportFormatStr = [reportFormatStr, reportFormats{i}];
    end
    fprintf('  Report Formats: %s\n', reportFormatStr);
    
    fprintf('\n');
end
end

%--------------------------------------------------------------------------
function stats = createExampleTestSuites(testRunner, exampleGroups, isVerbose)
% Creates test suites for each example group in the MFE Toolbox
%
% INPUTS:
%   testRunner   - TestRunner instance
%   exampleGroups - Cell array of example groups to run
%   isVerbose     - Flag for verbose output
%
% OUTPUTS:
%   stats - Example discovery statistics by group

% Initialize statistics structure for tracking discovery results
stats = struct();

% For each example group, create a test suite and discover examples
for i = 1:length(exampleGroups)
    groupName = exampleGroups{i};
    
    % Create a test suite for this group
    testSuite = testRunner.createTestSuite(['Example Tests: ' groupName]);
    
    % Build path to examples directory
    examplesPath = fullfile('src', 'test', 'examples', groupName);
    
    % Check if directory exists
    if exist(examplesPath, 'dir')
        % Discover example files with naming pattern '*Example.m'
        numDiscovered = testRunner.discoverTestsInFolder(examplesPath, '*Example.m', ['Example Tests: ' groupName]);
        
        % Store discovery statistics
        stats.(groupName) = numDiscovered;
        
        if isVerbose
            fprintf('Discovered %d example tests in group: %s\n', numDiscovered, groupName);
        end
    else
        stats.(groupName) = 0;
        if isVerbose
            fprintf('Examples directory not found: %s\n', examplesPath);
        end
    end
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