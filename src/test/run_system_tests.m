function results = run_system_tests(varargin)
% RUN_SYSTEM_TESTS Script that executes all system tests for the MFE Toolbox
%
% This script orchestrates the testing of system-level functionality including
% path configuration, toolbox initialization, MEX compilation, and end-to-end
% workflow validation. It validates the operation of the toolbox as a complete
% system across different platforms.
%
% USAGE:
%   run_system_tests
%   run_system_tests('verbose')
%   run_system_tests('stopOnFail')
%   run_system_tests('reportFormats', {'text', 'html'})
%   results = run_system_tests(...)
%
% INPUTS:
%   varargin - Optional inputs in parameter/value pairs or standalone flags:
%              'verbose' or '-v'    : Enable verbose output
%              'stopOnFail' or '-s' : Stop execution on first test failure
%              'reportFormats' or '-r' : Cell array of report formats
%
% OUTPUTS:
%   results  - Test execution results structure (if requested)
%
% EXAMPLES:
%   % Run all system tests with default settings
%   run_system_tests
%
%   % Run all system tests with verbose output
%   run_system_tests('verbose')
%
%   % Run all system tests and generate HTML report
%   run_system_tests('reportFormats', {'text', 'html'})
%
%   % Get test results structure
%   results = run_system_tests('verbose', 'stopOnFail')
%
% See also: TestRunner, BaseTest

% Parse and validate input arguments
config = parseInputArguments(varargin{:});

% Configuration for test execution
isVerbose = config.isVerbose;
stopOnFail = config.stopOnFail;
reportFormats = config.reportFormats;
systemTestDirectory = 'system';

% Ensure the MFE Toolbox is properly initialized
if ~ensureToolboxInitialized()
    error('Failed to initialize MFE Toolbox. Path configuration error.');
end

% Display execution header
displayHeader(isVerbose, stopOnFail, reportFormats);

% Create a TestRunner instance for managing test execution
testRunner = TestRunner();

% Configure TestRunner with user settings
testRunner.setVerbose(isVerbose);
testRunner.setStopOnFail(stopOnFail);
testRunner.setReportFormats(reportFormats);
testRunner.setReportTitle('MFE Toolbox System Test Report');

% Create system test suite
testSuite = testRunner.createTestSuite('System Tests');

% Build path to system test directory
systemTestPath = fullfile('src', 'test', systemTestDirectory);

% Discover and add system tests
testCount = testRunner.discoverTestsInFolder(systemTestPath, '*Test.m', 'System Tests');

if isVerbose
    fprintf('Discovered %d system tests in %s\n', testCount, systemTestPath);
end

% Start timer to measure execution time
tic;

% Execute all system tests
results = testRunner.run();

% Calculate total execution time
executionTime = toc;
if isVerbose
    fprintf('System tests completed in %.2f seconds\n', executionTime);
end

% Display test execution summary
testRunner.displaySummary();

% Generate test reports
reportInfo = testRunner.generateReport();

% Return execution results if requested
if nargout > 0
    % Results already assigned
else
    % Check if we're running in standalone mode (not called by another script)
    callStack = dbstack;
    if length(callStack) <= 1
        % Set exit code based on test success for CI/CD integration
        if results.summary.failCount > 0
            exit(1);  % Failure exit code
        else
            exit(0);  % Success exit code
        end
    end
end

end

function config = parseInputArguments(varargin)
% PARSEINPUTARGUMENTS Parses and validates input arguments for the test script
%
% INPUTS:
%   varargin - Variable input arguments
%
% OUTPUTS:
%   config   - Parsed configuration options structure

% Initialize default configuration
config = struct('isVerbose', false, 'stopOnFail', false, 'reportFormats', {'text'});

% If no arguments, return default configuration
if nargin == 0
    return;
end

% Process input arguments
i = 1;
while i <= nargin
    arg = varargin{i};
    
    % Handle verbose flag
    if ischar(arg) && (strcmpi(arg, 'verbose') || strcmpi(arg, '-v'))
        config.isVerbose = true;
        i = i + 1;
        
    % Handle stopOnFail flag
    elseif ischar(arg) && (strcmpi(arg, 'stopOnFail') || strcmpi(arg, '-s'))
        config.stopOnFail = true;
        i = i + 1;
        
    % Handle reportFormats parameter
    elseif ischar(arg) && (strcmpi(arg, 'reportFormats') || strcmpi(arg, '-r'))
        if i + 1 <= nargin && iscell(varargin{i+1})
            config.reportFormats = varargin{i+1};
            i = i + 2;
        else
            error('reportFormats parameter requires a cell array value');
        end
        
    % Handle direct specification of boolean values for parameters
    elseif ischar(arg) && i + 1 <= nargin
        paramName = arg;
        paramValue = varargin{i+1};
        
        if strcmpi(paramName, 'isVerbose')
            if islogical(paramValue) || (isnumeric(paramValue) && (paramValue == 0 || paramValue == 1))
                config.isVerbose = logical(paramValue);
            else
                error('isVerbose parameter must be a logical value');
            end
            i = i + 2;
            
        elseif strcmpi(paramName, 'stopOnFail')
            if islogical(paramValue) || (isnumeric(paramValue) && (paramValue == 0 || paramValue == 1))
                config.stopOnFail = logical(paramValue);
            else
                error('stopOnFail parameter must be a logical value');
            end
            i = i + 2;
            
        else
            error('Unknown parameter: %s', paramName);
        end
        
    else
        error('Invalid input argument at position %d: %s', i, arg);
    end
end

% Validate reportFormats values
if ~iscell(config.reportFormats)
    error('reportFormats must be a cell array');
end

for i = 1:length(config.reportFormats)
    if ~ischar(config.reportFormats{i})
        error('Each element in reportFormats must be a string');
    end
end

end

function success = ensureToolboxInitialized()
% ENSURETOOLBOXINITIALIZED Ensures the MFE Toolbox is properly initialized for testing
%
% OUTPUTS:
%   success - Logical indicating if initialization was successful

% Check if a core function from the MFE Toolbox is on the path
success = exist('addToPath', 'file') == 2;

% If not on path, attempt to initialize
if ~success
    % Try to find and execute addToPath function
    % First check if we're in a subdirectory of the MFE Toolbox
    currentDir = pwd;
    
    % Try to find addToPath.m in relative paths
    potentialPaths = {
        fullfile('..', 'backend', 'addToPath.m'),
        fullfile('..', '..', 'backend', 'addToPath.m'),
        fullfile('src', 'backend', 'addToPath.m')
    };
    
    for i = 1:length(potentialPaths)
        if exist(potentialPaths{i}, 'file') == 2
            % Add directory to path temporarily
            [pathDir, ~, ~] = fileparts(potentialPaths{i});
            addpath(pathDir);
            break;
        end
    end
    
    % Now try to call addToPath
    if exist('addToPath', 'file') == 2
        success = addToPath(false, true);
    else
        success = false;
        warning('Cannot find addToPath.m. MFE Toolbox initialization failed.');
    end
end

% Verify that critical components are accessible
if success
    criticalFunctions = {'parametercheck', 'matrixdiagnostics'};
    for i = 1:length(criticalFunctions)
        if exist(criticalFunctions{i}, 'file') ~= 2
            success = false;
            warning('Critical function "%s" not found. MFE Toolbox initialization incomplete.', criticalFunctions{i});
            break;
        end
    end
end

end

function displayHeader(isVerbose, stopOnFail, reportFormats)
% DISPLAYHEADER Displays a formatted header for the system test execution
%
% INPUTS:
%   isVerbose     - Flag for verbose output
%   stopOnFail    - Flag for stopping on first failure
%   reportFormats - Cell array of report formats
%
% OUTPUTS:
%   void - No return value

% Display separator line
fprintf('==========================================================\n');

% Display centered title
titleText = 'MFE Toolbox System Tests';
padding = floor((58 - length(titleText)) / 2);
fprintf('%s%s%s\n', repmat(' ', 1, padding), titleText, repmat(' ', 1, padding));

% Display current date and time
currentTime = datestr(now, 'yyyy-mm-dd HH:MM:SS');
fprintf('Timestamp: %s\n', currentTime);

% Display separator line
fprintf('==========================================================\n');

% Display configuration information if verbose mode is enabled
if isVerbose
    fprintf('\nTest Configuration:\n');
    fprintf('  Verbose Mode: Enabled\n');
    
    if stopOnFail
        stopText = 'Enabled';
    else
        stopText = 'Disabled';
    end
    fprintf('  Stop On Fail: %s\n', stopText);
    
    formatStr = '';
    for i = 1:length(reportFormats)
        if i > 1
            formatStr = [formatStr, ', ']; %#ok<AGROW>
        end
        formatStr = [formatStr, reportFormats{i}]; %#ok<AGROW>
    end
    fprintf('  Report Formats: %s\n', formatStr);
    
    % Platform and MATLAB version information
    fprintf('  Platform: %s\n', computer);
    fprintf('  MATLAB Version: %s\n', version);
    fprintf('\n');
end

end