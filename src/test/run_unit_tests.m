function results = run_unit_tests(varargin)
% RUN_UNIT_TESTS Executes all unit tests for the MFE Toolbox
%
% Script that executes all unit tests for the MFE Toolbox, discovering and running 
% tests from the src/test/unit directory hierarchy. Organizes tests by component 
% category, collects results, and generates formatted reports.
%
% USAGE:
%   results = run_unit_tests()
%   results = run_unit_tests('verbose', true)
%   results = run_unit_tests('stopOnFail', true)
%   results = run_unit_tests('reportFormats', {'text', 'html'})
%   results = run_unit_tests('components', {'utility', 'distributions'})
%
% INPUTS:
%   Parameters can be provided as name-value pairs or flags:
%   'verbose'/'-v'        - Enable verbose output (default: false)
%   'stopOnFail'/'-s'     - Stop on first test failure (default: false)
%   'reportFormats'/'-r'  - Cell array of report formats (default: {'text'})
%                          Supported formats: 'text', 'html'
%   'components'/'-c'     - Cell array of component categories to test
%                          (default: all components)
%
% OUTPUTS:
%   results - Structure with test execution results
%
% NOTES:
%   Component categories correspond to the MFE Toolbox directory structure:
%   - utility: Core utility functions
%   - distributions: Statistical distribution functions
%   - bootstrap: Bootstrap methods
%   - timeseries: Time series analysis
%   - realized: High-frequency analysis
%   - univariate: Univariate volatility models
%   - crosssection: Cross-sectional analysis
%   - multivariate: Multivariate volatility models
%   - gui: Graphical user interface
%   - tests: Statistical tests
%
% See also: TestRunner, BaseTest, addToPath

% Parse input arguments
config = parseInputArguments(varargin{:});

% Ensure toolbox is initialized
if ~ensureToolboxInitialized()
    error('Failed to initialize MFE Toolbox. Check installation and path.');
end

% Display header
displayHeader();

% Initialize global variables
global isVerbose stopOnFail reportFormats componentCategories testRunner;
isVerbose = config.isVerbose;
stopOnFail = config.stopOnFail;
reportFormats = config.reportFormats;

% Define component categories to test
if isfield(config, 'components')
    componentCategories = config.components;
else
    componentCategories = {'utility', 'distributions', 'bootstrap', 'timeseries', ...
                          'realized', 'univariate', 'crosssection', 'multivariate', ...
                          'gui', 'tests'};
end

% Create TestRunner instance
testRunner = TestRunner();
testRunner.setVerbose(isVerbose);
testRunner.setStopOnFail(stopOnFail);
testRunner.setReportFormats(reportFormats);

% Create test suites for each component
stats = createComponentTestSuites(testRunner, componentCategories);

% Run all tests and measure execution time
tic;
results = testRunner.run();
executionTime = toc;

% Display summary and generate reports
testRunner.displaySummary();
testRunner.generateReport();

% Return execution results if running in external context
if ~nargout
    clear results;
end

% Set exit code based on test success if running as standalone script
if ~isdeployed && isfield(results, 'summary') && ...
   isfield(results.summary, 'failCount') && results.summary.failCount > 0
    exit(1);  % Non-zero exit code indicates test failures
end
end

function config = parseInputArguments(varargin)
% Parses and validates input arguments for the test script
%
% INPUTS:
%   varargin - Variable input arguments
%
% OUTPUTS:
%   config   - Parsed configuration options

% Initialize default configuration
config.isVerbose = false;
config.stopOnFail = false;
config.reportFormats = {'text'};

% If no arguments, return default config
if nargin == 0
    return;
end

% Process arguments as pairs or individual flags
i = 1;
while i <= numel(varargin)
    arg = varargin{i};
    
    % Handle flags or name-value pairs
    if ischar(arg)
        switch lower(arg)
            case {'verbose', '-v'}
                % Check if next argument is logical, otherwise treat as flag
                if i+1 <= numel(varargin) && (islogical(varargin{i+1}) || ...
                   (isnumeric(varargin{i+1}) && (varargin{i+1} == 0 || varargin{i+1} == 1)))
                    config.isVerbose = logical(varargin{i+1});
                    i = i + 2;
                else
                    config.isVerbose = true;
                    i = i + 1;
                end
            case {'stoponfail', '-s'}
                if i+1 <= numel(varargin) && (islogical(varargin{i+1}) || ...
                   (isnumeric(varargin{i+1}) && (varargin{i+1} == 0 || varargin{i+1} == 1)))
                    config.stopOnFail = logical(varargin{i+1});
                    i = i + 2;
                else
                    config.stopOnFail = true;
                    i = i + 1;
                end
            case {'reportformats', '-r'}
                if i+1 <= numel(varargin) && iscell(varargin{i+1})
                    config.reportFormats = varargin{i+1};
                    i = i + 2;
                else
                    error('reportFormats must be a cell array of strings');
                end
            case {'components', '-c'}
                if i+1 <= numel(varargin) && iscell(varargin{i+1})
                    config.components = varargin{i+1};
                    i = i + 2;
                else
                    error('components must be a cell array of strings');
                end
            otherwise
                error('Unknown parameter: %s', arg);
        end
    else
        error('Parameters must be specified as name-value pairs or flags');
    end
end

% Validate configuration
if ~islogical(config.isVerbose)
    error('verbose must be a logical value');
end

if ~islogical(config.stopOnFail)
    error('stopOnFail must be a logical value');
end

if ~iscell(config.reportFormats)
    error('reportFormats must be a cell array');
end

if isfield(config, 'components') && ~iscell(config.components)
    error('components must be a cell array');
end
end

function success = ensureToolboxInitialized()
% Ensures the MFE Toolbox is properly initialized for testing
%
% OUTPUTS:
%   success - Initialization success status

success = true;

try
    % Call addToPath to ensure toolbox is initialized
    addToPath(false, true);
    
    % Verify that critical toolbox components are accessible
    if exist('TestRunner', 'class') ~= 8
        fprintf('Warning: TestRunner class not found. Path may not be correctly configured.\n');
        success = false;
    end
catch err
    fprintf('Error initializing MFE Toolbox: %s\n', err.message);
    success = false;
end
end

function displayHeader()
% Displays a formatted header for the unit test execution
%
% OUTPUTS:
%   void - No return value

fprintf('================================================================\n');
fprintf('                    MFE Toolbox Unit Tests                      \n');
fprintf('                  %s                   \n', datestr(now));
fprintf('================================================================\n\n');

% Print configuration information if verbose mode is enabled
global isVerbose stopOnFail reportFormats componentCategories;
if isVerbose
    fprintf('Test Configuration:\n');
    fprintf('  Verbose mode: %s\n', mat2str(isVerbose));
    fprintf('  Stop on failure: %s\n', mat2str(stopOnFail));
    fprintf('  Report formats: %s\n', cell2mat(cellfun(@(f) [f ' '], reportFormats, 'UniformOutput', false)));
    fprintf('  Components: %s\n', strjoin(componentCategories, ', '));
    fprintf('\n');
end
end

function stats = createComponentTestSuites(testRunner, componentCategories)
% Creates test suites for each component category in the MFE Toolbox
%
% INPUTS:
%   testRunner - TestRunner instance
%   componentCategories - Cell array of component categories
%
% OUTPUTS:
%   stats - Test discovery statistics by component

% Initialize statistics structure for tracking discovery results
stats = struct();

% For each component category in componentCategories:
for i = 1:length(componentCategories)
    category = componentCategories{i};
    
    % Create test suite with the category name
    testSuite = testRunner.createTestSuite(category);
    
    % Build path to component test directory
    testPath = fullfile('src', 'test', 'unit', category);
    
    % If directory exists, discover tests
    if exist(testPath, 'dir')
        numTests = testRunner.discoverTestsInFolder(testPath, '*Test.m', category);
        stats.(category) = numTests;
        fprintf('Discovered %d tests for component: %s\n', numTests, category);
    else
        stats.(category) = 0;
        fprintf('No test directory found for component: %s\n', category);
    end
end
end