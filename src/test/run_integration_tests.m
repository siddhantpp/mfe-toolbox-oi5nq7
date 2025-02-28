% Run integration tests for the MFE Toolbox
% This script orchestrates the execution of integration tests that validate the interactions between various toolbox components, ensuring proper functionality across distribution functions, time series models, volatility models, and other interconnected features.

% Parse input arguments to configure test execution options
config = parseInputArguments(varargin{:});

% Display header information for the integration test execution
displayHeader();

% Verify that the test environment is properly configured for integration testing
if ~checkTestEnvironment()
    error('Test environment is not properly configured. See output for details.');
end

% Create a TestRunner instance for integration test execution
runner = TestRunner();

% Configure the TestRunner with verbose, stopOnFail and report settings
runner.setVerbose(config.verbose);
runner.setStopOnFail(config.stopOnFail);
runner.setReportFormats(config.reportFormats);
runner.setReportTitle(config.reportTitle);

% Create an integration test suite with 'Integration Tests' name
suite = runner.createTestSuite('Integration Tests');

% Discover all integration tests in the integration folder
integrationTestPath = fullfile('src', 'test', 'integration');
numTests = runner.discoverTestsInFolder(integrationTestPath);

% Start timing test execution using tic
tic;

% Run all discovered integration tests
results = runner.run();

% Measure execution time using toc
executionTime = toc;

% Display test execution summary using runner.displaySummary
runner.displaySummary();

% Generate test report using runner.generateReport
reportInfo = runner.generateReport();

% Return test success status (0 for success, 1 for failures)
if results.summary.failCount > 0 || results.summary.errorCount > 0
    exit(1);
else
    exit(0);
end

function config = parseInputArguments(varargin)
    %parseInputArguments Parses command-line arguments to configure integration test execution options
    %   config = parseInputArguments(varargin) parses command-line arguments to configure integration test execution options.
    %
    %   INPUTS:
    %       varargin - input arguments
    %
    %   OUTPUTS:
    %       struct - Configuration options including verbose mode, report formats, and stop-on-fail flag
    
    % Initialize default configuration with verbose=false, reportFormats={'text', 'html'}, stopOnFail=false
    config = struct('verbose', false, 'reportFormats', {{'text', 'html'}}, 'reportTitle', 'MFE Toolbox Integration Test Report', 'stopOnFail', false);
    
    % Parse varargin for command-line flags and parameters
    i = 1;
    while i <= length(varargin)
        arg = varargin{i};
        
        % Set verbose flag if '-v' or '--verbose' is present
        if strcmp(arg, '-v') || strcmp(arg, '--verbose')
            config.verbose = true;
            i = i + 1;
            
        % Set stop on failure if '-s' or '--stop-on-fail' is present
        elseif strcmp(arg, '-s') || strcmp(arg, '--stop-on-fail')
            config.stopOnFail = true;
            i = i + 1;
            
        % Set report formats if '-r' or '--report' is present with format list
        elseif strcmp(arg, '-r') || strcmp(arg, '--report')
            if i + 1 <= length(varargin)
                formats = regexpi(varargin{i+1}, ',', 'split');
                config.reportFormats = formats;
                i = i + 2;
            else
                fprintf('Error: Missing report format list after -r/--report flag.\n');
                i = i + 1;
            end
        else
            fprintf('Warning: Ignoring invalid command-line argument: %s\n', arg);
            i = i + 1;
        end
    end
    
    % Return the parsed configuration options as a struct
end

function displayHeader()
    %displayHeader Displays a formatted header for the integration test execution
    %   displayHeader() displays a formatted header for the integration test execution, including the title, date, and time.
    
    % Print separator line of equals signs
    fprintf('==========================================================\n');
    
    % Print centered title 'MFE Toolbox Integration Tests'
    fprintf('%s\n', centerText('MFE Toolbox Integration Tests'));
    
    % Print current date and time
    fprintf('%s\n', centerText(datestr(now)));
    
    % Print separator line of equals signs
    fprintf('==========================================================\n');
end

function centeredText = centerText(text)
    %centerText Centers a text string within a 60-character width
    %   centeredText = centerText(text) centers the given text string within a 60-character width by padding with spaces.
    
    width = 60;
    padding = (width - length(text)) / 2;
    centeredText = [repmat(' ', 1, floor(padding)), text, repmat(' ', 1, ceil(padding))];
end

function isValid = checkTestEnvironment()
    %checkTestEnvironment Checks that the test environment is properly configured for integration testing
    %   isValid = checkTestEnvironment() checks that the test environment is properly configured for integration testing, including directory existence and component accessibility.
    
    isValid = true;
    
    % Check that integration test directory exists
    if ~exist(fullfile('src', 'test', 'integration'), 'dir')
        fprintf('Error: Integration test directory not found: %s\n', fullfile('src', 'test', 'integration'));
        isValid = false;
    end
    
    % Verify essential test framework components are accessible
    if ~exist('TestRunner', 'class') || ~exist('TestSuite', 'class') || ~exist('BaseTest', 'class')
        fprintf('Error: Essential test framework components are missing.\n');
        isValid = false;
    end
    
    % Check that critical MFE Toolbox components are accessible
    if ~exist('armaxfilter', 'file') || ~exist('sacf', 'file') || ~exist('pp_test', 'file')
        fprintf('Error: Critical MFE Toolbox components are missing.\n');
        isValid = false;
    end
    
    % Return boolean indicating environment validation status
end