function exitCode = run_cross_platform_tests(varargin)
% RUN_CROSS_PLATFORM_TESTS Script that executes cross-platform tests for the MFE Toolbox
%
% This script orchestrates the execution of cross-platform tests to validate
% the consistency and compatibility of MEX implementations across Windows
% and Unix environments. It generates comprehensive compatibility reports.
%
% USAGE:
%   run_cross_platform_tests
%   run_cross_platform_tests('-v')  % Verbose output
%   run_cross_platform_tests('-r', 'text,html') % Specify report formats
%   run_cross_platform_tests('-o', 'path/to/output') % Specify output directory
%
% OPTIONS:
%   -v, --verbose       Enable verbose output
%   -r, --report FORMAT Specify report format(s) (text,html,xml)
%   -o, --output DIR    Specify output directory
%
% RETURNS:
%   exitCode - 0 if all tests pass, 1 if any fail

% Parse command-line arguments
config = parseInputArguments(varargin{:});

% Display header information
displayHeader();

% Set up cross-platform test environment
testRunner = setupCrossPlatformTests(config);

% Start timing the test execution
tic;

% Execute test suite
results = testRunner.run();

% Measure execution time
executionTime = toc;
results.summary.executionTime = executionTime;

% Generate cross-platform compatibility reports
reportInfo = generateCrossPlatformReports(testRunner, config);

% Display summary of test results
displaySummary(results);

% Return test success status as exit code
if results.summary.failCount > 0
    exitCode = 1;
else
    exitCode = 0;
end
end

function config = parseInputArguments(varargin)
% Parses command-line arguments to configure test execution options

% Initialize default configuration
config = struct();
config.verbose = false;
config.reportFormats = {'text', 'html'};
config.outputDirectory = 'src/test/results/cross_platform/';
config.testComponents = {'agarch_core', 'tarch_core', 'egarch_core', 'igarch_core', 'armaxerrors', 'composite_likelihood'};

% Parse input arguments
i = 1;
while i <= length(varargin)
    arg = varargin{i};
    
    if ischar(arg)
        if strcmp(arg, '-v') || strcmp(arg, '--verbose')
            config.verbose = true;
        elseif (strcmp(arg, '-r') || strcmp(arg, '--report')) && i < length(varargin)
            i = i + 1;
            if ischar(varargin{i})
                config.reportFormats = strsplit(varargin{i}, ',');
            end
        elseif (strcmp(arg, '-o') || strcmp(arg, '--output')) && i < length(varargin)
            i = i + 1;
            if ischar(varargin{i})
                config.outputDirectory = varargin{i};
            end
        end
    end
    
    i = i + 1;
end
end

function displayHeader()
% Displays a formatted header for the cross-platform test execution

% Print separator line
fprintf('==========================================================\n');
fprintf('                MFE Toolbox Cross-Platform Tests          \n');
fprintf('==========================================================\n');

% Print platform information
fprintf('Platform: %s\n', computer());
fprintf('Date: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf('==========================================================\n\n');
end

function testRunner = setupCrossPlatformTests(config)
% Configures and prepares the test environment for cross-platform testing

% Create a new TestRunner instance
testRunner = TestRunner();

% Set verbosity based on config
testRunner.setVerbose(config.verbose);

% Set report formats
testRunner.setReportFormats(config.reportFormats);

% Set output directory for reports
testRunner.setOutputDirectory(config.outputDirectory);

% Create output directory if it doesn't exist
if ~exist(config.outputDirectory, 'dir')
    mkdir(config.outputDirectory);
end

% Create the cross-platform test suite
createCrossPlatformSuite(testRunner);

% Return configured TestRunner instance
end

function createCrossPlatformSuite(testRunner)
% Creates the test suite containing all cross-platform test classes and test methods

% Create a test suite for cross-platform tests
testSuite = testRunner.createTestSuite('Cross-Platform Tests');

% Determine current platform
currentPlatform = computer();

% Platform-specific tests
if strncmpi(currentPlatform, 'PCW', 3) % Windows platform
    % Add Windows-specific MEX tests
    testSuite.addTest(WindowsMEXTest());
    
    % Add platform-specific compatibility tests
    fprintf('Adding Windows-specific tests...\n');
    
elseif strncmpi(currentPlatform, 'GLN', 3) % Unix platform
    % Add Unix-specific MEX tests
    testSuite.addTest(UnixMEXTest());
    
    % Add platform-specific compatibility tests
    fprintf('Adding Unix-specific tests...\n');
end

% Add PlatformCompatibilityTest class to the suite
testSuite.addTest(PlatformCompatibilityTest());

% Add cross-platform validation tests
crossValidator = CrossPlatformValidator();
testSuite.addTest(crossValidator);
end

function reportInfo = generateCrossPlatformReports(testRunner, config)
% Generates reports based on the test execution results

% Set report title
testRunner.setReportTitle('MFE Toolbox Cross-Platform Compatibility Report');

% Configure report formats
testRunner.setReportFormats(config.reportFormats);

% Set output directory
testRunner.setOutputDirectory(config.outputDirectory);

% Generate reports
reportInfo = testRunner.generateReport();

% Print report generation status message
fprintf('\nGenerated cross-platform test reports in %s\n\n', config.outputDirectory);
end

function displaySummary(results)
% Displays a summary of cross-platform test execution results

% Print summary header
fprintf('==========================================================\n');
fprintf('            CROSS-PLATFORM TEST SUMMARY                   \n');
fprintf('==========================================================\n');

% Extract test statistics
totalTests = results.summary.totalTests;
passCount = results.summary.passCount;
failCount = results.summary.failCount;
passRate = (passCount / totalTests) * 100;

% Print test counts
fprintf('Total Tests: %d\n', totalTests);
fprintf('Passed: %d (%.1f%%)\n', passCount, passRate);
fprintf('Failed: %d\n', failCount);

% Print failed tests if any
if failCount > 0
    fprintf('\nFailed Tests:\n');
    
    % Loop through suite results to find failed tests
    suiteNames = fieldnames(results.suiteResults);
    for i = 1:length(suiteNames)
        suite = results.suiteResults.(suiteNames{i});
        if isfield(suite, 'methods')
            methodNames = fieldnames(suite.methods);
            for j = 1:length(methodNames)
                method = suite.methods.(methodNames{j});
                if ~strcmp(method.status, 'passed')
                    fprintf('  %s::%s\n', suiteNames{i}, methodNames{j});
                end
            end
        end
    end
end

% Platform-specific test breakdown
fprintf('\nPlatform-Specific Test Breakdown:\n');

% Count tests by platform type
windowsTests = 0;
unixTests = 0;
crossPlatformTests = 0;

suiteNames = fieldnames(results.suiteResults);
for i = 1:length(suiteNames)
    suite = results.suiteResults.(suiteNames{i});
    if contains(suiteNames{i}, 'Windows')
        windowsTests = windowsTests + suite.summary.totalTests;
    elseif contains(suiteNames{i}, 'Unix')
        unixTests = unixTests + suite.summary.totalTests;
    else
        crossPlatformTests = crossPlatformTests + suite.summary.totalTests;
    end
end

fprintf('  Windows Tests: %d\n', windowsTests);
fprintf('  Unix Tests: %d\n', unixTests);
fprintf('  Cross-Platform Tests: %d\n', crossPlatformTests);

% Cross-platform compatibility metrics
fprintf('\nCross-Platform Compatibility Metrics:\n');
% Would normally display compatibility metrics extracted from PlatformCompatibilityTest results

% Total execution time
fprintf('\nTotal Test Execution Time: %.2f seconds\n', results.summary.executionTime);
fprintf('==========================================================\n');
end