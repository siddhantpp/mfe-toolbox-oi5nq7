function results = run_mex_tests(options)
% RUN_MEX_TESTS Main function that orchestrates the execution of MEX-related tests for the MFE Toolbox
%
% INPUTS:
%   options - Optional structure with configuration settings
%
% OUTPUTS:
%   results - Test execution results and statistics

    % Initialize settings and parse options parameter
    defaultOptions = struct('verbose', true, 'generateReport', true, 'outputDirectory', '../test/reports', 'reportTitle', 'MFE Toolbox MEX Test Report');
    if nargin < 1 || isempty(options)
        options = defaultOptions;
    else
        options = parse_options(options);
    end

    % Display start message with timestamp and platform information
    disp(['Starting MEX tests at ' datestr(now) ' on ' computer()]);

    % Create a TestRunner instance for test orchestration
    runner = TestRunner();

    % Configure TestRunner with verbosity, report settings, and output path
    runner.setConfig(options);

    % Create a MEX test suite for organizing MEX-specific tests
    mexSuite = setup_mex_suite(runner, options);

    % Execute the test suite and collect results
    results = runner.run();

    % Validate that all MEX files exist and can be loaded
    validationResults = validate_mex_binaries();

    % Display summary of test execution results
    runner.displaySummary();

    % Generate detailed test report if GENERATE_REPORT is true
    if options.generateReport
        runner.generateReport();
    end

    % Return test execution results and statistics
end

function mexSuite = setup_mex_suite(runner, options)
% SETUP_MEX_SUITE Sets up and configures the MEX test suite with all required test cases
%
% INPUTS:
%   runner - TestRunner instance
%   options - Options structure
%
% OUTPUTS:
%   mexSuite - Configured MEX test suite

    % Create a new test suite named 'MEX Tests' using the runner
    mexSuite = runner.createTestSuite('MEX Tests');

    % Discover all tests in the mex directory matching '*Test.m'
    runner.discoverTestsInFolder('src/test/mex', '*Test.m', 'MEX Tests');

    % Add specific MEXCompilationTest instance for compilation testing
    mexSuite.addTest(MEXCompilationTest());

    % Add MEXPerformanceTest instance configured with performance thresholds
    mexSuite.addTest(MEXPerformanceTest());

    % Configure the suite with verbosity and stop-on-fail settings from options
    mexSuite.setVerbose(options.verbose);
    mexSuite.setStopOnFail(options.stopOnFail);

    % Return the fully configured test suite ready for execution
end

function validationResults = validate_mex_binaries()
% VALIDATE_MEX_BINARIES Validates that all required MEX binaries exist and can be loaded
%
% OUTPUTS:
%   validationResults - Validation results for all MEX binaries

    % Create a MEXValidator instance for validation
    validator = MEXValidator();

    % Determine the current platform using computer()
    currentPlatform = computer();

    % Get the correct MEX file extension using getMEXExtension
    mexExtension = validator.getMEXExtension();

    % Define the required MEX binaries
    requiredMEXFiles = {'agarch_core', 'armaxerrors', 'composite_likelihood', 'egarch_core', 'igarch_core', 'tarch_core'};
    
    % Initialize validation results structure
    validationResults = struct();

    % Check each required MEX binary exists in the dlls directory
    for i = 1:length(requiredMEXFiles)
        mexFileName = requiredMEXFiles{i};
        fullPath = fullfile('src/backend/dlls', [mexFileName, '.', mexExtension]);

        % Check if the file exists
        if exist(fullPath, 'file') == 3  % 3 indicates a MEX file
            % Verify MEX binaries can be loaded without errors
            try
                % Attempt to load the MEX file
                clear(mexFileName);  % Clear MEX file from memory
                % Check if the function exists
                if exist(mexFileName, 'file') == 3
                    % If it exists, display a success message
                    fprintf('MEX file %s loaded successfully.\n', mexFileName);
                    validationResults.(mexFileName) = true;
                else
                    % If it does not exist, display an error message
                    fprintf('MEX file %s not found.\n', mexFileName);
                    validationResults.(mexFileName) = false;
                end
            catch ME
                % If an error occurs, display the error message
                fprintf('Error loading MEX file %s: %s\n', mexFileName, ME.message);
                validationResults.(mexFileName) = false;
            end
        else
            fprintf('MEX file %s not found.\n', mexFileName);
            validationResults.(mexFileName) = false;
        end
    end

    % Return validation results with details for each MEX binary
end

function options = parse_options(options)
% PARSE_OPTIONS Parses and validates test execution options with defaults
%
% INPUTS:
%   options - Structure with options
%
% OUTPUTS:
%   options - Validated and defaulted options

    % Initialize default options (verbose=VERBOSE_MODE, generateReport=GENERATE_REPORT)
    options.verbose = true;
    options.generateReport = true;

    % If options provided, override defaults with provided values
    if isfield(options, 'verbose')
        options.verbose = logical(options.verbose);
    end
    if isfield(options, 'generateReport')
        options.generateReport = logical(options.generateReport);
    end
    if isfield(options, 'stopOnFail')
        options.stopOnFail = logical(options.stopOnFail);
    else
        options.stopOnFail = false;
    end

    % Validate option types (verbose is logical, etc.)
    if ~islogical(options.verbose)
        error('verbose option must be a logical value');
    end
    if ~islogical(options.generateReport)
        error('generateReport option must be a logical value');
    end
    if ~islogical(options.stopOnFail)
        error('stopOnFail option must be a logical value');
    end

    % Set output directory to OUTPUT_DIRECTORY or from options
    if isfield(options, 'outputDirectory')
        options.outputDirectory = options.outputDirectory;
    else
        options.outputDirectory = '../test/reports';
    end

    % Set report title to REPORT_TITLE or from options
    if isfield(options, 'reportTitle')
        options.reportTitle = options.reportTitle;
    else
        options.reportTitle = 'MFE Toolbox MEX Test Report';
    end

    % Return validated options structure
end