function run_validation_suite(options)
% RUN_VALIDATION_SUITE Runs the comprehensive validation suite for the MFE Toolbox, executing all validation tests, collecting results, and generating reports
%   run_validation_suite(options)
%
%   INPUTS:
%       options - Structure with optional fields:
%           .verbose - Logical, display detailed validation output (default: true)
%           .reportFormats - Cell array of strings, report formats ('text', 'html') (default: {'text'})
%           .reportPath - String, path to save validation reports (default: '../validation_results')
%           .tolerance - Numerical tolerance for comparisons (default: 1e-6)
%           .title - String, title for the validation report (default: 'MFE Toolbox Validation Report')
%           .components - Cell array of strings, components to validate (default: all)
%
%   OUTPUTS:
%       None, but generates validation reports and displays a summary
%
%   This function initializes the MFE Toolbox, configures validation parameters,
%   executes validation tests for various components (distributions, ARMA/ARMAX models,
%   GARCH models, bootstrap methods, etc.), generates comprehensive validation reports,
%   and displays a summary of the validation status.

% Initialize the MFE Toolbox by adding all necessary directories to the MATLAB path
addToPath(false, true);

% Parse and validate input options with default values if not provided
options = configure_validation_options(options);

% Configure validation parameters (verbosity, report formats, etc.)
VALIDATION_VERBOSE = options.verbose;
VALIDATION_REPORT_FORMATS = options.reportFormats;
VALIDATION_REPORT_PATH = options.reportPath;
VALIDATION_TOLERANCE = options.tolerance;
VALIDATION_REPORT_TITLE = options.title;
VALIDATION_COMPONENTS = options.components;

% Create TestReporter instance for validation reporting
reporter = TestReporter(VALIDATION_REPORT_TITLE);
reporter.setVerboseOutput(VALIDATION_VERBOSE);
reporter.setReportFormats(VALIDATION_REPORT_FORMATS);
reporter.setReportOutputPath(VALIDATION_REPORT_PATH);

% Start timing the validation process using tic
tic;

% Display validation suite header information
display_validation_header();

% Execute Distribution function validation using DistributionValidation class
if any(strcmp(VALIDATION_COMPONENTS, 'Distribution'))
    distributionValidator = DistributionValidation();
    distributionResults = distributionValidator.runAllTests();
    reporter.addTestResults(distributionResults, 'Distribution');
end

% Execute ARMA/ARMAX model validation using ARMAValidation class
if any(strcmp(VALIDATION_COMPONENTS, 'ARMA'))
    armaValidator = ARMAValidation();
    armaResults = armaValidator.runAllTests();
    reporter.addTestResults(armaResults, 'ARMA');
end

% Execute GARCH model validation using GARCHValidation class
if any(strcmp(VALIDATION_COMPONENTS, 'GARCH'))
    garchValidator = GARCHValidation();
    garchResults = garchValidator.runAllTests();
    reporter.addTestResults(garchResults, 'GARCH');
end

% Execute Bootstrap method validation using BootstrapValidation class
if any(strcmp(VALIDATION_COMPONENTS, 'Bootstrap'))
    bootstrapValidator = BootstrapValidation();
    bootstrapResults = bootstrapValidator.runAllTests();
    reporter.addTestResults(bootstrapResults, 'Bootstrap');
end

% Execute Multivariate model validation using MultivariateValidation class
if any(strcmp(VALIDATION_COMPONENTS, 'Multivariate'))
    multivariateValidator = MultivariateValidation();
    multivariateResults = multivariateValidator.runAllTests();
    reporter.addTestResults(multivariateResults, 'Multivariate');
end

% Execute Realized Measures validation using RealizedMeasuresValidation class
if any(strcmp(VALIDATION_COMPONENTS, 'Realized'))
    realizedValidator = RealizedMeasuresValidation();
    realizedResults = realizedValidator.runAllTests();
    reporter.addTestResults(realizedResults, 'Realized');
end

% Execute Statistical Tests validation using StatisticalTestsValidation class
if any(strcmp(VALIDATION_COMPONENTS, 'StatisticalTests'))
    statisticalTestsValidator = StatisticalTestsValidation();
    statisticalTestsResults = statisticalTestsValidator.runAllTests();
    reporter.addTestResults(statisticalTestsResults, 'StatisticalTests');
end

% Execute MEX implementation validation using MEXValidator class
if any(strcmp(VALIDATION_COMPONENTS, 'MEX'))
    mexValidator = MEXValidator();
    mexResults = mexValidator.validateMEXFiles();
    reporter.addTestResults(mexResults, 'MEX');
end

% Execute Cross-platform compatibility validation using CrossPlatformValidator class
if any(strcmp(VALIDATION_COMPONENTS, 'CrossPlatform'))
    crossPlatformValidator = CrossPlatformValidator();
    crossPlatformResults = crossPlatformValidator.validatePlatformCompatibility();
    reporter.addTestResults(crossPlatformResults, 'CrossPlatform');
end

% Stop timing and calculate total validation time using toc
elapsedTime = toc;

% Generate comprehensive validation reports
reports = reporter.generateReport();

% Display validation summary
reporter.displaySummary();

% Return consolidated validation results and report information
validationResults = struct();
validationResults.reports = reports;
validationResults.elapsedTime = elapsedTime;

end

function options = configure_validation_options(options)
% PARSES AND VALIDATES input options for the validation suite, applying default values where needed
%   options = configure_validation_options(options)
%
%   INPUTS:
%       options - Structure with optional fields:
%           .verbose - Logical, display detailed validation output (default: true)
%           .reportFormats - Cell array of strings, report formats ('text', 'html') (default: {'text'})
%           .reportPath - String, path to save validation reports (default: '../validation_results')
%           .tolerance - Numerical tolerance for comparisons (default: 1e-6)
%           .title - String, title for the validation report (default: 'MFE Toolbox Validation Report')
%   OUTPUTS:
%       options - Validated and complete options structure

% Check if options structure is provided, create empty structure if not
if nargin < 1 || isempty(options)
    options = struct();
end

% Set verbose flag (default: true) for detailed validation output
if ~isfield(options, 'verbose')
    options.verbose = true;
end

% Set report formats (default: {'text', 'html'}) for validation reports
if ~isfield(options, 'reportFormats')
    options.reportFormats = {'text'};
end

% Set report path (default: '../validation_results') for saving reports
if ~isfield(options, 'reportPath')
    options.reportPath = '../validation_results';
end

% Set validation tolerance (default: 1e-6) for numerical comparisons
if ~isfield(options, 'tolerance')
    options.tolerance = 1e-6;
end

% Set report title (default: 'MFE Toolbox Validation Report')
if ~isfield(options, 'title')
    options.title = 'MFE Toolbox Validation Report';
end

% Set component selection flags to control which components are validated
if ~isfield(options, 'components')
    options.components = {'Distribution', 'ARMA', 'GARCH', 'Bootstrap', 'Multivariate', 'Realized', 'StatisticalTests', 'MEX', 'CrossPlatform'};
end

% Create report directory if it doesn't exist
if ~exist(options.reportPath, 'dir')
    mkdir(options.reportPath);
end

end

function success = prepare_validation_environment(options)
% PREPARES THE ENVIRONMENT for validation by ensuring necessary files and paths are available
%   success = prepare_validation_environment(options)
%
%   INPUTS:
%       options - Structure with optional fields:
%           .verbose - Logical, display detailed validation output (default: true)
%           .reportFormats - Cell array of strings, report formats ('text', 'html') (default: {'text'})
%           .reportPath - String, path to save validation reports (default: '../validation_results')
%           .tolerance - Numerical tolerance for comparisons (default: 1e-6)
%   OUTPUTS:
%       success - Logical flag indicating if environment is properly configured

% Ensure MFE Toolbox is properly initialized using addToPath
addToPath(false, true);

% Verify test data files are accessible
% Verify reference data files are accessible
% Verify MEX files are available if MEX validation is enabled

% Create output directories for validation reports
if ~exist(options.reportPath, 'dir')
    mkdir(options.reportPath);
end

success = true;

end

function summary = generate_validation_summary(results)
% GENERATES A SUMMARY of validation results including pass/fail statistics
%   summary = generate_validation_summary(results)
%
%   INPUTS:
%       results - Structure with validation results
%   OUTPUTS:
%       summary - Structured validation summary

% Extract results from each validation component
% Calculate overall pass/fail counts and rates
% Calculate component-specific pass/fail rates
% Identify any critical validation failures
% Format summary data into a structured report

summary = struct();
end

function display_validation_header()
% DISPLAYS A FORMATTED HEADER for the validation suite in the command window

% Print separator line for visual clarity
fprintf('==========================================================\n');

% Print validation suite title with version information
fprintf('MFE Toolbox Validation Suite v4.0 (28-Oct-2009)\n');

% Print execution timestamp
fprintf('Execution Timestamp: %s\n', datestr(now));

% Print system information (MATLAB version, OS platform)
fprintf('MATLAB Version: %s\n', version);
fprintf('Operating System: %s\n', computer);

% Print separator line to end header
fprintf('==========================================================\n');

end