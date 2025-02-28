% A MATLAB script that runs all validation tests for the MFE Toolbox.
% Validation tests verify the correctness of the toolbox's statistical implementations
% by comparing results against known reference values or alternative implementations.

function run_validation_tests(options)
%RUN_VALIDATION_TESTS Runs all validation tests for the MFE Toolbox, verifying the correctness of statistical implementations
%
% INPUTS:
%   options - Optional structure with configuration settings
%
% OUTPUTS:
%   Test execution results and statistics

% Display start message with timestamp
fprintf('Starting MFE Toolbox validation tests at %s\\n', datestr(now));

% Initialize test configuration including verbose mode and report formats
isVerbose = false;
reportFormats = {'text', 'html'};

% Check if options structure is provided
if nargin > 0 && ~isempty(options) && isstruct(options)
    % Check if verbose option is provided
    if isfield(options, 'isVerbose')
        isVerbose = options.isVerbose;
    end
    
    % Check if reportFormats option is provided
    if isfield(options, 'reportFormats')
        reportFormats = options.reportFormats;
    end
end

% Add MFE Toolbox components to MATLAB path using addToPath()
addToPath();

% Create TestRunner instance for managing validation tests
runner = TestRunner();

% Configure TestRunner with verbose mode, report formats, and title
runner.setVerbose(isVerbose);
runner.reporter.setReportFormats(reportFormats);
runner.reporter.setReportTitle('MFE Toolbox Validation Report');

% Start timing the validation process
tic;

% Determine validation approach - either use explicit validation classes or discover tests from validation folder
useExplicitValidation = true; % Set to true for explicit validation, false for discovery

if useExplicitValidation
    % Explicit validation approach: Instantiate and run each validation class
    
    % Instantiate validation classes
    distributionValidation = DistributionValidation();
    armaValidation = ARMAValidation();
    garchValidation = GARCHValidation();
    bootstrapValidation = BootstrapValidation();
    multivariateValidation = MultivariateValidation();
    realizedMeasuresValidation = RealizedMeasuresValidation();
    statisticalTestsValidation = StatisticalTestsValidation();
    
    % Execute validation tests for each component
    distributionResults = distributionValidation.runAllTests();
    armaResults = armaValidation.runAllTests();
    garchResults = garchValidation.runAllTests();
    bootstrapResults = bootstrapValidation.runAllTests();
    multivariateResults = multivariateValidation.runAllMultivariateModels();
    realizedMeasuresResults = realizedMeasuresValidation.runAllValidations();
    statisticalTestsResults = statisticalTestsValidation.runAllValidationTests();
    
    % Aggregate results
    results = struct( ...
        'DistributionValidation', distributionResults, ...
        'ARMAValidation', armaResults, ...
        'GARCHValidation', garchResults, ...
        'BootstrapValidation', bootstrapResults, ...
        'MultivariateValidation', multivariateResults, ...
        'RealizedMeasuresValidation', realizedMeasuresResults, ...
        'StatisticalTestsValidation', statisticalTestsResults);
else
    % Discovery approach: Create TestSuite and use addTestsFromFolder to load validation tests
    suite = TestSuite('MFE Toolbox Validation Tests');
    runner.addTestSuite(suite);
    
    % Add tests from the validation folder
    testCount = suite.addTestsFromFolder('src/test/validation');
    fprintf('Discovered %d validation tests in src/test/validation\\n', testCount);
    
    % Execute the validation tests
    results = runner.run();
end

% Measure total execution time
totalTime = toc;

% Generate validation report in specified formats (text, html)
reportInfo = runner.generateReport();

% Display validation summary with pass/fail counts and statistics
runner.displaySummary();

% Display total execution time
fprintf('Total validation time: %.2f seconds\\n', totalTime);

% Return comprehensive results structure with validation status
return;