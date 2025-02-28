function run_performance_tests(varargin)
    %RUN_PERFORMANCE_TESTS Runs performance tests for the MFE Toolbox
    %   This script orchestrates the execution of performance benchmark tests for the MFE Toolbox.
    %   It measures execution time, memory usage, and scalability across various toolbox components.
    
    % Initialize default configuration
    config.verbose = false;
    config.visualizationsEnabled = true;
    config.reportFormats = {'text', 'html'};
    config.speedupThreshold = 1.5;
    config.resultsDir = './performance_results';

    % Parse varargin for command-line options and parameters
    i = 1;
    while i <= nargin
        arg = varargin{i};
        switch arg
            case {'-v', '--verbose'}
                config.verbose = true;
                i = i + 1;
            case {'--no-visuals'}
                config.visualizationsEnabled = false;
                i = i + 1;
            case {'--visuals'}
                config.visualizationsEnabled = true;
                i = i + 1;
            case {'-r', '--report'}
                if i + 1 <= nargin
                    config.reportFormats = strsplit(varargin{i+1}, ',');
                    i = i + 2;
                else
                    fprintf('Error: Report formats must be specified after -r or --report.\n');
                    return;
                end
            case {'-t', '--threshold'}
                if i + 1 <= nargin
                    config.speedupThreshold = str2double(varargin{i+1});
                    i = i + 2;
                else
                    fprintf('Error: Speedup threshold must be specified after -t or --threshold.\n');
                    return;
                end
            case {'-d', '--dir'}
                if i + 1 <= nargin
                    config.resultsDir = varargin{i+1};
                    i = i + 2;
                else
                    fprintf('Error: Results directory must be specified after -d or --dir.\n');
                    return;
                end
            otherwise
                fprintf('Warning: Unknown argument %s.\n', arg);
                i = i + 1;
        end
    end

    % Display header information
    displayHeader();

    % Create TestRunner instance
    runner = TestRunner();
    runner.setVerbose(config.verbose);
    runner.setTrackPerformance(true);

    % Configure global performance test settings
    configurePerformanceTests(config);

    % Create test suites
    distributionSuite = createPerformanceTestSuite(runner, 'Distribution Performance Tests', config);
    timeSeriesSuite = createPerformanceTestSuite(runner, 'Time Series Performance Tests', config);
    volatilitySuite = createPerformanceTestSuite(runner, 'Volatility Model Performance Tests', config);
    multivariateSuite = createPerformanceTestSuite(runner, 'Multivariate Model Performance Tests', config);
    bootstrapSuite = createPerformanceTestSuite(runner, 'Bootstrap Method Performance Tests', config);
    largeScaleSuite = createPerformanceTestSuite(runner, 'Large-Scale Data Tests', config);
    mexSpeedupSuite = createPerformanceTestSuite(runner, 'MEX Speedup Comparison Tests', config);
    memoryUsageSuite = createPerformanceTestSuite(runner, 'Memory Usage Tests', config);

    % Add tests to suites
    addDistributionPerformanceTests(distributionSuite, config);
    addTimeSeriesPerformanceTests(timeSeriesSuite, config);
    addVolatilityPerformanceTests(volatilitySuite, config);
    addMultivariatePerformanceTests(multivariateSuite, config);
    addBootstrapPerformanceTests(bootstrapSuite, config);
    addLargeScaleDataTests(largeScaleSuite, config);
    addMEXSpeedupTests(mexSpeedupSuite, config);
    addMemoryUsageTests(memoryUsageSuite, config);

    % Run tests
    startTime = tic;
    results = runner.run();
    executionTime = toc(startTime);

    % Generate performance report
    reportInfo = generatePerformanceReport(runner, config, results);

    % Display summary
    runner.displaySummary();

    % Return execution results and status
end

function config = parseInputArguments(varargin)
    %PARSEINPUTARGUMENTS Parses command-line arguments to configure performance test execution
    %   Parses command-line arguments to configure performance test execution

    % Initialize default configuration
    config.verbose = false;
    config.visualizationsEnabled = true;
    config.reportFormats = {'text', 'html'};
    config.speedupThreshold = 1.5;
    config.resultsDir = './performance_results';

    % Parse varargin for command-line options and parameters
    i = 1;
    while i <= nargin
        arg = varargin{i};
        switch arg
            case {'-v', '--verbose'}
                config.verbose = true;
                i = i + 1;
            case {'--no-visuals'}
                config.visualizationsEnabled = false;
                i = i + 1;
            case {'--visuals'}
                config.visualizationsEnabled = true;
                i = i + 1;
            case {'-r', '--report'}
                if i + 1 <= nargin
                    config.reportFormats = strsplit(varargin{i+1}, ',');
                    i = i + 2;
                else
                    fprintf('Error: Report formats must be specified after -r or --report.\n');
                    return;
                end
            case {'-t', '--threshold'}
                if i + 1 <= nargin
                    config.speedupThreshold = str2double(varargin{i+1});
                    i = i + 2;
                else
                    fprintf('Error: Speedup threshold must be specified after -t or --threshold.\n');
                    return;
                end
            case {'-d', '--dir'}
                if i + 1 <= nargin
                    config.resultsDir = varargin{i+1};
                    i = i + 2;
                else
                    fprintf('Error: Results directory must be specified after -d or --dir.\n');
                    return;
                end
            otherwise
                fprintf('Warning: Unknown argument %s.\n', arg);
                i = i + 1;
        end
    end

    % Validate configuration options and set defaults for missing values
    % (Defaults are already set at the beginning of the function)

    % Return the parsed configuration as a struct
end

function displayHeader()
    %DISPLAYHEADER Displays a formatted header for the performance test execution
    %   Displays a formatted header for the performance test execution

    % Print separator line of equals signs
    fprintf('==========================================================\\n');

    % Print centered title 'MFE Toolbox Performance Test Suite'
    fprintf('             MFE Toolbox Performance Test Suite             \\n');

    % Print current date and time using datestr
    fprintf('                       %s                       \\n', datestr(now));

    % Print separator line of equals signs
    fprintf('==========================================================\\n');
end

function suite = createPerformanceTestSuite(runner, suiteName, config)
    %CREATEPERFORMANCETESTSUITE Creates and configures a test suite for performance tests
    %   Creates and configures a test suite for performance tests

    % Use runner.createTestSuite to create a new test suite with the specified name
    suite = runner.createTestSuite(suiteName);

    % Configure the suite with verbose mode from config
    suite.setVerbose(config.verbose);

    % Configure parallel execution based on test type
    % (Parallel execution is not applicable for performance tests)

    % Return the configured test suite
end

function configurePerformanceTests(config)
    %CONFIGUREPERFORMANCETESTS Configures global performance test settings for all test classes
    %   Configures global performance test settings for all test classes

    % Create PerformanceBenchmark instance for global configuration
    benchmark = PerformanceBenchmark();

    % Set speedup threshold using setSpeedupThreshold method
    benchmark.setSpeedupThreshold(config.speedupThreshold);

    % Configure visualization settings using enableVisualizationSaving method
    benchmark.enableVisualizationSaving(config.visualizationsEnabled, config.resultsDir);

    % Set output directory for performance results
    % Create output directory if it doesn't exist
    if ~exist(config.resultsDir, 'dir')
        mkdir(config.resultsDir);
    end

    % If MEXSpeedupTest exists, configure its threshold and visualization settings
    if exist('MEXSpeedupTest', 'class')
        MEXSpeedupTest.setSpeedupThreshold(config.speedupThreshold);
        MEXSpeedupTest.enableVisualizations(config.visualizationsEnabled);
    end
end

function success = addDistributionPerformanceTests(suite, config)
    %ADDDISTRIBUTIONPERFORMANCETESTS Adds distribution function performance tests to the suite
    %   Adds distribution function performance tests to the suite

    % Check if DistributionsPerformanceTest class exists
    if exist('DistributionsPerformanceTest', 'class')
        % Create instance of DistributionsPerformanceTest
        testInstance = DistributionsPerformanceTest();

        % Add test instance to the suite
        suite.addTest(testInstance);

        % If verbose is true, print message about added tests
        if config.verbose
            fprintf('Added DistributionsPerformanceTest to suite: %s\\n', suite.name);
        end

        % Return true if successful
        success = true;
    else
        % Return false if DistributionsPerformanceTest class doesn't exist
        fprintf('Warning: DistributionsPerformanceTest class not found. Skipping.\\n');
        success = false;
    end
end

function success = addTimeSeriesPerformanceTests(suite, config)
    %ADDTIMESERIESPERFORMANCETESTS Adds time series function performance tests to the suite
    %   Adds time series function performance tests to the suite

    % Check if TimeSeriesPerformanceTest class exists
    if exist('TimeSeriesPerformanceTest', 'class')
        % Create instance of TimeSeriesPerformanceTest
        testInstance = TimeSeriesPerformanceTest();

        % Add test instance to the suite
        suite.addTest(testInstance);

        % If verbose is true, print message about added tests
        if config.verbose
            fprintf('Added TimeSeriesPerformanceTest to suite: %s\\n', suite.name);
        end

        % Return true if successful
        success = true;
    else
        % Return false if TimeSeriesPerformanceTest class doesn't exist
        fprintf('Warning: TimeSeriesPerformanceTest class not found. Skipping.\\n');
        success = false;
    end
end

function success = addVolatilityPerformanceTests(suite, config)
    %ADDVOLATILITYPERFORMANCETESTS Adds volatility model performance tests to the suite
    %   Adds volatility model performance tests to the suite

    % Check if VolatilityModelPerformanceTest class exists
    if exist('VolatilityModelPerformanceTest', 'class')
        % Create instance of VolatilityModelPerformanceTest
        testInstance = VolatilityModelPerformanceTest();

        % Add test instance to the suite
        suite.addTest(testInstance);

        % If verbose is true, print message about added tests
        if config.verbose
            fprintf('Added VolatilityModelPerformanceTest to suite: %s\\n', suite.name);
        end

        % Return true if successful
        success = true;
    else
        % Return false if VolatilityModelPerformanceTest class doesn't exist
        fprintf('Warning: VolatilityModelPerformanceTest class not found. Skipping.\\n');
        success = false;
    end
end

function success = addMultivariatePerformanceTests(suite, config)
    %ADDMULTIVARIATEPERFORMANCETESTS Adds multivariate analysis performance tests to the suite
    %   Adds multivariate analysis performance tests to the suite

    % Check if MultivariatePerformanceTest class exists
    if exist('MultivariatePerformanceTest', 'class')
        % Create instance of MultivariatePerformanceTest
        testInstance = MultivariatePerformanceTest();

        % Add test instance to the suite
        suite.addTest(testInstance);

        % If verbose is true, print message about added tests
        if config.verbose
            fprintf('Added MultivariatePerformanceTest to suite: %s\\n', suite.name);
        end

        % Return true if successful
        success = true;
    else
        % Return false if MultivariatePerformanceTest class doesn't exist
        fprintf('Warning: MultivariatePerformanceTest class not found. Skipping.\\n');
        success = false;
    end
end

function success = addBootstrapPerformanceTests(suite, config)
    %ADDBOOTSTRAPPERFORMANCETESTS Adds bootstrap method performance tests to the suite
    %   Adds bootstrap method performance tests to the suite

    % Check if BootstrapPerformanceTest class exists
    if exist('BootstrapPerformanceTest', 'class')
        % Create instance of BootstrapPerformanceTest
        testInstance = BootstrapPerformanceTest();

        % Add test instance to the suite
        suite.addTest(testInstance);

        % If verbose is true, print message about added tests
        if config.verbose
            fprintf('Added BootstrapPerformanceTest to suite: %s\\n', suite.name);
        end

        % Return true if successful
        success = true;
    else
        % Return false if BootstrapPerformanceTest class doesn't exist
        fprintf('Warning: BootstrapPerformanceTest class not found. Skipping.\\n');
        success = false;
    end
end

function success = addLargeScaleDataTests(suite, config)
    %ADDLARGESCALEDATATESTS Adds large-scale data processing performance tests to the suite
    %   Adds large-scale data processing performance tests to the suite

    % Check if LargeScaleDataTest class exists
    if exist('LargeScaleDataTest', 'class')
        % Create instance of LargeScaleDataTest
        testInstance = LargeScaleDataTest();

        % Add test instance to the suite
        suite.addTest(testInstance);

        % If verbose is true, print message about added tests
        if config.verbose
            fprintf('Added LargeScaleDataTest to suite: %s\\n', suite.name);
        end

        % Return true if successful
        success = true;
    else
        % Return false if LargeScaleDataTest class doesn't exist
        fprintf('Warning: LargeScaleDataTest class not found. Skipping.\\n');
        success = false;
    end
end

function success = addMEXSpeedupTests(suite, config)
    %ADDMEXSPEEDUPTESTS Adds MEX implementation speedup comparison tests to the suite
    %   Adds MEX implementation speedup comparison tests to the suite

    % Check if MEXSpeedupTest class exists
    if exist('MEXSpeedupTest', 'class')
        % Create instance of MEXSpeedupTest
        testInstance = MEXSpeedupTest();

        % Configure speedup threshold from config
        testInstance.setSpeedupThreshold(config.speedupThreshold);

        % Configure visualization settings from config
        testInstance.enableVisualizationSaving(config.visualizationsEnabled, config.resultsDir);

        % Add test instance to the suite
        suite.addTest(testInstance);

        % If verbose is true, print message about added tests
        if config.verbose
            fprintf('Added MEXSpeedupTest to suite: %s\\n', suite.name);
        end

        % Return true if successful
        success = true;
    else
        % Return false if MEXSpeedupTest class doesn't exist
        fprintf('Warning: MEXSpeedupTest class not found. Skipping.\\n');
        success = false;
    end
end

function success = addMemoryUsageTests(suite, config)
    %ADDMEMORYUSAGETESTS Adds memory utilization tests to the suite
    %   Adds memory utilization tests to the suite

    % Check if MemoryUsageTest class exists
    if exist('MemoryUsageTest', 'class')
        % Create instance of MemoryUsageTest
        testInstance = MemoryUsageTest();

        % Add test instance to the suite
        suite.addTest(testInstance);

        % If verbose is true, print message about added tests
        if config.verbose
            fprintf('Added MemoryUsageTest to suite: %s\\n', suite.name);
        end

        % Return true if successful
        success = true;
    else
        % Return false if MemoryUsageTest class doesn't exist
        fprintf('Warning: MemoryUsageTest class not found. Skipping.\\n');
        success = false;
    end
end

function reportInfo = generatePerformanceReport(runner, config, results)
    %GENERATEPERFORMANCEREPORT Generates a comprehensive performance report based on test results
    %   Generates a comprehensive performance report based on test results

    % Configure runner to generate reports in specified formats
    runner.setReportFormats(config.reportFormats);

    % Set report title to 'MFE Toolbox Performance Benchmark Report'
    runner.setReportTitle('MFE Toolbox Performance Benchmark Report');

    % Set output directory to config.resultsDir
    runner.setOutputDirectory(config.resultsDir);

    % Generate performance report with runner.generateReport
    reportInfo = runner.generateReport();

    % If verbose is true, print message about report generation
    if config.verbose
        fprintf('Performance report generated: %s\\n', reportInfo.html);
    end

    % Return report generation status and file paths
end