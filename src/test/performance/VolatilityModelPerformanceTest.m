classdef VolatilityModelPerformanceTest < BaseTest
    %VOLATILITYMODELPERFORMANCETEST A specialized test class for benchmarking the performance of volatility model implementations in the MFE Toolbox.
    %   This class tests the computational efficiency of various GARCH-type volatility models under different conditions, error distributions, and data sizes, with comprehensive performance metrics reporting and visualization.

    properties
        dataGenerator         % TestDataGenerator instance for generating test data
        benchmarker           % PerformanceBenchmark instance for measuring performance
        reporter              % TestReporter instance for reporting test results
        testData              % Structure to store test data
        volatilityModels      % Cell array of volatility model names
        errorDistributions    % Cell array of error distribution names
        requiredSpeedupFactor % Required speedup factor for MEX implementations
        generateVisualizations % Logical flag to enable performance visualization graphs
        reportOutputPath      % String path to the output directory for performance reports
        dataSizes             % Cell array of data sizes for scalability testing
    end

    methods
        function obj = VolatilityModelPerformanceTest()
            % Initializes the VolatilityModelPerformanceTest class with default settings and test configuration

            % Call superclass (BaseTest) constructor with 'Volatility Model Performance Test' name
            obj = obj@BaseTest('Volatility Model Performance Test');

            % Initialize dataGenerator with TestDataGenerator instance
            obj.dataGenerator = TestDataGenerator();

            % Initialize benchmarker with PerformanceBenchmark instance
            obj.benchmarker = PerformanceBenchmark();

            % Initialize reporter with TestReporter instance
            obj.reporter = TestReporter();

            % Set requiredSpeedupFactor to 1.5 (50% performance improvement)
            obj.requiredSpeedupFactor = 1.5;

            % Configure benchmarker to use requiredSpeedupFactor as threshold
            obj.benchmarker.setSpeedupThreshold(obj.requiredSpeedupFactor);

            % Define volatilityModels cell array with list of models to test: 'AGARCH', 'EGARCH', 'IGARCH', 'TARCH', 'NAGARCH'
            obj.volatilityModels = {'AGARCH', 'EGARCH', 'IGARCH', 'TARCH', 'NAGARCH'};

            % Define errorDistributions cell array with distributions to test: 'Normal', 'Student', 'GED', 'Skewed-t'
            obj.errorDistributions = {'Normal', 'Student', 'GED', 'Skewed-t'};

            % Set dataSizes for scalability testing: [500, 1000, 2000, 5000, 10000]
            obj.dataSizes = {500, 1000, 2000, 5000, 10000};

            % Set generateVisualizations to true for performance visualization graphs
            obj.generateVisualizations = true;

            % Set default reportOutputPath to current directory
            obj.reportOutputPath = pwd;
        end

        function setUp(obj)
            % Prepares the test environment before executing tests

            % Call superclass setUp method
            setUp@BaseTest(obj);

            % Verify that all required volatility model functions are available
            obj.assertTrue(exist('agarchfit', 'file') == 2, 'AGARCH fitting function not found');
            obj.assertTrue(exist('egarchfit', 'file') == 2, 'EGARCH fitting function not found');
            obj.assertTrue(exist('igarchfit', 'file') == 2, 'IGARCH fitting function not found');
            obj.assertTrue(exist('tarchfit', 'file') == 2, 'TARCH fitting function not found');
            obj.assertTrue(exist('nagarchfit', 'file') == 2, 'NAGARCH fitting function not found');

            % Check for MEX implementations of core functions
            hasAgarchMex = exist('agarchcore', 'file') == 3;
            hasEgarchMex = exist('egarchcore', 'file') == 3;
            hasIgarchMex = exist('igarchcore', 'file') == 3;
            hasTarchMex = exist('tarchcore', 'file') == 3;

            % Generate test data for different model types and sizes
            obj.testData = struct();
            for i = 1:length(obj.volatilityModels)
                modelName = obj.volatilityModels{i};
                obj.testData.(modelName) = struct();
                for j = 1:length(obj.errorDistributions)
                    distributionName = obj.errorDistributions{j};
                    switch distributionName
                        case 'Normal'
                            obj.testData.(modelName).(distributionName) = obj.dataGenerator.generateFinancialReturns(1000, 1, struct('distribution', 'normal'));
                        case 'Student'
                            obj.testData.(modelName).(distributionName) = obj.dataGenerator.generateFinancialReturns(1000, 1, struct('distribution', 't', 'distParams', 7));
                        case 'GED'
                            obj.testData.(modelName).(distributionName) = obj.dataGenerator.generateFinancialReturns(1000, 1, struct('distribution', 'ged', 'distParams', 1.5));
                        case 'Skewed-t'
                            obj.testData.(modelName).(distributionName) = obj.dataGenerator.generateFinancialReturns(1000, 1, struct('distribution', 'skewt', 'distParams', [7, 0.2]));
                    end
                end
            end

            % Generate standard financial returns sample for volatility models
            obj.testData.returns = obj.dataGenerator.generateFinancialReturns(1000, 1, struct('distribution', 'normal'));

            % Configure benchmarker with appropriate iteration count
            obj.benchmarker.setIterations(50);

            % Initialize test results structure
            obj.testResults = struct();
        end

        function tearDown(obj)
            % Cleans up after test execution

            % Generate performance report if any tests were executed
            if ~isempty(fieldnames(obj.testResults))
                obj.generatePerformanceReport(obj.testResults);
            end

            % Clean up test data to free memory
            obj.testData = struct();

            % Reset benchmarker state
            obj.benchmarker = PerformanceBenchmark();

            % Call superclass tearDown method
            tearDown@BaseTest(obj);
        end

        function testAgarchPerformance(obj)
            % Tests the performance of AGARCH model estimation across different error distributions

            % Loop through each error distribution type
            for i = 1:length(obj.errorDistributions)
                distributionName = obj.errorDistributions{i};

                % Generate appropriate test data for the current distribution
                returns = obj.testData.AGARCH.(distributionName);

                % Configure AGARCH model options (p=1, q=1, distribution)
                options = struct('p', 1, 'q', 1, 'distribution', distributionName);

                % Define function handle for AGARCH model estimation
                agarchFunction = @() agarchfit(returns, options);

                % Measure execution time using benchmarker.benchmarkFunction
                results = obj.benchmarker.benchmarkFunction(agarchFunction);

                % If MEX implementation exists, compare with MATLAB implementation
                if exist('agarchcore', 'file') == 3
                    % Define function handle for MATLAB implementation
                    options.useMEX = false;
                    agarchFunctionMatlab = @() agarchfit(returns, options);

                    % Compare with MATLAB implementation
                    comparisonResults = obj.benchmarker.compareImplementations(agarchFunctionMatlab, agarchFunction);

                    % Verify consistency of results between implementations
                    obj.assertTrue(comparisonResults.outputsMatch, 'AGARCH: MEX and MATLAB implementations produce inconsistent results');

                    % Assert that MEX implementation meets or exceeds performance threshold
                    obj.assertTrue(comparisonResults.meetsThreshold, 'AGARCH: MEX implementation does not meet performance threshold');
                end

                % Generate visualization if generateVisualizations is true
                if obj.generateVisualizations
                    obj.benchmarker.visualizeResults(results, 'timeseries', ['AGARCH Performance - ' distributionName]);
                end

                % Store performance results for reporting
                obj.testResults.(['AGARCH_' distributionName]) = results;

                % Return comprehensive performance metrics
            end
        end

        function testEgarchPerformance(obj)
            % Tests the performance of EGARCH model estimation across different error distributions

            % Loop through each error distribution type
            for i = 1:length(obj.errorDistributions)
                distributionName = obj.errorDistributions{i};

                % Generate appropriate test data for the current distribution
                returns = obj.testData.EGARCH.(distributionName);

                % Configure EGARCH model options (p=1, q=1, distribution)
                options = struct('p', 1, 'q', 1, 'distribution', distributionName);

                % Define function handle for EGARCH model estimation
                egarchFunction = @() egarchfit(returns, 1, 1, 1, options);

                % Measure execution time using benchmarker.benchmarkFunction
                results = obj.benchmarker.benchmarkFunction(egarchFunction);

                % If MEX implementation exists, compare with MATLAB implementation
                if exist('egarchcore', 'file') == 3
                    % Define function handle for MATLAB implementation
                    options.useMEX = false;
                    egarchFunctionMatlab = @() egarchfit(returns, 1, 1, 1, options);

                    % Compare with MATLAB implementation
                    comparisonResults = obj.benchmarker.compareImplementations(egarchFunctionMatlab, egarchFunction);

                    % Verify consistency of results between implementations
                    obj.assertTrue(comparisonResults.outputsMatch, 'EGARCH: MEX and MATLAB implementations produce inconsistent results');

                    % Assert that MEX implementation meets or exceeds performance threshold
                    obj.assertTrue(comparisonResults.meetsThreshold, 'EGARCH: MEX implementation does not meet performance threshold');
                end

                % Generate visualization if generateVisualizations is true
                if obj.generateVisualizations
                    obj.benchmarker.visualizeResults(results, 'timeseries', ['EGARCH Performance - ' distributionName]);
                end

                % Store performance results for reporting
                obj.testResults.(['EGARCH_' distributionName]) = results;

                % Return comprehensive performance metrics
            end
        end

        function testIgarchPerformance(obj)
            % Tests the performance of IGARCH model estimation across different error distributions

            % Loop through each error distribution type
            for i = 1:length(obj.errorDistributions)
                distributionName = obj.errorDistributions{i};

                % Generate appropriate test data for the current distribution
                returns = obj.testData.IGARCH.(distributionName);

                % Configure IGARCH model options (p=1, q=1, distribution)
                options = struct('p', 1, 'q', 1, 'distribution', distributionName);

                % Define function handle for IGARCH model estimation
                igarchFunction = @() igarchfit(returns, options);

                % Measure execution time using benchmarker.benchmarkFunction
                results = obj.benchmarker.benchmarkFunction(igarchFunction);

                % If MEX implementation exists, compare with MATLAB implementation
                if exist('igarchcore', 'file') == 3
                    % Define function handle for MATLAB implementation
                    options.useMEX = false;
                    igarchFunctionMatlab = @() igarchfit(returns, options);

                    % Compare with MATLAB implementation
                    comparisonResults = obj.benchmarker.compareImplementations(igarchFunctionMatlab, igarchFunction);

                    % Verify consistency of results between implementations
                    obj.assertTrue(comparisonResults.outputsMatch, 'IGARCH: MEX and MATLAB implementations produce inconsistent results');

                    % Assert that MEX implementation meets or exceeds performance threshold
                    obj.assertTrue(comparisonResults.meetsThreshold, 'IGARCH: MEX implementation does not meet performance threshold');
                end

                % Generate visualization if generateVisualizations is true
                if obj.generateVisualizations
                    obj.benchmarker.visualizeResults(results, 'timeseries', ['IGARCH Performance - ' distributionName]);
                end

                % Store performance results for reporting
                obj.testResults.(['IGARCH_' distributionName]) = results;

                % Return comprehensive performance metrics
            end
        end

        function testTarchPerformance(obj)
            % Tests the performance of TARCH model estimation across different error distributions

            % Loop through each error distribution type
            for i = 1:length(obj.errorDistributions)
                distributionName = obj.errorDistributions{i};

                % Generate appropriate test data for the current distribution
                returns = obj.testData.TARCH.(distributionName);

                % Configure TARCH model options (p=1, q=1, distribution)
                options = struct('p', 1, 'q', 1, 'distribution', distributionName);

                % Define function handle for TARCH model estimation
                tarchFunction = @() tarchfit(returns, options);

                % Measure execution time using benchmarker.benchmarkFunction
                results = obj.benchmarker.benchmarkFunction(tarchFunction);

                % If MEX implementation exists, compare with MATLAB implementation
                if exist('tarchcore', 'file') == 3
                    % Define function handle for MATLAB implementation
                    options.useMEX = false;
                    tarchFunctionMatlab = @() tarchfit(returns, options);

                    % Compare with MATLAB implementation
                    comparisonResults = obj.benchmarker.compareImplementations(tarchFunctionMatlab, tarchFunction);

                    % Verify consistency of results between implementations
                    obj.assertTrue(comparisonResults.outputsMatch, 'TARCH: MEX and MATLAB implementations produce inconsistent results');

                    % Assert that MEX implementation meets or exceeds performance threshold
                    obj.assertTrue(comparisonResults.meetsThreshold, 'TARCH: MEX implementation does not meet performance threshold');
                end

                % Generate visualization if generateVisualizations is true
                if obj.generateVisualizations
                    obj.benchmarker.visualizeResults(results, 'timeseries', ['TARCH Performance - ' distributionName]);
                end

                % Store performance results for reporting
                obj.testResults.(['TARCH_' distributionName]) = results;

                % Return comprehensive performance metrics
            end
        end

        function testNagarchPerformance(obj)
            % Tests the performance of NAGARCH model estimation across different error distributions

            % Loop through each error distribution type
            for i = 1:length(obj.errorDistributions)
                distributionName = obj.errorDistributions{i};

                % Generate appropriate test data for the current distribution
                returns = obj.testData.NAGARCH.(distributionName);

                % Configure NAGARCH model options (p=1, q=1, distribution)
                options = struct('p', 1, 'q', 1, 'distribution', distributionName);

                % Define function handle for NAGARCH model estimation
                nagarchFunction = @() nagarchfit(returns, options);

                % Measure execution time using benchmarker.benchmarkFunction
                results = obj.benchmarker.benchmarkFunction(nagarchFunction);

                % Compare with MATLAB implementation if MEX implementation exists
                % NAGARCH does not have a MEX implementation
                % Generate visualization if generateVisualizations is true
                if obj.generateVisualizations
                    obj.benchmarker.visualizeResults(results, 'timeseries', ['NAGARCH Performance - ' distributionName]);
                end

                % Store performance results for reporting
                obj.testResults.(['NAGARCH_' distributionName]) = results;

                % Return comprehensive performance metrics
            end
        end

        function testModelOrderScalability(obj)
            % Tests how performance scales with increasing model order (p,q)

            % Define array of model orders to test: [(1,1), (1,2), (2,1), (2,2)]
            modelOrders = {[1, 1], [1, 2], [2, 1], [2, 2]};

            % Loop through each volatility model type
            for i = 1:length(obj.volatilityModels)
                modelName = obj.volatilityModels{i};

                % For each model order, measure execution time
                timings = zeros(length(modelOrders), 1);
                for j = 1:length(modelOrders)
                    order = modelOrders{j};
                    p = order(1);
                    q = order(2);

                    % Configure model options
                    options = struct('p', p, 'q', q, 'distribution', 'Normal');

                    % Define function handle for model estimation
                    switch modelName
                        case 'AGARCH'
                            modelFunction = @() agarchfit(obj.testData.returns, options);
                        case 'EGARCH'
                            modelFunction = @() egarchfit(obj.testData.returns, p, p, q, options);
                        case 'IGARCH'
                            modelFunction = @() igarchfit(obj.testData.returns, options);
                        case 'TARCH'
                            modelFunction = @() tarchfit(obj.testData.returns, options);
                        case 'NAGARCH'
                            modelFunction = @() nagarchfit(obj.testData.returns, options);
                    end

                    % Measure execution time
                    benchResults = obj.benchmarker.benchmarkFunction(modelFunction);
                    timings(j) = benchResults.mean;
                end

                % Compare execution time scaling across model orders
                % If MEX implementation exists, compare scaling with MATLAB implementation
                % Generate scalability visualization if generateVisualizations is true
                % Assert that performance degradation with higher orders is within acceptable limits
                % Store scalability results for reporting
                % Return comprehensive scalability metrics
            end
        end

        function testDataSizeScalability(obj)
            % Tests how performance scales with increasing data size

            % Define data generator function to create datasets of various sizes
            dataGenerator = @(size) obj.dataGenerator.generateFinancialReturns(size, 1, struct('distribution', 'normal'));

            % For each volatility model, test with increasing data sizes from dataSizes property
            % Measure execution time for each data size
            % Analyze execution time scaling behavior (linear, quadratic)
            % If MEX implementation exists, compare scaling behavior with MATLAB implementation
            % Generate scalability visualization if generateVisualizations is true
            % Assert that computational complexity is within theoretical bounds
            % Store scalability results for reporting
            % Return comprehensive scalability metrics
        end

        function testDistributionPerformanceImpact(obj)
            % Tests the performance impact of different error distributions

            % For each volatility model, test with all error distributions
            % Measure execution time for each distribution type
            % Compare relative computational cost across distributions
            % Generate performance comparison visualization if generateVisualizations is true
            % Verify that more complex distributions have expected performance characteristics
            % Store distribution performance results for reporting
            % Return comprehensive distribution performance metrics
        end

        function testMemoryUsage(obj)
            % Tests memory usage of volatility model implementations

            % For each volatility model, measure memory usage during execution
            % Use benchmarker.measureMemoryUsage for each model type
            % If MEX implementation exists, compare memory usage with MATLAB implementation
            % Calculate memory efficiency metrics
            % Assert that memory usage is efficient and does not leak
            % Store memory usage results for reporting
            % Return comprehensive memory usage metrics
        end

        function runAllTests(obj)
            % Runs all volatility model performance tests and generates comprehensive report

            % Execute individual test methods for each model type
            obj.testAgarchPerformance();
            obj.testEgarchPerformance();
            obj.testIgarchPerformance();
            obj.testTarchPerformance();
            obj.testNagarchPerformance();

            % Run scalability and memory usage tests
            obj.testModelOrderScalability();
            obj.testDataSizeScalability();
            obj.testDistributionPerformanceImpact();
            obj.testMemoryUsage();

            % Collect performance metrics across all models
            % Calculate comparative performance statistics
            % Generate comprehensive performance report
            % Create comparative visualizations across model types
            % Analyze overall MEX vs MATLAB performance improvement
            % Return aggregated test results with overall assessment
        end

        function reportFilePath = generatePerformanceReport(obj, results)
            % Generates a detailed performance report for volatility models

            % Configure reporter with performance-focused settings
            obj.reporter.setReportTitle('Volatility Model Performance Report');
            obj.reporter.setReportFormats({'html', 'text'});
            obj.reporter.setReportOutputPath(obj.reportOutputPath);
            obj.reporter.setIncludePerformanceData(true);

            % Add all test results to reporter
            resultFields = fieldnames(results);
            for i = 1:length(resultFields)
                result = results.(resultFields{i});
                obj.reporter.addTestResult(result.function, 'Volatility Model', true, result);
            end

            % Generate summary tables comparing models and distributions
            % Create performance visualizations comparing implementation methods
            % Include scalability analysis and memory usage comparison
            % Generate report in specified format (HTML, Text, etc.)
            reportFilePath = obj.reporter.generateReport();

            % Save report to reportOutputPath
            % Display summary statistics to console
            obj.reporter.displaySummary();

            % Return path to generated report file
        end

        function setSpeedupThreshold(obj, factor)
            % Sets the required speedup factor threshold for tests to pass

            % Validate that factor is greater than 1.0
            if factor <= 1.0
                error('Speedup factor must be greater than 1.0');
            end

            % Update requiredSpeedupFactor property
            obj.requiredSpeedupFactor = factor;

            % Update benchmarker's speedupThreshold setting
            obj.benchmarker.setSpeedupThreshold(factor);
        end

        function setReportOutputPath(obj, path)
            % Sets the output directory for performance reports

            % Validate that path is a valid directory or create it
            if ~exist(path, 'dir')
                mkdir(path);
            end

            % Update reportOutputPath property
            obj.reportOutputPath = path;

            % Configure reporter to use specified output path
            obj.reporter.setReportOutputPath(path);
        end

        function enableVisualizations(obj, enable)
            % Enables or disables performance visualization generation

            % Validate that enable is a logical value
            if ~islogical(enable)
                error('Enable must be a logical value');
            end

            % Update generateVisualizations property
            obj.generateVisualizations = enable;

            % Configure benchmarker visualization settings
            obj.benchmarker.enableVisualizationSaving(enable, obj.reportOutputPath);
        end
    end
end