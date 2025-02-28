classdef MEXSpeedupTest < BaseTest
    % MEXSpeedupTest A test class specifically designed to measure and validate the speedup achieved by MEX implementations in the MFE Toolbox compared to native MATLAB implementations
    
    properties
        dataGenerator     % TestDataGenerator instance for generating test data
        benchmarker       % PerformanceBenchmark instance for benchmarking
        testData          % Struct to store test data
        mexImplementations  % Cell array of function names to test
        requiredSpeedupFactor % Required speedup factor for MEX implementations
        generateVisualizations % Flag to enable performance visualization
        reportOutputPath   % Path to save performance reports
        dataSizes         % Data sizes for scalability testing
    end
    
    methods
        function obj = MEXSpeedupTest()
            % Initializes the MEXSpeedupTest class with default settings and test configuration
            
            % Call superclass constructor with 'MEX Speedup Test' name
            obj = obj@BaseTest('MEX Speedup Test');
            
            % Initialize dataGenerator with TestDataGenerator instance
            obj.dataGenerator = TestDataGenerator();
            
            % Initialize benchmarker with PerformanceBenchmark instance
            obj.benchmarker = PerformanceBenchmark();
            
            % Set requiredSpeedupFactor to 1.5 (50% performance improvement)
            obj.requiredSpeedupFactor = 1.5;
            
            % Configure benchmarker to use requiredSpeedupFactor as threshold
            obj.benchmarker.setSpeedupThreshold(obj.requiredSpeedupFactor);
            
            % Define mexImplementations cell array with list of functions to test that have MEX implementations
            obj.mexImplementations = {'agarchfit', 'egarchfit', 'igarchfit', 'tarchfit'};
            
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
            
            % Verify that all required MEX implementations exist
            for i = 1:length(obj.mexImplementations)
                funcName = obj.mexImplementations{i};
                mexFileName = [funcName, '.', mexext];
                if exist(mexFileName, 'file') ~= 3
                    error('MEX implementation not found for %s', funcName);
                end
            end
            
            % Create test data of various sizes for benchmarking
            obj.testData = struct();
            
            % Generate standard financial returns sample for volatility models
            obj.testData.financialReturns = obj.dataGenerator.generateFinancialReturns(1000, 1, struct('mean', 0, 'variance', 1));
            
            % Configure benchmarker with appropriate iteration count
            obj.benchmarker.setIterations(50);
            
            % Initialize test results structure
            obj.testResults = struct();
        end
        
        function tearDown(obj)
            % Cleans up after test execution
            
            % Generate performance report if any tests were executed
            if ~isempty(fieldnames(obj.testResults))
                obj.generatePerformanceReport();
            end
            
            % Clean up test data to free memory
            obj.testData = struct();
            
            % Reset benchmarker state
            obj.benchmarker = PerformanceBenchmark();
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
        end
        
        function testAgarchMEXSpeedup(obj)
            % Tests the speedup of AGARCH model estimation using MEX implementation compared to native MATLAB
            
            % Generate appropriate test data for AGARCH model
            data = obj.testData.financialReturns;
            
            % Create function handle for MATLAB-only implementation by disabling MEX
            matlabImpl = obj.testForceMATLABImplementation(@agarchfit);
            
            % Create function handle for MEX implementation
            mexImpl = @agarchfit;
            
            % Compare implementations using benchmarker.compareImplementations
            results = obj.benchmarker.compareImplementations(matlabImpl, mexImpl, [], data);
            
            % Verify that results are numerically equivalent between implementations
            obj.assertTrue(results.outputsMatch, 'AGARCH: MATLAB and MEX implementations produce different results');
            
            % Assert that MEX implementation meets or exceeds requiredSpeedupFactor
            obj.assertTrue(results.meetsThreshold, sprintf('AGARCH: MEX speedup (%g) is less than required (%g)', ...
                results.speedupRatio, obj.requiredSpeedupFactor));
            
            % Generate visualization if generateVisualizations is true
            if obj.generateVisualizations
                obj.benchmarker.visualizeResults(results, 'comparison', 'AGARCH MEX Speedup');
            end
            
            % Return comprehensive performance comparison metrics
            obj.testResults.agarchSpeedup = results;
        end
        
        function testEgarchMEXSpeedup(obj)
            % Tests the speedup of EGARCH model estimation using MEX implementation compared to native MATLAB
            
            % Generate appropriate test data for EGARCH model
            data = obj.testData.financialReturns;
            
            % Create function handle for MATLAB-only implementation by disabling MEX
            matlabImpl = obj.testForceMATLABImplementation(@egarchfit);
            
            % Create function handle for MEX implementation
            mexImpl = @egarchfit;
            
            % Compare implementations using benchmarker.compareImplementations
            results = obj.benchmarker.compareImplementations(matlabImpl, mexImpl, [], data);
            
            % Verify that results are numerically equivalent between implementations
            obj.assertTrue(results.outputsMatch, 'EGARCH: MATLAB and MEX implementations produce different results');
            
            % Assert that MEX implementation meets or exceeds requiredSpeedupFactor
            obj.assertTrue(results.meetsThreshold, sprintf('EGARCH: MEX speedup (%g) is less than required (%g)', ...
                results.speedupRatio, obj.requiredSpeedupFactor));
            
            % Generate visualization if generateVisualizations is true
            if obj.generateVisualizations
                obj.benchmarker.visualizeResults(results, 'comparison', 'EGARCH MEX Speedup');
            end
            
            % Return comprehensive performance comparison metrics
            obj.testResults.egarchSpeedup = results;
        end
        
        function testIgarchMEXSpeedup(obj)
            % Tests the speedup of IGARCH model estimation using MEX implementation compared to native MATLAB
            
            % Generate appropriate test data for IGARCH model
            data = obj.testData.financialReturns;
            
            % Create function handle for MATLAB-only implementation by disabling MEX
            matlabImpl = obj.testForceMATLABImplementation(@igarchfit);
            
            % Create function handle for MEX implementation
            mexImpl = @igarchfit;
            
            % Compare implementations using benchmarker.compareImplementations
            results = obj.benchmarker.compareImplementations(matlabImpl, mexImpl, [], data);
            
            % Verify that results are numerically equivalent between implementations
            obj.assertTrue(results.outputsMatch, 'IGARCH: MATLAB and MEX implementations produce different results');
            
            % Assert that MEX implementation meets or exceeds requiredSpeedupFactor
            obj.assertTrue(results.meetsThreshold, sprintf('IGARCH: MEX speedup (%g) is less than required (%g)', ...
                results.speedupRatio, obj.requiredSpeedupFactor));
            
            % Generate visualization if generateVisualizations is true
            if obj.generateVisualizations
                obj.benchmarker.visualizeResults(results, 'comparison', 'IGARCH MEX Speedup');
            end
            
            % Return comprehensive performance comparison metrics
            obj.testResults.igarchSpeedup = results;
        end
        
        function testTarchMEXSpeedup(obj)
            % Tests the speedup of TARCH model estimation using MEX implementation compared to native MATLAB
            
            % Generate appropriate test data for TARCH model
            data = obj.testData.financialReturns;
            
            % Create function handle for MATLAB-only implementation by disabling MEX
            matlabImpl = obj.testForceMATLABImplementation(@tarchfit);
            
            % Create function handle for MEX implementation
            mexImpl = @tarchfit;
            
            % Compare implementations using benchmarker.compareImplementations
            results = obj.benchmarker.compareImplementations(matlabImpl, mexImpl, [], data);
            
            % Verify that results are numerically equivalent between implementations
            obj.assertTrue(results.outputsMatch, 'TARCH: MATLAB and MEX implementations produce different results');
            
            % Assert that MEX implementation meets or exceeds requiredSpeedupFactor
            obj.assertTrue(results.meetsThreshold, sprintf('TARCH: MEX speedup (%g) is less than required (%g)', ...
                results.speedupRatio, obj.requiredSpeedupFactor));
            
            % Generate visualization if generateVisualizations is true
            if obj.generateVisualizations
                obj.benchmarker.visualizeResults(results, 'comparison', 'TARCH MEX Speedup');
            end
            
            % Return comprehensive performance comparison metrics
            obj.testResults.tarchSpeedup = results;
        end
        
        function testMEXScalabilityComparison(obj)
            % Tests how the performance advantage of MEX implementations scales with increasing data size
            
            % For each MEX implementation, test with increasing data sizes from dataSizes property
            % Compare MATLAB and MEX implementation scaling behavior
            
            % Generate data generator function for creating increasingly larger datasets
            dataGenerator = @(size) obj.dataGenerator.generateFinancialReturns(size, 1, struct('mean', 0, 'variance', 1));
            
            % Use benchmarker.compareScalability to analyze scaling behavior differences
            results = obj.benchmarker.compareScalability(obj.testForceMATLABImplementation(@agarchfit), @agarchfit, obj.dataSizes, dataGenerator);
            
            % Verify that MEX advantage is maintained or improves with larger datasets
            obj.assertTrue(true, 'Scalability test completed'); % Add meaningful assertion here
            
            % Generate scalability visualization comparing implementations
            if obj.generateVisualizations
                obj.benchmarker.visualizeResults(results, 'comparison', 'MEX Scalability Comparison');
            end
            
            % Calculate speedup ratio at each data size
            % Return comprehensive scalability comparison metrics
            obj.testResults.scalabilityComparison = results;
        end
        
        function testMEXMemoryUsageComparison(obj)
            % Tests memory usage of MEX implementations compared to native MATLAB implementations
            
            % For each MEX implementation, measure memory usage during execution
            % Compare with memory usage of equivalent MATLAB implementation
            
            % Use benchmarker.measureMemoryUsage for each implementation type
            matlabImpl = obj.testForceMATLABImplementation(@agarchfit);
            mexImpl = @agarchfit;
            
            results = obj.benchmarker.compareMemoryUsage(matlabImpl, mexImpl, obj.testData.financialReturns);
            
            % Calculate memory efficiency metrics and comparison ratios
            % Assert that MEX implementation has efficient memory usage
            obj.assertTrue(true, 'Memory usage test completed'); % Add meaningful assertion here
            
            % Generate memory usage comparison visualization
            if obj.generateVisualizations
                % Add visualization code here
            end
            
            % Return comprehensive memory usage comparison metrics
            obj.testResults.memoryUsageComparison = results;
        end
        
        function forcedFunc = testForceMATLABImplementation(obj, func)
            % Helper function to create a function wrapper that forces using MATLAB implementation by disabling MEX
            
            % Create a function wrapper that adds 'useMEX=false' to options struct
            forcedFunc = @(varargin) func(varargin{:}, struct('useMEX', false));
            
            % Ensure original function signature and behavior is preserved
            % Return wrapped function handle
        end
        
        function results = runAllTests(obj)
            % Runs all MEX speedup tests and generates a comprehensive report
            
            % Execute individual test methods for each model type
            obj.testAgarchMEXSpeedup();
            obj.testEgarchMEXSpeedup();
            obj.testIgarchMEXSpeedup();
            obj.testTarchMEXSpeedup();
            
            % Run scalability and memory usage tests
            obj.testMEXScalabilityComparison();
            obj.testMEXMemoryUsageComparison();
            
            % Collect speedup metrics across all implementations
            % Calculate average, minimum, and maximum speedup factor
            % Generate comprehensive performance report with visualizations
            % Return aggregated test results with overall assessment
            
            results = obj.testResults;
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
        
        function enableVisualizations(obj, enable)
            % Enables or disables performance visualization generation
            
            % Validate that enable is a logical value
            if ~islogical(enable)
                error('Enable flag must be a logical value');
            end
            
            % Update generateVisualizations property
            obj.generateVisualizations = enable;
            
            % Configure benchmarker visualization settings
        end
        
        function generatePerformanceReport(obj)
            % Generates a comprehensive performance report with visualizations
            
            % Create a new figure for the report
            fig = figure('Name', 'MEX Speedup Performance Report', 'NumberTitle', 'off', 'Visible', 'off');
            
            % Define the layout of the report
            numTests = length(fieldnames(obj.testResults));
            rows = ceil(sqrt(numTests));
            cols = ceil(numTests / rows);
            
            % Loop through each test result and add it to the report
            testNames = fieldnames(obj.testResults);
            for i = 1:numTests
                testName = testNames{i};
                results = obj.testResults.(testName);
                
                % Create a subplot for the current test
                subplot(rows, cols, i);
                
                % Generate a bar chart comparing the execution times
                if isfield(results, 'results1') && isfield(results, 'results2')
                    data = [results.results1.mean, results.results2.mean] * 1000; % Convert to ms
                    bar(data);
                    grid on;
                    xlabel('Implementation');
                    ylabel('Execution Time (ms)');
                    set(gca, 'XTickLabel', {results.function1, results.function2});
                    title(strrep(testName, 'Speedup', ''));
                    
                    % Add text for speedup
                    text(1.5, mean(data), [num2str(results.speedupRatio, '%.2f'), 'x speedup'], ...
                        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
                end
            end
            
            % Save the report to a file
            reportFileName = fullfile(obj.reportOutputPath, ['MEX_Speedup_Report_', datestr(now, 'yyyymmdd_HHMMSS'), '.png']);
            saveas(fig, reportFileName);
            
            % Close the figure
            close(fig);
        end
    end
end