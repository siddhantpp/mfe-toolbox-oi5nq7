classdef CrossSectionAnalysisTest < BaseTest
    %CROSSSECTIONANALYSISTEST Test class for validating the cross-sectional analysis functionality in the MFE Toolbox
    % This class contains unit tests for the cross_section_analysis module,
    % ensuring its statistical analysis capabilities, filtering operations,
    % and numerical properties are functioning correctly.

    properties
        testData
        numObservations
        numAssets
        numFactors
        testOptions
        comparator
        expectedResults
    end

    methods
        function obj = CrossSectionAnalysisTest()
            % Initializes the test class instance
            % Call the superclass constructor
            obj = obj@BaseTest();

            % Initialize test properties with default values
            obj.numObservations = 500;
            obj.numAssets = 50;
            obj.numFactors = 3;
            obj.testOptions = struct();
            obj.comparator = NumericalComparator();
        end

        function setUp(obj)
            % Setup method called before each test to prepare the test environment
            % Set random number generator seed for reproducibility
            rng(42);

            % Set test data dimensions
            obj.numObservations = 500;
            obj.numAssets = 50;
            obj.numFactors = 3;

            % Load cross-sectional test data from file or generate it
            if exist('src/test/data/cross_sectional_data.mat', 'file')
                % Load test data if available
                data = loadTestData(obj, 'cross_sectional_data.mat');
                obj.testData = data.csData.returns;
            else
                % Generate test data if not available
                obj.testData = generateCrossSectionalData(obj.numAssets, obj.numObservations, obj.numFactors);
            end

            % Initialize test options structure for analysis configuration
            obj.testOptions = struct();

            % Pre-compute expected results for validation
            obj.expectedResults = struct();
        end

        function tearDown(obj)
            % Cleanup method called after each test to restore the test environment
            % Clear test data to free memory
            clear obj.testData;

            % Reset any modified global settings
            rng('default');

            % Reset test properties to default values
            obj.numObservations = 500;
            obj.numAssets = 50;
            obj.numFactors = 3;
            obj.testOptions = struct();
        end

        function testBasicAnalysis(obj)
            % Tests the basic functionality of cross-sectional analysis with default options
            % Call analyze_cross_section with test data and default options
            results = analyze_cross_section(obj.testData);

            % Verify the structure of the returned results object
            obj.assertTrue(isstruct(results), 'Results must be a structure');
            obj.assertTrue(isfield(results, 'descriptive'), 'Descriptive statistics must be present');
            obj.assertTrue(isfield(results, 'correlations'), 'Correlation analysis must be present');

            % Validate that descriptive statistics are computed correctly
            obj.assertTrue(isstruct(results.descriptive), 'Descriptive statistics must be a structure');
            obj.assertTrue(isfield(results.descriptive, 'mean'), 'Mean must be present');
            obj.assertEqual(length(results.descriptive.mean), obj.numAssets, 'Mean length must match number of assets');

            % Check that correlation analysis results are valid
            obj.assertTrue(isstruct(results.correlations), 'Correlation analysis must be a structure');
            obj.assertTrue(isfield(results.correlations, 'pearson'), 'Pearson correlation must be present');
            obj.assertEqual(size(results.correlations.pearson), [obj.numAssets, obj.numAssets], 'Pearson correlation size must match number of assets');

            % Assert that results conform to expected statistical properties
            obj.assertTrue(all(diag(results.correlations.pearson) == 1), 'Diagonal elements of correlation matrix must be 1');
        end

        function testDescriptiveStatistics(obj)
            % Tests the calculation of descriptive statistics for cross-sectional data
            % Call compute_descriptive_statistics with test data
            stats = compute_descriptive_statistics(obj.testData);

            % Verify all expected statistics are present (mean, median, std, etc.)
            obj.assertTrue(isstruct(stats), 'Descriptive statistics must be a structure');
            obj.assertTrue(isfield(stats, 'mean'), 'Mean must be present');
            obj.assertTrue(isfield(stats, 'median'), 'Median must be present');
            obj.assertTrue(isfield(stats, 'std'), 'Standard deviation must be present');
            obj.assertTrue(isfield(stats, 'min'), 'Minimum must be present');
            obj.assertTrue(isfield(stats, 'max'), 'Maximum must be present');

            % Compare computed statistics with manually calculated values
            expectedMean = mean(obj.testData);
            obj.assertMatrixEqualsWithTolerance(stats.mean, expectedMean, 1e-10, 'Computed mean does not match expected value');

            expectedMedian = median(obj.testData);
            obj.assertMatrixEqualsWithTolerance(stats.median, expectedMedian, 1e-10, 'Computed median does not match expected value');

            expectedStd = std(obj.testData);
            obj.assertMatrixEqualsWithTolerance(stats.std, expectedStd, 1e-10, 'Computed standard deviation does not match expected value');

            % Test edge cases with single observation or single variable
            singleObsData = obj.testData(1, :);
            singleObsStats = compute_descriptive_statistics(singleObsData);
            obj.assertMatrixEqualsWithTolerance(singleObsStats.mean, singleObsData, 1e-10, 'Single observation mean incorrect');

            singleVarData = obj.testData(:, 1);
            singleVarStats = compute_descriptive_statistics(singleVarData);
            obj.assertMatrixEqualsWithTolerance(singleVarStats.mean, mean(singleVarData), 1e-10, 'Single variable mean incorrect');

            % Verify handling of various statistical edge cases
            zeroVarianceData = zeros(obj.numObservations, 1);
            zeroVarianceStats = compute_descriptive_statistics(zeroVarianceData);
            obj.assertEqual(zeroVarianceStats.std, 0, 'Zero variance standard deviation incorrect');
        end

        function testCorrelationAnalysis(obj)
            % Tests the correlation analysis functionality for cross-sectional variables
            % Call analyze_correlations with test data
            results = analyze_correlations(obj.testData);

            % Verify that correlation matrix is symmetric and has proper dimensions
            obj.assertTrue(isstruct(results), 'Correlation results must be a structure');
            obj.assertTrue(isfield(results, 'pearson'), 'Pearson correlation must be present');
            obj.assertEqual(size(results.pearson), [obj.numAssets, obj.numAssets], 'Pearson correlation size incorrect');

            % Check that diagonal elements are equal to 1
            obj.assertTrue(all(diag(results.pearson) == 1), 'Diagonal elements of correlation matrix must be 1');

            % Compare computed correlations with manually calculated values
            expectedCorr = corr(obj.testData);
            obj.assertMatrixEqualsWithTolerance(results.pearson, expectedCorr, 1e-10, 'Computed correlation does not match expected value');

            % Test different correlation methods (Pearson, Spearman)
            options = struct('type', 'spearman');
            spearmanResults = analyze_correlations(obj.testData, options);
            obj.assertTrue(isfield(spearmanResults, 'spearman'), 'Spearman correlation must be present');
            obj.assertEqual(size(spearmanResults.spearman), [obj.numAssets, obj.numAssets], 'Spearman correlation size incorrect');
        end

        function testWithFiltering(obj)
            % Tests cross-sectional analysis with various filtering options
            % Create test data with outliers and missing values
            testData = obj.testData;
            testData(1:5, 1:5) = NaN; % Introduce missing values
            testData(6:10, 6:10) = 1e6; % Introduce outliers

            % Configure filtering options for outlier detection and handling
            options = struct();
            options.preprocess = true;
            options.filter_options.missing_handling = 'remove';
            options.filter_options.outlier_detection = 'zscore';
            options.filter_options.outlier_handling = 'winsorize';

            % Call analyze_cross_section with filtering enabled
            results = analyze_cross_section(testData, options);

            % Verify that outliers are handled correctly
            obj.assertTrue(isstruct(results), 'Results must be a structure');
            obj.assertTrue(isfield(results, 'preprocessing'), 'Preprocessing results must be present');
            obj.assertTrue(isfield(results.preprocessing, 'data'), 'Filtered data must be present');

            % Test different filtering methods (winsorization, trimming)
            options.filter_options.outlier_handling = 'trim';
            resultsTrim = analyze_cross_section(testData, options);
            obj.assertTrue(size(resultsTrim.preprocessing.data, 1) < size(testData, 1), 'Trimming must reduce number of observations');

            % Compare filtered results with expected values
            obj.assertTrue(all(isfinite(results.preprocessing.data(:))), 'Filtered data must not contain NaN or Inf');
        end

        function testPortfolioStatistics(obj)
            % Tests the portfolio-level statistics calculation
            % Configure portfolio weights for test assets
            weights = rand(obj.numAssets, 1);
            weights = weights / sum(weights); % Normalize weights

            % Call analyze_cross_section with portfolio analysis enabled
            options = struct();
            options.portfolio = true;
            options.portfolio_options.weights = weights;
            results = analyze_cross_section(obj.testData, options);

            % Verify portfolio return calculation accuracy
            obj.assertTrue(isstruct(results), 'Results must be a structure');
            obj.assertTrue(isfield(results, 'portfolio'), 'Portfolio results must be present');
            obj.assertTrue(isfield(results.portfolio, 'expected_return'), 'Expected return must be present');

            % Check portfolio risk measures (variance, diversification ratio)
            obj.assertTrue(isfield(results.portfolio, 'variance'), 'Variance must be present');
            obj.assertTrue(isfield(results.portfolio, 'diversification'), 'Diversification must be present');

            % Test different weighting schemes (equal, market-cap)
            equalWeights = ones(obj.numAssets, 1) / obj.numAssets;
            options.portfolio_options.weights = equalWeights;
            resultsEqual = analyze_cross_section(obj.testData, options);
            obj.assertMatrixEqualsWithTolerance(mean(obj.testData) * equalWeights, resultsEqual.portfolio.expected_return, 1e-10, 'Equal weighted portfolio return incorrect');

            % Validate portfolio performance metrics
            obj.assertTrue(results.portfolio.sharpe_ratio > -100 && results.portfolio.sharpe_ratio < 100, 'Sharpe ratio must be within reasonable bounds');
        end

        function testBootstrapStatistics(obj)
            % Tests the bootstrap confidence interval generation
            % Configure bootstrap options (replications, confidence level)
            options = struct();
            options.bootstrap = true;
            options.bootstrap_options.replications = 100;
            options.bootstrap_options.conf_level = 0.90;

            % Call analyze_cross_section with bootstrap enabled
            results = analyze_cross_section(obj.testData, options);

            % Verify bootstrap confidence intervals contain true parameters
            obj.assertTrue(isstruct(results), 'Results must be a structure');
            obj.assertTrue(isfield(results, 'bootstrap'), 'Bootstrap results must be present');
            obj.assertTrue(isfield(results.bootstrap, 'confidence_intervals'), 'Confidence intervals must be present');

            % Check bootstrap distribution properties
            obj.assertTrue(isfield(results.bootstrap, 'standard_error'), 'Standard error must be present');
            obj.assertTrue(isfield(results.bootstrap, 'bias'), 'Bias must be present');

            % Test bootstrap statistical validity
            obj.assertTrue(results.bootstrap.confidence_intervals.lower < results.bootstrap.original_statistic && ...
                           results.bootstrap.confidence_intervals.upper > results.bootstrap.original_statistic, ...
                           'Confidence interval must contain original statistic');

            % Validate bootstrap implementation efficiency
            obj.assertTrue(length(results.bootstrap.bootstrap_statistics) == options.bootstrap_options.replications, ...
                           'Number of bootstrap statistics must match number of replications');
        end

        function testErrorHandling(obj)
            % Tests error handling for invalid inputs
            % Test with empty data matrix
            obj.assertThrows(@() analyze_cross_section([]), 'datacheck:DATA', 'Empty data matrix must throw an error');

            % Test with non-numeric data
            obj.assertThrows(@() analyze_cross_section({'a', 'b', 'c'}), 'datacheck:DATA', 'Non-numeric data must throw an error');

            % Test with data containing NaN/Inf without filtering
            dataWithNaN = obj.testData;
            dataWithNaN(1, 1) = NaN;
            obj.assertThrows(@() analyze_cross_section(dataWithNaN), 'datacheck:DATA', 'Data with NaN must throw an error');

            dataWithInf = obj.testData;
            dataWithInf(1, 1) = Inf;
            obj.assertThrows(@() analyze_cross_section(dataWithInf), 'datacheck:DATA', 'Data with Inf must throw an error');

            % Test with invalid options structure
            obj.assertThrows(@() analyze_cross_section(obj.testData, 123), 'MATLAB:unassignedOutputs', 'Invalid options structure must throw an error');

            % Verify appropriate error messages are generated
            try
                analyze_cross_section([]);
            catch ME
                obj.assertTrue(strcmp(ME.identifier, 'datacheck:DATA'), 'Incorrect error identifier');
            end

            % Check error message clarity and specificity
            try
                analyze_cross_section([]);
            catch ME
                obj.assertTrue(~isempty(strfind(ME.message, 'cannot be empty')), 'Error message must be clear and specific');
            end
        end

        function testPerformance(obj)
            % Tests the performance characteristics of the cross-sectional analysis
            % Generate large-scale test data (many observations and variables)
            numObs = 2000;
            numAssets = 200;
            largeData = randn(numObs, numAssets);

            % Measure execution time using measureExecutionTime method
            executionTime = obj.measureExecutionTime(@() analyze_cross_section(largeData));

            % Compare performance with reference benchmark
            referenceTime = 1; % Reference execution time in seconds
            obj.assertTrue(executionTime < 5 * referenceTime, 'Execution time must be within reasonable bounds');

            % Test memory usage efficiency
            memoryInfo = obj.checkMemoryUsage(@() analyze_cross_section(largeData));
            obj.assertTrue(memoryInfo.memoryDifferenceMB < 500, 'Memory usage must be within reasonable bounds');

            % Verify computational complexity scaling with data size
            executionTimeSmall = obj.measureExecutionTime(@() analyze_cross_section(obj.testData));
            complexityRatio = (numObs * numAssets) / (obj.numObservations * obj.numAssets);
            obj.assertTrue(executionTime / executionTimeSmall < 10 * complexityRatio, 'Performance scaling must be reasonable');

            % Evaluate performance with different analysis options
            options = struct();
            options.correlations = false;
            executionTimeNoCorr = obj.measureExecutionTime(@() analyze_cross_section(largeData, options));
            obj.assertTrue(executionTimeNoCorr < executionTime, 'Performance must improve with fewer options');
        end

        function testEdgeCases(obj)
            % Tests cross-sectional analysis with edge case inputs
            % Test with single observation
            singleObsData = obj.testData(1, :);
            resultsSingleObs = analyze_cross_section(singleObsData);
            obj.assertTrue(isstruct(resultsSingleObs), 'Single observation results must be a structure');

            % Test with single variable
            singleVarData = obj.testData(:, 1);
            resultsSingleVar = analyze_cross_section(singleVarData);
            obj.assertTrue(isstruct(resultsSingleVar), 'Single variable results must be a structure');

            % Test with perfectly correlated variables
            perfectCorrData = [obj.testData(:, 1), obj.testData(:, 1)];
            resultsPerfectCorr = analyze_cross_section(perfectCorrData);
            obj.assertTrue(isstruct(resultsPerfectCorr), 'Perfect correlation results must be a structure');

            % Test with zero-variance variables
            zeroVarianceData = zeros(obj.numObservations, 1);
            resultsZeroVariance = analyze_cross_section(zeroVarianceData);
            obj.assertTrue(isstruct(resultsZeroVariance), 'Zero variance results must be a structure');

            % Test with extreme outliers
            extremeOutliersData = obj.testData;
            extremeOutliersData(1, 1) = 1e10;
            resultsExtremeOutliers = analyze_cross_section(extremeOutliersData);
            obj.assertTrue(isstruct(resultsExtremeOutliers), 'Extreme outliers results must be a structure');

            % Verify correct handling of special cases
            obj.assertTrue(all(isfinite(resultsSingleObs.descriptive.mean)), 'Single observation mean must be finite');
            obj.assertTrue(all(isfinite(resultsSingleVar.descriptive.mean)), 'Single variable mean must be finite');
        end
    end
end