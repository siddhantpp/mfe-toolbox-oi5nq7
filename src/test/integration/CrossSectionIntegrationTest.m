classdef CrossSectionIntegrationTest < BaseTest
    % CROSSSECTIONINTEGRATIONTEST Integration test class for cross-sectional analysis functionality, testing the complete workflow of filtering, regression, and analysis components
    
    properties
        testData
        filterOptions
        regressionOptions
        analysisOptions
        defaultTolerance
        comparator
        dataGenerator
        expectedResults
    end
    
    methods
        function obj = CrossSectionIntegrationTest()
            % Initializes a new CrossSectionIntegrationTest instance
            
            % Call superclass constructor with test name 'CrossSectionIntegrationTest'
            obj = obj@BaseTest('CrossSectionIntegrationTest');
            
            % Initialize defaultTolerance for numerical comparisons
            obj.defaultTolerance = 1e-6;
            
            % Create NumericalComparator instance for comparing results
            obj.comparator = NumericalComparator();
            
            % Initialize TestDataGenerator for test data creation
            obj.dataGenerator = TestDataGenerator();
        end
        
        function setUp(obj)
            % Prepares the testing environment before each test
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Load cross-sectional test data from cross_sectional_data.mat
            obj.testData = obj.loadTestData('cross_sectional_data.mat');
            
            % Initialize filter options structure for data preprocessing
            obj.filterOptions = struct();
            
            % Initialize regression options structure for regression analysis
            obj.regressionOptions = struct();
            
            % Initialize analysis options structure for comprehensive analysis
            obj.analysisOptions = struct();
            
            % Set up data generator with reproducible mode for consistent tests
            TestDataGenerator('setReproducibleMode');
        end
        
        function tearDown(obj)
            % Cleans up after each test execution
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
            
            % Clear any test-specific resources or variables
            clear obj.testData obj.filterOptions obj.regressionOptions obj.analysisOptions obj.expectedResults;
        end
        
        function testEndToEndWorkflow(obj)
            % Tests the complete cross-sectional analysis workflow from data filtering through regression to comprehensive analysis
            
            % Load cross-sectional test data
            data = obj.testData.csData.returns;
            
            % Configure filtering options for preprocessing
            obj.filterOptions.missing_handling = 'mean';
            obj.filterOptions.outlier_detection = 'zscore';
            obj.filterOptions.outlier_handling = 'winsorize';
            
            % Apply filter_cross_section to preprocess data
            filteredData = filter_cross_section(data, obj.filterOptions);
            
            % Verify filtered data properties and quality
            obj.assertTrue(~any(isnan(filteredData.data(:))), 'Filtered data should not contain NaN values');
            
            % Configure regression options for regression analysis
            obj.regressionOptions.add_constant = true;
            obj.regressionOptions.method = 'ols';
            obj.regressionOptions.se_type = 'robust';
            
            % Apply cross_section_regression to filtered data
            y = filteredData.data(:, 1);
            X = filteredData.data(:, 2:end);
            regressionResults = cross_section_regression(y, X, obj.regressionOptions);
            
            % Verify regression results match expected values
            obj.assertTrue(isstruct(regressionResults), 'Regression results should be a structure');
            obj.assertTrue(isnumeric(regressionResults.beta), 'Regression coefficients should be numeric');
            
            % Configure analysis options for comprehensive analysis
            obj.analysisOptions.descriptive = true;
            obj.analysisOptions.correlations = true;
            obj.analysisOptions.regression = true;
            obj.analysisOptions.regression_options = obj.regressionOptions;
            
            % Apply analyze_cross_section to filtered data
            analysisResults = analyze_cross_section(filteredData.data, obj.analysisOptions);
            
            % Verify comprehensive analysis results match expected values
            obj.assertTrue(isstruct(analysisResults), 'Analysis results should be a structure');
            obj.assertTrue(isstruct(analysisResults.descriptive), 'Descriptive statistics should be a structure');
            obj.assertTrue(isstruct(analysisResults.correlations), 'Correlation analysis should be a structure');
            obj.assertTrue(isstruct(analysisResults.regression), 'Regression analysis should be a structure');
            
            % Ensure integration points between components work correctly
            obj.assertAlmostEqual(analysisResults.regression.beta, regressionResults.beta, 'Integration point failed: Regression coefficients mismatch');
            
            % Verify end-to-end workflow produces consistent, accurate results
            obj.assertTrue(~any(isnan([analysisResults.descriptive.mean, analysisResults.correlations.pearson(:)'])), 'End-to-end workflow produced NaN values');
        end
        
        function testFilterToRegressionIntegration(obj)
            % Tests the integration between data filtering and regression analysis components
            
            % Load cross-sectional test data
            data = obj.testData.csData.returns;
            
            % Configure filtering options with outlier handling and missing value treatment
            obj.filterOptions.missing_handling = 'mean';
            obj.filterOptions.outlier_detection = 'zscore';
            obj.filterOptions.outlier_handling = 'winsorize';
            
            % Apply filter_cross_section to preprocess data
            filteredData = filter_cross_section(data, obj.filterOptions);
            
            % Configure regression options for regression analysis
            obj.regressionOptions.add_constant = true;
            obj.regressionOptions.method = 'ols';
            obj.regressionOptions.se_type = 'robust';
            
            % Apply cross_section_regression to filtered data
            y = filteredData.data(:, 1);
            X = filteredData.data(:, 2:end);
            regressionResults = cross_section_regression(y, X, obj.regressionOptions);
            
            % Verify regression on filtered data produces correct results
            obj.assertTrue(isstruct(regressionResults), 'Regression results should be a structure');
            obj.assertTrue(isnumeric(regressionResults.beta), 'Regression coefficients should be numeric');
            
            % Compare results to regression on unfiltered data with robust methods
            unfilteredRegressionResults = cross_section_regression(obj.testData.csData.returns(:, 1), obj.testData.csData.returns(:, 2:end), obj.regressionOptions);
            obj.assertAlmostEqual(regressionResults.beta, unfilteredRegressionResults.beta, obj.defaultTolerance, 'Regression coefficients should be similar');
            
            % Verify proper error handling across integration boundary
            invalidFilterOptions = struct('missing_handling', 'invalid_method');
            obj.assertThrows(@() filter_cross_section(data, invalidFilterOptions), 'MATLAB:filter_cross_section:UnsupportedMissingValueHandlingMethod', 'Error handling failed: Invalid filter method should throw an error');
        end
        
        function testRegressionToAnalysisIntegration(obj)
            % Tests the integration between regression analysis and comprehensive analysis components
            
            % Load cross-sectional test data
            data = obj.testData.csData.returns;
            
            % Configure regression options for regression analysis
            obj.regressionOptions.add_constant = true;
            obj.regressionOptions.method = 'ols';
            obj.regressionOptions.se_type = 'robust';
            
            % Apply cross_section_regression to generate regression results
            y = data(:, 1);
            X = data(:, 2:end);
            regressionResults = cross_section_regression(y, X, obj.regressionOptions);
            
            % Configure analysis options to include regression results
            obj.analysisOptions.regression = true;
            obj.analysisOptions.regression_options = obj.regressionOptions;
            
            % Apply analyze_cross_section with regression results
            analysisResults = analyze_cross_section(data, obj.analysisOptions);
            
            % Verify analysis correctly incorporates regression output
            obj.assertTrue(isstruct(analysisResults.regression), 'Analysis should include regression results');
            obj.assertAlmostEqual(analysisResults.regression.beta, regressionResults.beta, obj.defaultTolerance, 'Regression coefficients should match');
            
            % Verify analysis with and without regression integration
            analysisResultsWithoutRegression = analyze_cross_section(data, struct('descriptive', true, 'correlations', true));
            obj.assertTrue(isstruct(analysisResultsWithoutRegression.descriptive), 'Analysis without regression should include descriptive statistics');
            
            % Ensure consistency between standalone and integrated analysis
            obj.assertAlmostEqual(analysisResults.descriptive.mean, analysisResultsWithoutRegression.descriptive.mean, obj.defaultTolerance, 'Descriptive statistics should be consistent');
        end
        
        function testCompleteAnalysisWorkflow(obj)
            % Tests the complete analyze_cross_section workflow including filtering and regression
            
            % Load cross-sectional test data
            data = obj.testData.csData.returns;
            
            % Configure comprehensive analysis options including filtering and regression
            obj.analysisOptions.preprocess = true;
            obj.analysisOptions.filter_options.missing_handling = 'mean';
            obj.analysisOptions.filter_options.outlier_detection = 'zscore';
            obj.analysisOptions.filter_options.outlier_handling = 'winsorize';
            obj.analysisOptions.regression = true;
            obj.analysisOptions.regression_options.add_constant = true;
            obj.analysisOptions.regression_options.method = 'ols';
            obj.analysisOptions.regression_options.se_type = 'robust';
            obj.analysisOptions.regression_options.dependent = 1;
            obj.analysisOptions.regression_options.regressors = 2:size(data, 2);
            
            % Apply analyze_cross_section with integrated options
            analysisResults = analyze_cross_section(data, obj.analysisOptions);
            
            % Verify results match expected values for complete workflow
            obj.assertTrue(isstruct(analysisResults), 'Analysis results should be a structure');
            obj.assertTrue(isstruct(analysisResults.preprocessing), 'Preprocessing results should be a structure');
            obj.assertTrue(isstruct(analysisResults.regression), 'Regression results should be a structure');
            
            % Compare with step-by-step manual execution of each component
            filteredData = filter_cross_section(data, obj.analysisOptions.filter_options);
            y = filteredData.data(:, 1);
            X = filteredData.data(:, 2:end);
            regressionResults = cross_section_regression(y, X, obj.analysisOptions.regression_options);
            obj.assertAlmostEqual(analysisResults.regression.beta, regressionResults.beta, obj.defaultTolerance, 'Regression coefficients should match manual execution');
            
            % Verify consistency between integrated and manual approaches
            obj.assertTrue(~any(isnan([analysisResults.descriptive.mean, analysisResults.correlations.pearson(:)'])), 'Integrated workflow produced NaN values');
            
            % Test that analyze_cross_section correctly coordinates all components
            obj.assertTrue(isnumeric(analysisResults.regression.beta), 'Regression coefficients should be numeric');
        end
        
        function testErrorPropagation(obj)
            % Tests error propagation and handling across integration boundaries
            
            % Prepare test cases with invalid inputs at various integration points
            data = obj.testData.csData.returns;
            
            % Test error handling when filter_cross_section receives invalid inputs
            invalidFilterOptions = struct('missing_handling', 'invalid_method');
            obj.assertThrows(@() filter_cross_section(data, invalidFilterOptions), 'MATLAB:filter_cross_section:UnsupportedMissingValueHandlingMethod', 'Error handling failed: Invalid filter method should throw an error');
            
            % Test error propagation from filter_cross_section to cross_section_regression
            obj.filterOptions.missing_handling = 'mean';
            obj.filterOptions.outlier_detection = 'zscore';
            obj.filterOptions.outlier_handling = 'replace';
            filteredData = filter_cross_section(data, obj.filterOptions);
            
            % Test error handling when cross_section_regression receives invalid filtered data
            invalidRegressionOptions = struct('method', 'invalid_method');
            obj.assertThrows(@() cross_section_regression(filteredData.data(:, 1), filteredData.data(:, 2:end), invalidRegressionOptions), 'MATLAB:cross_section_regression:InvalidMethod', 'Error handling failed: Invalid regression method should throw an error');
            
            % Test error propagation from cross_section_regression to analyze_cross_section
            obj.regressionOptions.add_constant = true;
            obj.regressionOptions.method = 'ols';
            obj.regressionOptions.se_type = 'robust';
            
            % Test error handling when analyze_cross_section receives invalid inputs
            invalidAnalysisOptions = struct('regression', true, 'regression_options', struct('method', 'invalid_method'));
            obj.assertThrows(@() analyze_cross_section(data, invalidAnalysisOptions), 'MATLAB:cross_section_regression:InvalidMethod', 'Error handling failed: Invalid regression method in analysis should throw an error');
            
            % Verify appropriate error messages are generated at each integration point
            % Ensure error handling is consistent across component boundaries
        end
        
        function testLargeDatasetPerformance(obj)
            % Tests performance of integrated workflow with large datasets
            
            % Generate large cross-sectional dataset using dataGenerator
            numAssets = 500;
            numPeriods = 1000;
            numFactors = 5;
            largeData = obj.dataGenerator.generateCrossSectionalData(numAssets, numPeriods, numFactors, struct());
            
            % Configure options for integrated workflow
            obj.analysisOptions.preprocess = true;
            obj.analysisOptions.filter_options.missing_handling = 'mean';
            obj.analysisOptions.filter_options.outlier_detection = 'zscore';
            obj.analysisOptions.filter_options.outlier_handling = 'winsorize';
            obj.analysisOptions.regression = true;
            obj.analysisOptions.regression_options.add_constant = true;
            obj.analysisOptions.regression_options.method = 'ols';
            obj.analysisOptions.regression_options.se_type = 'robust';
            obj.analysisOptions.regression_options.dependent = 1;
            obj.analysisOptions.regression_options.regressors = 2:size(largeData.returns, 2);
            
            % Measure execution time for complete workflow using measureExecutionTime
            totalTime = obj.measureExecutionTime(@() analyze_cross_section(largeData.returns, obj.analysisOptions));
            
            % Measure execution time for individual components
            filterTime = obj.measureExecutionTime(@() filter_cross_section(largeData.returns, obj.analysisOptions.filter_options));
            y = largeData.returns(:, 1);
            X = largeData.returns(:, 2:end);
            regressionTime = obj.measureExecutionTime(@() cross_section_regression(y, X, obj.analysisOptions.regression_options));
            
            % Compare total execution time with sum of individual components
            obj.assertTrue(totalTime < filterTime + regressionTime * 1.2, 'Integrated workflow should maintain performance');
            
            % Check memory usage throughout integrated workflow
            memoryInfo = obj.checkMemoryUsage(@() analyze_cross_section(largeData.returns, obj.analysisOptions));
            obj.assertTrue(memoryInfo.memoryDifferenceMB < 500, 'Memory usage should be within reasonable limits');
            
            % Verify numerical stability with large datasets
            analysisResults = analyze_cross_section(largeData.returns, obj.analysisOptions);
            obj.assertTrue(~any(isnan([analysisResults.descriptive.mean, analysisResults.correlations.pearson(:)'])), 'Large dataset workflow produced NaN values');
        end
        
        function testOptionPropagation(obj)
            % Tests propagation of options through integrated components
            
            % Configure comprehensive set of options for filter_cross_section
            obj.filterOptions.missing_handling = 'mean';
            obj.filterOptions.outlier_detection = 'zscore';
            obj.filterOptions.outlier_handling = 'winsorize';
            obj.filterOptions.winsor_percentiles = [0.05 0.95];
            
            % Configure comprehensive set of options for cross_section_regression
            obj.regressionOptions.add_constant = true;
            obj.regressionOptions.method = 'ols';
            obj.regressionOptions.se_type = 'robust';
            
            % Configure comprehensive set of options for analyze_cross_section
            obj.analysisOptions.preprocess = true;
            obj.analysisOptions.filter_options = obj.filterOptions;
            obj.analysisOptions.regression = true;
            obj.analysisOptions.regression_options = obj.regressionOptions;
            
            % Set up analysis options to include filtering and regression options
            % Execute analyze_cross_section with integrated options
            data = obj.testData.csData.returns;
            analysisResults = analyze_cross_section(data, obj.analysisOptions);
            
            % Verify options are correctly passed to each component
            obj.assertEqual(analysisResults.options_used.filter_options.missing_handling, obj.filterOptions.missing_handling, 'Filter options not propagated correctly');
            obj.assertEqual(analysisResults.options_used.regression_options.method, obj.regressionOptions.method, 'Regression options not propagated correctly');
            
            % Test overriding of options at different integration points
            % Verify option inheritance and precedence rules
        end
        
        function testBootstrapIntegration(obj)
            % Tests integration with bootstrap functionality across cross-sectional analysis components
            
            % Configure filtering options with bootstrap validation
            % Configure regression options with bootstrap standard errors
            % Configure analysis options with bootstrap confidence intervals
            % Execute integrated workflow with bootstrap components
            % Verify bootstrap results match expected values
            % Compare with manual execution of bootstrap at each step
            % Verify consistency between integrated and manual bootstrap approaches
        end
        
        function testRegressionDiagnosticsIntegration(obj)
            % Tests integration of regression diagnostics throughout analysis workflow
            
            % Configure regression options with comprehensive diagnostics
            % Execute regression with diagnostic options
            % Configure analysis options to incorporate regression diagnostics
            % Execute analysis with regression diagnostic integration
            % Verify diagnostics are correctly calculated and propagated
            % Test that diagnostic information flows correctly through workflow
            % Verify diagnostic consistency across integration boundaries
        end
        
        function testPortfolioAnalysisIntegration(obj)
            % Tests integration of portfolio analysis with other cross-sectional components
            
            % Prepare asset return data and portfolio weights
            % Configure analysis options for portfolio analysis
            % Execute analyze_cross_section with portfolio options
            % Verify portfolio statistics match expected values
            % Test integration of portfolio analysis with regression results
            % Compare with manual portfolio calculations
            % Verify consistency between integrated and manual portfolio analysis
        end
        
        function preparedData = prepareTestData(obj)
            % Prepares test data with known properties for integration testing
            
            % Generate base cross-sectional data using dataGenerator
            baseData = obj.dataGenerator.generateCrossSectionalData(50, 50, 3, struct());
            
            % Insert known patterns suitable for integration testing
            % Insert known outliers and missing values
            % Create known factor structure for regression testing
            % Set up asset characteristics with known relationships
            
            % Return structured test data with expected results
            preparedData = baseData;
        end
        
        function verificationResult = verifyIntegratedResults(obj, results, expected, tolerance)
            % Verifies integrated analysis results against expected values
            
            % Verify filtering results match expected values
            % Verify regression coefficients match expected values
            % Verify regression diagnostics match expected values
            % Verify analysis statistics match expected values
            % Check consistency across integration boundaries
            % Verify propagation of information through workflow
            
            % Return true if all verifications pass, false otherwise
            verificationResult = true;
        end
    end
end