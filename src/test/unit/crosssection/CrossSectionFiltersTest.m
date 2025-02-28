classdef CrossSectionFiltersTest < BaseTest
    % Test class for the cross_section_filters module that implements comprehensive data 
    % filtering methods for cross-sectional financial data analysis
    
    properties
        testData            % Matrix for test data
        testOptions         % Structure for test options
        defaultTolerance    % Default tolerance for numerical comparisons
        comparator          % NumericalComparator instance
        dataGenerator       % TestDataGenerator instance
        crossSectionalData  % Structure with test cross-sectional data
        expectedResults     % Structure with expected results
    end
    
    methods
        function obj = CrossSectionFiltersTest()
            % Initializes a new CrossSectionFiltersTest instance
            obj = obj@BaseTest('CrossSectionFiltersTest');
            obj.defaultTolerance = 1e-10;
            obj.comparator = NumericalComparator();
            obj.dataGenerator = TestDataGenerator();
        end
        
        function setUp(obj)
            % Prepares the testing environment before each test
            setUp@BaseTest(obj);
            
            % Load test data
            try
                testDataFile = 'cross_sectional_data.mat';
                obj.crossSectionalData = obj.loadTestData(testDataFile);
            catch
                % Generate fallback data if file not found
                obj.dataGenerator.setReproducibleMode(true);
                obj.crossSectionalData = obj.dataGenerator.generateCrossSectionalData(100, 10, 3, struct());
            end
            
            % Initialize test options
            obj.testOptions = struct();
            
            % Generate additional test data with controlled properties
            obj.testData = obj.dataGenerator.generateCrossSectionalData(50, 5, 2, struct());
            
            % Insert known missing values and outliers for filter testing
            [r, c] = size(obj.testData);
            % Insert some NaN values
            missing_mask = rand(r, c) < 0.05; % 5% missing values
            obj.testData(missing_mask) = NaN;
            
            % Insert some outliers (values far from the mean)
            outlier_mask = rand(r, c) < 0.03; % 3% outliers
            outlier_indices = find(outlier_mask);
            for i = 1:length(outlier_indices)
                obj.testData(outlier_indices(i)) = obj.testData(outlier_indices(i)) * 10; % Multiply by 10 to create outliers
            end
        end
        
        function tearDown(obj)
            % Cleans up after each test execution
            tearDown@BaseTest(obj);
        end
        
        function testMainFilterFunction(obj)
            % Tests the main filter_cross_section function with various options
            
            % Configure comprehensive filtering options
            options = struct(...
                'missing_handling', 'mean', ...
                'outlier_detection', 'zscore', ...
                'outlier_threshold', 3, ...
                'outlier_handling', 'winsorize', ...
                'winsor_percentiles', [0.01, 0.99], ...
                'transform', 'none', ...
                'normalize', 'standardize');
            
            % Call filter_cross_section with prepared test data
            result = filter_cross_section(obj.testData, options);
            
            % Verify filtered data has expected properties
            obj.assertTrue(isstruct(result), 'Result should be a structure');
            obj.assertTrue(isfield(result, 'data'), 'Result should have data field');
            obj.assertTrue(isfield(result, 'missing'), 'Result should have missing field');
            obj.assertTrue(isfield(result, 'outliers'), 'Result should have outliers field');
            obj.assertTrue(isfield(result, 'normalize'), 'Result should have normalize field');
            
            % Verify correct handling of missing values
            obj.assertFalse(any(isnan(result.data(:))), 'Filtered data should not contain NaN values');
            
            % Verify correct detection and handling of outliers
            obj.assertTrue(isfield(result.outliers, 'outlier_map'), 'Result should indicate detected outliers');
            
            % Verify correct normalization
            obj.assertTrue(abs(mean(mean(result.data)) - 0) < 0.1, 'Standardized data should have mean close to 0');
            obj.assertTrue(abs(std(reshape(result.data, [], 1)) - 1) < 0.1, 'Standardized data should have std close to 1');
            
            % Check that filtering statistics are correctly reported
            obj.assertTrue(result.missing.total_missing > 0, 'Should report some missing values');
            obj.assertTrue(result.outliers.total_outliers > 0, 'Should report some outliers');
        end
        
        function testMissingValueHandling(obj)
            % Tests the handle_missing_values function with different methods
            
            % Prepare test data with controlled missing values
            testData = obj.generateTestDataWithMissingValues(30, 5, 0.1, 'random');
            
            % Test 'remove' method for missing values
            options = struct('missing_handling', 'remove');
            result = filter_cross_section(testData, options);
            obj.assertTrue(size(result.data, 1) < size(testData, 1), 'Some rows should be removed');
            obj.assertFalse(any(isnan(result.data(:))), 'Result should not contain NaN values');
            
            % Test 'mean' imputation method
            options.missing_handling = 'mean';
            result = filter_cross_section(testData, options);
            obj.assertEqual(size(result.data, 1), size(testData, 1), 'Row count should be unchanged');
            obj.assertFalse(any(isnan(result.data(:))), 'Result should not contain NaN values');
            
            % Calculate column means for verification
            col_means = mean(testData, 'omitnan');
            
            % Check a few imputed values
            [rows, cols] = find(isnan(testData));
            if ~isempty(rows)
                sample_idx = min(length(rows), 5); % Check up to 5 imputed values
                for i = 1:sample_idx
                    row = rows(i);
                    col = cols(i);
                    obj.assertEqual(result.data(row, col), col_means(col), 'Imputed value should equal column mean');
                end
            end
            
            % Test 'median' imputation method
            options.missing_handling = 'median';
            result = filter_cross_section(testData, options);
            obj.assertFalse(any(isnan(result.data(:))), 'Result should not contain NaN values');
            
            % Test 'mode' imputation method
            options.missing_handling = 'mode';
            result = filter_cross_section(testData, options);
            obj.assertFalse(any(isnan(result.data(:))), 'Result should not contain NaN values');
            
            % Test 'knn' imputation method
            options.missing_handling = 'knn';
            options.missing_k = 3;
            result = filter_cross_section(testData, options);
            obj.assertFalse(any(isnan(result.data(:))), 'Result should not contain NaN values');
        end
        
        function testOutlierHandling(obj)
            % Tests the handle_outliers function with different detection and handling methods
            
            % Prepare test data with controlled outliers
            testData = obj.generateTestDataWithOutliers(30, 5, 0.1, 5.0);
            
            % Test Z-score outlier detection method
            options = struct('outlier_detection', 'zscore', 'outlier_threshold', 3, 'outlier_handling', 'none');
            result = filter_cross_section(testData, options);
            obj.assertTrue(isfield(result.outliers, 'outlier_map'), 'Should detect outliers');
            obj.assertTrue(sum(result.outliers.outlier_map(:)) > 0, 'Should identify some outliers');
            
            % Verify outliers are correctly identified using Z-score method
            % Calculate Z-scores manually for validation
            testDataCleaned = testData;
            testDataCleaned(isnan(testDataCleaned)) = 0; % Replace NaN with 0 for this test
            z_scores = abs((testDataCleaned - mean(testDataCleaned)) ./ std(testDataCleaned));
            expected_outliers = z_scores > 3;
            obj.assertTrue(sum(sum(abs(result.outliers.outlier_map - expected_outliers))) / numel(expected_outliers) < 0.1, ...
                'Z-score outlier detection should match expectation');
            
            % Test MAD outlier detection method
            options.outlier_detection = 'mad';
            result = filter_cross_section(testData, options);
            obj.assertTrue(sum(result.outliers.outlier_map(:)) > 0, 'Should identify some outliers');
            
            % Test IQR outlier detection method
            options.outlier_detection = 'iqr';
            options.outlier_threshold = 1.5;
            result = filter_cross_section(testData, options);
            obj.assertTrue(sum(result.outliers.outlier_map(:)) > 0, 'Should identify some outliers');
            
            % Test 'winsorize' outlier handling method
            options.outlier_handling = 'winsorize';
            options.winsor_percentiles = [0.05, 0.95];
            result = filter_cross_section(testData, options);
            
            % Verify winsorization
            for col = 1:size(testData, 2)
                lowerBound = prctile(testData(:, col), 5);
                upperBound = prctile(testData(:, col), 95);
                obj.assertTrue(all(result.data(:, col) >= lowerBound & result.data(:, col) <= upperBound), ...
                    'Winsorized data should be within percentile bounds');
            end
            
            % Test 'trim' outlier handling method
            options.outlier_handling = 'trim';
            result = filter_cross_section(testData, options);
            obj.assertTrue(size(result.data, 1) < size(testData, 1), 'Some rows should be removed');
            
            % Test 'replace' outlier handling method
            options.outlier_handling = 'replace';
            result = filter_cross_section(testData, options);
            obj.assertEqual(size(result.data), size(testData), 'Data dimensions should be unchanged');
            obj.assertTrue(sum(result.outliers.outlier_map(:)) > 0, 'Should identify some outliers');
        end
        
        function testDataTransformation(obj)
            % Tests the transform_data function with different transformation methods
            
            % Prepare test data for transformation
            testData = abs(obj.dataGenerator.generateCrossSectionalData(30, 5, 2, struct()));
            testData = testData + 0.1; % Ensure all values are positive for log transform
            
            % Test logarithmic transformation
            options = struct('transform', 'log', 'transform_offset', 0);
            result = filter_cross_section(testData, options);
            
            % Verify logarithmic transformation is correctly applied
            expectedTransformed = log(testData);
            obj.assertMatrixEqualsWithTolerance(result.data, expectedTransformed, 1e-10, 'Log transformation should match expected result');
            
            % Test square root transformation
            options.transform = 'sqrt';
            result = filter_cross_section(testData, options);
            
            % Verify square root transformation is correctly applied
            expectedTransformed = sqrt(testData);
            obj.assertMatrixEqualsWithTolerance(result.data, expectedTransformed, 1e-10, 'Square root transformation should match expected result');
            
            % Test Box-Cox transformation
            options.transform = 'boxcox';
            options.transform_lambda = 0.5;
            options.transform_estimate = false;
            result = filter_cross_section(testData, options);
            
            % Verify Box-Cox transformation is correctly applied
            expectedTransformed = (testData.^0.5 - 1) / 0.5;
            obj.assertMatrixEqualsWithTolerance(result.data, expectedTransformed, 1e-10, 'Box-Cox transformation should match expected result');
            
            % Test Yeo-Johnson transformation
            options.transform = 'yj';
            options.transform_lambda = 1.0;
            options.transform_estimate = false;
            result = filter_cross_section(testData, options);
            
            % Test rank transformation
            options.transform = 'rank';
            result = filter_cross_section(testData, options);
            
            % Verify rank transformation
            for col = 1:size(testData, 2)
                [~, ranks] = sort(testData(:, col));
                expected_ranks = zeros(size(ranks));
                expected_ranks(ranks) = (1:length(ranks))';
                expected_normalized = (expected_ranks - 0.5) / length(ranks);
                
                obj.assertMatrixEqualsWithTolerance(result.data(:, col), expected_normalized, 1e-10, ...
                    'Rank transformation should match expected result');
            end
            
            % Verify transformation validation using normality tests
            options.transform = 'boxcox';
            options.transform_estimate = true;
            result = filter_cross_section(testData, options);
            
            obj.assertTrue(isfield(result.transform, 'normality'), 'Transformation should include normality metrics');
            obj.assertTrue(isfield(result.transform.normality, 'improvement'), 'Should report improvement in normality');
        end
        
        function testDataNormalization(obj)
            % Tests the normalize_data function with different normalization methods
            
            % Prepare test data for normalization
            testData = obj.dataGenerator.generateCrossSectionalData(30, 5, 2, struct());
            
            % Test standardization method (z-score)
            options = struct('normalize', 'standardize');
            result = filter_cross_section(testData, options);
            
            % Verify data is correctly centered to zero mean and unit variance
            for col = 1:size(testData, 2)
                obj.assertTrue(abs(mean(result.data(:, col))) < 1e-10, 'Standardized data should have zero mean');
                obj.assertTrue(abs(std(result.data(:, col)) - 1) < 1e-10, 'Standardized data should have unit variance');
            end
            
            % Test min-max scaling method
            options.normalize = 'minmax';
            result = filter_cross_section(testData, options);
            
            % Verify data is correctly scaled to [0,1] range
            for col = 1:size(testData, 2)
                obj.assertTrue(min(result.data(:, col)) >= 0, 'Min-max scaled data should have minimum >= 0');
                obj.assertTrue(max(result.data(:, col)) <= 1, 'Min-max scaled data should have maximum <= 1');
            end
            
            % Test robust scaling method
            options.normalize = 'robust';
            result = filter_cross_section(testData, options);
            
            % Verify data is correctly scaled using median and MAD
            for col = 1:size(testData, 2)
                obj.assertTrue(abs(median(result.data(:, col))) < 1e-10, 'Robustly scaled data should have zero median');
            end
            
            % Test decimal scaling method
            options.normalize = 'decimal';
            result = filter_cross_section(testData, options);
            
            % Verify data is correctly scaled by powers of 10
            for col = 1:size(testData, 2)
                max_abs = max(abs(testData(:, col)));
                scale = 10^floor(log10(max_abs));
                expected = testData(:, col) / scale;
                obj.assertMatrixEqualsWithTolerance(result.data(:, col), expected, 1e-10, ...
                    'Decimal scaling should match expected result');
            end
            
            % Test custom scaling method
            custom_centers = mean(testData);
            custom_scales = std(testData);
            options.normalize = 'custom';
            options.normalize_center = custom_centers;
            options.normalize_scale = custom_scales;
            result = filter_cross_section(testData, options);
            
            % Verify data is correctly scaled using user-specified factors
            expected = zeros(size(testData));
            for col = 1:size(testData, 2)
                expected(:, col) = (testData(:, col) - custom_centers(col)) / custom_scales(col);
            end
            obj.assertMatrixEqualsWithTolerance(result.data, expected, 1e-10, 'Custom scaling should match expected result');
        end
        
        function testInputValidation(obj)
            % Tests input validation for all filtering functions
            
            % Prepare invalid inputs for each filtering function
            
            % Test with non-numeric data
            non_numeric_data = {'a', 'b'; 'c', 'd'};
            obj.assertThrows(@() filter_cross_section(non_numeric_data), 'MATLAB:invalidType', ...
                'Should throw error for non-numeric data');
            
            % Test with empty data
            empty_data = [];
            obj.assertThrows(@() filter_cross_section(empty_data), 'MATLAB:expectedNonempty', ...
                'Should throw error for empty data');
            
            % Test with invalid options structure
            invalid_options = 'not_a_struct';
            obj.assertThrows(@() filter_cross_section(obj.testData, invalid_options), 'MATLAB:invalidType', ...
                'Should throw error for non-struct options');
            
            % Test with incompatible parameters
            incompatible_options = struct(...
                'missing_handling', 'unknown_method', ...
                'outlier_detection', 'zscore');
            obj.assertThrows(@() filter_cross_section(obj.testData, incompatible_options), 'MATLAB:unrecognizedStringChoice', ...
                'Should throw error for unknown missing handling method');
            
            % Test boundary conditions for input validation
            boundary_options = struct(...
                'outlier_threshold', -1, ...  % Should be positive
                'outlier_detection', 'zscore');
            obj.assertThrows(@() filter_cross_section(obj.testData, boundary_options), 'MATLAB:expectedPositive', ...
                'Should throw error for negative outlier threshold');
        end
        
        function testSpecificOutlierDetectionMethods(obj)
            % Tests specific outlier detection methods in detail
            
            % Prepare test data with known outliers at specific positions
            testData = randn(20, 3);
            
            % Insert outliers at known positions
            testData(5, 1) = 10;  % Strong outlier in column 1
            testData(10, 2) = -8; % Strong outlier in column 2
            testData(15, 3) = 7;  % Strong outlier in column 3
            
            % Test detect_outliers_zscore function
            options = struct('outlier_detection', 'zscore', 'outlier_threshold', 3, 'outlier_handling', 'none');
            result = filter_cross_section(testData, options);
            
            % Verify correct identification of outliers using Z-score method
            outlier_map = result.outliers.outlier_map;
            obj.assertTrue(outlier_map(5, 1), 'Should detect outlier at (5,1)');
            obj.assertTrue(outlier_map(10, 2), 'Should detect outlier at (10,2)');
            obj.assertTrue(outlier_map(15, 3), 'Should detect outlier at (15,3)');
            
            % Test detect_outliers_iqr function
            options.outlier_detection = 'iqr';
            options.outlier_threshold = 1.5;
            result = filter_cross_section(testData, options);
            
            % Verify correct identification of outliers using IQR method
            outlier_map = result.outliers.outlier_map;
            obj.assertTrue(outlier_map(5, 1), 'Should detect outlier at (5,1) with IQR method');
            obj.assertTrue(outlier_map(10, 2), 'Should detect outlier at (10,2) with IQR method');
            obj.assertTrue(outlier_map(15, 3), 'Should detect outlier at (15,3) with IQR method');
            
            % Test different threshold values
            options.outlier_detection = 'zscore';
            options.outlier_threshold = 2; % Lower threshold should detect more outliers
            result = filter_cross_section(testData, options);
            
            % Count outliers with threshold=2
            outlier_count_2 = sum(result.outliers.outlier_map(:));
            
            options.outlier_threshold = 4; % Higher threshold should detect fewer outliers
            result = filter_cross_section(testData, options);
            
            % Count outliers with threshold=4
            outlier_count_4 = sum(result.outliers.outlier_map(:));
            
            obj.assertTrue(outlier_count_2 >= outlier_count_4, 'Lower threshold should detect more outliers');
            
            % Test multi-dimensional outlier detection
            multidim_data = testData;
            result = filter_cross_section(multidim_data, struct('outlier_detection', 'zscore', 'outlier_threshold', 3));
            
            % Verify that outlier detection works correctly across multiple dimensions
            obj.assertEqual(size(result.outliers.outlier_map), size(multidim_data), ...
                'Outlier map should match data dimensions');
        end
        
        function testWinsorization(obj)
            % Tests the winsorize function in detail
            
            % Prepare test data with known extreme values
            testData = randn(50, 3);
            
            % Insert extreme values
            testData(5, 1) = 10;   % High extreme in column 1
            testData(10, 2) = -8;  % Low extreme in column 2
            testData(15, 3) = 12;  % High extreme in column 3
            testData(20, 3) = -10; % Low extreme in column 3
            
            % Test symmetric winsorization (same percentiles)
            options = struct('outlier_detection', 'none', 'outlier_handling', 'winsorize', ...
                'winsor_percentiles', [0.05, 0.95]);
            result = filter_cross_section(testData, options);
            
            % Verify extreme values are correctly capped at symmetric percentiles
            for col = 1:size(testData, 2)
                lowerBound = prctile(testData(:, col), 5);
                upperBound = prctile(testData(:, col), 95);
                obj.assertTrue(all(result.data(:, col) >= lowerBound), 'All values should be >= lower bound');
                obj.assertTrue(all(result.data(:, col) <= upperBound), 'All values should be <= upper bound');
            end
            
            % Test asymmetric winsorization (different percentiles)
            options.winsor_percentiles = [0.1, 0.9];
            result = filter_cross_section(testData, options);
            
            % Verify extreme values are correctly capped at asymmetric percentiles
            for col = 1:size(testData, 2)
                lowerBound = prctile(testData(:, col), 10);
                upperBound = prctile(testData(:, col), 90);
                obj.assertTrue(all(result.data(:, col) >= lowerBound), 'All values should be >= lower bound');
                obj.assertTrue(all(result.data(:, col) <= upperBound), 'All values should be <= upper bound');
            end
            
            % Test column-specific winsorization
            % This test checks if the behavior is consistent for each column individually
            for col = 1:size(testData, 2)
                col_options = struct('outlier_detection', 'none', 'outlier_handling', 'winsorize', ...
                    'winsor_percentiles', [0.05, 0.95]);
                col_result = filter_cross_section(testData(:, col), col_options);
                
                lowerBound = prctile(testData(:, col), 5);
                upperBound = prctile(testData(:, col), 95);
                obj.assertTrue(all(col_result.data >= lowerBound), ...
                    sprintf('Column %d values should be >= lower bound', col));
                obj.assertTrue(all(col_result.data <= upperBound), ...
                    sprintf('Column %d values should be <= upper bound', col));
            end
            
            % Test winsorization with different percentile values
            percentiles_to_test = {[0.01, 0.99], [0.05, 0.95], [0.1, 0.9], [0.25, 0.75]};
            
            for i = 1:length(percentiles_to_test)
                p = percentiles_to_test{i};
                options.winsor_percentiles = p;
                result = filter_cross_section(testData, options);
                
                for col = 1:size(testData, 2)
                    lowerBound = prctile(testData(:, col), 100*p(1));
                    upperBound = prctile(testData(:, col), 100*p(2));
                    obj.assertTrue(all(result.data(:, col) >= lowerBound), ...
                        sprintf('With percentiles [%.2f, %.2f], values should be >= lower bound', p(1), p(2)));
                    obj.assertTrue(all(result.data(:, col) <= upperBound), ...
                        sprintf('With percentiles [%.2f, %.2f], values should be <= upper bound', p(1), p(2)));
                end
            end
        end
        
        function testBoxCoxOptimization(obj)
            % Tests the optimize_boxcox function for lambda parameter optimization
            
            % Prepare test data with known distribution properties
            % Log-normal distribution (becomes normal after log transform, so lambda should be ~0)
            lognormal_data = exp(randn(100, 1));
            
            % Configure options for Box-Cox with lambda estimation
            options = struct('transform', 'boxcox', 'transform_estimate', true);
            
            % Apply transformation
            result = filter_cross_section(lognormal_data, options);
            
            % Verify optimal lambda selection
            lambda = result.transform.parameters.lambda;
            obj.assertTrue(abs(lambda) < 0.3, 'Lambda should be close to 0 for log-normal data');
            
            % Verify transformation improves normality
            improvement = result.transform.normality.improvement;
            obj.assertTrue(improvement > 0, 'Normality should improve after Box-Cox transformation');
            
            % Test with different initial distributions
            % Chi-square distribution (right-skewed, lambda should be < 1)
            % Generate chi-square with degrees of freedom 4
            degrees = 4;
            chi_sq_data = sum(randn(100, degrees).^2, 2); % Sum of squared standard normals is chi-square
            
            result = filter_cross_section(chi_sq_data, options);
            lambda = result.transform.parameters.lambda;
            obj.assertTrue(lambda < 1, 'Lambda should be < 1 for right-skewed chi-square data');
            
            % Uniform distribution (lambda should be close to 1)
            uniform_data = rand(100, 1);
            result = filter_cross_section(uniform_data, options);
            lambda = result.transform.parameters.lambda;
            obj.assertTrue(abs(lambda - 1) < 0.5, 'Lambda should be close to 1 for uniform data');
            
            % Test with boundary distributions
            % Data with zeros (should err)
            zero_data = [lognormal_data; 0];
            obj.assertThrows(@() filter_cross_section(zero_data, options), 'MATLAB:positiverequired', ...
                'Should error with non-positive data');
        end
        
        function testLargeDatasetPerformance(obj)
            % Tests the performance of filtering functions with large datasets
            
            % Generate large cross-sectional dataset
            largeData = obj.dataGenerator.generateCrossSectionalData(1000, 20, 3, struct());
            
            % Insert some missing values and outliers
            [r, c] = size(largeData);
            missing_mask = rand(r, c) < 0.03; % 3% missing values
            largeData(missing_mask) = NaN;
            
            % Add some outliers
            outlier_mask = rand(r, c) < 0.02; % 2% outliers
            outlier_indices = find(outlier_mask);
            for i = 1:length(outlier_indices)
                largeData(outlier_indices(i)) = largeData(outlier_indices(i)) * 10;
            end
            
            % Define options for comprehensive filtering
            options = struct(...
                'missing_handling', 'mean', ...
                'outlier_detection', 'zscore', ...
                'outlier_handling', 'winsorize', ...
                'transform', 'none', ...
                'normalize', 'standardize');
            
            % Measure execution time for main filtering function
            executionTime = obj.measureExecutionTime(@filter_cross_section, largeData, options);
            
            % Verify that functions handle large datasets efficiently
            obj.assertTrue(executionTime < 10, 'Filtering should complete in reasonable time');
            
            % Test filter on large dataset and check results
            result = filter_cross_section(largeData, options);
            
            % Verify numerical stability with large datasets
            obj.assertFalse(any(isnan(result.data(:))), 'Result should not contain NaN values');
            obj.assertFalse(any(isinf(result.data(:))), 'Result should not contain Inf values');
            
            % Test algorithm efficiency with increasing dataset sizes
            sizes = [100, 500, 1000];
            times = zeros(length(sizes), 1);
            
            for i = 1:length(sizes)
                testData = obj.dataGenerator.generateCrossSectionalData(sizes(i), 10, 3, struct());
                times(i) = obj.measureExecutionTime(@filter_cross_section, testData, options);
            end
            
            % Check if execution time scales reasonably with data size
            % Execution time should increase less than quadratically with data size
            scaling_factor = times(end) / times(1);
            size_factor = sizes(end) / sizes(1);
            obj.assertTrue(scaling_factor < size_factor^2, 'Algorithm complexity should scale reasonably');
        end
        
        %% Helper methods
        
        function data = generateTestDataWithMissingValues(obj, numObservations, numVariables, missingRatio, missingPattern)
            % Generates test data with controlled missing value patterns
            
            % Generate base cross-sectional data using dataGenerator
            data = obj.dataGenerator.generateCrossSectionalData(numObservations, numVariables, 2, struct());
            
            % Calculate number of missing values to introduce
            totalElements = numObservations * numVariables;
            numMissing = round(totalElements * missingRatio);
            
            switch lower(missingPattern)
                case 'random'
                    % Completely random missing values
                    linearIndices = randperm(totalElements, numMissing);
                    data(linearIndices) = NaN;
                    
                case 'mcar' % Missing Completely At Random
                    % Same as random
                    linearIndices = randperm(totalElements, numMissing);
                    data(linearIndices) = NaN;
                    
                case 'mar' % Missing At Random
                    % Missing values depend on observed variables
                    % E.g., higher values have higher probability of being missing
                    data_abs = abs(data);
                    data_min = min(data_abs(:));
                    data_max = max(data_abs(:));
                    probs = (data_abs - data_min) / (data_max - data_min);
                    missingMask = rand(size(data)) < (probs * missingRatio * 2);
                    data(missingMask) = NaN;
                    
                case 'mnar' % Missing Not At Random
                    % Missing values depend on unobserved variables or the missing values themselves
                    % Simulate by making specific variables have more missing values
                    selectedVars = randperm(numVariables, max(1, round(numVariables/3)));
                    for var = selectedVars
                        numMissingInVar = round(numObservations * missingRatio * 3);
                        rows = randperm(numObservations, min(numMissingInVar, numObservations));
                        data(rows, var) = NaN;
                    end
                    
                otherwise
                    % Default to random
                    linearIndices = randperm(totalElements, numMissing);
                    data(linearIndices) = NaN;
            end
        end
        
        function data = generateTestDataWithOutliers(obj, numObservations, numVariables, outlierRatio, outlierMagnitude)
            % Generates test data with controlled outlier patterns
            
            % Generate base cross-sectional data using dataGenerator
            data = obj.dataGenerator.generateCrossSectionalData(numObservations, numVariables, 2, struct());
            
            % Calculate number of outliers to introduce
            totalElements = numObservations * numVariables;
            numOutliers = round(totalElements * outlierRatio);
            
            % Select positions for outliers based on outlierRatio
            linearIndices = randperm(totalElements, numOutliers);
            
            % Replace selected positions with outlier values of specified magnitude
            for i = 1:length(linearIndices)
                idx = linearIndices(i);
                [row, col] = ind2sub(size(data), idx);
                
                % Determine the direction of the outlier (positive or negative)
                direction = sign(randn(1));
                
                % Create outlier by multiplying by outlierMagnitude
                data(row, col) = data(row, col) * outlierMagnitude * direction;
            end
        end
        
        function isValid = verifyFilteredResults(obj, actual, expected, tolerance)
            % Verifies filtered data results against expected values
            
            if nargin < 4
                tolerance = obj.defaultTolerance;
            end
            
            % Initialize result
            isValid = true;
            
            % Compare filtered data matrices using assertMatrixEqualsWithTolerance
            if ~obj.comparator.compareMatrices(actual.data, expected.data, tolerance).isEqual
                isValid = false;
                return;
            end
            
            % Verify filtering statistics match expected values
            % Check for proper handling of missing values
            if isfield(actual, 'missing') && isfield(expected, 'missing')
                if actual.missing.total_missing ~= expected.missing.total_missing
                    isValid = false;
                    return;
                end
            end
            
            % Check for proper handling of outliers
            if isfield(actual, 'outliers') && isfield(expected, 'outliers')
                if actual.outliers.total_outliers ~= expected.outliers.total_outliers
                    isValid = false;
                    return;
                end
            end
            
            % Verify transformation and normalization parameters
            if isfield(actual, 'transform') && isfield(expected, 'transform')
                if isfield(actual.transform, 'parameters') && isfield(expected.transform, 'parameters')
                    if ~isequal(fieldnames(actual.transform.parameters), fieldnames(expected.transform.parameters))
                        isValid = false;
                        return;
                    end
                end
            end
            
            % Verify normalization parameters
            if isfield(actual, 'normalize') && isfield(expected, 'normalize')
                if isfield(actual.normalize, 'parameters') && isfield(expected.normalize, 'parameters')
                    if ~isequal(fieldnames(actual.normalize.parameters), fieldnames(expected.normalize.parameters))
                        isValid = false;
                        return;
                    end
                end
            end
        end
    end
end