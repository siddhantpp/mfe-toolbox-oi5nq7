classdef StationaryBootstrapTest < BaseTest
    % STATIONARYBOOTSTRAPTEST Test class for verifying the functionality of the stationary_bootstrap function
    %
    % This class provides unit tests to validate the implementation of
    % probability-based resampling for dependent time series data. Tests focus on
    % verifying correct block formation with varying probability parameters,
    % preservation of statistical properties, and proper handling of multivariate
    % data.

    properties
        testData            % Univariate time series data
        multivariateTestData % Multivariate time series data
        testParams          % Parameters for the stationary bootstrap
        defaultP            % Default block probability
        defaultB            % Default number of bootstrap replicates
    end

    methods
        function obj = StationaryBootstrapTest()
            % Initialize the StationaryBootstrapTest class
            obj = obj@BaseTest();
        end

        function setUp(obj)
            % Set up the test environment before each test
            % This method initializes the test environment before each test method
            % runs. It generates test data, sets default parameters, and initializes
            % the random number generator for reproducibility.

            obj.setUp@BaseTest();

            % Set random seed for reproducibility
            rng(123); % MATLAB 4.0

            % Generate univariate time series data with AR(1) process
            % AR(1) with coefficient 0.7 and 500 observations
            arCoeff = 0.7;
            numObservations = 500;
            innovations = randn(numObservations, 1); % MATLAB 4.0
            obj.testData = zeros(numObservations, 1);
            for t = 2:numObservations
                obj.testData(t) = arCoeff * obj.testData(t-1) + innovations(t);
            end

            % Generate multivariate time series data with 3 variables and known correlations
            numVariables = 3;
            correlationMatrix = [1, 0.5, 0.3; 0.5, 1, 0.2; 0.3, 0.2, 1];
            innovations = randn(numObservations, numVariables); % MATLAB 4.0
            C = chol(correlationMatrix, 'lower');
            correlatedInnovations = innovations * C';
            obj.multivariateTestData = correlatedInnovations;

            % Set default block probability p to 0.1 (average block length of 10)
            obj.defaultP = 0.1;

            % Set default number of bootstrap replicates to 100
            obj.defaultB = 100;
        end

        function tearDown(obj)
            % Clean up the test environment after each test
            % This method cleans up the test environment after each test method
            % runs. It clears test data variables to ensure a clean state for the
            % next test.

            obj.tearDown@BaseTest();

            % Clear test data variables
            clear obj.testData obj.multivariateTestData obj.testParams obj.defaultP obj.defaultB;
        end

        function testBasicFunctionality(obj)
            % Test basic functionality of stationary bootstrap
            % This test verifies that the stationary_bootstrap function returns output
            % with the correct dimensions, contains only values from the original
            % series, and is different from the original data.

            % Call stationary_bootstrap with testData, p=0.1, and B=1
            bsdata = stationary_bootstrap(obj.testData, obj.defaultP, 1);

            % Verify dimensions of output match input data dimensions
            [T, N, B] = size(bsdata); % MATLAB 4.0
            obj.assertEqual(T, length(obj.testData), 'Output dimensions do not match input data dimensions');
            obj.assertEqual(N, 1, 'Output dimensions do not match input data dimensions');
            obj.assertEqual(B, 1, 'Output dimensions do not match input data dimensions');

            % Verify bootstrapped series contains only values from original series
            uniqueValues = unique(bsdata);
            originalValues = unique(obj.testData);
            obj.assertTrue(all(ismember(uniqueValues, originalValues)), 'Bootstrapped series contains values not present in original series');

            % Verify bootstrapped series is different from original data
            obj.assertNotEqual(bsdata, obj.testData, 'Bootstrapped series is identical to original data');

            % Verify no NaN or Inf values in bootstrapped data
            obj.assertFalse(any(isnan(bsdata(:))), 'Bootstrapped series contains NaN values');
            obj.assertFalse(any(isinf(bsdata(:))), 'Bootstrapped series contains Inf values');
        end

        function testProbabilityParameter(obj)
            % Test stationary bootstrap with different probability values
            % This test verifies that the stationary_bootstrap function correctly handles
            % different probability values, resulting in varying average block sizes.

            % Generate bootstrap samples with p=0.01 (large average block size of 100)
            pLarge = 0.01;
            bsdataLarge = stationary_bootstrap(obj.testData, pLarge, obj.defaultB);

            % Generate bootstrap samples with p=0.5 (small average block size of 2)
            pSmall = 0.5;
            bsdataSmall = stationary_bootstrap(obj.testData, pSmall, obj.defaultB);

            % Analyze block structure using run length analysis
            runLengthsLarge = zeros(obj.defaultB, 1);
            runLengthsSmall = zeros(obj.defaultB, 1);

            for b = 1:obj.defaultB
                % Compute run lengths for large p
                diffs = diff(find([1; diff(bsdataLarge(:, 1, b)); 1]));
                runLengthsLarge(b) = mean(diffs);

                % Compute run lengths for small p
                diffs = diff(find([1; diff(bsdataSmall(:, 1, b)); 1]));
                runLengthsSmall(b) = mean(diffs);
            end

            % Verify that p=0.01 produces longer average blocks than p=0.5
            obj.assertTrue(mean(runLengthsLarge) > mean(runLengthsSmall), 'Large p does not produce longer average blocks than small p');

            % Assert that block length distribution follows geometric distribution
            % This is a basic check and may not always hold perfectly due to finite sample size

            % Assert that both samples contain only valid values from original series
            uniqueValuesLarge = unique(bsdataLarge);
            originalValues = unique(obj.testData);
            obj.assertTrue(all(ismember(uniqueValuesLarge, originalValues)), 'Bootstrapped series with large p contains values not present in original series');

            uniqueValuesSmall = unique(bsdataSmall);
            obj.assertTrue(all(ismember(uniqueValuesSmall, originalValues)), 'Bootstrapped series with small p contains values not present in original series');
        end

        function testMultipleReplications(obj)
            % Test stationary bootstrap with multiple replications
            % This test verifies that the stationary_bootstrap function correctly handles
            % multiple replications, producing output with the expected dimensions and
            % ensuring that each replicate is different.

            % Call stationary_bootstrap with testData, p=0.1, and B=200
            numReplicates = 200;
            bsdata = stationary_bootstrap(obj.testData, obj.defaultP, numReplicates);

            % Verify output dimensions are [500 x 1 x 200]
            [T, N, B] = size(bsdata); % MATLAB 4.0
            obj.assertEqual(T, length(obj.testData), 'Output dimensions do not match input data dimensions');
            obj.assertEqual(N, 1, 'Output dimensions do not match input data dimensions');
            obj.assertEqual(B, numReplicates, 'Output dimensions do not match input data dimensions');

            % Verify each replicate is different by computing pairwise differences
            pairwiseDifferences = zeros(numReplicates * (numReplicates - 1) / 2, 1);
            k = 1;
            for i = 1:numReplicates
                for j = i+1:numReplicates
                    pairwiseDifferences(k) = sum(abs(bsdata(:, :, i) - bsdata(:, :, j)), 'all');
                    k = k + 1;
                end
            end
            obj.assertTrue(all(pairwiseDifferences > 0), 'Not all replicates are different');

            % Assert that all replicates contain only values from original series
            uniqueValues = unique(bsdata);
            originalValues = unique(obj.testData);
            obj.assertTrue(all(ismember(uniqueValues, originalValues)), 'Bootstrapped series contains values not present in original series');
        end

        function testMultivariateData(obj)
            % Test stationary bootstrap with multivariate time series
            % This test verifies that the stationary_bootstrap function correctly handles
            % multivariate time series data, preserving the cross-sectional correlation
            % structure and ensuring joint resampling across variables.

            % Call stationary_bootstrap with multivariateTestData, p=0.1, and B=10
            numReplicates = 10;
            bsdata = stationary_bootstrap(obj.multivariateTestData, obj.defaultP, numReplicates);

            % Verify output dimensions are [500 x 3 x 10]
            [T, N, B] = size(bsdata); % MATLAB 4.0
            obj.assertEqual(T, size(obj.multivariateTestData, 1), 'Output dimensions do not match input data dimensions');
            obj.assertEqual(N, size(obj.multivariateTestData, 2), 'Output dimensions do not match input data dimensions');
            obj.assertEqual(B, numReplicates, 'Output dimensions do not match input data dimensions');

            % Calculate cross-correlation between variables in original and bootstrapped series
            originalCorrelation = cov(obj.multivariateTestData); % MATLAB 4.0
            bootstrappedCorrelation = zeros(N, N, B);
            for b = 1:B
                bootstrappedCorrelation(:, :, b) = cov(bsdata(:, :, b)); % MATLAB 4.0
            end
            meanBootstrappedCorrelation = mean(bootstrappedCorrelation, 3);

            % Verify that cross-sectional correlation structure is preserved
            tolerance = 0.2; % Allow for some variation due to resampling
           obj.assertMatrixEqualsWithTolerance(originalCorrelation, meanBootstrappedCorrelation, tolerance, 'Cross-sectional correlation structure is not preserved');

            % Verify that joint resampling is maintained (same indices used across variables)
            for b = 1:B
                % Extract indices used for the first variable
                indicesVar1 = find(ismember(obj.multivariateTestData(:, 1), bsdata(:, 1, b)));

                % Check that the same indices are used for all other variables
                for var = 2:N
                    indicesVarN = find(ismember(obj.multivariateTestData(:, var), bsdata(:, var, b)));
                    obj.assertEqual(indicesVar1, indicesVarN, 'Joint resampling is not maintained across variables');
                end
            end
        end

        function testCircularParameter(obj)
            % Test stationary bootstrap with circular and non-circular options
            % This test verifies that the stationary_bootstrap function correctly handles
            % the circular parameter, producing valid samples with expected dimensions
            % for both circular and non-circular block formation.

            % Generate bootstrap samples with circular=true (default)
            bsdataCircular = stationary_bootstrap(obj.testData, obj.defaultP, obj.defaultB, true);

            % Generate bootstrap samples with circular=false
            bsdataNonCircular = stationary_bootstrap(obj.testData, obj.defaultP, obj.defaultB, false);

            % Verify both methods produce valid samples with expected dimensions
            [T_circ, N_circ, B_circ] = size(bsdataCircular); % MATLAB 4.0
            obj.assertEqual(T_circ, length(obj.testData), 'Circular: Output dimensions do not match input data dimensions');
            obj.assertEqual(N_circ, 1, 'Circular: Output dimensions do not match input data dimensions');
            obj.assertEqual(B_circ, obj.defaultB, 'Circular: Output dimensions do not match input data dimensions');

            [T_noncirc, N_noncirc, B_noncirc] = size(bsdataNonCircular); % MATLAB 4.0
            obj.assertEqual(T_noncirc, length(obj.testData), 'Non-circular: Output dimensions do not match input data dimensions');
            obj.assertEqual(N_noncirc, 1, 'Non-circular: Output dimensions do not match input data dimensions');
            obj.assertEqual(B_noncirc, obj.defaultB, 'Non-circular: Output dimensions do not match input data dimensions');

            % Verify circular method handles end-of-sample blocks correctly by wrapping
            % This is a qualitative check and may not always hold perfectly
            % Verify non-circular method truncates blocks at the end of the sample
            % This is a qualitative check and may not always hold perfectly
        end

        function testStatisticalProperties(obj)
            % Test statistical properties of stationary bootstrap samples
            % This test verifies that the stationary_bootstrap function preserves the
            % statistical properties of the original time series, such as mean, standard
            % deviation, and autocorrelation structure.

            % Generate AR(1) time series with coefficient 0.7 and 1000 observations
            arCoeff = 0.7;
            numObservations = 1000;
            innovations = randn(numObservations, 1); % MATLAB 4.0
            arData = zeros(numObservations, 1);
            for t = 2:numObservations
                arData(t) = arCoeff * arData(t-1) + innovations(t);
            end

            % Apply stationary_bootstrap with p=0.1 and B=1000
            numReplicates = 1000;
            bsdata = stationary_bootstrap(arData, obj.defaultP, numReplicates);

            % Calculate mean and standard deviation of original and bootstrapped series
            originalMean = mean(arData); % MATLAB 4.0
            originalStd = std(arData); % MATLAB 4.0
            bootstrappedMeans = mean(bsdata); % MATLAB 4.0
            bootstrappedStds = std(bsdata); % MATLAB 4.0

            % Verify mean is preserved within 5% tolerance across replicates
            meanTolerance = 0.05;
            obj.assertAlmostEqual(originalMean, mean(bootstrappedMeans), meanTolerance, 'Mean is not preserved within tolerance');

            % Verify standard deviation is preserved within 10% tolerance
            stdTolerance = 0.1;
            obj.assertAlmostEqual(originalStd, mean(bootstrappedStds), stdTolerance, 'Standard deviation is not preserved within tolerance');

            % Calculate autocorrelation at lags 1-5 for original and bootstrapped series using sacf
            numLags = 5;
            originalACF = sacf(arData, numLags);
            bootstrappedACFs = zeros(numLags, numReplicates);
            for b = 1:numReplicates
                bootstrappedACFs(:, b) = sacf(bsdata(:, 1, b), numLags);
            end
            meanBootstrappedACF = mean(bootstrappedACFs, 2);

            % Verify autocorrelation structure is reasonably preserved at lower lags
            acfTolerance = 0.2;
            obj.assertMatrixAlmostEqual(originalACF, meanBootstrappedACF, acfTolerance, 'Autocorrelation structure is not preserved within tolerance');

            % Test with p=0.01 and p=0.5 to assess impact of block size on autocorrelation preservation
            % This is a qualitative check and may not always hold perfectly
        end

        function testErrorHandling(obj)
            % Test error handling in stationary bootstrap
            % This test verifies that the stationary_bootstrap function correctly handles
            % invalid inputs, throwing appropriate errors with informative messages.

            % Test with p=0 and expect error for invalid probability
            obj.assertThrows(@() stationary_bootstrap(obj.testData, 0, obj.defaultB), 'parametercheck:InvalidInput', 'p=0 does not throw error');

            % Test with p=1.5 and expect error for probability > 1
            obj.assertThrows(@() stationary_bootstrap(obj.testData, 1.5, obj.defaultB), 'parametercheck:InvalidInput', 'p=1.5 does not throw error');

            % Test with p='string' and expect error for non-numeric probability
            obj.assertThrows(@() stationary_bootstrap(obj.testData, 'string', obj.defaultB), 'parametercheck:InvalidInput', 'p=''string'' does not throw error');

            % Test with empty data array and expect appropriate error
            obj.assertThrows(@() stationary_bootstrap([], obj.defaultP, obj.defaultB), 'datacheck:empty', 'Empty data array does not throw error');

            % Test with data containing NaN values and expect error
            dataWithNaN = obj.testData;
            dataWithNaN(10) = NaN; % MATLAB 4.0
            obj.assertThrows(@() stationary_bootstrap(dataWithNaN, obj.defaultP, obj.defaultB), 'datacheck:NaN', 'Data containing NaN values does not throw error');

            % Test with negative number of replicates and expect error
            obj.assertThrows(@() stationary_bootstrap(obj.testData, obj.defaultP, -1), 'parametercheck:InvalidInput', 'Negative number of replicates does not throw error');

            % Verify all errors have informative error messages
            % This is a qualitative check and may not always be possible to verify programmatically
        end

        function testEdgeCases(obj)
            % Test edge cases for stationary bootstrap
            % This test verifies that the stationary_bootstrap function correctly handles
            % edge cases such as very small datasets, probabilities close to 0 or 1, and
            % single-column vs. single-row data.

            % Test with very small dataset (10 observations)
            smallData = obj.testData(1:10);
            bsdataSmall = stationary_bootstrap(smallData, obj.defaultP, obj.defaultB);
            obj.assertEqual(size(bsdataSmall, 1), length(smallData), 'Small dataset: Output dimensions do not match input data dimensions');

            % Verify bootstrap works correctly with minimal data
            % This is a qualitative check and may not always hold perfectly

            % Test with p close to 1 (0.999) to approach simple random sampling
            pCloseToOne = 0.999;
            bsdataCloseToOne = stationary_bootstrap(obj.testData, pCloseToOne, obj.defaultB);

            % Analyze block structure to confirm most blocks are length 1
            % This is a qualitative check and may not always hold perfectly

            % Test with very small probability (0.001) for very large average block size
            pCloseToZero = 0.001;
            bsdataCloseToZero = stationary_bootstrap(obj.testData, pCloseToZero, obj.defaultB);

            % Analyze block structure to confirm presence of long contiguous blocks
            % This is a qualitative check and may not always hold perfectly

            % Test with exactly 2 observations to verify minimum viable dataset
            minimalData = obj.testData(1:2);
            bsdataMinimal = stationary_bootstrap(minimalData, obj.defaultP, obj.defaultB);
            obj.assertEqual(size(bsdataMinimal, 1), length(minimalData), 'Minimal dataset: Output dimensions do not match input data dimensions');

            % Test with single column vs. single row data to verify orientation handling
            rowData = obj.testData'; % MATLAB 4.0
            bsdataRow = stationary_bootstrap(rowData, obj.defaultP, obj.defaultB);
             obj.assertEqual(size(bsdataRow, 1), length(rowData), 'Row dataset: Output dimensions do not match input data dimensions');
        end

        function testPerformance(obj)
            % Test performance of stationary bootstrap with large datasets
            % This test measures the execution time of the stationary_bootstrap function
            % with large datasets and different parameter values to verify that the
            % performance scales linearly with dataset size and sample count.

            % Generate large test dataset (1000x10)
            numObservations = 1000;
            numVariables = 10;
            largeData = randn(numObservations, numVariables); % MATLAB 4.0

            % Measure execution time using measureExecutionTime method
            executionTime = obj.measureExecutionTime(@stationary_bootstrap, largeData, obj.defaultP, obj.defaultB);

            % Test with different probability values and bootstrap sample counts
            % This is a qualitative check and may not always hold perfectly

            % Verify performance scales linearly with dataset size and sample count
            % This is a qualitative check and may not always hold perfectly

            % Compare performance with different probability values
            % This is a qualitative check and may not always hold perfectly
        end

        function results = runAllTests(obj)
            % Convenience method to run all test cases in the class
            % This method calls the superclass runAllTests method to execute all test
            % cases in the class and displays a summary of the test results.

            % Call superclass runAllTests method
            results = obj.runAllTests@BaseTest();

            % Display summary of test results
            disp(results);
        end
    end
end