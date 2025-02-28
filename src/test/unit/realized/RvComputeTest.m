classdef RvComputeTest < BaseTest
    % Test class for rv_compute function that calculates realized volatility from high-frequency returns
    
    properties
        testData            % Matrix containing high-frequency return test data
        referenceValues     % Struct containing reference values for testing
        tolerance           % Numerical tolerance for comparisons
        isMEXAvailable      % Flag indicating whether MEX implementation is available
    end
    
    methods
        function obj = RvComputeTest()
            % Constructor for RvComputeTest class
            
            % Call superclass constructor
            obj@BaseTest();
            
            % Initialize properties with default values
            obj.tolerance = 1e-10;
            obj.isMEXAvailable = false;
        end
        
        function setUp(obj)
            % Prepare test environment before each test method execution
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Load high-frequency test data
            data = obj.loadTestData('high_frequency_data.mat');
            obj.testData = data.high_frequency_returns;
            obj.referenceValues = data.reference_values;
            
            % Set numerical comparison tolerance
            obj.tolerance = 1e-10;
            
            % Check if MEX implementation is available
            obj.isMEXAvailable = exist('rv_compute_mex', 'file') == 3; % 3 means MEX-file
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
            
            % Clear test variables
            obj.testData = [];
            obj.referenceValues = [];
        end
        
        function testBasicOperation(obj)
            % Test basic functionality of rv_compute with standard inputs
            
            % Call rv_compute with standard high-frequency returns
            result = rv_compute(obj.testData);
            
            % Compare result with reference value
            expected = obj.referenceValues.realized_volatility;
            obj.assertAlmostEqual(expected, result, 'Basic realized volatility calculation failed');
        end
        
        function testSamplingFrequencies(obj)
            % Test rv_compute with different sampling frequencies
            
            % Create options structure with 5-minute sampling frequency
            options = struct('method', 'subsample', 'subSample', 5);
            result5min = rv_compute(obj.testData, options);
            
            % Test with 10-minute sampling frequency
            options.subSample = 10;
            result10min = rv_compute(obj.testData, options);
            
            % Test with 30-minute sampling frequency
            options.subSample = 30;
            result30min = rv_compute(obj.testData, options);
            
            % Check relative magnitudes - different sampling frequencies should 
            % result in similar estimates but with some variation
            numAssets = size(obj.testData, 2);
            for i = 1:numAssets
                % Calculate relative differences between sampling frequencies
                relativeDiff1 = abs(result5min(i) - result10min(i)) / result5min(i);
                relativeDiff2 = abs(result5min(i) - result30min(i)) / result5min(i);
                
                % Differences shouldn't exceed reasonable thresholds
                obj.assertTrue(relativeDiff1 < 0.3, ...
                    sprintf('Excessive difference between 5-min and 10-min sampling for asset %d', i));
                obj.assertTrue(relativeDiff2 < 0.5, ...
                    sprintf('Excessive difference between 5-min and 30-min sampling for asset %d', i));
            end
            
            % Verify correct shape of output
            obj.assertEqual(size(result5min), [1, numAssets], 'Subsample output has incorrect dimensions');
            obj.assertEqual(size(result10min), [1, numAssets], 'Subsample output has incorrect dimensions');
            obj.assertEqual(size(result30min), [1, numAssets], 'Subsample output has incorrect dimensions');
        end
        
        function testInputValidation(obj)
            % Test rv_compute input validation mechanisms
            
            % Test with non-numeric input
            nonNumericInput = {'not', 'a', 'numeric', 'input'};
            errorOccurred = false;
            try
                rv_compute(nonNumericInput);
            catch
                errorOccurred = true;
            end
            obj.assertTrue(errorOccurred, 'Non-numeric input should throw an error');
            
            % Test with NaN values
            nanInput = obj.testData;
            nanInput(1,1) = NaN;
            errorOccurred = false;
            try
                rv_compute(nanInput);
            catch
                errorOccurred = true;
            end
            obj.assertTrue(errorOccurred, 'NaN input should throw an error');
            
            % Test with empty array
            errorOccurred = false;
            try
                rv_compute([]);
            catch
                errorOccurred = true;
            end
            obj.assertTrue(errorOccurred, 'Empty input should throw an error');
            
            % Test with row vector (should be handled correctly)
            rowVector = obj.testData(1,:);
            errorOccurred = false;
            try
                rv_compute(rowVector);
            catch
                errorOccurred = true;
            end
            obj.assertFalse(errorOccurred, 'Row vector input should be handled without errors');
            
            % Test with invalid options
            invalidOptions = struct('method', 'invalid_method');
            errorOccurred = false;
            try
                rv_compute(obj.testData, invalidOptions);
            catch
                errorOccurred = true;
            end
            obj.assertTrue(errorOccurred, 'Invalid method should throw an error');
        end
        
        function testEdgeCases(obj)
            % Test rv_compute with edge cases
            
            % Test with single observation
            singleObs = obj.testData(1,:);
            result = rv_compute(singleObs);
            % For a single observation, RV should equal the squared return
            expected = singleObs.^2;
            obj.assertAlmostEqual(expected, result, 'Single observation test failed');
            
            % Test with very small values
            smallValues = obj.testData * 1e-10;
            resultSmall = rv_compute(smallValues);
            % For small values, RV should still be correctly computed as sum of squares
            expectedSmall = sum(smallValues.^2);
            obj.assertAlmostEqual(expectedSmall, resultSmall, 'Small values test failed');
            
            % Test with very large values
            largeValues = obj.testData * 1e10;
            resultLarge = rv_compute(largeValues);
            % For large values, RV should still be correctly computed as sum of squares
            expectedLarge = sum(largeValues.^2);
            obj.assertAlmostEqual(expectedLarge, resultLarge, 'Large values test failed');
        end
        
        function testConsistencyWithTheory(obj)
            % Test that rv_compute results are consistent with theoretical properties
            
            % Generate data with known volatility
            numObs = 1000;
            trueVol = 0.1;
            rng(42); % Set random seed for reproducibility
            simulatedReturns = sqrt(trueVol) * randn(numObs, 1);
            
            % Compute realized volatility
            result = rv_compute(simulatedReturns);
            
            % In theory, the realized volatility should converge to the true volatility
            % as the number of observations increases. Use reasonable threshold for difference.
            absDiff = abs(trueVol - result);
            obj.assertTrue(absDiff < 0.03, sprintf('Theoretical consistency test failed: |%g - %g| = %g', trueVol, result, absDiff));
            
            % Test asymptotic property: variance of RV estimator decreases with sample size
            numSamples = 30;
            resultSmall = zeros(numSamples, 1);
            resultLarge = zeros(numSamples, 1);
            
            for i = 1:numSamples
                smallSample = sqrt(trueVol) * randn(100, 1);
                largeSample = sqrt(trueVol) * randn(1000, 1);
                
                resultSmall(i) = rv_compute(smallSample);
                resultLarge(i) = rv_compute(largeSample);
            end
            
            varSmall = var(resultSmall);
            varLarge = var(resultLarge);
            
            % Variance of the estimator should be smaller for larger sample
            obj.assertTrue(varLarge < varSmall, sprintf('Asymptotic variance property failed: var(large)=%g, var(small)=%g', varLarge, varSmall));
        end
        
        function testPerformance(obj)
            % Test performance of rv_compute with large datasets
            
            % Generate large test dataset
            numObs = 10000;
            numAssets = 10;
            largeData = randn(numObs, numAssets);
            
            % Measure execution time
            executionTime = obj.measureExecutionTime(@() rv_compute(largeData));
            
            % Verify execution time is within acceptable limits
            % This is a somewhat arbitrary threshold and may need adjustment
            maxAllowedTime = 1.0; % seconds
            obj.assertTrue(executionTime < maxAllowedTime, ...
                sprintf('Performance test failed: execution time %.4f seconds exceeds threshold %.4f seconds', ...
                executionTime, maxAllowedTime));
                
            % Test with different options to ensure good performance
            options = struct('method', 'subsample', 'subSample', 5);
            executionTimeSubsample = obj.measureExecutionTime(@() rv_compute(largeData, options));
            
            obj.assertTrue(executionTimeSubsample < maxAllowedTime * 2, ...
                sprintf('Subsample performance test failed: execution time %.4f seconds exceeds threshold', ...
                executionTimeSubsample));
                
            % Test with jackknife correction
            options = struct('jackknife', true);
            executionTimeJackknife = obj.measureExecutionTime(@() rv_compute(largeData, options));
            
            obj.assertTrue(executionTimeJackknife < maxAllowedTime * 1.5, ...
                sprintf('Jackknife performance test failed: execution time %.4f seconds exceeds threshold', ...
                executionTimeJackknife));
        end
        
        function testMEXImplementation(obj)
            % Test MEX implementation if available
            
            % Skip test if MEX not available
            if ~obj.isMEXAvailable
                warning('Skipping MEX implementation test as it is not available');
                return;
            end
            
            % Run both MATLAB and MEX implementations
            resultMATLAB = rv_compute(obj.testData);
            resultMEX = rv_compute_mex(obj.testData);
            
            % Compare results from both implementations
            obj.assertAlmostEqual(resultMATLAB, resultMEX, 'MATLAB and MEX implementations produce different results');
            
            % Compare performance between implementations
            timeMATLAB = obj.measureExecutionTime(@() rv_compute(obj.testData));
            timeMEX = obj.measureExecutionTime(@() rv_compute_mex(obj.testData));
            
            % MEX should be faster than MATLAB
            obj.assertTrue(timeMEX < timeMATLAB, sprintf('MEX implementation is not faster than MATLAB implementation: MEX: %.4fs, MATLAB: %.4fs', timeMEX, timeMATLAB));
            
            % Calculate speedup
            speedup = timeMATLAB / timeMEX;
            fprintf('MEX speedup: %.2fx\n', speedup);
        end
    end
end