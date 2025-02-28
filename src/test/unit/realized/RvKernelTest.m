classdef RvKernelTest < BaseTest
    % RvKernelTest Test class for the rv_kernel function that implements kernel-based realized volatility estimation
    
    properties
        testData        % Structure to store test data
        returns         % Matrix of high-frequency returns for testing
        referenceValues % Structure with reference values for validation
    end
    
    methods
        function obj = RvKernelTest()
            % Initializes a new RvKernelTest instance
            obj@BaseTest();  % Call the parent class constructor
            
            % Initialize class properties to empty
            obj.testData = struct();
            obj.returns = [];
            obj.referenceValues = struct();
        end
        
        function setUp(obj)
            % Set up test environment before each test method execution
            
            % Call the parent class setUp method
            setUp@BaseTest(obj);
            
            % Load high frequency data using loadTestData method
            obj.testData = obj.loadTestData('rv_kernel_test_data.mat');
            
            % Extract high frequency returns from test data
            obj.returns = obj.testData.returns;
            
            % Extract reference values for kernel estimators from test data
            obj.referenceValues = obj.testData.referenceValues;
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            
            % Call the parent class tearDown method
            tearDown@BaseTest(obj);
            
            % Clear temporary variables and test data
        end
        
        function testDefaultParameters(obj)
            % Tests rv_kernel function with default parameters
            
            % Call rv_kernel with only returns data, using all default parameters
            result = rv_kernel(obj.returns);
            
            % Verify that the result is a positive scalar
            obj.assertTrue(isscalar(result) && result > 0, 'Default rv_kernel should return a positive scalar value');
            
            % Compare the result with the reference value for default parameters
            expectedValue = obj.referenceValues.default;
            obj.assertAlmostEqual(result, expectedValue, 'Default parameter result does not match expected value');
        end
        
        function testBandwidthParameter(obj)
            % Tests rv_kernel with different bandwidth parameter values
            
            % Test with specific bandwidth values
            bandwidthValues = [5, 10, 20];
            
            for i = 1:length(bandwidthValues)
                bandwidth = bandwidthValues(i);
                
                % Create options structure with bandwidth parameter
                options = struct('bandwidth', bandwidth);
                
                % Call rv_kernel with the bandwidth option
                result = rv_kernel(obj.returns, options);
                
                % Verify result is a positive scalar
                obj.assertTrue(isscalar(result) && result > 0, ...
                    sprintf('rv_kernel with bandwidth=%d should return a positive scalar', bandwidth));
                
                % Compare the result with the corresponding reference value
                expectedValue = obj.referenceValues.bandwidth.(sprintf('b%d', bandwidth));
                obj.assertAlmostEqual(result, expectedValue, ...
                    sprintf('Bandwidth=%d result does not match expected value', bandwidth));
            end
            
            % Test that increasing bandwidth affects smoothing effect
            options1 = struct('bandwidth', 5);
            options2 = struct('bandwidth', 20);
            
            result1 = rv_kernel(obj.returns, options1);
            result2 = rv_kernel(obj.returns, options2);
            
            % Different bandwidths should produce measurably different results
            obj.assertTrue(abs(result1 - result2) > 1e-6, ...
                'Different bandwidths should produce measurably different results');
        end
        
        function testKernelTypes(obj)
            % Tests rv_kernel with different kernel type options
            
            % List of available kernel types to test
            kernelTypes = {'Bartlett-Parzen', 'Quadratic', 'Cubic', 'Exponential', 'Tukey-Hanning'};
            
            for i = 1:length(kernelTypes)
                kernelType = kernelTypes{i};
                
                % Create options structure with kernel type
                options = struct('kernelType', kernelType, 'bandwidth', 10);
                
                % Call rv_kernel with the kernel type option
                result = rv_kernel(obj.returns, options);
                
                % Verify result is a positive scalar
                obj.assertTrue(isscalar(result) && result > 0, ...
                    sprintf('rv_kernel with kernel=%s should return a positive scalar', kernelType));
                
                % Compare the result with the corresponding reference value
                % Creating a field name from kernel type by removing non-alphanumeric chars
                fieldName = lower(strrep(strrep(kernelType, '-', ''), ' ', ''));
                expectedValue = obj.referenceValues.kernelType.(fieldName);
                
                obj.assertAlmostEqual(result, expectedValue, ...
                    sprintf('Kernel=%s result does not match expected value', kernelType));
            end
            
            % Verify that different kernel types produce different results
            options1 = struct('kernelType', 'Bartlett-Parzen', 'bandwidth', 10);
            options2 = struct('kernelType', 'Tukey-Hanning', 'bandwidth', 10);
            
            result1 = rv_kernel(obj.returns, options1);
            result2 = rv_kernel(obj.returns, options2);
            
            obj.assertTrue(abs(result1 - result2) > 1e-6, ...
                'Different kernel types should produce measurably different results');
        end
        
        function testAsymptoticCorrection(obj)
            % Tests rv_kernel with asymptotic correction option
            
            % Create options for testing with and without correction
            optionsWithCorrection = struct('autoCorrection', true, 'bandwidth', 10);
            optionsWithoutCorrection = struct('autoCorrection', false, 'bandwidth', 10);
            
            % Calculate realized volatility with and without asymptotic correction
            resultWithCorrection = rv_kernel(obj.returns, optionsWithCorrection);
            resultWithoutCorrection = rv_kernel(obj.returns, optionsWithoutCorrection);
            
            % Verify both results are positive scalars
            obj.assertTrue(isscalar(resultWithCorrection) && resultWithCorrection > 0, ...
                'rv_kernel with autoCorrection=true should return a positive scalar');
            obj.assertTrue(isscalar(resultWithoutCorrection) && resultWithoutCorrection > 0, ...
                'rv_kernel with autoCorrection=false should return a positive scalar');
            
            % Compare with reference values
            obj.assertAlmostEqual(resultWithCorrection, obj.referenceValues.correction.withCorrection, ...
                'Result with correction does not match expected value');
            obj.assertAlmostEqual(resultWithoutCorrection, obj.referenceValues.correction.withoutCorrection, ...
                'Result without correction does not match expected value');
            
            % Ensure the correction makes a difference in the results
            obj.assertTrue(abs(resultWithCorrection - resultWithoutCorrection) > 1e-6, ...
                'Asymptotic correction should affect the results');
        end
        
        function testMultipleAssets(obj)
            % Tests rv_kernel with multi-asset return data
            
            % Create multi-asset return data by duplicating returns and adding noise
            numAssets = 3;
            multiReturns = repmat(obj.returns, 1, numAssets);
            
            % Add different small noise to each asset column to make them distinct
            for i = 1:numAssets
                multiReturns(:, i) = multiReturns(:, i) + 0.0001 * i * randn(size(obj.returns));
            end
            
            % Set fixed random seed for reproducibility
            rng(42);
            
            % Process multi-asset data with rv_kernel
            options = struct('bandwidth', 10);
            result = rv_kernel(multiReturns, options);
            
            % Verify result is a row vector with numAssets elements
            obj.assertEqual(size(result), [1, numAssets], ...
                'Multi-asset result should be a row vector with one estimate per asset');
            
            % Verify all results are positive
            obj.assertTrue(all(result > 0), 'All multi-asset results should be positive');
            
            % Compare with reference values
            obj.assertAlmostEqual(result, obj.referenceValues.multiAsset, ...
                'Multi-asset results do not match expected values');
            
            % Reset random seed
            rng('default');
        end
        
        function testInvalidInputs(obj)
            % Tests error handling for invalid inputs to rv_kernel
            
            % Test with empty returns
            obj.assertThrows(@() rv_kernel([]), 'MATLAB:InputSizeMismatch', ...
                'Empty returns should throw an error');
            
            % Test with NaN values in returns
            nanReturns = obj.returns;
            nanReturns(5) = NaN;
            obj.assertThrows(@() rv_kernel(nanReturns), 'MATLAB:InputSizeMismatch', ...
                'Returns with NaN values should throw an error');
            
            % Test with invalid bandwidth parameter
            options = struct('bandwidth', -1);
            obj.assertThrows(@() rv_kernel(obj.returns, options), 'MATLAB:InputSizeMismatch', ...
                'Negative bandwidth should throw an error');
            
            options = struct('bandwidth', 1.5);
            obj.assertThrows(@() rv_kernel(obj.returns, options), 'MATLAB:InputSizeMismatch', ...
                'Non-integer bandwidth should throw an error');
            
            % Test with invalid kernel type
            options = struct('kernelType', 'InvalidKernel');
            obj.assertThrows(@() rv_kernel(obj.returns, options), 'MATLAB:InputSizeMismatch', ...
                'Invalid kernel type should throw an error');
            
            % Test with invalid options structure
            obj.assertThrows(@() rv_kernel(obj.returns, 'not_a_struct'), 'MATLAB:InputSizeMismatch', ...
                'Non-struct options should throw an error');
        end
        
        function testPerformance(obj)
            % Tests performance characteristics of rv_kernel
            
            % Create a large sample for performance testing
            largeSample = repmat(obj.returns, 10, 1);
            
            % Measure execution time
            options = struct('bandwidth', 20);
            executionTime = obj.measureExecutionTime(@() rv_kernel(largeSample, options));
            
            % Check that execution time is reasonable
            obj.assertTrue(executionTime < 5.0, ...
                sprintf('Execution time (%.2f sec) exceeds performance threshold', executionTime));
            
            % Check memory usage
            memoryInfo = obj.checkMemoryUsage(@() rv_kernel(largeSample, options));
            
            % Display memory usage information if in verbose mode
            if obj.verbose
                fprintf('Memory usage: %.2f MB\n', memoryInfo.memoryDifferenceMB);
            end
            
            % Verify that memory usage doesn't grow excessively
            obj.assertTrue(memoryInfo.memoryDifferenceMB < 100, ...
                'Memory usage exceeds performance threshold');
            
            % Test execution time scaling with data size
            smallSample = obj.returns(1:100);
            mediumSample = obj.returns;
            
            smallTime = obj.measureExecutionTime(@() rv_kernel(smallSample, options));
            mediumTime = obj.measureExecutionTime(@() rv_kernel(mediumSample, options));
            largeTime = executionTime;
            
            % Verify that execution time scales reasonably with data size
            ratioMediumToSmall = mediumTime / smallTime;
            ratioLargeToMedium = largeTime / mediumTime;
            
            obj.assertTrue(ratioMediumToSmall > 1 && ratioLargeToMedium > 1, ...
                'Execution time should scale positively with data size');
        end
        
        function testNumericalStability(obj)
            % Tests numerical stability of rv_kernel with extreme inputs
            
            % Test with very small returns
            smallReturns = obj.returns * 1e-5;
            resultSmall = rv_kernel(smallReturns);
            obj.assertTrue(isfinite(resultSmall) && resultSmall >= 0, ...
                'rv_kernel should handle very small returns stably');
            
            % Test with very large returns
            largeReturns = obj.returns * 1e5;
            resultLarge = rv_kernel(largeReturns);
            obj.assertTrue(isfinite(resultLarge) && resultLarge >= 0, ...
                'rv_kernel should handle very large returns stably');
            
            % Test with returns containing extreme outliers
            outliersReturns = obj.returns;
            outliersReturns(10) = outliersReturns(10) * 100;  % Extreme positive outlier
            outliersReturns(20) = outliersReturns(20) * -100; % Extreme negative outlier
            
            % Test with outlier removal option
            options = struct('removeOutliers', true);
            resultOutliers = rv_kernel(outliersReturns, options);
            obj.assertTrue(isfinite(resultOutliers) && resultOutliers >= 0, ...
                'rv_kernel should handle extreme outliers stably when removeOutliers=true');
            
            % Test with returns having high autocorrelation
            autoCorr = obj.returns;
            windowSize = 5;
            for i = windowSize+1:length(autoCorr)
                autoCorr(i) = mean(obj.returns(i-windowSize:i));
            end
            resultAutoCorr = rv_kernel(autoCorr);
            obj.assertTrue(isfinite(resultAutoCorr) && resultAutoCorr >= 0, ...
                'rv_kernel should handle returns with high autocorrelation stably');
            
            % Verify scaling properties (volatility should scale with square of returns)
            scale = 2;
            result1 = rv_kernel(obj.returns);
            result2 = rv_kernel(obj.returns * scale);
            
            % Volatility (variance) should scale with square of returns
            expectedRatio = scale^2;
            actualRatio = result2 / result1;
            
            obj.assertAlmostEqual(actualRatio, expectedRatio, ...
                'Volatility should scale with square of returns');
        end
    end
end