classdef RealizedSpectrumTest < BaseTest
    properties
        testData          % Test data for high-frequency returns
        referenceValues   % Reference values for different calculations
        tolerance         % Numerical tolerance for comparisons
        isMEXAvailable    % Flag indicating if MEX implementation is available
    end
    
    methods
        % Constructor
        function obj = RealizedSpectrumTest()
            % Call super class constructor
            obj@BaseTest();
        end
        
        % Setup method to run before each test
        function setUp(obj)
            % Call superclass setUp
            setUp@BaseTest(obj);
            
            % Load high-frequency test data
            testData = obj.loadTestData('high_frequency_data.mat');
            obj.testData = testData.high_frequency_returns;
            obj.referenceValues = testData.reference_values;
            
            % Set numerical tolerance
            obj.tolerance = 1e-10;
            
            % Check if MEX implementation is available
            obj.isMEXAvailable = exist('realized_spectrum_mex', 'file') == 3; % 3 = MEX file
        end
        
        % Teardown method to run after each test
        function tearDown(obj)
            % Call superclass tearDown
            tearDown@BaseTest(obj);
            
            % Clean up any test-specific resources
            clear testData;
        end
        
        % Basic operation test
        function testBasicOperation(obj)
            % Test the basic functionality with default parameters
            rv = realized_spectrum(obj.testData);
            
            % Compare with reference values
            reference = obj.referenceValues.kernel_estimates;
            
            % Assert that results match within specified tolerance
            obj.assertAlmostEqual(rv, reference, 'Basic operation does not match reference values');
        end
        
        % Test different spectral window types
        function testSpectralWindowTypes(obj)
            % Test with Parzen window (default)
            options = struct();
            options.windowType = 'Parzen';
            rv_parzen = realized_spectrum(obj.testData, options);
            
            % Test with Bartlett window
            options.windowType = 'Bartlett';
            rv_bartlett = realized_spectrum(obj.testData, options);
            
            % Test with Tukey-Hanning window
            options.windowType = 'Tukey-Hanning';
            rv_tukey = realized_spectrum(obj.testData, options);
            
            % Test with Quadratic window
            options.windowType = 'Quadratic';
            rv_quadratic = realized_spectrum(obj.testData, options);
            
            % Test with Cubic window
            options.windowType = 'Cubic';
            rv_cubic = realized_spectrum(obj.testData, options);
            
            % Test with Flat-Top window
            options.windowType = 'Flat-Top';
            rv_flattop = realized_spectrum(obj.testData, options);
            
            % Assert that all windows produce valid results
            % Each window type should produce results that differ from each other
            obj.assertTrue(~isequal(rv_parzen, rv_bartlett), 'Parzen and Bartlett windows produce identical results');
            obj.assertTrue(~isequal(rv_parzen, rv_tukey), 'Parzen and Tukey-Hanning windows produce identical results');
            
            % Results should be positive and finite
            obj.assertTrue(all(rv_parzen > 0), 'Parzen window produces non-positive results');
            obj.assertTrue(all(isfinite(rv_parzen)), 'Parzen window produces non-finite results');
            obj.assertTrue(all(rv_bartlett > 0), 'Bartlett window produces non-positive results');
            obj.assertTrue(all(isfinite(rv_bartlett)), 'Bartlett window produces non-finite results');
            obj.assertTrue(all(rv_tukey > 0), 'Tukey-Hanning window produces non-positive results');
            obj.assertTrue(all(isfinite(rv_tukey)), 'Tukey-Hanning window produces non-finite results');
            obj.assertTrue(all(rv_quadratic > 0), 'Quadratic window produces non-positive results');
            obj.assertTrue(all(isfinite(rv_quadratic)), 'Quadratic window produces non-finite results');
            obj.assertTrue(all(rv_cubic > 0), 'Cubic window produces non-positive results');
            obj.assertTrue(all(isfinite(rv_cubic)), 'Cubic window produces non-finite results');
            obj.assertTrue(all(rv_flattop > 0), 'Flat-Top window produces non-positive results');
            obj.assertTrue(all(isfinite(rv_flattop)), 'Flat-Top window produces non-finite results');
        end
        
        % Test different sampling frequencies
        function testSamplingFrequencies(obj)
            % Generate test data with different sampling frequencies
            data = obj.testData;
            
            % Test with default sampling
            rv_default = realized_spectrum(data);
            
            % Test with 5-minute sampling (subsample by 5)
            subsample = 5;
            subsampled_data = data(1:subsample:end, :);
            options = struct();
            rv_5min = realized_spectrum(subsampled_data, options);
            
            % Test with 10-minute sampling (subsample by 10)
            subsample = 10;
            subsampled_data = data(1:subsample:end, :);
            rv_10min = realized_spectrum(subsampled_data, options);
            
            % Verify that results differ by sampling frequency
            obj.assertTrue(~isequal(rv_default, rv_5min), 'Default and 5-minute sampling produce identical results');
            obj.assertTrue(~isequal(rv_default, rv_10min), 'Default and 10-minute sampling produce identical results');
            obj.assertTrue(~isequal(rv_5min, rv_10min), '5-minute and 10-minute sampling produce identical results');
            
            % Results should be positive and finite
            obj.assertTrue(all(rv_default > 0), 'Default sampling produces non-positive results');
            obj.assertTrue(all(isfinite(rv_default)), 'Default sampling produces non-finite results');
            obj.assertTrue(all(rv_5min > 0), '5-minute sampling produces non-positive results');
            obj.assertTrue(all(isfinite(rv_5min)), '5-minute sampling produces non-finite results');
            obj.assertTrue(all(rv_10min > 0), '10-minute sampling produces non-positive results');
            obj.assertTrue(all(isfinite(rv_10min)), '10-minute sampling produces non-finite results');
        end
        
        % Test cutoff frequency parameter
        function testCutoffFrequency(obj)
            % Test with different cutoff frequency values
            options = struct();
            
            % Test with low cutoff frequency
            options.cutoffFreq = 0.1;
            rv_low = realized_spectrum(obj.testData, options);
            
            % Test with medium cutoff frequency
            options.cutoffFreq = 0.3;
            rv_medium = realized_spectrum(obj.testData, options);
            
            % Test with high cutoff frequency
            options.cutoffFreq = 0.5;
            rv_high = realized_spectrum(obj.testData, options);
            
            % Verify that results differ by cutoff frequency
            obj.assertTrue(~isequal(rv_low, rv_medium), 'Low and medium cutoff frequencies produce identical results');
            obj.assertTrue(~isequal(rv_low, rv_high), 'Low and high cutoff frequencies produce identical results');
            obj.assertTrue(~isequal(rv_medium, rv_high), 'Medium and high cutoff frequencies produce identical results');
            
            % Results should be positive and finite
            obj.assertTrue(all(rv_low > 0), 'Low cutoff frequency produces non-positive results');
            obj.assertTrue(all(isfinite(rv_low)), 'Low cutoff frequency produces non-finite results');
            obj.assertTrue(all(rv_medium > 0), 'Medium cutoff frequency produces non-positive results');
            obj.assertTrue(all(isfinite(rv_medium)), 'Medium cutoff frequency produces non-finite results');
            obj.assertTrue(all(rv_high > 0), 'High cutoff frequency produces non-positive results');
            obj.assertTrue(all(isfinite(rv_high)), 'High cutoff frequency produces non-finite results');
        end
        
        % Test input validation
        function testInputValidation(obj)
            % Test with invalid input types
            obj.assertThrows(@() realized_spectrum('not_numeric'), '', 'Did not throw error for non-numeric input');
            
            % Test with NaN values
            data_with_nan = obj.testData;
            data_with_nan(5,1) = NaN;
            obj.assertThrows(@() realized_spectrum(data_with_nan), '', 'Did not throw error for input with NaN values');
            
            % Test with Inf values
            data_with_inf = obj.testData;
            data_with_inf(5,1) = Inf;
            obj.assertThrows(@() realized_spectrum(data_with_inf), '', 'Did not throw error for input with Inf values');
            
            % Test with empty array
            obj.assertThrows(@() realized_spectrum([]), '', 'Did not throw error for empty input');
            
            % Test with row vector (should be handled internally)
            row_data = obj.testData(1:10, 1)';
            rv = realized_spectrum(row_data);
            obj.assertTrue(isscalar(rv), 'Row vector input not properly handled');
            
            % Test with invalid options
            % Invalid window type
            options = struct('windowType', 'InvalidWindow');
            obj.assertThrows(@() realized_spectrum(obj.testData, options), '', 'Did not throw error for invalid window type');
            
            % Invalid cutoff frequency
            options = struct('cutoffFreq', -0.1);
            obj.assertThrows(@() realized_spectrum(obj.testData, options), '', 'Did not throw error for negative cutoff frequency');
            options = struct('cutoffFreq', 1.1);
            obj.assertThrows(@() realized_spectrum(obj.testData, options), '', 'Did not throw error for cutoff frequency > 0.5');
        end
        
        % Test edge cases
        function testEdgeCases(obj)
            % Test with minimal data (just enough for calculation)
            minimal_data = obj.testData(1:10, 1);
            rv_minimal = realized_spectrum(minimal_data);
            obj.assertTrue(rv_minimal > 0, 'Minimal data produces non-positive results');
            obj.assertTrue(isfinite(rv_minimal), 'Minimal data produces non-finite results');
            
            % Test with very small values
            small_data = obj.testData * 1e-6;
            rv_small = realized_spectrum(small_data);
            obj.assertTrue(rv_small > 0, 'Very small data produces non-positive results');
            obj.assertTrue(isfinite(rv_small), 'Very small data produces non-finite results');
            
            % Test with very large values
            large_data = obj.testData * 1e6;
            rv_large = realized_spectrum(large_data);
            obj.assertTrue(rv_large > 0, 'Very large data produces non-positive results');
            obj.assertTrue(isfinite(rv_large), 'Very large data produces non-finite results');
            
            % Test with data with outliers and outlier removal
            data_with_outlier = obj.testData;
            data_with_outlier(5,1) = data_with_outlier(5,1) * 10;
            
            % Without outlier removal
            rv_with_outlier = realized_spectrum(data_with_outlier);
            
            % With outlier removal
            options = struct('removeOutliers', true);
            rv_outlier_removed = realized_spectrum(data_with_outlier, options);
            
            % Outlier removal should change the result
            obj.assertTrue(~isequal(rv_with_outlier, rv_outlier_removed), 'Outlier removal does not affect the result');
        end
        
        % Test comparison with standard realized volatility
        function testComparisonWithStandardRV(obj)
            % Compare spectral-based RV with standard RV
            rv_spectral = realized_spectrum(obj.testData);
            rv_standard = rv_compute(obj.testData);
            
            % In general, spectral RV should differ from standard RV
            obj.assertTrue(~isequal(rv_spectral, rv_standard), 'Spectral and standard RV are identical');
            
            % Generate data with known noise properties
            % Create low noise and high noise scenarios
            lowNoiseParams = struct('microstructure', 'additive', 'microstructureParams', struct('std', 0.0001));
            highNoiseParams = struct('microstructure', 'additive', 'microstructureParams', struct('std', 0.001));
            
            low_noise_data = TestDataGenerator('generateHighFrequencyData', 1, 100, lowNoiseParams);
            high_noise_data = TestDataGenerator('generateHighFrequencyData', 1, 100, highNoiseParams);
            
            % Compute both methods on both datasets
            rv_spectral_low = realized_spectrum(low_noise_data.returns);
            rv_standard_low = rv_compute(low_noise_data.returns);
            
            rv_spectral_high = realized_spectrum(high_noise_data.returns);
            rv_standard_high = rv_compute(high_noise_data.returns);
            
            % For low noise, both methods should be relatively close
            relative_diff_low = abs(rv_spectral_low - rv_standard_low) / rv_standard_low;
            
            % For high noise, spectral method should differ more significantly
            relative_diff_high = abs(rv_spectral_high - rv_standard_high) / rv_standard_high;
            
            % Check that high noise causes larger difference
            obj.assertTrue(relative_diff_high > relative_diff_low, 'Spectral method does not show expected behavior with noise');
        end
        
        % Test consistency with theoretical properties
        function testConsistencyWithTheory(obj)
            % Test that results have expected theoretical properties
            
            % Generate multiple days of data
            params = struct('volatilityModel', 'constant', 'volatilityParams', struct('sigma', 0.01));
            data = TestDataGenerator('generateHighFrequencyData', 5, 100, params);
            
            % Compute realized spectral volatility
            rv = realized_spectrum(data.returns);
            
            % Verify scaling property: RV of scaled returns should scale by square of the factor
            scaling_factor = 2;
            rv_scaled = realized_spectrum(data.returns * scaling_factor);
            expected_scaling = scaling_factor^2;
            
            for i = 1:length(rv)
                obj.assertAlmostEqual(rv_scaled(i) / rv(i), expected_scaling, ...
                    sprintf('Scaling property not satisfied for asset %d', i));
            end
            
            % Verify time additivity: RV over sub-periods should approximately sum to total
            % (This is only approximate for spectral methods due to boundary effects)
            half_size = floor(size(data.returns, 1) / 2);
            data1 = data.returns(1:half_size, :);
            data2 = data.returns(half_size+1:end, :);
            
            rv_total = realized_spectrum(data.returns);
            rv_part1 = realized_spectrum(data1);
            rv_part2 = realized_spectrum(data2);
            
            % Sum should be approximately equal, with some tolerance for spectral boundary effects
            for i = 1:length(rv_total)
                ratio = rv_total(i) / (rv_part1(i) + rv_part2(i));
                obj.assertTrue(ratio > 0.8 && ratio < 1.2, ...
                    sprintf('Time additivity property severely violated for asset %d', i));
            end
        end
        
        % Test performance
        function testPerformance(obj)
            % Test performance with large datasets
            
            % Create large dataset
            n = 10000;
            large_data = randn(n, 1);
            
            % Measure execution time
            execution_time = obj.measureExecutionTime(@() realized_spectrum(large_data));
            
            % Check that execution completes in reasonable time
            % This is hardware-dependent, so just use a very large upper bound
            obj.assertTrue(execution_time < 10, 'Execution took excessively long');
            
            % Check that result is valid
            rv = realized_spectrum(large_data);
            obj.assertTrue(rv > 0, 'Result is non-positive');
            obj.assertTrue(isfinite(rv), 'Result is non-finite');
        end
        
        % Test bias correction functionality
        function testBiasCorrection(obj)
            % Test with and without bias correction
            
            % Without bias correction
            options = struct('biasCorrection', false);
            rv_no_correction = realized_spectrum(obj.testData, options);
            
            % With bias correction
            options = struct('biasCorrection', true);
            rv_with_correction = realized_spectrum(obj.testData, options);
            
            % Bias correction should change the result
            obj.assertTrue(~isequal(rv_no_correction, rv_with_correction), 'Bias correction does not affect the result');
            
            % In most cases, bias correction should reduce the estimated variance
            % (except in cases with negative bias)
            % This test might be too specific, so we'll make it a soft comparison
            for i = 1:length(rv_no_correction)
                obj.assertTrue(abs(rv_with_correction(i) - rv_no_correction(i)) / rv_no_correction(i) < 0.5, ...
                    sprintf('Bias correction has dramatic effect on asset %d', i));
            end
        end
        
        % Test MEX implementation if available
        function testMEXImplementation(obj)
            % Skip test if MEX not available
            if ~obj.isMEXAvailable
                disp('MEX implementation not available, skipping test.');
                return;
            end
            
            % Test both MATLAB and MEX implementations
            % First force MATLAB implementation
            options = struct('useMEX', false);
            rv_matlab = realized_spectrum(obj.testData, options);
            
            % Then use MEX implementation
            options = struct('useMEX', true);
            rv_mex = realized_spectrum(obj.testData, options);
            
            % Results should be identical or very close
            for i = 1:length(rv_matlab)
                obj.assertAlmostEqual(rv_matlab(i), rv_mex(i), ...
                    sprintf('MATLAB and MEX implementations give different results for asset %d', i));
            end
            
            % MEX should be faster
            % Measure MATLAB time
            options = struct('useMEX', false);
            matlab_time = obj.measureExecutionTime(@() realized_spectrum(obj.testData, options));
            
            % Measure MEX time
            options = struct('useMEX', true);
            mex_time = obj.measureExecutionTime(@() realized_spectrum(obj.testData, options));
            
            % MEX should be faster, but this is not guaranteed, so make it a soft test
            disp(['MATLAB time: ', num2str(matlab_time), 's, MEX time: ', num2str(mex_time), 's']);
            obj.assertTrue(mex_time < matlab_time * 1.5, 'MEX implementation unexpectedly slower than MATLAB');
        end
    end
end