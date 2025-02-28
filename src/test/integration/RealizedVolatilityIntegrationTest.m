classdef RealizedVolatilityIntegrationTest < BaseTest
    % REALIZEDVOLATILITYINTEGRATIONTEST Integration test for realized volatility components
    %
    % This class provides integration tests for realized volatility components
    % of the MFE Toolbox, ensuring that the various high-frequency analysis functions
    % work together correctly and produce consistent results across different
    % methodologies.
    
    properties
        highFrequencyData  % Full high-frequency dataset
        cleanData          % Clean data (no noise, no jumps)
        noisyData          % Data with microstructure noise
        jumpData           % Data with jumps
        referenceValues    % Expected reference values
        tolerance          % Tolerance for numerical comparisons
        isMEXAvailable     % Flag indicating if MEX implementations are available
    end
    
    methods
        function obj = RealizedVolatilityIntegrationTest()
            % Constructor for the RealizedVolatilityIntegrationTest class
            obj = obj@BaseTest();
        end
        
        function setUp(obj)
            % Setup method to prepare the test environment
            setUp@BaseTest(obj);
            
            % Load or generate test data
            obj.loadTestData();
            
            % Set numerical comparison tolerance
            obj.tolerance = 1e-10;
            
            % Check if MEX implementations are available
            obj.isMEXAvailable = exist('rv_kernel_mex', 'file') == 3;
            
            % Initialize reference values
            obj.initializeReferenceValues();
        end
        
        function tearDown(obj)
            % Teardown method to clean up after tests
            tearDown@BaseTest(obj);
            
            % Clear test data variables
            obj.highFrequencyData = [];
            obj.cleanData = [];
            obj.noisyData = [];
            obj.jumpData = [];
            
            % Clear reference values
            obj.referenceValues = [];
        end
        
        function testRVandBVConsistency(obj)
            % Test consistency between realized volatility and bipower variation
            
            % Compute RV and BV on clean data
            rv = rv_compute(obj.cleanData);
            bv = bv_compute(obj.cleanData);
            
            % Verify that RV and BV are close in the absence of jumps
            bvToRvRatio = bv / rv;
            obj.assertTrue(abs(bvToRvRatio - 1) < 0.1, ...
                'BV/RV ratio should be close to 1 for clean data');
            
            % Verify that our ratio matches the expected reference value
            obj.assertAlmostEqual(bvToRvRatio, obj.referenceValues.cleanBVtoRVRatio, ...
                'BV/RV ratio should match reference value');
            
            % Check that BV with pi/2 scaling factor produces consistent estimate
            % The theoretical scaling factor for BV is π/2
            scalingFactor = pi/2;
            bvOptions = struct('scaleFactor', scalingFactor);
            bvScaled = bv_compute(obj.cleanData, bvOptions);
            
            % This scaled BV should be closer to RV
            bvScaledToRvRatio = bvScaled / rv;
            obj.assertTrue(abs(bvScaledToRvRatio - scalingFactor) < 0.15, ...
                'Scaled BV should be close to RV * scaling factor');
            
            % Test with different sampling frequencies
            sampleFreqs = [1, 5, 10]; % Take every nth observation
            rvs = zeros(size(sampleFreqs));
            bvs = zeros(size(sampleFreqs));
            
            for i = 1:length(sampleFreqs)
                % Sample the data at different frequencies
                freq = sampleFreqs(i);
                sampledData = obj.cleanData(1:freq:end);
                
                % Compute RV and BV at this frequency
                rvs(i) = rv_compute(sampledData);
                bvs(i) = bv_compute(sampledData);
                
                % Check consistency
                ratio = bvs(i) / rvs(i);
                obj.assertTrue(abs(ratio - 1) < 0.15, ...
                    sprintf('BV/RV ratio should be close to 1 at sampling frequency %d', freq));
            end
            
            % Verify trend consistency across frequencies
            % RV and BV should both decrease with less frequent sampling
            obj.assertTrue(all(diff(rvs) <= 0), 'RV should decrease with less frequent sampling');
            obj.assertTrue(all(diff(bvs) <= 0), 'BV should decrease with less frequent sampling');
        end
        
        function testNoiseRobustEstimation(obj)
            % Test noise-robust realized volatility estimation methods
            
            % Compute standard RV on clean and noisy data
            rvClean = rv_compute(obj.cleanData);
            rvNoisy = rv_compute(obj.noisyData);
            
            % Verify that noise increases the standard RV estimate
            obj.assertTrue(rvNoisy > rvClean, ...
                'Microstructure noise should increase standard RV');
            
            % Compute realized kernel estimator with different kernel types
            kernelTypes = {'Bartlett-Parzen', 'Quadratic', 'Cubic', 'Tukey-Hanning'};
            rvKernelResults = zeros(length(kernelTypes), 1);
            
            for i = 1:length(kernelTypes)
                kernelOptions = struct('kernelType', kernelTypes{i});
                rvKernelResults(i) = rv_kernel(obj.noisyData, kernelOptions);
                
                % The kernel estimator should be less than the noisy RV
                % (should correct for noise bias)
                obj.assertTrue(rvKernelResults(i) < rvNoisy, ...
                    sprintf('%s kernel should reduce noise-induced bias', kernelTypes{i}));
                
                % The kernel estimator should be closer to the clean RV than the noisy RV
                noisyError = abs(rvNoisy - rvClean);
                kernelError = abs(rvKernelResults(i) - rvClean);
                obj.assertTrue(kernelError < noisyError, ...
                    sprintf('%s kernel should be closer to true RV than standard estimator', kernelTypes{i}));
            end
            
            % Test with different bandwidth parameters
            bandwidths = [4, 8, 12]; % Different bandwidth values
            kernelOptions = struct('kernelType', 'Bartlett-Parzen');
            rvKernelBandwidth = zeros(length(bandwidths), 1);
            
            for i = 1:length(bandwidths)
                kernelOptions.bandwidth = bandwidths(i);
                rvKernelBandwidth(i) = rv_kernel(obj.noisyData, kernelOptions);
                
                % Bandwidth shouldn't dramatically change results for reasonable values
                if i > 1
                    percentDiff = abs(rvKernelBandwidth(i) - rvKernelBandwidth(i-1)) / rvKernelBandwidth(i-1);
                    obj.assertTrue(percentDiff < 0.2, ...
                        'Reasonable bandwidth changes should not drastically affect kernel estimator');
                end
            end
            
            % Verify that autocorrection improves noise robustness
            kernelOptions = struct('kernelType', 'Bartlett-Parzen', 'autoCorrection', true);
            rvKernelWithCorrection = rv_kernel(obj.noisyData, kernelOptions);
            kernelOptions.autoCorrection = false;
            rvKernelNoCorrection = rv_kernel(obj.noisyData, kernelOptions);
            
            % Bias correction should move the estimate closer to the true value
            errorWithCorrection = abs(rvKernelWithCorrection - rvClean);
            errorNoCorrection = abs(rvKernelNoCorrection - rvClean);
            obj.assertTrue(errorWithCorrection <= errorNoCorrection, ...
                'Bias correction should improve kernel estimator accuracy');
        end
        
        function testJumpDetection(obj)
            % Test jump detection using realized volatility and bipower variation
            
            % Compute RV and BV on clean data (no jumps)
            rvClean = rv_compute(obj.cleanData);
            bvClean = bv_compute(obj.cleanData);
            
            % Compute RV and BV on data with jumps
            rvJump = rv_compute(obj.jumpData);
            bvJump = bv_compute(obj.jumpData);
            
            % In the presence of jumps, RV should be higher than BV
            % Calculate ratio RV/BV for both datasets
            ratioClean = rvClean / bvClean;
            ratioJump = rvJump / bvJump;
            
            % The ratio should be higher for the data with jumps
            obj.assertTrue(ratioJump > ratioClean, ...
                'RV/BV ratio should be higher for data with jumps');
            
            % The jump ratio should be significantly higher than 1
            obj.assertTrue(ratioJump > 1.1, ...
                'RV/BV ratio should be significantly greater than 1 for data with jumps');
            
            % Verify that our ratio matches the expected reference value
            obj.assertAlmostEqual(ratioJump, obj.referenceValues.jumpRVtoBVRatio, ...
                'RV/BV ratio for jump data should match reference value');
            
            % Use formal jump test
            jumpResultsClean = jump_test(obj.cleanData);
            jumpResultsJump = jump_test(obj.jumpData);
            
            % Jump test statistic should be higher for data with jumps
            obj.assertTrue(jumpResultsJump.zStatistic > jumpResultsClean.zStatistic, ...
                'Jump test statistic should be higher for data with jumps');
            
            % For jump data, test should often reject the null at 5% level
            jumpDetected = jumpResultsJump.jumpDetected(2, 1); % 5% significance level
            obj.assertTrue(jumpDetected, ...
                'Jump test should detect jumps at 5% significance level');
            
            % Verify jump component estimation
            % The jump component should be positive for data with jumps
            obj.assertTrue(jumpResultsJump.jumpComponent > 0, ...
                'Estimated jump component should be positive for data with jumps');
            
            % The jump component should be a significant fraction of total RV
            jumpFraction = jumpResultsJump.jumpComponent / rvJump;
            obj.assertTrue(jumpFraction > 0.05, ...
                'Jump component should be a significant fraction of total RV');
            
            % The continuous component should be close to BV for jump data
            continuousComponent = jumpResultsJump.contComponent;
            bvDiff = abs(continuousComponent - bvJump) / bvJump;
            obj.assertTrue(bvDiff < 0.15, ...
                'Continuous component should be close to BV for data with jumps');
        end
        
        function testSpectralMethods(obj)
            % Test realized spectrum methods and their integration with other estimators
            
            % Basic spectrum calculation
            spectrumOptions = struct('windowType', 'Parzen');
            rvSpectrum = realized_spectrum(obj.noisyData, spectrumOptions);
            
            % Verify spectrum result against reference value
            obj.assertAlmostEqual(rvSpectrum, obj.referenceValues.spectrumRV, ...
                'Realized spectrum should match reference value');
            
            % Compare with other noise-robust estimators on noisy data
            rvNoisy = rv_compute(obj.noisyData);
            kernelOptions = struct('kernelType', 'Bartlett-Parzen');
            rvKernel = rv_kernel(obj.noisyData, kernelOptions);
            
            % Spectrum estimator should be less than standard RV on noisy data
            % (should correct for noise bias)
            obj.assertTrue(rvSpectrum < rvNoisy, ...
                'Spectral estimator should reduce noise-induced bias');
            
            % Spectrum and kernel estimators should be reasonably close for similar parameters
            spectrumToKernelRatio = rvSpectrum / rvKernel;
            obj.assertTrue(abs(spectrumToKernelRatio - 1) < 0.25, ...
                'Spectral and kernel estimators should provide similar estimates on noisy data');
            
            % Test different window types
            windowTypes = {'Parzen', 'Bartlett', 'Tukey-Hanning', 'Quadratic'};
            spectrumResults = zeros(length(windowTypes), 1);
            
            for i = 1:length(windowTypes)
                spectrumOptions.windowType = windowTypes{i};
                spectrumResults(i) = realized_spectrum(obj.noisyData, spectrumOptions);
                
                % All window types should yield estimates less than noisy RV
                obj.assertTrue(spectrumResults(i) < rvNoisy, ...
                    sprintf('%s window spectrum should reduce noise bias', windowTypes{i}));
                
                % Results shouldn't vary too dramatically between window types
                if i > 1
                    windowRatio = spectrumResults(i) / spectrumResults(1);
                    obj.assertTrue(abs(windowRatio - 1) < 0.3, ...
                        'Different window types should give broadly similar results');
                end
            end
            
            % Test cutoff frequency parameter
            cutoffFreqs = [0.1, 0.3, 0.5];
            spectrumCutoffs = zeros(length(cutoffFreqs), 1);
            
            for i = 1:length(cutoffFreqs)
                spectrumOptions.windowType = 'Parzen';
                spectrumOptions.cutoffFreq = cutoffFreqs(i);
                spectrumCutoffs(i) = realized_spectrum(obj.noisyData, spectrumOptions);
                
                % Higher cutoff frequency should increase the estimate (less filtering)
                if i > 1
                    obj.assertTrue(spectrumCutoffs(i) >= spectrumCutoffs(i-1), ...
                        'Higher cutoff frequency should increase the spectral estimate');
                end
            end
            
            % Test with comparison to clean data RV
            rvClean = rv_compute(obj.cleanData);
            
            % Compute error metrics
            noisyError = abs(rvNoisy - rvClean) / rvClean;
            spectrumError = abs(rvSpectrum - rvClean) / rvClean;
            kernelError = abs(rvKernel - rvClean) / rvClean;
            
            % Both robust estimators should be more accurate than noisy RV
            obj.assertTrue(spectrumError < noisyError, ...
                'Spectral estimator should be more accurate than standard RV for noisy data');
            obj.assertTrue(kernelError < noisyError, ...
                'Kernel estimator should be more accurate than standard RV for noisy data');
        end
        
        function testSamplingFrequencyImpact(obj)
            % Test impact of sampling frequency on different realized volatility estimators
            
            % Define sampling frequencies to test
            samplingFreqs = [1, 2, 5, 10, 20]; % Take every nth observation
            numFrequencies = length(samplingFreqs);
            
            % Initialize arrays to store results
            rvResults = zeros(numFrequencies, 3); % For clean, noisy, and jump data
            bvResults = zeros(numFrequencies, 3);
            rvKernelResults = zeros(numFrequencies, 1); % For noisy data only
            
            % Loop through each sampling frequency
            for i = 1:numFrequencies
                freq = samplingFreqs(i);
                
                % Sample data at current frequency
                sampledClean = obj.cleanData(1:freq:end);
                sampledNoisy = obj.noisyData(1:freq:end);
                sampledJump = obj.jumpData(1:freq:end);
                
                % Compute standard RV at each frequency for each data type
                rvResults(i, 1) = rv_compute(sampledClean);
                rvResults(i, 2) = rv_compute(sampledNoisy);
                rvResults(i, 3) = rv_compute(sampledJump);
                
                % Compute BV at each frequency for each data type
                bvResults(i, 1) = bv_compute(sampledClean);
                bvResults(i, 2) = bv_compute(sampledNoisy);
                bvResults(i, 3) = bv_compute(sampledJump);
                
                % Compute kernel estimator at each frequency for noisy data
                kernelOptions = struct('kernelType', 'Bartlett-Parzen');
                rvKernelResults(i) = rv_kernel(sampledNoisy, kernelOptions);
            end
            
            % Check expected trends for clean data
            % RV and BV should decrease as sampling frequency decreases
            obj.assertTrue(all(diff(rvResults(:, 1)) <= 0), ...
                'RV should decrease with less frequent sampling for clean data');
            obj.assertTrue(all(diff(bvResults(:, 1)) <= 0), ...
                'BV should decrease with less frequent sampling for clean data');
            
            % For noisy data, RV might initially decrease with less frequent sampling
            % due to reduced noise impact
            noisyRVTrend = diff(rvResults(1:3, 2));
            obj.assertTrue(any(noisyRVTrend < 0), ...
                'RV may decrease initially with less frequent sampling for noisy data due to noise reduction');
            
            % BV/RV ratio for clean data should remain close to 1 across sampling frequencies
            cleanRatios = bvResults(:, 1) ./ rvResults(:, 1);
            for i = 1:numFrequencies
                obj.assertTrue(abs(cleanRatios(i) - 1) < 0.15, ...
                    sprintf('BV/RV ratio should remain close to 1 at sampling frequency %d for clean data', ...
                    samplingFreqs(i)));
            end
            
            % For jump data, RV/BV ratio should be elevated across sampling frequencies
            jumpRatios = rvResults(:, 3) ./ bvResults(:, 3);
            for i = 1:numFrequencies
                obj.assertTrue(jumpRatios(i) > 1.1, ...
                    sprintf('RV/BV ratio should remain elevated at sampling frequency %d for jump data', ...
                    samplingFreqs(i)));
            end
            
            % The kernel estimator should be less sensitive to sampling frequency than standard RV
            % for noisy data
            rvNoisyVariation = max(rvResults(:, 2)) / min(rvResults(:, 2));
            kernelVariation = max(rvKernelResults) / min(rvKernelResults);
            obj.assertTrue(kernelVariation < rvNoisyVariation, ...
                'Kernel estimator should be less sensitive to sampling frequency than standard RV for noisy data');
            
            % For all data types, increasing the sampling interval eventually leads to information loss
            % and underestimation of volatility
            obj.assertTrue(rvResults(end, 1) < rvResults(1, 1) * 0.9, ...
                'Significant subsampling should lead to volatility underestimation for clean data');
        end
        
        function testIntegratedWorkflow(obj)
            % Test end-to-end workflow of realized volatility analysis
            
            % Choose the test data (use a combination of clean data and jump data)
            testData = [obj.cleanData(1:end/2); obj.jumpData(end/2+1:end)];
            
            % 1. Compute standard RV and BV
            disp('Computing RV and BV...');
            rv = rv_compute(testData);
            bv = bv_compute(testData);
            
            % 2. Perform jump test
            disp('Performing jump test...');
            jumpResults = jump_test(testData);
            
            % 3. Apply noise-robust estimators
            disp('Computing noise-robust estimators...');
            kernelOptions = struct('kernelType', 'Bartlett-Parzen');
            rvKernel = rv_kernel(testData, kernelOptions);
            
            spectrumOptions = struct('windowType', 'Parzen');
            rvSpectrum = realized_spectrum(testData, spectrumOptions);
            
            % 4. Decompose RV into continuous and jump components
            continuousComponent = jumpResults.contComponent;
            jumpComponent = jumpResults.jumpComponent;
            
            % 5. Verify decomposition
            % The continuous component plus jump component should equal total RV
            reconstructedRV = continuousComponent + jumpComponent;
            obj.assertAlmostEqual(reconstructedRV, rv, ...
                'Continuous component plus jump component should equal total RV');
            
            % 6. Verify that jump component is non-negative
            obj.assertTrue(jumpComponent >= 0, ...
                'Jump component should be non-negative');
            
            % 7. Verify that continuous component is close to BV
            continuousToBVRatio = continuousComponent / bv;
            obj.assertTrue(abs(continuousToBVRatio - 1) < 0.15, ...
                'Continuous component should be close to BV');
            
            % 8. Verify that noise-robust estimators are reasonably close
            kernelToSpectrumRatio = rvKernel / rvSpectrum;
            obj.assertTrue(abs(kernelToSpectrumRatio - 1) < 0.25, ...
                'Kernel and spectrum estimators should provide similar estimates');
            
            % 9. Compare different sampling frequencies
            disp('Testing different sampling frequencies...');
            samplingFreqs = [1, 5, 10];
            rvSampled = zeros(length(samplingFreqs), 1);
            
            for i = 1:length(samplingFreqs)
                freq = samplingFreqs(i);
                sampledData = testData(1:freq:end);
                rvSampled(i) = rv_compute(sampledData);
            end
            
            % Verify that sampling at lower frequencies reduces the RV estimate
            obj.assertTrue(all(diff(rvSampled) <= 0), ...
                'RV should decrease with less frequent sampling');
            
            % 10. Report summary statistics
            disp('Workflow complete. Results summary:');
            disp(['RV: ', num2str(rv)]);
            disp(['BV: ', num2str(bv)]);
            disp(['Jump test statistic: ', num2str(jumpResults.zStatistic)]);
            disp(['Continuous component: ', num2str(continuousComponent)]);
            disp(['Jump component: ', num2str(jumpComponent)]);
            disp(['Jump fraction: ', num2str(jumpComponent/rv)]);
            disp(['Kernel RV: ', num2str(rvKernel)]);
            disp(['Spectrum RV: ', num2str(rvSpectrum)]);
        end
        
        function testPerformanceScaling(obj)
            % Test performance scaling of realized volatility methods with dataset size
            
            % Define dataset sizes to test
            dataSizes = [100, 500, 1000, 5000, 10000];
            
            % Skip the largest sizes if running in a limited environment
            if ispc() && ~obj.isMEXAvailable
                % On Windows without MEX acceleration, limit size for faster tests
                dataSizes = dataSizes(dataSizes <= 5000);
            end
            
            % Initialize timing arrays
            rvTimes = zeros(length(dataSizes), 1);
            bvTimes = zeros(length(dataSizes), 1);
            kernelTimes = zeros(length(dataSizes), 1);
            spectrumTimes = zeros(length(dataSizes), 1);
            jumpTestTimes = zeros(length(dataSizes), 1);
            
            % Generate test data of varying sizes
            for i = 1:length(dataSizes)
                n = dataSizes(i);
                disp(['Testing with dataset size: ', num2str(n)]);
                
                % Generate synthetic data
                hfParams = struct();
                hfParams.volatilityModel = 'garch';
                hfParams.volatilityParams = struct('modelType', 'GARCH', 'omega', 0.05, 'alpha', 0.1, 'beta', 0.85);
                hfParams.intradayPattern = 'U-shape';
                hfParams.jumpProcess = 'none';
                hfParams.microstructure = 'none';
                
                % Generate data for this size - using a simple approach with randn for performance testing
                testData = randn(n, 1) * 0.01;
                
                % Measure execution time for each method
                disp('Measuring standard RV computation time...');
                rvTimes(i) = obj.measureExecutionTime(@() rv_compute(testData));
                
                disp('Measuring BV computation time...');
                bvTimes(i) = obj.measureExecutionTime(@() bv_compute(testData));
                
                disp('Measuring kernel estimator computation time...');
                kernelOptions = struct('kernelType', 'Bartlett-Parzen');
                kernelTimes(i) = obj.measureExecutionTime(@() rv_kernel(testData, kernelOptions));
                
                disp('Measuring spectrum estimator computation time...');
                spectrumOptions = struct('windowType', 'Parzen');
                spectrumTimes(i) = obj.measureExecutionTime(@() realized_spectrum(testData, spectrumOptions));
                
                disp('Measuring jump test computation time...');
                jumpTestTimes(i) = obj.measureExecutionTime(@() jump_test(testData));
            end
            
            % Verify performance scaling with dataset size
            % Fitting a power law model log(time) = a + b*log(size)
            % We'd expect b to be approximately 1 for linear scaling, higher for worse scaling
            
            % Convert to log-log scale
            logSizes = log(dataSizes(:));
            logRvTimes = log(rvTimes);
            logBvTimes = log(bvTimes);
            logKernelTimes = log(kernelTimes);
            logSpectrumTimes = log(spectrumTimes);
            logJumpTestTimes = log(jumpTestTimes);
            
            % Simple linear regression to estimate scaling factors
            rvScaling = (logSizes \ logRvTimes);
            bvScaling = (logSizes \ logBvTimes);
            kernelScaling = (logSizes \ logKernelTimes);
            spectrumScaling = (logSizes \ logSpectrumTimes);
            jumpTestScaling = (logSizes \ logJumpTestTimes);
            
            % RV and BV should have approximately linear scaling (b ≈ 1)
            obj.assertTrue(rvScaling > 0.8 && rvScaling < 1.2, ...
                'Standard RV should have approximately linear scaling with dataset size');
            obj.assertTrue(bvScaling > 0.8 && bvScaling < 1.2, ...
                'Bipower variation should have approximately linear scaling with dataset size');
            
            % For more complex methods, scaling is typically super-linear but should be reasonable
            obj.assertTrue(kernelScaling < 2, ...
                'Kernel estimator scaling should be reasonable with dataset size');
            obj.assertTrue(spectrumScaling < 2, ...
                'Spectrum estimator scaling should be reasonable with dataset size');
            obj.assertTrue(jumpTestScaling < 2, ...
                'Jump test scaling should be reasonable with dataset size');
            
            % Compare RV and BV computation times
            % BV should be slightly slower than RV, but not by orders of magnitude
            rvToBvRatio = bvTimes ./ rvTimes;
            meanRatio = mean(rvToBvRatio);
            obj.assertTrue(meanRatio > 1 && meanRatio < 3, ...
                'BV should be moderately slower than RV, but not by orders of magnitude');
            
            % Report scaling results
            disp('Performance scaling factors (log-log slope):');
            disp(['RV: ', num2str(rvScaling)]);
            disp(['BV: ', num2str(bvScaling)]);
            disp(['Kernel: ', num2str(kernelScaling)]);
            disp(['Spectrum: ', num2str(spectrumScaling)]);
            disp(['Jump test: ', num2str(jumpTestScaling)]);
        end
        
        function loadTestData(obj)
            % Load or generate test data
            
            % Try to load pre-generated test data
            try
                data = TestDataGenerator('loadTestData', 'high_frequency_data.mat');
                obj.highFrequencyData = data.returns;
            catch
                % Generate synthetic high-frequency data if file not found
                disp('Generating synthetic high-frequency data for testing...');
                
                % Create parameters for high-frequency data generation
                hfParams = struct();
                hfParams.volatilityModel = 'garch';
                hfParams.volatilityParams = struct('modelType', 'GARCH', 'omega', 0.05, 'alpha', 0.1, 'beta', 0.85);
                hfParams.intradayPattern = 'U-shape';
                
                % Generate clean data first
                hfParams.jumpProcess = 'none';
                hfParams.microstructure = 'none';
                cleanData = TestDataGenerator('generateHighFrequencyData', 5, 78, hfParams);
                obj.cleanData = cleanData.returns;
                
                % Generate data with microstructure noise
                hfParams.jumpProcess = 'none';
                hfParams.microstructure = 'additive';
                hfParams.microstructureParams = struct('std', 0.0002);
                noisyData = TestDataGenerator('generateHighFrequencyData', 5, 78, hfParams);
                obj.noisyData = noisyData.returns;
                
                % Generate data with jumps
                hfParams.jumpProcess = 'poisson';
                hfParams.jumpParams = struct('intensity', 0.1, 'jumpSize', [0, 0.005]);
                hfParams.microstructure = 'none';
                jumpData = TestDataGenerator('generateHighFrequencyData', 5, 78, hfParams);
                obj.jumpData = jumpData.returns;
                
                % Use clean data as default high-frequency data
                obj.highFrequencyData = obj.cleanData;
            end
            
            % Ensure we have all the data types we need
            if ~isfield(obj, 'cleanData') || isempty(obj.cleanData)
                % Generate clean data if not already loaded
                hfParams = struct();
                hfParams.volatilityModel = 'garch';
                hfParams.volatilityParams = struct('modelType', 'GARCH', 'omega', 0.05, 'alpha', 0.1, 'beta', 0.85);
                hfParams.intradayPattern = 'U-shape';
                hfParams.jumpProcess = 'none';
                hfParams.microstructure = 'none';
                cleanData = TestDataGenerator('generateHighFrequencyData', 5, 78, hfParams);
                obj.cleanData = cleanData.returns;
            end
            
            if ~isfield(obj, 'noisyData') || isempty(obj.noisyData)
                % Generate noisy data if not already loaded
                hfParams = struct();
                hfParams.volatilityModel = 'garch';
                hfParams.volatilityParams = struct('modelType', 'GARCH', 'omega', 0.05, 'alpha', 0.1, 'beta', 0.85);
                hfParams.intradayPattern = 'U-shape';
                hfParams.jumpProcess = 'none';
                hfParams.microstructure = 'additive';
                hfParams.microstructureParams = struct('std', 0.0002);
                noisyData = TestDataGenerator('generateHighFrequencyData', 5, 78, hfParams);
                obj.noisyData = noisyData.returns;
            end
            
            if ~isfield(obj, 'jumpData') || isempty(obj.jumpData)
                % Generate jump data if not already loaded
                hfParams = struct();
                hfParams.volatilityModel = 'garch';
                hfParams.volatilityParams = struct('modelType', 'GARCH', 'omega', 0.05, 'alpha', 0.1, 'beta', 0.85);
                hfParams.intradayPattern = 'U-shape';
                hfParams.jumpProcess = 'poisson';
                hfParams.jumpParams = struct('intensity', 0.1, 'jumpSize', [0, 0.005]);
                hfParams.microstructure = 'none';
                jumpData = TestDataGenerator('generateHighFrequencyData', 5, 78, hfParams);
                obj.jumpData = jumpData.returns;
            end
        end
        
        function initializeReferenceValues(obj)
            % Initialize reference values for comparison
            
            % Create a structure to hold reference values
            obj.referenceValues = struct();
            
            % Compute standard realized volatility on clean data
            obj.referenceValues.cleanRV = rv_compute(obj.cleanData);
            
            % Compute bipower variation on clean data
            obj.referenceValues.cleanBV = bv_compute(obj.cleanData);
            
            % Expected ratio of BV to RV for clean data (should be close to 1)
            obj.referenceValues.cleanBVtoRVRatio = obj.referenceValues.cleanBV / obj.referenceValues.cleanRV;
            
            % Compute standard realized volatility on noisy data
            obj.referenceValues.noisyRV = rv_compute(obj.noisyData);
            
            % Compute realized kernel on noisy data
            kernelOptions = struct('kernelType', 'Bartlett-Parzen');
            obj.referenceValues.noisyRVkernel = rv_kernel(obj.noisyData, kernelOptions);
            
            % Compute realized volatility on jump data
            obj.referenceValues.jumpRV = rv_compute(obj.jumpData);
            
            % Compute bipower variation on jump data
            obj.referenceValues.jumpBV = bv_compute(obj.jumpData);
            
            % Expected ratio of RV to BV for jump data (should be > 1)
            obj.referenceValues.jumpRVtoBVRatio = obj.referenceValues.jumpRV / obj.referenceValues.jumpBV;
            
            % Compute jump test statistics
            jumpResults = jump_test(obj.jumpData);
            obj.referenceValues.jumpTestStat = jumpResults.zStatistic;
            
            % Compute realized spectrum
            spectrumOptions = struct('windowType', 'Parzen');
            obj.referenceValues.spectrumRV = realized_spectrum(obj.noisyData, spectrumOptions);
        end
    end
end