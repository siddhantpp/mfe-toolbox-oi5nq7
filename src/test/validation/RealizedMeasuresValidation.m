classdef RealizedMeasuresValidation < BaseTest
    % REALIZEDMEASURESVALIDATION Test suite for realized measures components of MFE Toolbox
    %
    % This class provides comprehensive validation of high-frequency financial
    % econometrics calculations, testing the accuracy, performance, and numerical
    % stability of realized volatility, bipower variation, kernel-based estimators,
    % and jump tests against known theoretical properties and benchmark values.
    %
    % The test suite validates implementations against:
    %   1. Theoretical properties of estimators (consistency, robustness to noise/jumps)
    %   2. Known reference values for controlled simulated processes
    %   3. Performance benchmarks for large datasets
    %   4. Numerical stability under extreme conditions
    %
    % Key validation components:
    %   - Realized volatility computation (rv_compute)
    %   - Bipower variation computation (bv_compute)
    %   - Kernel-based volatility estimation (rv_kernel)
    %   - Jump detection tests (jump_test)
    %
    % Example:
    %   % Run all validation tests
    %   validator = RealizedMeasuresValidation();
    %   results = validator.runAllValidations();
    %   report = validator.getSummaryReport();
    %
    % See also: BASETEST, RV_COMPUTE, BV_COMPUTE, RV_KERNEL, JUMP_TEST
    
    properties
        % Test data structure containing high-frequency data for validation
        testData
        
        % Matrix of high-frequency returns for testing
        highFrequencyReturns
        
        % Matrix of corresponding timestamps
        timestamps
        
        % Structure with reference/benchmark values
        referenceValues
        
        % Sampling frequency of the test data
        samplingFrequency
        
        % Structure storing validation results
        validationResults
        
        % Numerical comparator instance for precise floating-point validation
        comparator
    end
    
    methods
        function obj = RealizedMeasuresValidation()
            % Constructor initializes the RealizedMeasuresValidation test suite
            %
            % Creates a test suite for validating the realized measures components
            % of the MFE Toolbox with appropriate test configuration and tolerance settings.
            
            % Call superclass constructor
            obj@BaseTest('RealizedMeasuresValidation');
            
            % Initialize empty property values
            obj.testData = struct();
            obj.highFrequencyReturns = [];
            obj.timestamps = [];
            obj.referenceValues = struct();
            obj.samplingFrequency = [];
            obj.validationResults = struct();
            
            % Create numerical comparator with appropriate tolerances for volatility calculations
            % Use tighter tolerances for financial calculations
            comparatorOptions = struct('absoluteTolerance', 1e-12, 'relativeTolerance', 1e-10);
            obj.comparator = NumericalComparator(comparatorOptions);
            
            % Set verbose mode to true for detailed output
            obj.setVerbose(true);
        end
        
        function setUp(obj)
            % Prepare the test environment before each test execution
            %
            % Loads test data, validates its integrity, and initializes
            % test structures for validation results.
            
            % Call superclass setUp
            setUp@BaseTest(obj);
            
            % Load high-frequency test data from the test data directory
            try
                obj.testData = obj.loadTestData('high_frequency_data.mat');
                
                % Extract test components (may vary based on actual test data format)
                if isfield(obj.testData, 'returns')
                    obj.highFrequencyReturns = obj.testData.returns;
                elseif isfield(obj.testData, 'highFrequencyReturns')
                    obj.highFrequencyReturns = obj.testData.highFrequencyReturns;
                else
                    error('Test data must contain returns or highFrequencyReturns field');
                end
                
                if isfield(obj.testData, 'timestamps')
                    obj.timestamps = obj.testData.timestamps;
                end
                
                if isfield(obj.testData, 'referenceValues')
                    obj.referenceValues = obj.testData.referenceValues;
                else
                    % Create empty reference values structure if not in test data
                    obj.referenceValues = struct();
                end
                
                if isfield(obj.testData, 'samplingFrequency')
                    obj.samplingFrequency = obj.testData.samplingFrequency;
                else
                    % Default to 5-min (assuming standard high-frequency data)
                    obj.samplingFrequency = 288; % Default: 288 observations per day (5-min data)
                end
                
                % Verify that data is valid for testing
                if isempty(obj.highFrequencyReturns)
                    error('High-frequency returns data is empty');
                end
                
                % Initialize validation results structure
                obj.validationResults = struct(...
                    'realizedVolatility', struct(), ...
                    'bipowerVariation', struct(), ...
                    'kernelEstimation', struct(), ...
                    'jumpTest', struct(), ...
                    'asymptoticProperties', struct(), ...
                    'numericalStability', struct() ...
                );
                
            catch ME
                % If test data is not available, generate simulated data for testing
                warning(['Failed to load test data: ' ME.message ...
                    '. Generating simulated data for testing.']);
                
                % Generate simulated diffusion process with known properties
                simulationOptions = struct('includeJumps', true, 'microstructureNoise', false);
                simData = obj.generateSimulatedProcess(0.2, 1, 288, simulationOptions);
                
                % Set up test data from simulation
                obj.highFrequencyReturns = simData.returns;
                obj.timestamps = simData.timestamps;
                obj.referenceValues = simData.referenceValues;
                obj.samplingFrequency = 288;
                
                % Initialize validation results structure
                obj.validationResults = struct(...
                    'realizedVolatility', struct(), ...
                    'bipowerVariation', struct(), ...
                    'kernelEstimation', struct(), ...
                    'jumpTest', struct(), ...
                    'asymptoticProperties', struct(), ...
                    'numericalStability', struct() ...
                );
            end
        end
        
        function tearDown(obj)
            % Clean up after test execution
            %
            % Saves validation results and cleans up any temporary resources.
            
            % Call superclass tearDown
            tearDown@BaseTest(obj);
            
            % Store validation results in test results structure
            if isfield(obj.testResults, 'validationResults')
                obj.testResults.validationResults = obj.validationResults;
            else
                obj.testResults.validationResults = struct();
            end
            
            % Clear any temporary variables created during testing
            % (No specific cleanup needed for this test class)
        end
        
        function results = validateRealizedVolatility(obj)
            % Validates the accuracy and performance of realized volatility computation
            %
            % OUTPUTS:
            %   results - Struct containing validation results for realized volatility
            %
            % This function tests whether the realized volatility computations:
            %   1. Produce accurate results compared to reference values
            %   2. Exhibit appropriate asymptotic properties
            %   3. Handle various data inputs correctly
            %   4. Perform efficiently with large datasets
            
            disp('Validating realized volatility (rv_compute) implementation...');
            
            % Initialize results structure
            results = struct(...
                'accuracyTests', struct(), ...
                'robustnessTests', struct(), ...
                'performanceTests', struct(), ...
                'errorHandlingTests', struct(), ...
                'overallResult', 'pending' ...
            );
            
            try
                % ACCURACY TESTS
                % --------------
                
                % Test 1: Basic computation - compare with reference values if available
                basicOptions = struct('scale', 1);
                basicRV = rv_compute(obj.highFrequencyReturns, basicOptions);
                
                if isfield(obj.referenceValues, 'realizedVolatility')
                    % Compare with reference values using the appropriate tolerance
                    compResult = obj.comparator.compareMatrices(...
                        obj.referenceValues.realizedVolatility, basicRV, 1e-10);
                    
                    results.accuracyTests.basicComputation = struct(...
                        'passed', compResult.isEqual, ...
                        'referenceValue', obj.referenceValues.realizedVolatility, ...
                        'computedValue', basicRV, ...
                        'absoluteDifference', compResult.maxAbsoluteDifference, ...
                        'relativeDifference', compResult.maxRelativeDifference ...
                    );
                else
                    % If no reference value, store computed value for reference
                    results.accuracyTests.basicComputation = struct(...
                        'passed', true, ...
                        'referenceValue', 'Not available', ...
                        'computedValue', basicRV, ...
                        'notes', 'No reference value available for comparison' ...
                    );
                end
                
                % Test 2: Verify that RV scales correctly with the scaling factor
                scaleOptions = struct('scale', 252);  % Annual scaling
                scaledRV = rv_compute(obj.highFrequencyReturns, scaleOptions);
                
                % Check scaling relationship: scaled RV should be scale*basic RV
                expectedScaled = basicRV * 252;
                compResult = obj.comparator.compareMatrices(expectedScaled, scaledRV, 1e-10);
                
                results.accuracyTests.scalingProperty = struct(...
                    'passed', compResult.isEqual, ...
                    'expectedValue', expectedScaled, ...
                    'computedValue', scaledRV, ...
                    'absoluteDifference', compResult.maxAbsoluteDifference, ...
                    'notes', 'Verifies correct scaling by constant factor' ...
                );
                
                % Test 3: Verify consistency with theoretical properties
                % For simulated Brownian motion, RV should approximate integrated variance
                if isfield(obj.referenceValues, 'integratedVariance')
                    % RV should converge to integrated variance
                    intVar = obj.referenceValues.integratedVariance;
                    
                    % Allow for sampling error with wider tolerance
                    rvDeviation = abs(basicRV - intVar) / intVar;
                    results.accuracyTests.theoreticalConsistency = struct(...
                        'passed', rvDeviation < 0.1, ...  % 10% tolerance for sampling error
                        'integratedVariance', intVar, ...
                        'realizedVolatility', basicRV, ...
                        'relativeDeviation', rvDeviation, ...
                        'notes', 'Verifies RV approximates integrated variance' ...
                    );
                else
                    results.accuracyTests.theoreticalConsistency = struct(...
                        'passed', true, ...
                        'notes', 'Theoretical consistency check skipped - no reference integrated variance' ...
                    );
                end
                
                % ROBUSTNESS TESTS
                % ---------------
                
                % Test 4: Different sampling frequencies
                if size(obj.highFrequencyReturns, 1) >= 10
                    % Subsample returns to simulate lower frequency
                    subsampledReturns = obj.highFrequencyReturns(1:2:end, :);
                    
                    % Compute RV at original and lower frequency
                    rvOriginal = rv_compute(obj.highFrequencyReturns);
                    rvSubsampled = rv_compute(subsampledReturns);
                    
                    % Subsampled RV should be reasonably close to original
                    % (wider tolerance due to sampling differences)
                    relDiff = abs(rvSubsampled - rvOriginal) ./ max(abs(rvOriginal), 1e-10);
                    
                    results.robustnessTests.frequencySensitivity = struct(...
                        'passed', all(relDiff < 0.3), ...  % 30% tolerance for subsampling
                        'originalRV', rvOriginal, ...
                        'subsampledRV', rvSubsampled, ...
                        'relativeDeviation', relDiff, ...
                        'notes', 'Checks sensitivity to sampling frequency' ...
                    );
                else
                    results.robustnessTests.frequencySensitivity = struct(...
                        'passed', true, ...
                        'notes', 'Insufficient data for frequency sensitivity test' ...
                    );
                end
                
                % Test 5: Subsampled computation
                subsampleOptions = struct('method', 'subsample', 'subSample', 3);
                rvSubsampled = rv_compute(obj.highFrequencyReturns, subsampleOptions);
                
                % Ensure subsampling produces valid results (should be positive)
                results.robustnessTests.subsamplingMethod = struct(...
                    'passed', all(rvSubsampled > 0), ...
                    'computedValue', rvSubsampled, ...
                    'notes', 'Validates subsampling method produces valid results' ...
                );
                
                % Test 6: Jackknife bias correction
                jackknifeOptions = struct('jackknife', true);
                rvJackknife = rv_compute(obj.highFrequencyReturns, jackknifeOptions);
                
                % Ensure jackknife correction produces valid results (should be positive)
                results.robustnessTests.jackknifeBiasCorrection = struct(...
                    'passed', all(rvJackknife > 0), ...
                    'computedValue', rvJackknife, ...
                    'notes', 'Validates jackknife correction produces valid results' ...
                );
                
                % PERFORMANCE TESTS
                % ----------------
                
                % Test 7: Execution time measurement
                n = 100; % Number of times to repeat for reliable timing
                executionTime = obj.measureExecutionTime(@() rv_compute(obj.highFrequencyReturns, basicOptions));
                
                results.performanceTests.executionTime = struct(...
                    'timeInSeconds', executionTime, ...
                    'returnsProcessed', numel(obj.highFrequencyReturns), ...
                    'returnsPerSecond', numel(obj.highFrequencyReturns) / max(executionTime, 1e-10), ...
                    'notes', 'Basic performance benchmark' ...
                );
                
                % ERROR HANDLING TESTS
                % -------------------
                
                % Test 8: Invalid inputs
                invalidTestPassed = false;
                try
                    % Attempt to call with invalid returns (containing NaN)
                    badReturns = obj.highFrequencyReturns;
                    badReturns(1, 1) = NaN;
                    rv_compute(badReturns);
                    
                    % If we got here, the test failed
                    invalidTestPassed = false;
                catch
                    % Error was correctly thrown
                    invalidTestPassed = true;
                end
                
                results.errorHandlingTests.invalidInputs = struct(...
                    'passed', invalidTestPassed, ...
                    'notes', 'Validates error handling for invalid inputs' ...
                );
                
                % Determine overall result
                allTests = [
                    results.accuracyTests.basicComputation.passed,
                    results.accuracyTests.scalingProperty.passed,
                    results.robustnessTests.subsamplingMethod.passed,
                    results.robustnessTests.jackknifeBiasCorrection.passed,
                    results.errorHandlingTests.invalidInputs.passed
                ];
                
                if all(allTests)
                    results.overallResult = 'passed';
                else
                    results.overallResult = 'failed';
                end
                
                % Display overall result
                disp(['Realized volatility validation ', results.overallResult]);
                
            catch ME
                % Handle unexpected errors
                results.overallResult = 'error';
                results.errorMessage = ME.message;
                results.errorIdentifier = ME.identifier;
                results.errorStack = ME.stack;
                
                warning(['Error in realized volatility validation: ' ME.message]);
            end
            
            % Store results in validation results structure
            obj.validationResults.realizedVolatility = results;
        end
        
        function results = validateBipowerVariation(obj)
            % Validates the accuracy and performance of bipower variation computation
            %
            % OUTPUTS:
            %   results - Struct containing validation results for bipower variation
            %
            % This function tests whether the bipower variation computations:
            %   1. Produce accurate results compared to reference values
            %   2. Exhibit appropriate jump robustness properties
            %   3. Apply correct scaling factors
            %   4. Perform efficiently with large datasets
            
            disp('Validating bipower variation (bv_compute) implementation...');
            
            % Initialize results structure
            results = struct(...
                'accuracyTests', struct(), ...
                'jumpRobustnessTests', struct(), ...
                'scalingTests', struct(), ...
                'performanceTests', struct(), ...
                'errorHandlingTests', struct(), ...
                'overallResult', 'pending' ...
            );
            
            try
                % ACCURACY TESTS
                % --------------
                
                % Test 1: Basic computation - compare with reference values if available
                basicBV = bv_compute(obj.highFrequencyReturns);
                
                if isfield(obj.referenceValues, 'bipowerVariation')
                    % Compare with reference values using the appropriate tolerance
                    compResult = obj.comparator.compareMatrices(...
                        obj.referenceValues.bipowerVariation, basicBV, 1e-10);
                    
                    results.accuracyTests.basicComputation = struct(...
                        'passed', compResult.isEqual, ...
                        'referenceValue', obj.referenceValues.bipowerVariation, ...
                        'computedValue', basicBV, ...
                        'absoluteDifference', compResult.maxAbsoluteDifference, ...
                        'relativeDifference', compResult.maxRelativeDifference ...
                    );
                else
                    % If no reference value, store computed value for reference
                    results.accuracyTests.basicComputation = struct(...
                        'passed', true, ...
                        'referenceValue', 'Not available', ...
                        'computedValue', basicBV, ...
                        'notes', 'No reference value available for comparison' ...
                    );
                end
                
                % Test 2: Verify that BV calculation applies correct pi/2 scaling factor
                customOptions = struct('scaleFactor', 1); % No scaling
                unscaledBV = bv_compute(obj.highFrequencyReturns, customOptions);
                
                % BV with default scaling should be (pi/2) * unscaled BV
                expectedBV = (pi/2) * unscaledBV;
                compResult = obj.comparator.compareMatrices(expectedBV, basicBV, 1e-10);
                
                results.scalingTests.defaultScalingFactor = struct(...
                    'passed', compResult.isEqual, ...
                    'expectedValue', expectedBV, ...
                    'computedValue', basicBV, ...
                    'absoluteDifference', compResult.maxAbsoluteDifference, ...
                    'notes', 'Verifies default pi/2 scaling factor is applied correctly' ...
                );
                
                % Test 3: Custom scaling factor
                customOptions = struct('scaleFactor', 0.8);
                customBV = bv_compute(obj.highFrequencyReturns, customOptions);
                
                % BV with custom scaling should be 0.8 * unscaled BV
                expectedCustomBV = 0.8 * unscaledBV;
                compResult = obj.comparator.compareMatrices(expectedCustomBV, customBV, 1e-10);
                
                results.scalingTests.customScalingFactor = struct(...
                    'passed', compResult.isEqual, ...
                    'expectedValue', expectedCustomBV, ...
                    'computedValue', customBV, ...
                    'absoluteDifference', compResult.maxAbsoluteDifference, ...
                    'notes', 'Verifies custom scaling factor is applied correctly' ...
                );
                
                % JUMP ROBUSTNESS TESTS
                % --------------------
                
                % Test 4: Generate data with and without jumps to test BV robustness
                % For controlled testing of jump robustness
                
                % Generate simulated return series without jumps
                noJumpOptions = struct('includeJumps', false, 'microstructureNoise', false);
                noJumpData = obj.generateSimulatedProcess(0.2, 1, 288, noJumpOptions);
                
                % Generate simulated return series with jumps
                jumpOptions = struct('includeJumps', true, 'microstructureNoise', false, ...
                    'jumpIntensity', 3, 'jumpSize', 5);
                jumpData = obj.generateSimulatedProcess(0.2, 1, 288, jumpOptions);
                
                % Compute RV and BV for both series
                rv_noJump = rv_compute(noJumpData.returns);
                bv_noJump = bv_compute(noJumpData.returns);
                
                rv_withJump = rv_compute(jumpData.returns);
                bv_withJump = bv_compute(jumpData.returns);
                
                % For no-jump data, RV and BV should be similar
                rvBvRatio_noJump = rv_noJump ./ bv_noJump;
                
                % For jump data, RV should be larger than BV
                rvBvRatio_withJump = rv_withJump ./ bv_withJump;
                
                % BV should be robust to jumps: ratio should be near 1 for no jumps,
                % and significantly > 1 when jumps are present
                results.jumpRobustnessTests.noJumpCase = struct(...
                    'passed', all(abs(rvBvRatio_noJump - 1) < 0.2), ... % 20% tolerance
                    'rvValue', rv_noJump, ...
                    'bvValue', bv_noJump, ...
                    'rvBvRatio', rvBvRatio_noJump, ...
                    'notes', 'RV/BV ratio should be close to 1 without jumps' ...
                );
                
                results.jumpRobustnessTests.withJumpCase = struct(...
                    'passed', all(rvBvRatio_withJump > 1.2), ... % Should be at least 20% higher
                    'rvValue', rv_withJump, ...
                    'bvValue', bv_withJump, ...
                    'rvBvRatio', rvBvRatio_withJump, ...
                    'notes', 'RV/BV ratio should be greater than 1 with jumps' ...
                );
                
                % PERFORMANCE TESTS
                % ----------------
                
                % Test 5: Execution time measurement
                executionTime = obj.measureExecutionTime(@() bv_compute(obj.highFrequencyReturns));
                
                results.performanceTests.executionTime = struct(...
                    'timeInSeconds', executionTime, ...
                    'returnsProcessed', numel(obj.highFrequencyReturns), ...
                    'returnsPerSecond', numel(obj.highFrequencyReturns) / max(executionTime, 1e-10), ...
                    'notes', 'Basic performance benchmark' ...
                );
                
                % ERROR HANDLING TESTS
                % -------------------
                
                % Test 6: Invalid inputs
                invalidTestPassed = false;
                try
                    % Attempt to call with invalid returns (containing NaN)
                    badReturns = obj.highFrequencyReturns;
                    badReturns(1, 1) = NaN;
                    bv_compute(badReturns);
                    
                    % If we got here, the test failed
                    invalidTestPassed = false;
                catch
                    % Error was correctly thrown
                    invalidTestPassed = true;
                end
                
                results.errorHandlingTests.invalidInputs = struct(...
                    'passed', invalidTestPassed, ...
                    'notes', 'Validates error handling for invalid inputs' ...
                );
                
                % Test 7: Insufficient observations
                if size(obj.highFrequencyReturns, 1) > 2
                    insufficientTestPassed = false;
                    try
                        % Attempt with only one observation (BV needs at least 2)
                        bv_compute(obj.highFrequencyReturns(1,:));
                        
                        % If we got here, the test failed
                        insufficientTestPassed = false;
                    catch
                        % Error was correctly thrown
                        insufficientTestPassed = true;
                    end
                    
                    results.errorHandlingTests.insufficientObservations = struct(...
                        'passed', insufficientTestPassed, ...
                        'notes', 'Validates error handling for insufficient observations' ...
                    );
                else
                    results.errorHandlingTests.insufficientObservations = struct(...
                        'passed', true, ...
                        'notes', 'Skipped test due to insufficient test data' ...
                    );
                end
                
                % Determine overall result
                allTests = [
                    results.accuracyTests.basicComputation.passed,
                    results.scalingTests.defaultScalingFactor.passed,
                    results.scalingTests.customScalingFactor.passed,
                    results.jumpRobustnessTests.noJumpCase.passed,
                    results.jumpRobustnessTests.withJumpCase.passed,
                    results.errorHandlingTests.invalidInputs.passed
                ];
                
                if all(allTests)
                    results.overallResult = 'passed';
                else
                    results.overallResult = 'failed';
                end
                
                % Display overall result
                disp(['Bipower variation validation ', results.overallResult]);
                
            catch ME
                % Handle unexpected errors
                results.overallResult = 'error';
                results.errorMessage = ME.message;
                results.errorIdentifier = ME.identifier;
                results.errorStack = ME.stack;
                
                warning(['Error in bipower variation validation: ' ME.message]);
            end
            
            % Store results in validation results structure
            obj.validationResults.bipowerVariation = results;
        end
        
        function results = validateKernelEstimation(obj)
            % Validates the accuracy and performance of kernel-based realized volatility estimation
            %
            % OUTPUTS:
            %   results - Struct containing validation results for kernel estimators
            %
            % This function tests whether the kernel-based estimations:
            %   1. Produce accurate results compared to reference values
            %   2. Exhibit appropriate noise-robustness properties
            %   3. Handle different kernel types correctly
            %   4. Perform efficiently with large datasets
            
            disp('Validating kernel-based volatility estimation (rv_kernel) implementation...');
            
            % Initialize results structure
            results = struct(...
                'accuracyTests', struct(), ...
                'noiseRobustnessTests', struct(), ...
                'kernelTests', struct(), ...
                'performanceTests', struct(), ...
                'errorHandlingTests', struct(), ...
                'overallResult', 'pending' ...
            );
            
            try
                % ACCURACY TESTS
                % --------------
                
                % Test 1: Default kernel estimation - compare with reference values if available
                defaultOptions = struct();
                defaultKernel = rv_kernel(obj.highFrequencyReturns, defaultOptions);
                
                if isfield(obj.referenceValues, 'kernelRV')
                    % Compare with reference values using the appropriate tolerance
                    compResult = obj.comparator.compareMatrices(...
                        obj.referenceValues.kernelRV, defaultKernel, 1e-8);
                    
                    results.accuracyTests.defaultKernel = struct(...
                        'passed', compResult.isEqual, ...
                        'referenceValue', obj.referenceValues.kernelRV, ...
                        'computedValue', defaultKernel, ...
                        'absoluteDifference', compResult.maxAbsoluteDifference, ...
                        'relativeDifference', compResult.maxRelativeDifference ...
                    );
                else
                    % If no reference value, store computed value for reference
                    results.accuracyTests.defaultKernel = struct(...
                        'passed', true, ...
                        'referenceValue', 'Not available', ...
                        'computedValue', defaultKernel, ...
                        'notes', 'No reference value available for comparison' ...
                    );
                end
                
                % Test 2: Standard RV vs Kernel RV - kernel should produce valid estimate
                standardRV = rv_compute(obj.highFrequencyReturns);
                
                % Kernel RV should be positive and comparable to standard RV in magnitude
                relDiff = abs(defaultKernel - standardRV) ./ max(abs(standardRV), 1e-10);
                
                results.accuracyTests.comparisonWithStandardRV = struct(...
                    'passed', all(defaultKernel > 0) && all(relDiff < 0.5), ... % 50% tolerance
                    'standardRV', standardRV, ...
                    'kernelRV', defaultKernel, ...
                    'relativeDeviation', relDiff, ...
                    'notes', 'Kernel RV should be comparable to standard RV' ...
                );
                
                % NOISE ROBUSTNESS TESTS
                % ---------------------
                
                % Test 3: Generate data with and without microstructure noise
                % For controlled testing of noise robustness
                
                % Generate simulated return series without noise
                noNoiseOptions = struct('includeJumps', false, 'microstructureNoise', false);
                noNoiseData = obj.generateSimulatedProcess(0.2, 1, 288, noNoiseOptions);
                
                % Generate simulated return series with noise
                noiseOptions = struct('includeJumps', false, 'microstructureNoise', true, ...
                    'noiseRatio', 0.2); % 20% noise-to-signal ratio
                noiseData = obj.generateSimulatedProcess(0.2, 1, 288, noiseOptions);
                
                % Compute standard RV and kernel RV for both series
                rv_noNoise = rv_compute(noNoiseData.returns);
                kernel_noNoise = rv_kernel(noNoiseData.returns);
                
                rv_withNoise = rv_compute(noiseData.returns);
                kernel_withNoise = rv_kernel(noiseData.returns);
                
                % True integrated variance from the simulation
                trueIV = noNoiseData.referenceValues.integratedVariance;
                
                % For noise-free data, both estimators should be accurate
                error_rv_noNoise = abs(rv_noNoise - trueIV) / trueIV;
                error_kernel_noNoise = abs(kernel_noNoise - trueIV) / trueIV;
                
                % For noisy data, kernel should be more accurate than RV
                error_rv_withNoise = abs(rv_withNoise - trueIV) / trueIV;
                error_kernel_withNoise = abs(kernel_withNoise - trueIV) / trueIV;
                
                results.noiseRobustnessTests.noNoiseCase = struct(...
                    'passed', error_kernel_noNoise < 0.3, ... % 30% tolerance
                    'trueIV', trueIV, ...
                    'rvValue', rv_noNoise, ...
                    'kernelValue', kernel_noNoise, ...
                    'rvError', error_rv_noNoise, ...
                    'kernelError', error_kernel_noNoise, ...
                    'notes', 'Both estimators should be accurate without noise' ...
                );
                
                results.noiseRobustnessTests.withNoiseCase = struct(...
                    'passed', error_kernel_withNoise < error_rv_withNoise, ...
                    'trueIV', trueIV, ...
                    'rvValue', rv_withNoise, ...
                    'kernelValue', kernel_withNoise, ...
                    'rvError', error_rv_withNoise, ...
                    'kernelError', error_kernel_withNoise, ...
                    'notes', 'Kernel estimator should be more robust to noise than standard RV' ...
                );
                
                % KERNEL TYPE TESTS
                % ----------------
                
                % Test 4: Different kernel types
                kernelTypes = {'Bartlett-Parzen', 'Quadratic', 'Exponential', 'Tukey-Hanning'};
                kernelResults = struct();
                
                for i = 1:length(kernelTypes)
                    kernelType = kernelTypes{i};
                    kernelOpt = struct('kernelType', kernelType, 'bandwidth', 10);
                    
                    % Measure execution time for this kernel
                    kernelExecutionTime = obj.measureExecutionTime(@() rv_kernel(obj.highFrequencyReturns, kernelOpt));
                    
                    % Compute kernel estimate
                    kernelRV = rv_kernel(obj.highFrequencyReturns, kernelOpt);
                    
                    % Kernel estimate should be positive
                    kernelResults.(strrep(kernelType, '-', '_')) = struct(...
                        'passed', all(kernelRV > 0), ...
                        'computedValue', kernelRV, ...
                        'executionTime', kernelExecutionTime, ...
                        'notes', ['Validates ' kernelType ' kernel estimation'] ...
                    );
                end
                
                results.kernelTests = kernelResults;
                
                % Test 5: Bandwidth sensitivity
                bandwidths = [5, 10, 20];
                bandwidthResults = struct();
                
                for i = 1:length(bandwidths)
                    bw = bandwidths(i);
                    bwOpt = struct('bandwidth', bw);
                    
                    % Compute kernel estimate with this bandwidth
                    bwKernelRV = rv_kernel(obj.highFrequencyReturns, bwOpt);
                    
                    % Kernel estimate should be positive
                    bandwidthResults.(['bandwidth_' num2str(bw)]) = struct(...
                        'passed', all(bwKernelRV > 0), ...
                        'computedValue', bwKernelRV, ...
                        'notes', ['Validates bandwidth = ' num2str(bw)] ...
                    );
                end
                
                results.kernelTests.bandwidthSensitivity = bandwidthResults;
                
                % PERFORMANCE TESTS
                % ----------------
                
                % Test 6: Execution time measurement
                executionTime = obj.measureExecutionTime(@() rv_kernel(obj.highFrequencyReturns));
                
                results.performanceTests.executionTime = struct(...
                    'timeInSeconds', executionTime, ...
                    'returnsProcessed', numel(obj.highFrequencyReturns), ...
                    'returnsPerSecond', numel(obj.highFrequencyReturns) / max(executionTime, 1e-10), ...
                    'notes', 'Basic performance benchmark' ...
                );
                
                % ERROR HANDLING TESTS
                % -------------------
                
                % Test 7: Invalid kernel type
                invalidKernelTestPassed = false;
                try
                    % Attempt to call with invalid kernel type
                    badOptions = struct('kernelType', 'InvalidKernel');
                    rv_kernel(obj.highFrequencyReturns, badOptions);
                    
                    % If we got here, the test failed
                    invalidKernelTestPassed = false;
                catch
                    % Error was correctly thrown
                    invalidKernelTestPassed = true;
                end
                
                results.errorHandlingTests.invalidKernelType = struct(...
                    'passed', invalidKernelTestPassed, ...
                    'notes', 'Validates error handling for invalid kernel type' ...
                );
                
                % Test 8: Invalid inputs
                invalidInputTestPassed = false;
                try
                    % Attempt to call with invalid returns (containing NaN)
                    badReturns = obj.highFrequencyReturns;
                    badReturns(1, 1) = NaN;
                    rv_kernel(badReturns);
                    
                    % If we got here, the test failed
                    invalidInputTestPassed = false;
                catch
                    % Error was correctly thrown
                    invalidInputTestPassed = true;
                end
                
                results.errorHandlingTests.invalidInputs = struct(...
                    'passed', invalidInputTestPassed, ...
                    'notes', 'Validates error handling for invalid inputs' ...
                );
                
                % Determine overall result
                coreTests = [
                    results.accuracyTests.defaultKernel.passed,
                    results.accuracyTests.comparisonWithStandardRV.passed,
                    results.noiseRobustnessTests.withNoiseCase.passed,
                    results.errorHandlingTests.invalidKernelType.passed,
                    results.errorHandlingTests.invalidInputs.passed
                ];
                
                if all(coreTests)
                    results.overallResult = 'passed';
                else
                    results.overallResult = 'failed';
                end
                
                % Display overall result
                disp(['Kernel-based estimation validation ', results.overallResult]);
                
            catch ME
                % Handle unexpected errors
                results.overallResult = 'error';
                results.errorMessage = ME.message;
                results.errorIdentifier = ME.identifier;
                results.errorStack = ME.stack;
                
                warning(['Error in kernel-based estimation validation: ' ME.message]);
            end
            
            % Store results in validation results structure
            obj.validationResults.kernelEstimation = results;
        end
        
        function results = validateJumpTest(obj)
            % Validates the accuracy and performance of jump detection test
            %
            % OUTPUTS:
            %   results - Struct containing validation results for jump test
            %
            % This function tests whether the jump test:
            %   1. Correctly identifies jumps in price processes 
            %   2. Maintains appropriate size (false positive rate)
            %   3. Provides accurate decomposition into continuous and jump components
            %   4. Performs efficiently with large datasets
            
            disp('Validating jump detection test (jump_test) implementation...');
            
            % Initialize results structure
            results = struct(...
                'accuracyTests', struct(), ...
                'sizeAndPowerTests', struct(), ...
                'decompositionTests', struct(), ...
                'performanceTests', struct(), ...
                'errorHandlingTests', struct(), ...
                'overallResult', 'pending' ...
            );
            
            try
                % ACCURACY TESTS
                % --------------
                
                % Test 1: Basic jump test - compare with reference values if available
                jumpTestResults = jump_test(obj.highFrequencyReturns);
                
                if isfield(obj.referenceValues, 'jumpTest')
                    % Compare z-statistic with reference value
                    compResult = obj.comparator.compareMatrices(...
                        obj.referenceValues.jumpTest.zStatistic, ...
                        jumpTestResults.zStatistic, 1e-8);
                    
                    results.accuracyTests.basicTest = struct(...
                        'passed', compResult.isEqual, ...
                        'referenceValue', obj.referenceValues.jumpTest.zStatistic, ...
                        'computedValue', jumpTestResults.zStatistic, ...
                        'absoluteDifference', compResult.maxAbsoluteDifference, ...
                        'notes', 'Validates basic jump test statistic computation' ...
                    );
                else
                    % If no reference value, store computed value for reference
                    results.accuracyTests.basicTest = struct(...
                        'passed', true, ...
                        'referenceValue', 'Not available', ...
                        'computedValue', jumpTestResults.zStatistic, ...
                        'notes', 'No reference value available for comparison' ...
                    );
                end
                
                % Test 2: RV-BV relationship in jump test
                % The ratio of RV/BV should match the ratio returned by jump_test
                myRV = rv_compute(obj.highFrequencyReturns);
                myBV = bv_compute(obj.highFrequencyReturns);
                myRatio = myRV ./ myBV;
                
                compResult = obj.comparator.compareMatrices(myRatio, jumpTestResults.ratio, 1e-10);
                
                results.accuracyTests.rvBvRatio = struct(...
                    'passed', compResult.isEqual, ...
                    'expectedRatio', myRatio, ...
                    'computedRatio', jumpTestResults.ratio, ...
                    'absoluteDifference', compResult.maxAbsoluteDifference, ...
                    'notes', 'Validates RV/BV ratio computation in jump test' ...
                );
                
                % SIZE AND POWER TESTS
                % -------------------
                
                % Test 3: Generate data with and without jumps to test detection power
                
                % No-jump data
                noJumpOptions = struct('includeJumps', false, 'microstructureNoise', false);
                noJumpData = obj.generateSimulatedProcess(0.2, 1, 288, noJumpOptions);
                noJumpResults = jump_test(noJumpData.returns);
                
                % Jump data with high jumps
                highJumpOptions = struct('includeJumps', true, 'microstructureNoise', false, ...
                    'jumpIntensity', 3, 'jumpSize', 5);
                highJumpData = obj.generateSimulatedProcess(0.2, 1, 288, highJumpOptions);
                highJumpResults = jump_test(highJumpData.returns);
                
                % Jump data with small jumps
                smallJumpOptions = struct('includeJumps', true, 'microstructureNoise', false, ...
                    'jumpIntensity', 2, 'jumpSize', 2);
                smallJumpData = obj.generateSimulatedProcess(0.2, 1, 288, smallJumpOptions);
                smallJumpResults = jump_test(smallJumpData.returns);
                
                % SIZE: False positive rate at 5% level for no-jump data
                % should be approximately 5% (within reasonable bounds)
                falsePositiveRate = mean(noJumpResults.jumpDetected(2,:));
                
                results.sizeAndPowerTests.size = struct(...
                    'passed', falsePositiveRate < 0.15, ... % Allow up to 15% false positive rate
                    'expectedFalsePositiveRate', 0.05, ...
                    'actualFalsePositiveRate', falsePositiveRate, ...
                    'notes', 'Validates false positive rate (size) of the test' ...
                );
                
                % POWER: True positive rate for high-jump data should be high
                truePositiveRateHigh = mean(highJumpResults.jumpDetected(2,:));
                
                results.sizeAndPowerTests.powerHighJumps = struct(...
                    'passed', truePositiveRateHigh > 0.7, ... % Power should be at least 70%
                    'expectedTruePositiveRate', 'High (>70%)', ...
                    'actualTruePositiveRate', truePositiveRateHigh, ...
                    'notes', 'Validates detection power for large jumps' ...
                );
                
                % POWER: True positive rate for small-jump data should be moderate
                truePositiveRateSmall = mean(smallJumpResults.jumpDetected(2,:));
                
                results.sizeAndPowerTests.powerSmallJumps = struct(...
                    'passed', truePositiveRateSmall > 0.3, ... % Power should be at least 30%
                    'expectedTruePositiveRate', 'Moderate (>30%)', ...
                    'actualTruePositiveRate', truePositiveRateSmall, ...
                    'notes', 'Validates detection power for small jumps' ...
                );
                
                % DECOMPOSITION TESTS
                % ------------------
                
                % Test 4: Verify decomposition into continuous and jump components
                
                % The sum of continuous and jump components should equal RV
                sumComponents = jumpTestResults.contComponent + jumpTestResults.jumpComponent .* jumpTestResults.rv;
                
                compResult = obj.comparator.compareMatrices(jumpTestResults.rv, sumComponents, 1e-10);
                
                results.decompositionTests.componentSum = struct(...
                    'passed', compResult.isEqual, ...
                    'rvValue', jumpTestResults.rv, ...
                    'sumComponents', sumComponents, ...
                    'absoluteDifference', compResult.maxAbsoluteDifference, ...
                    'notes', 'Continuous + Jump components should equal RV' ...
                );
                
                % For no-jump data, continuous component should be close to RV
                % and jump component should be close to zero
                contRatio = noJumpResults.contComponent ./ noJumpResults.rv;
                jumpComp = noJumpResults.jumpComponent .* noJumpResults.rv;
                
                results.decompositionTests.noJumpDecomposition = struct(...
                    'passed', all(contRatio > 0.9) && all(jumpComp < 0.1 * noJumpResults.rv), ...
                    'contRatio', contRatio, ...
                    'jumpComponent', jumpComp, ...
                    'notes', 'For no-jump data, continuous component should dominate' ...
                );
                
                % For high-jump data, jump component should be significant
                highJumpComp = highJumpResults.jumpComponent .* highJumpResults.rv;
                
                results.decompositionTests.highJumpDecomposition = struct(...
                    'passed', all(highJumpComp > 0.1 * highJumpResults.rv), ...
                    'contComponent', highJumpResults.contComponent, ...
                    'jumpComponent', highJumpComp, ...
                    'notes', 'For high-jump data, jump component should be significant' ...
                );
                
                % PERFORMANCE TESTS
                % ----------------
                
                % Test 5: Execution time measurement
                executionTime = obj.measureExecutionTime(@() jump_test(obj.highFrequencyReturns));
                
                results.performanceTests.executionTime = struct(...
                    'timeInSeconds', executionTime, ...
                    'returnsProcessed', numel(obj.highFrequencyReturns), ...
                    'returnsPerSecond', numel(obj.highFrequencyReturns) / max(executionTime, 1e-10), ...
                    'notes', 'Basic performance benchmark' ...
                );
                
                % ERROR HANDLING TESTS
                % -------------------
                
                % Test 6: Invalid alpha
                invalidAlphaTestPassed = false;
                try
                    % Attempt to call with invalid alpha
                    badOptions = struct('alpha', 2);  % Alpha should be between 0 and 1
                    jump_test(obj.highFrequencyReturns, badOptions);
                    
                    % If we got here, the test failed
                    invalidAlphaTestPassed = false;
                catch
                    % Error was correctly thrown
                    invalidAlphaTestPassed = true;
                end
                
                results.errorHandlingTests.invalidAlpha = struct(...
                    'passed', invalidAlphaTestPassed, ...
                    'notes', 'Validates error handling for invalid alpha parameter' ...
                );
                
                % Test 7: Invalid inputs
                invalidInputTestPassed = false;
                try
                    % Attempt to call with invalid returns (containing NaN)
                    badReturns = obj.highFrequencyReturns;
                    badReturns(1, 1) = NaN;
                    jump_test(badReturns);
                    
                    % If we got here, the test failed
                    invalidInputTestPassed = false;
                catch
                    % Error was correctly thrown
                    invalidInputTestPassed = true;
                end
                
                results.errorHandlingTests.invalidInputs = struct(...
                    'passed', invalidInputTestPassed, ...
                    'notes', 'Validates error handling for invalid inputs' ...
                );
                
                % Determine overall result
                coreTests = [
                    results.accuracyTests.basicTest.passed,
                    results.accuracyTests.rvBvRatio.passed,
                    results.decompositionTests.componentSum.passed,
                    results.errorHandlingTests.invalidAlpha.passed,
                    results.errorHandlingTests.invalidInputs.passed
                ];
                
                if all(coreTests)
                    results.overallResult = 'passed';
                else
                    results.overallResult = 'failed';
                end
                
                % Display overall result
                disp(['Jump test validation ', results.overallResult]);
                
            catch ME
                % Handle unexpected errors
                results.overallResult = 'error';
                results.errorMessage = ME.message;
                results.errorIdentifier = ME.identifier;
                results.errorStack = ME.stack;
                
                warning(['Error in jump test validation: ' ME.message]);
            end
            
            % Store results in validation results structure
            obj.validationResults.jumpTest = results;
        end
        
        function results = testAsymptoticProperties(obj)
            % Tests the asymptotic properties of realized measures with increasing sample size
            %
            % OUTPUTS:
            %   results - Structure containing asymptotic property test results
            %
            % This function validates that realized measures converge to their theoretical
            % limits at the appropriate rates with increasing sample sizes, confirming:
            %   1. RV's consistency as an estimator of integrated variance
            %   2. BV's robustness to jumps
            %   3. Kernel estimators' noise resistance
            
            disp('Testing asymptotic properties of realized measures...');
            
            % Initialize results structure
            results = struct(...
                'consistencyTests', struct(), ...
                'convergenceRateTests', struct(), ...
                'noiseRobustnessTests', struct(), ...
                'jumpRobustnessTests', struct(), ...
                'overallResult', 'pending' ...
            );
            
            try
                % Define sampling frequencies to test (observations per day)
                samplingFrequencies = [48, 96, 192, 384, 768]; % 30min, 15min, 7.5min, 3.75min, 1.875min
                
                % Simulate a diffusion process with known integrated variance
                baseVolatility = 0.2;  % 20% annual volatility
                timeSpan = 1;          % 1 day
                
                % Set up simulation results storage
                simResults = struct('samplingFrequency', [], 'rvError', [], 'bvError', [], ...
                    'kernelError', [], 'rvWithNoiseError', [], 'kernelWithNoiseError', []);
                
                % For each sampling frequency
                for i = 1:length(samplingFrequencies)
                    freq = samplingFrequencies(i);
                    simResults.samplingFrequency(i) = freq;
                    
                    % 1. Clean diffusion process (no jumps, no noise)
                    cleanOptions = struct('includeJumps', false, 'microstructureNoise', false);
                    cleanData = obj.generateSimulatedProcess(baseVolatility, timeSpan, freq, cleanOptions);
                    
                    % 2. Jump diffusion process (with jumps, no noise)
                    jumpOptions = struct('includeJumps', true, 'microstructureNoise', false, ...
                        'jumpIntensity', 3, 'jumpSize', 4);
                    jumpData = obj.generateSimulatedProcess(baseVolatility, timeSpan, freq, jumpOptions);
                    
                    % 3. Noisy diffusion process (no jumps, with noise)
                    noiseOptions = struct('includeJumps', false, 'microstructureNoise', true, ...
                        'noiseRatio', 0.1);
                    noiseData = obj.generateSimulatedProcess(baseVolatility, timeSpan, freq, noiseOptions);
                    
                    % True IV from the simulation
                    trueIV = cleanData.referenceValues.integratedVariance;
                    
                    % Compute measures for each process
                    rv_clean = rv_compute(cleanData.returns);
                    bv_clean = bv_compute(cleanData.returns);
                    kernel_clean = rv_kernel(cleanData.returns);
                    
                    rv_jump = rv_compute(jumpData.returns);
                    bv_jump = bv_compute(jumpData.returns);
                    
                    rv_noise = rv_compute(noiseData.returns);
                    kernel_noise = rv_kernel(noiseData.returns);
                    
                    % Calculate relative errors
                    simResults.rvError(i) = abs(rv_clean - trueIV) / trueIV;
                    simResults.bvError(i) = abs(bv_clean - trueIV) / trueIV;
                    simResults.kernelError(i) = abs(kernel_clean - trueIV) / trueIV;
                    
                    simResults.rvJumpError(i) = abs(rv_jump - trueIV) / trueIV;
                    simResults.bvJumpError(i) = abs(bv_jump - trueIV) / trueIV;
                    
                    simResults.rvNoiseError(i) = abs(rv_noise - trueIV) / trueIV;
                    simResults.kernelNoiseError(i) = abs(kernel_noise - trueIV) / trueIV;
                end
                
                % CONSISTENCY TESTS
                % ----------------
                
                % Test 1: RV should converge to IV as sampling frequency increases (clean case)
                % Errors should decrease as sampling frequency increases
                rvErrorDecreasing = all(diff(simResults.rvError) < 0);
                finalRvError = simResults.rvError(end);
                
                results.consistencyTests.rvConsistency = struct(...
                    'passed', rvErrorDecreasing && finalRvError < 0.1, ... % Final error < 10%
                    'samplingFrequencies', simResults.samplingFrequency, ...
                    'errors', simResults.rvError, ...
                    'errorDecreasing', rvErrorDecreasing, ...
                    'finalError', finalRvError, ...
                    'notes', 'RV should converge to IV with increasing sampling frequency' ...
                );
                
                % Test 2: BV should also converge to IV as sampling frequency increases (clean case)
                bvErrorDecreasing = all(diff(simResults.bvError) < 0);
                finalBvError = simResults.bvError(end);
                
                results.consistencyTests.bvConsistency = struct(...
                    'passed', bvErrorDecreasing && finalBvError < 0.15, ... % Allow slightly higher error for BV
                    'samplingFrequencies', simResults.samplingFrequency, ...
                    'errors', simResults.bvError, ...
                    'errorDecreasing', bvErrorDecreasing, ...
                    'finalError', finalBvError, ...
                    'notes', 'BV should converge to IV with increasing sampling frequency' ...
                );
                
                % Test 3: Kernel RV should converge to IV as sampling frequency increases (clean case)
                kernelErrorDecreasing = all(diff(simResults.kernelError) < 0);
                finalKernelError = simResults.kernelError(end);
                
                results.consistencyTests.kernelConsistency = struct(...
                    'passed', kernelErrorDecreasing && finalKernelError < 0.1, ... % Final error < 10%
                    'samplingFrequencies', simResults.samplingFrequency, ...
                    'errors', simResults.kernelError, ...
                    'errorDecreasing', kernelErrorDecreasing, ...
                    'finalError', finalKernelError, ...
                    'notes', 'Kernel RV should converge to IV with increasing frequency' ...
                );
                
                % CONVERGENCE RATE TESTS
                % ---------------------
                
                % Test 4: RV convergence rate should be approximately n^(-1/2)
                % Log-log regression of error vs. frequency should have slope close to -0.5
                logFreq = log(simResults.samplingFrequency);
                logRvError = log(simResults.rvError);
                rvSlope = (length(logFreq)*sum(logFreq.*logRvError) - sum(logFreq)*sum(logRvError)) / ...
                    (length(logFreq)*sum(logFreq.^2) - sum(logFreq)^2);
                
                results.convergenceRateTests.rvConvergenceRate = struct(...
                    'passed', abs(rvSlope + 0.5) < 0.2, ... % Slope should be close to -0.5
                    'expectedSlope', -0.5, ...
                    'actualSlope', rvSlope, ...
                    'notes', 'RV should converge at rate n^(-1/2)' ...
                );
                
                % JUMP ROBUSTNESS TESTS
                % -------------------
                
                % Test 5: RV should be affected by jumps
                % RV error should be higher with jumps than without
                rvJumpRobustness = mean(simResults.rvJumpError) > mean(simResults.rvError);
                
                results.jumpRobustnessTests.rvJumpSensitivity = struct(...
                    'passed', rvJumpRobustness, ...
                    'cleanErrors', simResults.rvError, ...
                    'jumpErrors', simResults.rvJumpError, ...
                    'notes', 'RV should be affected by jumps (higher error with jumps)' ...
                );
                
                % Test 6: BV should be more robust to jumps than RV
                % BV error should be lower than RV error in presence of jumps
                bvJumpRobustness = mean(simResults.bvJumpError) < mean(simResults.rvJumpError);
                
                results.jumpRobustnessTests.bvJumpRobustness = struct(...
                    'passed', bvJumpRobustness, ...
                    'rvErrors', simResults.rvJumpError, ...
                    'bvErrors', simResults.bvJumpError, ...
                    'notes', 'BV should be more robust to jumps than RV' ...
                );
                
                % NOISE ROBUSTNESS TESTS
                % --------------------
                
                % Test 7: RV should be affected by microstructure noise
                % RV error should be higher with noise than without
                rvNoiseSensitivity = mean(simResults.rvNoiseError) > mean(simResults.rvError);
                
                results.noiseRobustnessTests.rvNoiseSensitivity = struct(...
                    'passed', rvNoiseSensitivity, ...
                    'cleanErrors', simResults.rvError, ...
                    'noiseErrors', simResults.rvNoiseError, ...
                    'notes', 'RV should be affected by microstructure noise' ...
                );
                
                % Test 8: Kernel RV should be more robust to noise than standard RV
                % Kernel error should be lower than RV error in presence of noise
                kernelNoiseRobustness = mean(simResults.kernelNoiseError) < mean(simResults.rvNoiseError);
                
                results.noiseRobustnessTests.kernelNoiseRobustness = struct(...
                    'passed', kernelNoiseRobustness, ...
                    'rvErrors', simResults.rvNoiseError, ...
                    'kernelErrors', simResults.kernelNoiseError, ...
                    'notes', 'Kernel RV should be more robust to noise than standard RV' ...
                );
                
                % Determine overall result
                coreTests = [
                    results.consistencyTests.rvConsistency.passed,
                    results.consistencyTests.bvConsistency.passed,
                    results.consistencyTests.kernelConsistency.passed,
                    results.jumpRobustnessTests.bvJumpRobustness.passed,
                    results.noiseRobustnessTests.kernelNoiseRobustness.passed
                ];
                
                if all(coreTests)
                    results.overallResult = 'passed';
                else
                    results.overallResult = 'failed';
                end
                
                % Display overall result
                disp(['Asymptotic properties testing ', results.overallResult]);
                
            catch ME
                % Handle unexpected errors
                results.overallResult = 'error';
                results.errorMessage = ME.message;
                results.errorIdentifier = ME.identifier;
                results.errorStack = ME.stack;
                
                warning(['Error in asymptotic properties testing: ' ME.message]);
            end
            
            % Store results in validation results structure
            obj.validationResults.asymptoticProperties = results;
        end
        
        function results = testNumericalStability(obj)
            % Tests numerical stability of realized measures under extreme conditions
            %
            % OUTPUTS:
            %   results - Structure containing numerical stability test results
            %
            % This function validates the numerical robustness of realized measures by:
            %   1. Testing with extreme volatility levels
            %   2. Testing with very sparse and dense sampling
            %   3. Testing with outliers and extreme price jumps
            %   4. Checking behavior with near-zero values
            
            disp('Testing numerical stability of realized measures...');
            
            % Initialize results structure
            results = struct(...
                'extremeVolatilityTests', struct(), ...
                'extremeSamplingTests', struct(), ...
                'outlierTests', struct(), ...
                'nearZeroTests', struct(), ...
                'overallResult', 'pending' ...
            );
            
            try
                % EXTREME VOLATILITY TESTS
                % -----------------------
                
                % Test 1: Very high volatility
                highVolOptions = struct('includeJumps', false, 'microstructureNoise', false);
                highVolData = obj.generateSimulatedProcess(2.0, 1, 288, highVolOptions); % 200% volatility
                
                highVolRV = rv_compute(highVolData.returns);
                highVolBV = bv_compute(highVolData.returns);
                highVolKernel = rv_kernel(highVolData.returns);
                highVolJump = jump_test(highVolData.returns);
                
                % All measures should produce finite, positive values
                highVolStability = struct(...
                    'rvFinite', all(isfinite(highVolRV)) && all(highVolRV > 0), ...
                    'bvFinite', all(isfinite(highVolBV)) && all(highVolBV > 0), ...
                    'kernelFinite', all(isfinite(highVolKernel)) && all(highVolKernel > 0), ...
                    'jumpFinite', all(isfinite(highVolJump.zStatistic)) ...
                );
                
                results.extremeVolatilityTests.highVolatility = struct(...
                    'passed', all(struct2array(highVolStability)), ...
                    'stability', highVolStability, ...
                    'rvValue', highVolRV, ...
                    'bvValue', highVolBV, ...
                    'kernelValue', highVolKernel, ...
                    'jumpStat', highVolJump.zStatistic, ...
                    'notes', 'All measures should be finite and positive with high volatility' ...
                );
                
                % Test 2: Very low volatility
                lowVolOptions = struct('includeJumps', false, 'microstructureNoise', false);
                lowVolData = obj.generateSimulatedProcess(0.01, 1, 288, lowVolOptions); % 1% volatility
                
                lowVolRV = rv_compute(lowVolData.returns);
                lowVolBV = bv_compute(lowVolData.returns);
                lowVolKernel = rv_kernel(lowVolData.returns);
                lowVolJump = jump_test(lowVolData.returns);
                
                % All measures should produce finite, non-negative values
                lowVolStability = struct(...
                    'rvFinite', all(isfinite(lowVolRV)) && all(lowVolRV >= 0), ...
                    'bvFinite', all(isfinite(lowVolBV)) && all(lowVolBV >= 0), ...
                    'kernelFinite', all(isfinite(lowVolKernel)) && all(lowVolKernel >= 0), ...
                    'jumpFinite', all(isfinite(lowVolJump.zStatistic)) ...
                );
                
                results.extremeVolatilityTests.lowVolatility = struct(...
                    'passed', all(struct2array(lowVolStability)), ...
                    'stability', lowVolStability, ...
                    'rvValue', lowVolRV, ...
                    'bvValue', lowVolBV, ...
                    'kernelValue', lowVolKernel, ...
                    'jumpStat', lowVolJump.zStatistic, ...
                    'notes', 'All measures should be finite and non-negative with low volatility' ...
                );
                
                % EXTREME SAMPLING TESTS
                % ---------------------
                
                % Test 3: Very sparse sampling
                sparseOptions = struct('includeJumps', false, 'microstructureNoise', false);
                sparseData = obj.generateSimulatedProcess(0.2, 1, 24, sparseOptions); % Daily/hourly data
                
                sparseRV = rv_compute(sparseData.returns);
                sparseBV = bv_compute(sparseData.returns);
                sparseKernel = rv_kernel(sparseData.returns);
                sparseJump = jump_test(sparseData.returns);
                
                % All measures should produce finite values
                sparseStability = struct(...
                    'rvFinite', all(isfinite(sparseRV)), ...
                    'bvFinite', all(isfinite(sparseBV)), ...
                    'kernelFinite', all(isfinite(sparseKernel)), ...
                    'jumpFinite', all(isfinite(sparseJump.zStatistic)) ...
                );
                
                results.extremeSamplingTests.sparseSampling = struct(...
                    'passed', all(struct2array(sparseStability)), ...
                    'stability', sparseStability, ...
                    'rvValue', sparseRV, ...
                    'bvValue', sparseBV, ...
                    'kernelValue', sparseKernel, ...
                    'jumpStat', sparseJump.zStatistic, ...
                    'notes', 'All measures should handle sparse sampling' ...
                );
                
                % Test 4: Very dense sampling (less dense than tick data)
                denseOptions = struct('includeJumps', false, 'microstructureNoise', false);
                denseData = obj.generateSimulatedProcess(0.2, 1, 1440, denseOptions); % Minute data
                
                denseRV = rv_compute(denseData.returns);
                denseBV = bv_compute(denseData.returns);
                denseKernel = rv_kernel(denseData.returns);
                denseJump = jump_test(denseData.returns);
                
                % All measures should produce finite values
                denseStability = struct(...
                    'rvFinite', all(isfinite(denseRV)), ...
                    'bvFinite', all(isfinite(denseBV)), ...
                    'kernelFinite', all(isfinite(denseKernel)), ...
                    'jumpFinite', all(isfinite(denseJump.zStatistic)) ...
                );
                
                results.extremeSamplingTests.denseSampling = struct(...
                    'passed', all(struct2array(denseStability)), ...
                    'stability', denseStability, ...
                    'rvValue', denseRV, ...
                    'bvValue', denseBV, ...
                    'kernelValue', denseKernel, ...
                    'jumpStat', denseJump.zStatistic, ...
                    'notes', 'All measures should handle dense sampling' ...
                );
                
                % OUTLIER TESTS
                % -------------
                
                % Test 5: Extreme price jumps
                extremeJumpOptions = struct('includeJumps', true, 'microstructureNoise', false, ...
                    'jumpIntensity', 1, 'jumpSize', 10); % Few but very large jumps
                extremeJumpData = obj.generateSimulatedProcess(0.2, 1, 288, extremeJumpOptions);
                
                % Add one extreme outlier (10-sigma event)
                extremeJumpData.returns(144, :) = extremeJumpData.returns(144, :) + 10 * 0.2 / sqrt(288);
                
                extremeJumpRV = rv_compute(extremeJumpData.returns);
                extremeJumpBV = bv_compute(extremeJumpData.returns);
                extremeJumpKernel = rv_kernel(extremeJumpData.returns);
                extremeJumpTest = jump_test(extremeJumpData.returns);
                
                % All measures should produce finite values
                extremeJumpStability = struct(...
                    'rvFinite', all(isfinite(extremeJumpRV)), ...
                    'bvFinite', all(isfinite(extremeJumpBV)), ...
                    'kernelFinite', all(isfinite(extremeJumpKernel)), ...
                    'jumpFinite', all(isfinite(extremeJumpTest.zStatistic)), ...
                    'jumpDetected', any(extremeJumpTest.jumpDetected(2,:)) ... % Should detect the jump
                );
                
                results.outlierTests.extremeJumps = struct(...
                    'passed', all(struct2array(extremeJumpStability)), ...
                    'stability', extremeJumpStability, ...
                    'rvValue', extremeJumpRV, ...
                    'bvValue', extremeJumpBV, ...
                    'kernelValue', extremeJumpKernel, ...
                    'jumpStat', extremeJumpTest.zStatistic, ...
                    'notes', 'All measures should handle extreme price jumps' ...
                );
                
                % NEAR-ZERO TESTS
                % --------------
                
                % Test 6: Near-zero returns
                % Generate very low volatility returns, then scale down further
                tinyReturnsData = obj.generateSimulatedProcess(0.001, 1, 288, struct());
                tinyReturns = tinyReturnsData.returns * 0.001;  % Scale down further
                
                % Ensure some returns are exactly zero for testing edge cases
                tinyReturns(1:10:end) = 0;
                
                tinyRV = rv_compute(tinyReturns);
                tinyBV = bv_compute(tinyReturns);
                tinyKernel = rv_kernel(tinyReturns);
                tinyJump = jump_test(tinyReturns);
                
                % All measures should produce finite values
                tinyStability = struct(...
                    'rvFinite', all(isfinite(tinyRV)), ...
                    'bvFinite', all(isfinite(tinyBV)), ...
                    'kernelFinite', all(isfinite(tinyKernel)), ...
                    'jumpFinite', all(isfinite(tinyJump.zStatistic)) ...
                );
                
                results.nearZeroTests.tinyReturns = struct(...
                    'passed', all(struct2array(tinyStability)), ...
                    'stability', tinyStability, ...
                    'rvValue', tinyRV, ...
                    'bvValue', tinyBV, ...
                    'kernelValue', tinyKernel, ...
                    'jumpStat', tinyJump.zStatistic, ...
                    'notes', 'All measures should handle near-zero returns' ...
                );
                
                % Determine overall result
                coreTests = [
                    results.extremeVolatilityTests.highVolatility.passed,
                    results.extremeVolatilityTests.lowVolatility.passed,
                    results.extremeSamplingTests.sparseSampling.passed,
                    results.extremeSamplingTests.denseSampling.passed,
                    results.outlierTests.extremeJumps.passed,
                    results.nearZeroTests.tinyReturns.passed
                ];
                
                if all(coreTests)
                    results.overallResult = 'passed';
                else
                    results.overallResult = 'failed';
                end
                
                % Display overall result
                disp(['Numerical stability testing ', results.overallResult]);
                
            catch ME
                % Handle unexpected errors
                results.overallResult = 'error';
                results.errorMessage = ME.message;
                results.errorIdentifier = ME.identifier;
                results.errorStack = ME.stack;
                
                warning(['Error in numerical stability testing: ' ME.message]);
            end
            
            % Store results in validation results structure
            obj.validationResults.numericalStability = results;
        end
        
        function simData = generateSimulatedProcess(obj, volatility, timeSpan, samplingFrequency, options)
            % Generates simulated diffusion process with controlled properties for validation
            %
            % INPUTS:
            %   volatility - Base volatility level (annualized)
            %   timeSpan - Time span in days
            %   samplingFrequency - Number of observations per day
            %   options - Structure with simulation options:
            %       .includeJumps - Whether to include price jumps [default: false]
            %       .jumpIntensity - Average number of jumps per day [default: 3]
            %       .jumpSize - Average jump size in volatility units [default: 3]
            %       .microstructureNoise - Whether to add noise [default: false]
            %       .noiseRatio - Ratio of noise to return std [default: 0.1]
            %
            % OUTPUTS:
            %   simData - Structure containing:
            %       .prices - Simulated price path
            %       .returns - Simulated returns
            %       .timestamps - Time points
            %       .jumpTimes - Times of jumps (if included)
            %       .jumpSizes - Sizes of jumps (if included)
            %       .referenceValues - Known true values for validation
            
            % Set default options if not provided
            if nargin < 5 || isempty(options)
                options = struct();
            end
            
            % Default values for options
            if ~isfield(options, 'includeJumps')
                options.includeJumps = false;
            end
            
            if ~isfield(options, 'jumpIntensity')
                options.jumpIntensity = 3;  % jumps per day on average
            end
            
            if ~isfield(options, 'jumpSize')
                options.jumpSize = 3;  % in volatility units
            end
            
            if ~isfield(options, 'microstructureNoise')
                options.microstructureNoise = false;
            end
            
            if ~isfield(options, 'noiseRatio')
                options.noiseRatio = 0.1;  % 10% noise-to-signal ratio
            end
            
            % Calculate number of observations
            numObs = round(timeSpan * samplingFrequency);
            
            % Time increments
            dt = timeSpan / numObs;
            timePoints = (0:numObs) * dt;
            
            % Annualized volatility to per-period volatility
            periodVol = volatility / sqrt(252 * samplingFrequency);
            
            % Initialize price path with log-normal diffusion
            prices = zeros(numObs + 1, 1);
            prices(1) = 100;  % Initial price
            
            % Generate diffusion component
            diffusion = exp(cumsum(randn(numObs, 1) * periodVol - 0.5 * periodVol^2));
            prices(2:end) = prices(1) * diffusion;
            
            % Storage for jump information
            jumpTimes = [];
            jumpSizes = [];
            
            % Add jumps if requested
            if options.includeJumps
                % Poisson process for jump times
                jumpProb = options.jumpIntensity / samplingFrequency;
                jumpIndicator = rand(numObs, 1) < jumpProb;
                jumpIndices = find(jumpIndicator);
                
                if ~isempty(jumpIndices)
                    % Generate jump sizes
                    jumpTimes = timePoints(jumpIndices + 1);
                    jumpSizes = options.jumpSize * periodVol * randn(length(jumpIndices), 1);
                    
                    % Apply jumps to price path
                    for i = 1:length(jumpIndices)
                        idx = jumpIndices(i) + 1;
                        jumpSize = jumpSizes(i);
                        prices(idx:end) = prices(idx:end) * exp(jumpSize);
                    end
                end
            end
            
            % Calculate returns
            returns = diff(log(prices));
            
            % Add microstructure noise if requested
            if options.microstructureNoise
                % Calculate signal standard deviation
                signalStd = std(returns);
                
                % Generate noise with specified ratio to signal
                noiseStd = signalStd * options.noiseRatio;
                noise = randn(numObs, 1) * noiseStd;
                
                % Add noise to returns
                returns = returns + noise;
            end
            
            % Calculate integrated variance (daily)
            integratedVariance = volatility^2 / 252;
            
            % Calculate expected jump variation if jumps are included
            jumpVariation = 0;
            if options.includeJumps && ~isempty(jumpSizes)
                jumpVariation = sum(jumpSizes.^2);
            end
            
            % Calculate reference values for validation
            referenceValues = struct(...
                'integratedVariance', integratedVariance, ...
                'jumpVariation', jumpVariation, ...
                'totalVariation', integratedVariance + jumpVariation ...
            );
            
            % Return simulation data
            simData = struct(...
                'prices', prices, ...
                'returns', returns, ...
                'timestamps', timePoints(2:end)', ...
                'jumpTimes', jumpTimes, ...
                'jumpSizes', jumpSizes, ...
                'referenceValues', referenceValues, ...
                'simulationOptions', options ...
            );
        end
        
        function results = runAllValidations(obj)
            % Runs all validation tests for realized measures components
            %
            % OUTPUTS:
            %   results - Comprehensive validation results structure
            %
            % This is the main entry point for running the full validation test suite.
            % It executes all individual validation tests and compiles the results.
            
            disp('Starting comprehensive validation of realized measures components...');
            
            try
                % Run individual validation tests
                rvResults = obj.validateRealizedVolatility();
                bvResults = obj.validateBipowerVariation();
                kernelResults = obj.validateKernelEstimation();
                jumpResults = obj.validateJumpTest();
                
                % Run theoretical property tests
                asymptoticResults = obj.testAsymptoticProperties();
                stabilityResults = obj.testNumericalStability();
                
                % Compile all results
                results = struct(...
                    'realizedVolatility', rvResults, ...
                    'bipowerVariation', bvResults, ...
                    'kernelEstimation', kernelResults, ...
                    'jumpTest', jumpResults, ...
                    'asymptoticProperties', asymptoticResults, ...
                    'numericalStability', stabilityResults, ...
                    'overallSummary', struct(), ...
                    'validationTime', datetime('now') ...
                );
                
                % Determine overall validation status
                componentResults = {
                    rvResults.overallResult,
                    bvResults.overallResult,
                    kernelResults.overallResult,
                    jumpResults.overallResult,
                    asymptoticResults.overallResult,
                    stabilityResults.overallResult
                };
                
                passedCount = sum(strcmp(componentResults, 'passed'));
                failedCount = sum(strcmp(componentResults, 'failed'));
                errorCount = sum(strcmp(componentResults, 'error'));
                
                % Overall validation summary
                results.overallSummary = struct(...
                    'componentsValidated', length(componentResults), ...
                    'componentsPassed', passedCount, ...
                    'componentsFailed', failedCount, ...
                    'componentsError', errorCount, ...
                    'successRate', passedCount / length(componentResults), ...
                    'overallStatus', '' ...
                );
                
                % Set overall status based on results
                if errorCount > 0
                    results.overallSummary.overallStatus = 'ERROR';
                elseif failedCount > 0
                    results.overallSummary.overallStatus = 'FAILED';
                else
                    results.overallSummary.overallStatus = 'PASSED';
                end
                
                % Store complete results in validation results
                obj.validationResults = results;
                
                % Display overall validation summary
                disp('===== Realized Measures Validation Summary =====');
                disp(['Components validated: ' num2str(length(componentResults))]);
                disp(['Components passed: ' num2str(passedCount) ' (' ...
                    num2str(100 * passedCount / length(componentResults), '%.1f') '%)']);
                disp(['Overall status: ' results.overallSummary.overallStatus]);
                disp('===============================================');
                
            catch ME
                % Handle unexpected errors in validation process
                results = struct(...
                    'overallSummary', struct(...
                        'overallStatus', 'ERROR', ...
                        'errorMessage', ME.message, ...
                        'errorIdentifier', ME.identifier, ...
                        'errorStack', ME.stack ...
                    ), ...
                    'validationTime', datetime('now') ...
                );
                
                warning(['Error in validation process: ' ME.message]);
            end
        end
        
        function summary = getSummaryReport(obj)
            % Generates comprehensive summary report of validation results
            %
            % OUTPUTS:
            %   summary - Structure containing summary of all validation tests
            %
            % This function compiles validation results into a concise report format,
            % highlighting key metrics, issues, and overall validation status.
            
            % Initialize summary structure
            summary = struct(...
                'overallStatus', 'Unknown', ...
                'componentsValidated', 0, ...
                'componentsPassed', 0, ...
                'componentsFailed', 0, ...
                'componentsWithErrors', 0, ...
                'successRate', 0, ...
                'accuracyMetrics', struct(), ...
                'performanceMetrics', struct(), ...
                'issuesIdentified', struct(), ...
                'recommendations', {}, ...
                'validationTime', datetime('now') ...
            );
            
            % Check if validationResults is populated
            if ~isstruct(obj.validationResults) || isempty(fieldnames(obj.validationResults))
                summary.overallStatus = 'NOT_RUN';
                summary.recommendations{end+1} = 'Run validation tests using runAllValidations() before generating report.';
                return;
            end
            
            % If results include overallSummary field, use that information
            if isfield(obj.validationResults, 'overallSummary')
                summary.overallStatus = obj.validationResults.overallSummary.overallStatus;
                summary.componentsValidated = obj.validationResults.overallSummary.componentsValidated;
                summary.componentsPassed = obj.validationResults.overallSummary.componentsPassed;
                summary.componentsFailed = obj.validationResults.overallSummary.componentsFailed;
                summary.componentsWithErrors = obj.validationResults.overallSummary.componentsError;
                summary.successRate = obj.validationResults.overallSummary.successRate;
            else
                % Manual calculation if overallSummary not available
                components = {'realizedVolatility', 'bipowerVariation', 'kernelEstimation', ...
                    'jumpTest', 'asymptoticProperties', 'numericalStability'};
                
                validComponents = 0;
                passedComponents = 0;
                failedComponents = 0;
                errorComponents = 0;
                
                for i = 1:length(components)
                    if isfield(obj.validationResults, components{i})
                        validComponents = validComponents + 1;
                        
                        if isfield(obj.validationResults.(components{i}), 'overallResult')
                            result = obj.validationResults.(components{i}).overallResult;
                            
                            if strcmp(result, 'passed')
                                passedComponents = passedComponents + 1;
                            elseif strcmp(result, 'failed')
                                failedComponents = failedComponents + 1;
                            elseif strcmp(result, 'error')
                                errorComponents = errorComponents + 1;
                            end
                        end
                    end
                end
                
                summary.componentsValidated = validComponents;
                summary.componentsPassed = passedComponents;
                summary.componentsFailed = failedComponents;
                summary.componentsWithErrors = errorComponents;
                summary.successRate = passedComponents / max(validComponents, 1);
                
                % Set overall status
                if errorComponents > 0
                    summary.overallStatus = 'ERROR';
                elseif failedComponents > 0
                    summary.overallStatus = 'FAILED';
                elseif passedComponents == validComponents && validComponents > 0
                    summary.overallStatus = 'PASSED';
                else
                    summary.overallStatus = 'UNKNOWN';
                end
            end
            
            % Compile accuracy metrics across components
            accuracyMetrics = struct();
            
            % Realized Volatility accuracy
            if isfield(obj.validationResults, 'realizedVolatility') && ...
                    isfield(obj.validationResults.realizedVolatility, 'accuracyTests')
                
                rvAcc = obj.validationResults.realizedVolatility.accuracyTests;
                
                if isfield(rvAcc, 'basicComputation') && isfield(rvAcc.basicComputation, 'absoluteDifference')
                    accuracyMetrics.realizedVolatility = rvAcc.basicComputation.absoluteDifference;
                end
            end
            
            % Bipower Variation accuracy
            if isfield(obj.validationResults, 'bipowerVariation') && ...
                    isfield(obj.validationResults.bipowerVariation, 'accuracyTests')
                
                bvAcc = obj.validationResults.bipowerVariation.accuracyTests;
                
                if isfield(bvAcc, 'basicComputation') && isfield(bvAcc.basicComputation, 'absoluteDifference')
                    accuracyMetrics.bipowerVariation = bvAcc.basicComputation.absoluteDifference;
                end
            end
            
            % Kernel accuracy
            if isfield(obj.validationResults, 'kernelEstimation') && ...
                    isfield(obj.validationResults.kernelEstimation, 'accuracyTests')
                
                kernelAcc = obj.validationResults.kernelEstimation.accuracyTests;
                
                if isfield(kernelAcc, 'defaultKernel') && isfield(kernelAcc.defaultKernel, 'absoluteDifference')
                    accuracyMetrics.kernelEstimation = kernelAcc.defaultKernel.absoluteDifference;
                end
            end
            
            % Jump test accuracy
            if isfield(obj.validationResults, 'jumpTest') && ...
                    isfield(obj.validationResults.jumpTest, 'accuracyTests')
                
                jumpAcc = obj.validationResults.jumpTest.accuracyTests;
                
                if isfield(jumpAcc, 'basicTest') && isfield(jumpAcc.basicTest, 'absoluteDifference')
                    accuracyMetrics.jumpTest = jumpAcc.basicTest.absoluteDifference;
                end
            end
            
            summary.accuracyMetrics = accuracyMetrics;
            
            % Compile performance metrics
            performanceMetrics = struct();
            
            % Extract execution times from each component
            components = {'realizedVolatility', 'bipowerVariation', 'kernelEstimation', 'jumpTest'};
            
            for i = 1:length(components)
                if isfield(obj.validationResults, components{i}) && ...
                        isfield(obj.validationResults.(components{i}), 'performanceTests') && ...
                        isfield(obj.validationResults.(components{i}).performanceTests, 'executionTime')
                    
                    perfTest = obj.validationResults.(components{i}).performanceTests.executionTime;
                    
                    if isfield(perfTest, 'timeInSeconds')
                        performanceMetrics.(components{i}) = struct(...
                            'executionTime', perfTest.timeInSeconds ...
                        );
                        
                        if isfield(perfTest, 'returnsPerSecond')
                            performanceMetrics.(components{i}).returnsPerSecond = perfTest.returnsPerSecond;
                        end
                    end
                end
            end
            
            summary.performanceMetrics = performanceMetrics;
            
            % Identify issues
            issuesIdentified = struct();
            
            % Check each component for failed tests
            for i = 1:length(components)
                component = components{i};
                
                if isfield(obj.validationResults, component)
                    compResults = obj.validationResults.(component);
                    compIssues = {};
                    
                    % Check if component has overall failure
                    if isfield(compResults, 'overallResult') && ~strcmp(compResults.overallResult, 'passed')
                        % Find specific failed tests
                        testTypes = {'accuracyTests', 'robustnessTests', 'performanceTests', ...
                            'errorHandlingTests', 'jumpRobustnessTests', 'noiseRobustnessTests', ...
                            'sizeAndPowerTests', 'decompositionTests', 'scalingTests', 'kernelTests'};
                        
                        for j = 1:length(testTypes)
                            testType = testTypes{j};
                            
                            if isfield(compResults, testType)
                                testResults = compResults.(testType);
                                testFields = fieldnames(testResults);
                                
                                for k = 1:length(testFields)
                                    field = testFields{k};
                                    
                                    % Check for test failure
                                    if isfield(testResults.(field), 'passed') && ...
                                            ~testResults.(field).passed
                                        
                                        if isfield(testResults.(field), 'notes')
                                            compIssues{end+1} = [testType '.' field ': ' ...
                                                testResults.(field).notes];
                                        else
                                            compIssues{end+1} = [testType '.' field];
                                        end
                                    end
                                end
                            end
                        end
                    end
                    
                    % Add issues to summary if any were found
                    if ~isempty(compIssues)
                        issuesIdentified.(component) = compIssues;
                    end
                end
            end
            
            summary.issuesIdentified = issuesIdentified;
            
            % Generate recommendations based on issues
            recommendations = {};
            
            % Check if any components failed
            if summary.componentsFailed > 0 || summary.componentsWithErrors > 0
                recommendations{end+1} = 'Review failed validation tests and component errors.';
            end
            
            % Check for specific issues and provide recommendations
            if isfield(issuesIdentified, 'realizedVolatility')
                recommendations{end+1} = 'Review realized volatility implementation for accuracy issues.';
            end
            
            if isfield(issuesIdentified, 'bipowerVariation')
                recommendations{end+1} = 'Check bipower variation implementation for correct scaling factor and jump robustness.';
            end
            
            if isfield(issuesIdentified, 'kernelEstimation')
                recommendations{end+1} = 'Verify kernel-based estimator implementation for noise robustness.';
            end
            
            if isfield(issuesIdentified, 'jumpTest')
                recommendations{end+1} = 'Examine jump test implementation for statistical properties and decomposition.';
            end
            
            if isfield(issuesIdentified, 'numericalStability')
                recommendations{end+1} = 'Address numerical stability issues with extreme values or unusual market conditions.';
            end
            
            % If no specific issues were found but tests failed, provide general recommendation
            if isempty(recommendations) && summary.componentsFailed > 0
                recommendations{end+1} = 'Run specific component validations to identify detailed issues.';
            end
            
            % If all passed, provide positive recommendation
            if strcmp(summary.overallStatus, 'PASSED')
                recommendations{end+1} = 'All validation tests passed. Implementation is ready for production use.';
            end
            
            summary.recommendations = recommendations;
            summary.validationTime = datetime('now');
        end
    end
end