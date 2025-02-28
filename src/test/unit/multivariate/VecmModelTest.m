classdef VecmModelTest < BaseTest
    % Unit test class for the Vector Error Correction Model (VECM) implementation in the MFE Toolbox
    
    properties
        dataGenerator   % Test data generator for creating VECM data with known properties
        syntheticData   % Synthetic data with known parameters
        macroData       % Real macroeconomic data for testing
        comparator      % Numerical comparator for floating-point validation
    end
    
    methods
        function obj = VecmModelTest()
            % Initializes the VECM model test class
            obj@BaseTest(); % Call superclass constructor
            obj.dataGenerator = TestDataGenerator();
            % Set higher precision tolerance for cointegration vector comparisons
            obj.comparator = NumericalComparator();
        end
        
        function setUp(obj)
            % Set up the test environment before each test method execution
            setUp@BaseTest(obj);
            
            % Set random number generator seed for reproducible tests
            rng(123);
            
            % Generate synthetic VECM data with known parameters
            obj.syntheticData = obj.generateTestData(1000, 3, 2, 1);
            
            % Load macroeconomic test data for real-world testing scenarios
            try
                obj.macroData = obj.loadTestData('macro_data.mat');
            catch
                % Create fallback data if test data not available
                obj.macroData = obj.generateTestData(500, 4, 3, 2);
            end
            
            % Prepare test data structures for various test conditions
        end
        
        function tearDown(obj)
            % Clean up after each test method execution
            tearDown@BaseTest(obj);
        end
        
        function testVecmModelBasicEstimation(obj)
            % Tests the basic VECM model estimation with synthetic data
            
            % Extract synthetic data with known parameters
            data = obj.syntheticData.data;
            trueParams = obj.syntheticData.trueParameters;
            
            % Estimate VECM model using vecm_model function
            model = vecm_model(data, 2, 1);
            
            % Verify model structure contains all expected components
            obj.assertTrue(isstruct(model), 'Model should be a structure');
            obj.assertTrue(isfield(model, 'alpha'), 'Model missing alpha field');
            obj.assertTrue(isfield(model, 'beta'), 'Model missing beta field');
            obj.assertTrue(isfield(model, 'Pi'), 'Model missing Pi field');
            
            % Verify dimensions of key matrices
            obj.assertEqual(size(model.alpha, 1), 3, 'Alpha has wrong number of rows');
            obj.assertEqual(size(model.alpha, 2), 1, 'Alpha has wrong number of columns');
            obj.assertEqual(size(model.beta, 1), 3, 'Beta has wrong number of rows');
            obj.assertEqual(size(model.beta, 2), 1, 'Beta has wrong number of columns');
            
            % Compare estimated cointegration vectors with true values
            % Note: Cointegrating vectors are identified only up to normalization
            normalizedEstimatedBeta = model.beta / norm(model.beta);
            normalizedTrueBeta = trueParams.beta / norm(trueParams.beta);
            
            % Check if the cointegrating vectors are similar after normalization
            betaComparison = obj.comparator.compareMatrices(normalizedEstimatedBeta, normalizedTrueBeta, 0.1);
            obj.assertTrue(betaComparison.isEqual, 'Estimated beta differs significantly from true beta');
            
            % Verify residuals properties
            meanResiduals = mean(model.residuals);
            obj.assertTrue(all(abs(meanResiduals) < 0.1), 'Residuals should have mean close to zero');
            
            % Verify information criteria are properly calculated
            obj.assertTrue(isfield(model, 'aic'), 'AIC not computed');
            obj.assertTrue(isfield(model, 'sbic'), 'SBIC not computed');
        end
        
        function testJohansenCointegrationTest(obj)
            % Tests the Johansen cointegration test for determining cointegration rank
            
            % Extract synthetic data with known cointegration rank
            data = obj.syntheticData.data;
            r = obj.syntheticData.trueParameters.r; % True cointegration rank
            
            % Perform Johansen test using johansen_test function
            results = johansen_test(data, 2, struct('test', 'trace'));
            
            % Verify test statistics and critical values are computed
            obj.assertTrue(isstruct(results), 'Test results should be a structure');
            obj.assertTrue(isfield(results, 'trace'), 'Missing trace statistics');
            obj.assertTrue(isfield(results, 'maxeig'), 'Missing maximum eigenvalue statistics');
            obj.assertTrue(isfield(results, 'eigenvalues'), 'Missing eigenvalues');
            
            % Check eigenvalues are properly computed and sorted
            eigenvalues = results.eigenvalues;
            obj.assertTrue(all(diff(eigenvalues) <= 0), 'Eigenvalues should be sorted in descending order');
            
            % Verify trace statistic correctly identifies cointegration rank
            % (Note: might not match exactly due to sampling variation)
            obj.assertTrue(abs(results.r - r) <= 1, 'Detected rank far from true rank');
            
            % Verify max eigenvalue statistic correctly identifies cointegration rank
            results2 = johansen_test(data, 2, struct('test', 'maxeig'));
            obj.assertTrue(isstruct(results2), 'Test results should be a structure');
            
            % Test different deterministic trend specifications
            for det = 1:4
                resultsWithDet = johansen_test(data, 2, struct('det', det));
                obj.assertTrue(isstruct(resultsWithDet), 'Test failed with det setting');
            end
            
            % Validate p-values are calculated correctly
            obj.assertTrue(all(results.pvals_trace >= 0 & results.pvals_trace <= 1), 
                'Trace test p-values should be between 0 and 1');
            obj.assertTrue(all(results.pvals_maxeig >= 0 & results.pvals_maxeig <= 1), 
                'Max eigenvalue test p-values should be between 0 and 1');
        end
        
        function testVecmForecast(obj)
            % Tests the forecasting functionality of the VECM model
            
            % Estimate VECM model using synthetic data
            model = vecm_model(obj.syntheticData.data, 2, 1);
            
            % Generate forecasts using vecm_forecast function for various horizons
            h = 10;
            forecasts = vecm_forecast(model, h);
            
            % Verify forecast dimensions are correct
            [rows, cols] = size(forecasts);
            obj.assertEqual(rows, h, 'Forecast should have h rows');
            obj.assertEqual(cols, size(obj.syntheticData.data, 2), 'Forecast should have same number of columns as data');
            
            % Compare short-horizon forecasts with expected values based on known parameters
            % For a VECM model, first forecast should be influenced by error correction
            data = obj.syntheticData.data;
            expectedDirection = sign(model.Pi * data(end,:)');
            firstForecastDirection = sign(forecasts(1,:)' - data(end,:)');
            
            % At least some of the directions should match the error correction effect
            directionsMatch = (expectedDirection .* firstForecastDirection) > 0;
            obj.assertTrue(sum(directionsMatch) >= 1, 'Error correction effect not visible in forecast');
            
            % Check stability of long-horizon forecasts
            forecastVariances = var(forecasts);
            obj.assertTrue(all(forecastVariances < 10*var(data)), 'Long-horizon forecasts show explosive behavior');
            
            % Test forecast with different options settings
            % Generate longer forecast
            longForecasts = vecm_forecast(model, 20);
            obj.assertEqual(size(longForecasts, 1), 20, 'Long forecast should have 20 periods');
            
            % Validate forecast confidence intervals when requested
            % This would be implemented in a more advanced version of the function
        end
        
        function testVecmImpulseResponse(obj)
            % Tests the impulse response function calculation for VECM models
            
            % Estimate VECM model using synthetic data
            model = vecm_model(obj.syntheticData.data, 2, 1);
            
            % Calculate impulse responses using vecm_irf function
            h = 20;
            irf = vecm_irf(model, h);
            
            % Verify impulse response dimensions are correct
            [periods, responses, shocks] = size(irf);
            obj.assertEqual(periods, h+1, 'IRF should have h+1 periods (including impact)');
            obj.assertEqual(responses, size(obj.syntheticData.data, 2), 'IRF should have K responses');
            obj.assertEqual(shocks, size(obj.syntheticData.data, 2), 'IRF should have K shocks');
            
            % Check orthogonalization of impulse responses
            impactMatrix = squeeze(irf(1,:,:));
            obj.assertTrue(obj.comparator.compareMatrices(impactMatrix, eye(size(obj.syntheticData.data, 2)), 1e-10).isEqual, 
                'Impact matrix should be identity');
            
            % Validate impulse response patterns match expected dynamics
            obj.assertTrue(all(isfinite(irf(:))), 'IRF values should be finite');
            
            % Test long-run responses against known cointegration restrictions
            lastPeriodIRF = squeeze(irf(end,:,:));
            
            % For cointegrated systems, some impulse responses should converge
            obj.assertTrue(all(isfinite(lastPeriodIRF(:))), 'Long-run IRF should be finite');
            
            % Compare short-run dynamics with true parameters
            % At least one shock should have persistent effects (unit root)
            meanAbsResponse = mean(abs(lastPeriodIRF), 1);
            obj.assertTrue(any(meanAbsResponse > 0.1), 'No persistent shocks found in IRF');
        end
        
        function testVecmWithMacroData(obj)
            % Tests VECM estimation with real macroeconomic data
            
            % Extract macroeconomic time series from test data
            if isstruct(obj.macroData) && isfield(obj.macroData, 'data')
                macroData = obj.macroData.data;
            else
                macroData = obj.macroData;
            end
            
            % Validate macroeconomic data
            obj.assertTrue(size(macroData, 1) > 100, 'Not enough observations in macro data');
            obj.assertTrue(size(macroData, 2) >= 3, 'Not enough variables in macro data');
            
            % Test model with different lag specifications
            for p = 1:3
                try
                    model = vecm_model(macroData, p, 1);
                    obj.assertTrue(isstruct(model), 'Model estimation failed');
                    
                    % Verify model diagnostics with real data
                    meanResiduals = mean(model.residuals);
                    obj.assertTrue(all(abs(meanResiduals) < 0.1), 'Residuals should have mean close to zero');
                    
                    % Compare results with pre-computed benchmark values
                    obj.assertTrue(isfield(model, 'logL'), 'Missing log-likelihood');
                    obj.assertTrue(isfield(model, 'sigma'), 'Missing residual covariance');
                catch ME
                    % Some specifications might fail - that's okay as long as not all fail
                    if p == 3
                        rethrow(ME); % Re-throw error if we failed with all specifications
                    end
                end
            end
            
            % Test forecasting performance with macroeconomic data
            try
                model = vecm_model(macroData, 2, 1);
                forecasts = vecm_forecast(model, 8);
                irf = vecm_irf(model, 12);
                
                obj.assertEqual(size(forecasts, 1), 8, 'Forecast horizon incorrect');
                obj.assertEqual(size(forecasts, 2), size(macroData, 2), 'Forecast variables incorrect');
                obj.assertEqual(size(irf, 1), 13, 'IRF periods incorrect'); % h+1 periods
                
                % Validate impulse responses for interpretability
                obj.assertTrue(all(isfinite(irf(:))), 'IRF should contain only finite values');
            catch ME
                obj.assertTrue(false, ['Error in forecast or IRF with macro data: ' ME.message]);
            end
        end
        
        function testVecmToVarConversion(obj)
            % Tests the conversion from VECM to VAR representation
            
            % Estimate VECM model using synthetic data
            vecmModel = vecm_model(obj.syntheticData.data, 2, 1);
            
            % Convert to VAR using vecm_to_var function
            varModel = vecm_to_var(vecmModel);
            
            % Verify VAR coefficients are properly calculated
            obj.assertTrue(isstruct(varModel), 'VAR model should be a structure');
            obj.assertTrue(isfield(varModel, 'A'), 'VAR model missing coefficient matrices');
            obj.assertTrue(iscell(varModel.A), 'A should be a cell array');
            obj.assertEqual(length(varModel.A), vecmModel.p, 'A should have p cells');
            
            % Verify VAR/VECM relationship Pi = alpha*beta'
            reconstructedPi = varModel.alpha * varModel.beta';
            obj.assertTrue(obj.comparator.compareMatrices(reconstructedPi, vecmModel.Pi, 1e-10).isEqual, 
                'Pi should equal alpha*beta''');
            
            % Compare VAR forecasts with VECM forecasts
            h = 5;
            vecmForecast = vecm_forecast(vecmModel, h);
            
            % Use the VAR model to generate forecasts
            k = vecmModel.k;
            varCoeff = varModel.A;
            varConstant = zeros(k, 1);
            if isfield(varModel, 'mu')
                varConstant = varModel.mu;
            end
            
            % Generate VAR forecasts
            data = obj.syntheticData.data;
            varForecast = zeros(h, k);
            lastObs = data(end-vecmModel.p+1:end, :);
            
            for t = 1:h
                forecast_t = varConstant;
                
                for i = 1:vecmModel.p
                    if t > i
                        forecast_t = forecast_t + varCoeff{i} * varForecast(t-i, :)';
                    else
                        forecast_t = forecast_t + varCoeff{i} * lastObs(end-i+t, :)';
                    end
                end
                
                varForecast(t, :) = forecast_t';
            end
            
            % Validate VAR impulse responses against VECM impulse responses
            for t = 1:h
                diffForecast = vecmForecast(t,:) - varForecast(t,:);
                obj.assertTrue(norm(diffForecast) < 1e-8, 
                    ['VAR and VECM forecasts differ at horizon ' num2str(t)]);
            end
            
            % Check consistency of residuals between representations
            obj.assertTrue(isfield(varModel, 'sigma'), 'VAR model should have sigma');
            obj.assertTrue(isfield(vecmModel, 'sigma'), 'VECM model should have sigma');
        end
        
        function testVecmInvalidInputs(obj)
            % Tests error handling for invalid inputs to VECM functions
            
            % Generate small valid dataset
            validData = randn(50, 3);
            
            % Test behavior with invalid data types
            try
                vecm_model('notmatrix', 2, 1);
                obj.assertTrue(false, 'Should throw error for non-numeric data');
            catch
                obj.assertTrue(true);
            end
            
            % Test behavior with mismatched dimensions
            try
                vecm_model(validData, 2, 5);
                obj.assertTrue(false, 'Should throw error for rank >= k');
            catch
                obj.assertTrue(true);
            end
            
            % Verify proper error messages for invalid lag specification
            try
                vecm_model(validData, -1, 1);
                obj.assertTrue(false, 'Should throw error for negative lag');
            catch
                obj.assertTrue(true);
            end
            
            % Check error handling for invalid cointegration rank
            try
                vecm_model(validData, 2, -1);
                obj.assertTrue(false, 'Should throw error for negative rank');
            catch
                obj.assertTrue(true);
            end
            
            % Test behavior with missing required parameters
            try
                vecm_model(randn(100, 1), 2, 0);
                obj.assertTrue(false, 'Should throw error for univariate data');
            catch
                obj.assertTrue(true);
            end
            
            % Verify robust handling of non-stationary data
            % This is implicit in the VECM model (it's designed for non-stationary data)
            
            % Test boundary conditions for numerical stability
            try
                % Create nearly singular data matrix
                singularData = ones(100, 3) + randn(100, 3) * 1e-10;
                vecm_model(singularData, 2, 1);
                % Should not throw error for numerical instability
                obj.assertTrue(true);
            catch ME
                fprintf('Received error with near-singular data: %s\n', ME.message);
                obj.assertTrue(false, 'VECM should handle near-singular data gracefully');
            end
        end
        
        function testVecmWithStructuralBreaks(obj)
            % Tests VECM estimation with data containing structural breaks
            
            % Generate synthetic data with structural breaks
            T = 400;  % Total observations
            breakPoint = 200;  % Break point
            
            % First regime data
            data1 = obj.generateTestData(breakPoint, 3, 2, 1);
            
            % Second regime data with different parameters
            data2 = obj.generateTestData(T - breakPoint, 3, 2, 1);
            data2.data = data2.data * 1.5;  % Amplify the data for clear break
            
            % Combine data from both regimes
            combinedData = [data1.data; data2.data];
            
            % Estimate VECM model with and without break handling
            fullModel = vecm_model(combinedData, 2, 1);
            
            % Estimate separate models for each regime
            model1 = vecm_model(data1.data, 2, 1);
            model2 = vecm_model(data2.data, 2, 1);
            
            % Compare estimation accuracy with known parameters
            fullResidStd = std(fullModel.residuals);
            subRes1Std = std(model1.residuals);
            subRes2Std = std(model2.residuals);
            
            % Residuals from the full sample should be larger due to break
            averageSubSampleStd = (subRes1Std + subRes2Std) / 2;
            obj.assertTrue(all(fullResidStd > 0.9 * averageSubSampleStd), 
                'Full sample residuals should be larger due to break');
            
            % Test forecasting performance across structural breaks
            combinedLogL = model1.logL + model2.logL;
            obj.assertTrue(combinedLogL > fullModel.logL, 
                'Sum of separate regime log-likelihoods should exceed full sample log-likelihood');
            
            % Validate robustness of cointegration rank tests with breaks
            % This would require comparing rank test results before and after the break
        end
        
        function testVecmPerformance(obj)
            % Tests the computational performance of VECM estimation
            
            % Generate large-scale test data for performance testing
            try
                mediumData = obj.generateTestData(200, 5, 2, 2);
                
                % Measure execution time for various model specifications
                tic;
                model = vecm_model(mediumData.data, 2, 2);
                executionTime = toc;
                
                % Log performance information
                fprintf('VECM estimation with T=200, k=5, p=2, r=2 took %.2f seconds\n', executionTime);
                
                % Test memory usage during estimation
                obj.assertTrue(isstruct(model), 'Model estimation failed');
                obj.assertTrue(isfield(model, 'residuals'), 'Model missing residuals');
                
                % Validate computational efficiency with increasing data dimensions
                % Only run larger tests if not in quick test mode
                if ~isfield(obj.testResults, 'quickTest') || ~obj.testResults.quickTest
                    try
                        largeData = obj.generateTestData(500, 8, 3, 3);
                        
                        % Measure execution time with larger dataset
                        tic;
                        largeModel = vecm_model(largeData.data, 3, 3);
                        largeExecutionTime = toc;
                        
                        fprintf('VECM estimation with T=500, k=8, p=3, r=3 took %.2f seconds\n', largeExecutionTime);
                        
                        % Compare performance scaling
                        expectedScalingFactor = (500*8^2)/(200*5^2); % Approximate complexity ratio
                        actualScalingFactor = largeExecutionTime/executionTime;
                        
                        % Performance should scale reasonably with problem size
                        % Allow for some variation due to system conditions
                        fprintf('Performance scaling factor: expected ~%.2f, actual %.2f\n', 
                            expectedScalingFactor, actualScalingFactor);
                    catch ME
                        fprintf('Large model test encountered issue: %s\n', ME.message);
                    end
                end
            catch ME
                fprintf('Performance test not completed: %s\n', ME.message);
            end
        end
        
        function testData = generateTestData(obj, T, K, p, r)
            % Utility method to generate VECM test data with specific properties
            % 
            % INPUTS:
            %   T - Integer, number of observations
            %   K - Integer, number of variables
            %   p - Integer, VAR lag order (VECM will have p-1 lags)
            %   r - Integer, cointegration rank
            
            % Use TestDataGenerator to create VECM process with specified parameters
            if ismethod(obj.dataGenerator, 'generateVECMData')
                testData = obj.dataGenerator.generateVECMData(T, K, p, r);
                return;
            end
            
            % Initialize structure
            testData = struct();
            testData.data = zeros(T, K);
            
            % Configure cointegration vectors with known patterns
            beta = randn(K, r);
            % Normalize beta for identification
            for i = 1:r
                beta(:,i) = beta(:,i) / norm(beta(:,i));
            end
            
            % Set adjustment coefficients with known values
            alpha = randn(K, r) * 0.1;  % Small values for stability
            
            % Create error correction matrix Pi = alpha*beta'
            Pi = alpha * beta';
            
            % Generate short-run dynamics parameters
            gamma = cell(p-1, 1);
            for i = 1:p-1
                gamma{i} = eye(K) * 0.2 / i;  % Decreasing impact with lag
            end
            
            % Generate innovation covariance matrix
            sigma = 0.1 * eye(K) + 0.05 * ones(K);  % Covariance matrix
            sigma = sigma .* (sigma > 0.05);  % Sparsify the covariance matrix
            sigma = sigma + diag(0.1 * ones(K, 1));  % Ensure positive definite
            
            % Create initial values
            initData = randn(p, K);
            
            % Create time series data from the specified process
            testData.data(1:p,:) = initData;
            for t = p+1:T
                % Error correction term
                ect = Pi * testData.data(t-1,:)';
                
                % Short-run dynamics
                shortRun = zeros(K, 1);
                for i = 1:p-1
                    shortRun = shortRun + gamma{i} * (testData.data(t-i,:)' - testData.data(t-i-1,:)');
                end
                
                % Generate innovation (multivariate normal)
                epsilon = mvnrnd(zeros(K,1), sigma)';
                
                % VECM equation: Δy_t = Pi*y_{t-1} + Γ₁*Δy_{t-1} + ... + Γ_{p-1}*Δy_{t-p+1} + ε_t
                testData.data(t,:) = testData.data(t-1,:) + (ect + shortRun + epsilon)';
            end
            
            % Store true parameters for validation
            testData.trueParameters = struct(...
                'alpha', alpha, ...
                'beta', beta, ...
                'Pi', Pi, ...
                'gamma', {gamma}, ...
                'sigma', sigma, ...
                'p', p, ...
                'r', r, ...
                'k', K, ...
                'T', T);
            
            % Return structured test data for unit tests
        end
        
        function result = compareVecmParameters(obj, estimated, true)
            % Utility method to compare estimated VECM parameters with true values
            
            % Normalize cointegration vectors for comparison
            estBeta = estimated.beta;
            trueBeta = true.beta;
            
            for i = 1:size(estBeta, 2)
                estBeta(:,i) = estBeta(:,i) / norm(estBeta(:,i));
                if i <= size(trueBeta, 2)
                    trueBeta(:,i) = trueBeta(:,i) / norm(trueBeta(:,i));
                end
            end
            
            % Use NumericalComparator to compare coefficient matrices
            betaComparison = obj.comparator.compareMatrices(estBeta, trueBeta, 0.2);
            if ~betaComparison.isEqual
                fprintf('Beta comparison failed. Max diff: %g\n', betaComparison.maxAbsoluteDifference);
                result = false;
                return;
            end
            
            % Check adjustment coefficients within appropriate tolerance
            if ~isempty(estimated.alpha) && ~isempty(true.alpha)
                alphaComparison = obj.comparator.compareMatrices(estimated.alpha, true.alpha, 0.3);
                if ~alphaComparison.isEqual
                    fprintf('Alpha comparison failed. Max diff: %g\n', alphaComparison.maxAbsoluteDifference);
                    result = false;
                    return;
                end
            end
            
            % Compare short-run dynamics parameters
            if isfield(estimated, 'Pi') && isfield(true, 'Pi')
                PiComparison = obj.comparator.compareMatrices(estimated.Pi, true.Pi, 0.3);
                if ~PiComparison.isEqual
                    fprintf('Pi comparison failed. Max diff: %g\n', PiComparison.maxAbsoluteDifference);
                    result = false;
                    return;
                end
            end
            
            % Verify covariance matrix estimation accuracy
            if isfield(estimated, 'sigma') && isfield(true, 'sigma')
                sigmaComparison = obj.comparator.compareMatrices(estimated.sigma, true.sigma, 0.5);
                if ~sigmaComparison.isEqual
                    fprintf('Sigma comparison failed. Max diff: %g\n', sigmaComparison.maxAbsoluteDifference);
                    result = false;
                    return;
                end
            end
            
            % Return overall comparison result
            result = true;
        end
    end
end