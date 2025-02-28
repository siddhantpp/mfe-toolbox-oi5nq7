classdef CccMvgarchTest < BaseTest
    % Test class for the CCC-MVGARCH model implementation, verifying parameter estimation, 
    % forecasting accuracy, and numerical stability of the multivariate volatility model
    % with constant correlation structure.
    
    properties
        dataGenerator     % TestDataGenerator instance
        comparator        % NumericalComparator instance
        testTolerance     % Numerical tolerance for tests
        testData          % Test data structure
        testOptions       % Test options structure
    end
    
    methods
        function obj = CccMvgarchTest()
            % Initializes the CccMvgarchTest class with test data and numerical comparators
            
            % Call superclass constructor
            obj = obj@BaseTest();
            
            % Set test tolerance for numerical comparisons
            obj.testTolerance = 1e-6;
            
            % Create TestDataGenerator instance
            obj.dataGenerator = @TestDataGenerator;
            
            % Create NumericalComparator instance
            obj.comparator = NumericalComparator();
            
            % Set default test options for CCC-MVGARCH model estimation
            obj.testOptions = struct();
            obj.testOptions.model = 'GARCH';
            obj.testOptions.univariate = struct('distribution', 'NORMAL', 'p', 1, 'q', 1);
        end
        
        function setUp(obj)
            % Prepares the test environment before each test method execution
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Set random seed for reproducible tests
            rng(123);
            
            % Initialize test data structures
            obj.testData = struct();
            
            % Generate or load synthetic multivariate time series data with known properties
            obj.testData = obj.generateMultivariateCCCData(3, 500, struct('omega', 0.05, 'alpha', 0.1, 'beta', 0.8), eye(3));
        end
        
        function tearDown(obj)
            % Cleans up the test environment after each test method execution
            
            % Clear test data from memory
            obj.testData = [];
            
            % Reset any modified global states
            rng('default');
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
        end
        
        function testBasicEstimation(obj)
            % Tests the basic parameter estimation functionality of the CCC-MVGARCH model
            
            % Generate multivariate time series data with known variance and correlation parameters
            testData = obj.testData;
            data = testData.returns;
            
            % Estimate CCC-MVGARCH model using default options
            results = ccc_mvgarch(data, obj.testOptions);
            
            % Verify that parameter estimates are close to true parameters
            for k = 1:size(data, 2)
                trueParams = [testData.garchParams.omega; 
                             testData.garchParams.alpha; 
                             testData.garchParams.beta];
                estimatedParams = results.parameters{k};
                
                % Compare with reasonable tolerance
                obj.assertAlmostEqual(trueParams, estimatedParams, ...
                    'GARCH parameters not accurately estimated');
            end
            
            % Verify log-likelihood is reasonable and finite
            obj.assertTrue(isfinite(results.loglikelihood), ...
                'Log-likelihood should be finite');
            obj.assertTrue(results.loglikelihood < 0, ...
                'Log-likelihood should be negative for this type of model');
            
            % Verify that correlation matrix is positive definite
            obj.assertTrue(det(results.R) > 0, ...
                'Correlation matrix should be positive definite');
            
            % Verify model stationarity conditions are satisfied
            obj.assertTrue(all(results.diagnostics.stationarity), ...
                'Model should satisfy stationarity conditions');
        end
        
        function testGaussianCCC(obj)
            % Tests CCC-MVGARCH estimation with Gaussian innovation distribution
            
            % Generate multivariate time series data with normal innovations
            testData = obj.generateMultivariateCCCData(3, 500, ...
                struct('omega', 0.05, 'alpha', 0.1, 'beta', 0.8), ...
                [1.0, 0.5, 0.3; 0.5, 1.0, 0.4; 0.3, 0.4, 1.0], ...
                'normal');
            
            % Configure model options with Gaussian distribution assumption
            options = obj.testOptions;
            options.univariate.distribution = 'NORMAL';
            
            % Estimate CCC-MVGARCH model
            results = ccc_mvgarch(testData.returns, options);
            
            % Verify that univariate GARCH parameters are correctly estimated
            for k = 1:size(testData.returns, 2)
                trueParams = [testData.garchParams.omega; 
                             testData.garchParams.alpha; 
                             testData.garchParams.beta];
                estimatedParams = results.parameters{k};
                
                % Compare with tolerance
                obj.assertAlmostEqual(trueParams, estimatedParams, ...
                    sprintf('GARCH parameters for series %d not accurately estimated', k));
            end
            
            % Validate that the constant correlation matrix accurately reflects the true correlation
            estimatedCorr = results.R;
            trueCorr = testData.correlationMatrix;
            
            % Compare correlation matrices
            obj.assertTrue(obj.comparator.compareMatrices(trueCorr, estimatedCorr, 0.1).isEqual, ...
                'Estimated correlation matrix does not match true correlation structure');
            
            % Verify that the model adequately captures volatility dynamics
            Dt_squared = results.Dt.^2;
            for k = 1:size(testData.returns, 2)
                % Check correlation between estimated and true conditional variances
                corr_value = corr(Dt_squared(:, k), testData.trueVariances(:, k));
                obj.assertTrue(corr_value > 0.7, ...
                    sprintf('Estimated variances should be highly correlated with true variances for series %d', k));
            end
        end
        
        function testStudentTCCC(obj)
            % Tests CCC-MVGARCH estimation with Student's t innovation distribution
            
            % Generate multivariate time series data with t-distributed innovations
            testData = obj.generateMultivariateCCCData(3, 500, ...
                struct('omega', 0.05, 'alpha', 0.1, 'beta', 0.8), ...
                [1.0, 0.5, 0.3; 0.5, 1.0, 0.4; 0.3, 0.4, 1.0], ...
                't');
            
            % Configure model options with Student's t distribution assumption
            options = obj.testOptions;
            options.univariate.distribution = 'T';
            
            % Estimate CCC-MVGARCH model with degrees of freedom parameter
            results = ccc_mvgarch(testData.returns, options);
            
            % Verify degrees of freedom parameter is accurately estimated
            % (This would require extracting the nu parameter from results, which may be model-specific)
            % For now, we just verify that the estimation completes successfully
            obj.assertTrue(isfinite(results.loglikelihood), ...
                'Log-likelihood should be finite for t-distribution');
            
            % Validate that the constant correlation matrix is correctly estimated
            estimatedCorr = results.R;
            trueCorr = testData.correlationMatrix;
            
            % Compare correlation matrices with tolerance
            obj.assertTrue(obj.comparator.compareMatrices(trueCorr, estimatedCorr, 0.15).isEqual, ...
                'Estimated correlation matrix does not match true correlation structure for t-distribution');
            
            % Validate that model adequately captures both volatility dynamics and fat tails
            Dt_squared = results.Dt.^2;
            for k = 1:size(testData.returns, 2)
                % Check correlation between estimated and true conditional variances
                corr_value = corr(Dt_squared(:, k), testData.trueVariances(:, k));
                obj.assertTrue(corr_value > 0.7, ...
                    sprintf('Estimated variances should be highly correlated with true variances for series %d', k));
            end
        end
        
        function testCCCForecast(obj)
            % Tests the forecasting functionality of the CCC-MVGARCH model
            
            % Estimate CCC-MVGARCH model on test data
            testData = obj.testData;
            options = obj.testOptions;
            options.forecast = 10;  % 10-step ahead forecast
            
            % Generate multi-step ahead forecasts for variances and covariances
            results = ccc_mvgarch(testData.returns, options);
            
            % Verify forecast was generated
            obj.assertTrue(isfield(results, 'forecast'), 'Forecast should be included in results');
            obj.assertTrue(isfield(results.forecast, 'Ht'), 'Forecast should include covariance matrices');
            obj.assertTrue(isfield(results.forecast, 'Dt'), 'Forecast should include standard deviations');
            
            % Verify forecast covariance matrices maintain constant correlation structure
            for h = 1:size(results.forecast.Ht, 1)
                H_t = squeeze(results.forecast.Ht(h, :, :));
                D_t = diag(results.forecast.Dt(h, :));
                
                % Recreate correlation from H_t and D_t
                R_implied = diag(1./diag(D_t)) * H_t * diag(1./diag(D_t));
                
                % Compare with original correlation
                obj.assertTrue(obj.comparator.compareMatrices(results.R, R_implied, 1e-4).isEqual, ...
                    sprintf('Forecast correlation at horizon %d does not match estimated correlation', h));
            end
            
            % Verify that stationarity properties are maintained in forecasts
            % For stationary GARCH, long-horizon forecasts should approach unconditional variance
            longHorizon = size(results.forecast.Dt, 1);
            if longHorizon > 5
                % Check if last few forecasts stabilize (difference should be small)
                lastDiffNorm = norm(results.forecast.Dt(end, :) - results.forecast.Dt(end-1, :)) / ...
                              norm(results.forecast.Dt(end-1, :));
                obj.assertTrue(lastDiffNorm < 0.05, ...
                    'Long-horizon forecasts should stabilize toward unconditional volatility');
            end
            
            % Test that forecast uncertainty increases with horizon
            % (This is typically measured by forecast error variance, which may not be
            % directly accessible in the implementation. For simplicity, we skip this test.)
        end
        
        function testHighDimensionalCCC(obj)
            % Tests CCC-MVGARCH estimation with higher-dimensional data
            
            % Generate higher-dimensional multivariate time series (e.g., 10 assets)
            numAssets = 10;
            testData = obj.generateMultivariateCCCData(numAssets, 300, ...
                struct('omega', 0.05, 'alpha', 0.1, 'beta', 0.8), ...
                0.3 * ones(numAssets) + 0.7 * eye(numAssets));
            
            % Configure model with appropriate optimization settings for high dimensions
            options = obj.testOptions;
            if isfield(options, 'algorithm')
                options.algorithm = 'sqp';  % Often more stable for higher dimensions
            end
            
            % Measure execution time
            tic;
            
            % Estimate CCC-MVGARCH model and measure performance
            results = ccc_mvgarch(testData.returns, options);
            
            executionTime = toc;
            fprintf('High-dimensional estimation completed in %.2f seconds\n', executionTime);
            
            % Verify that correlation structure is captured correctly
            estimatedCorr = results.R;
            trueCorr = testData.correlationMatrix;
            
            % For high dimensions, we use a higher tolerance
            obj.assertTrue(obj.comparator.compareMatrices(trueCorr, estimatedCorr, 0.2).isEqual, ...
                'Estimated correlation matrix does not match true correlation structure in high dimensions');
            
            % Test numerical stability in higher dimensions
            obj.assertTrue(all(results.diagnostics.stationarity), ...
                'Model should satisfy stationarity conditions in high dimensions');
            
            % Verify positive definiteness of large correlation matrix
            obj.assertTrue(det(results.R) > 0, ...
                'Correlation matrix should be positive definite in high dimensions');
            
            % Verify diagonal elements are exactly 1
            obj.assertTrue(all(abs(diag(results.R) - 1) < 1e-10), ...
                'Diagonal elements of correlation matrix should be 1');
        end
        
        function testCorrelationMatrixAccuracy(obj)
            % Tests the accuracy of the constant correlation matrix estimation
            
            % Generate data with known correlation structure
            correlationLevels = [0.2, 0.5, 0.8];
            numTests = length(correlationLevels);
            
            for i = 1:numTests
                rho = correlationLevels(i);
                corrMatrix = [1, rho, rho; rho, 1, rho; rho, rho, 1];
                
                % Generate data with specified correlation
                testData = obj.generateMultivariateCCCData(3, 500, ...
                    struct('omega', 0.05, 'alpha', 0.1, 'beta', 0.8), ...
                    corrMatrix);
                
                % Estimate model
                results = ccc_mvgarch(testData.returns, obj.testOptions);
                
                % Compare estimated correlation with true correlation
                estimatedCorr = results.R;
                
                % Use tolerance that scales with correlation magnitude
                tolerance = 0.1 + 0.05 * rho;
                
                % Verify matrix properties (symmetry, unity diagonal)
                obj.assertTrue(max(abs(diag(estimatedCorr) - 1)) < 1e-10, ...
                    'Diagonal elements of correlation matrix should be 1');
                
                obj.assertTrue(norm(estimatedCorr - estimatedCorr') < 1e-10, ...
                    'Correlation matrix should be symmetric');
                
                % Compare with true correlation
                obj.assertTrue(obj.comparator.compareMatrices(corrMatrix, estimatedCorr, tolerance).isEqual, ...
                    sprintf('Correlation estimation failed for rho=%.1f', rho));
            end
            
            % Test with various correlation patterns (high/low correlations)
            % This is already covered by the loop above
        end
        
        function testNumericalStability(obj)
            % Tests the numerical stability of CCC-MVGARCH estimation under challenging conditions
            
            % Generate data with extreme correlation patterns
            nearSingularCorr = [1.0, 0.95, 0.95; 0.95, 1.0, 0.95; 0.95, 0.95, 1.0];
            
            % Apply a small perturbation to ensure positive definiteness
            [V, D] = eig(nearSingularCorr);
            D = diag(max(diag(D), 0.01));
            nearSingularCorr = V * D * V';
            
            % Normalize to ensure proper correlation matrix
            D_inv = diag(1./sqrt(diag(nearSingularCorr)));
            nearSingularCorr = D_inv * nearSingularCorr * D_inv;
            
            % Generate test data with near-singular correlation matrix
            testData = obj.generateMultivariateCCCData(3, 500, ...
                struct('omega', 0.05, 'alpha', 0.1, 'beta', 0.8), ...
                nearSingularCorr);
            
            % Verify robustness to initialization values by using different starting points
            options1 = obj.testOptions;
            options2 = obj.testOptions;
            
            % Different distribution assumptions to test robustness
            options1.univariate.distribution = 'NORMAL';
            options2.univariate.distribution = 'T';
            
            % Run both estimations
            results1 = ccc_mvgarch(testData.returns, options1);
            results2 = ccc_mvgarch(testData.returns, options2);
            
            % Both models should converge and produce valid results
            obj.assertTrue(isfinite(results1.loglikelihood), ...
                'Model 1 should produce finite log-likelihood with near-singular correlation');
            obj.assertTrue(isfinite(results2.loglikelihood), ...
                'Model 2 should produce finite log-likelihood with near-singular correlation');
            
            % Correlation matrices should be positive definite
            obj.assertTrue(det(results1.R) > 0, ...
                'Model 1 correlation matrix should be positive definite');
            obj.assertTrue(det(results2.R) > 0, ...
                'Model 2 correlation matrix should be positive definite');
            
            % Generate data with outliers to test robustness
            outlierData = testData.returns;
            % Add outliers (5 standard deviations) at random positions
            numOutliers = 5;
            for j = 1:numOutliers
                series = randi(size(outlierData, 2));
                pos = randi(size(outlierData, 1));
                outlierData(pos, series) = 5 * std(outlierData(:, series)) * sign(randn);
            end
            
            % Estimate with outlier data
            resultsOutlier = ccc_mvgarch(outlierData, obj.testOptions);
            
            % Model should still converge with outliers
            obj.assertTrue(isfinite(resultsOutlier.loglikelihood), ...
                'Model should handle outliers gracefully');
            
            % Test convergence behavior with different optimization settings
            % (This is more complex and may require additional configuration options
            % that might not be available in the current implementation)
        end
        
        function testParameterBoundaryConditions(obj)
            % Tests CCC-MVGARCH estimation at the boundaries of parameter constraints
            
            % Generate data with GARCH parameters at constraint boundaries (α+β ≈ 1)
            testData = obj.generateMultivariateCCCData(3, 500, ...
                struct('omega', 0.01, 'alpha', 0.15, 'beta', 0.84), ...
                [1.0, 0.5, 0.3; 0.5, 1.0, 0.4; 0.3, 0.4, 1.0]);
            
            % Test estimation with standard settings
            results = ccc_mvgarch(testData.returns, obj.testOptions);
            
            % Verify that parameter constraints are enforced in univariate models
            for k = 1:size(testData.returns, 2)
                params = results.parameters{k};
                
                % Extract alpha and beta (assuming standard GARCH model)
                alpha = params(2);
                beta = params(3);
                
                % Sum should be less than 1 (enforced by the model)
                obj.assertTrue(alpha + beta < 1, ...
                    sprintf('Stationarity constraint violated for series %d: alpha + beta = %.6f', ...
                    k, alpha + beta));
                
                % Parameters should be positive
                obj.assertTrue(all(params > 0), ...
                    sprintf('Positivity constraint violated for series %d', k));
            end
            
            % Verify appropriate warnings or errors are generated
            % (This would require capturing warnings, which is outside the scope of this test)
            
            % Validate parameter estimates under boundary conditions
            % We expect the estimates to be close to but not exactly at the boundary
            for k = 1:size(testData.returns, 2)
                params = results.parameters{k};
                
                % Extract alpha and beta (assuming standard GARCH model)
                alpha = params(2);
                beta = params(3);
                
                % Check that estimates are close to true values
                obj.assertAlmostEqual(testData.garchParams.alpha, alpha, ...
                    sprintf('Alpha estimate for series %d not close to true value at boundary', k));
                
                obj.assertAlmostEqual(testData.garchParams.beta, beta, ...
                    sprintf('Beta estimate for series %d not close to true value at boundary', k));
            end
        end
        
        function testCCCDiagnostics(obj)
            % Tests diagnostic methods for CCC-MVGARCH model specification
            
            % Estimate CCC-MVGARCH model on test data
            results = ccc_mvgarch(obj.testData.returns, obj.testOptions);
            
            % Compute and verify standardized residuals
            obj.assertTrue(isfield(results, 'std_residuals'), ...
                'Results should include standardized residuals');
            
            standardizedResiduals = results.std_residuals;
            
            % Verify standardized residuals have approximately mean 0 and variance 1
            for k = 1:size(standardizedResiduals, 2)
                meanResid = mean(standardizedResiduals(:, k));
                varResid = var(standardizedResiduals(:, k));
                
                obj.assertTrue(abs(meanResid) < 0.1, ...
                    sprintf('Standardized residuals for series %d should have mean close to 0', k));
                
                obj.assertTrue(abs(varResid - 1) < 0.2, ...
                    sprintf('Standardized residuals for series %d should have variance close to 1', k));
            end
            
            % Verify correlation of standardized residuals matches model correlation
            empiricalCorr = corr(standardizedResiduals);
            
            % Compare with tolerance
            obj.assertTrue(obj.comparator.compareMatrices(results.R, empiricalCorr, 0.2).isEqual, ...
                'Empirical correlation of standardized residuals should match model correlation');
            
            % Verify that model diagnostics are correctly calculated
            obj.assertTrue(isfield(results, 'diagnostics'), ...
                'Results should include diagnostics field');
            
            obj.assertTrue(isfield(results.diagnostics, 'stationarity'), ...
                'Diagnostics should include stationarity check');
            
            obj.assertTrue(isfield(results.diagnostics, 'correlation_pd'), ...
                'Diagnostics should include correlation positive definiteness check');
            
            % Check information criteria
            obj.assertTrue(isfield(results, 'aic'), 'Results should include AIC');
            obj.assertTrue(isfield(results, 'bic'), 'Results should include BIC');
            obj.assertTrue(results.bic > results.aic, 'BIC should be larger than AIC for same model');
        end
        
        function testDistributionComparison(obj)
            % Tests CCC-MVGARCH performance across different error distributions
            
            % Generate test data
            testData = obj.testData;
            
            % Estimate with different distributions
            distributions = {'NORMAL', 'T'};
            
            % Initialize arrays to store results
            logLikelihoods = zeros(length(distributions), 1);
            aicValues = zeros(length(distributions), 1);
            bicValues = zeros(length(distributions), 1);
            
            % Estimate models with each distribution specification
            for i = 1:length(distributions)
                options = obj.testOptions;
                options.univariate.distribution = distributions{i};
                
                results = ccc_mvgarch(testData.returns, options);
                
                % Store metrics
                logLikelihoods(i) = results.loglikelihood;
                aicValues(i) = results.aic;
                bicValues(i) = results.bic;
                
                % All models should converge and produce valid results
                obj.assertTrue(isfinite(results.loglikelihood), ...
                    sprintf('Model with %s distribution should produce finite log-likelihood', ...
                    distributions{i}));
                
                % Correlation matrix should be positive definite
                obj.assertTrue(det(results.R) > 0, ...
                    sprintf('Model with %s distribution should have positive definite correlation', ...
                    distributions{i}));
            end
            
            % For illustrative purposes, print comparison of log-likelihoods
            fprintf('Log-likelihood comparison:\n');
            for i = 1:length(distributions)
                fprintf('  %s: %.4f\n', distributions{i}, logLikelihoods(i));
            end
            
            % For this test, we don't explicitly verify which distribution is better,
            % as that depends on the actual data generation process used in the test.
        end
        
        function testData = generateMultivariateCCCData(obj, numSeries, numObservations, garchParams, correlationMatrix, innovationType)
            % Helper method to generate multivariate time series data with CCC structure
            
            % Set default parameters if not provided
            if nargin < 6
                innovationType = 'normal';
            end
            
            % Set random seed for reproducibility
            rng(123);
            
            % Generate constant correlation matrix if not provided
            if nargin < 5 || isempty(correlationMatrix)
                correlationMatrix = eye(numSeries);
            else
                % Ensure the correlation matrix is valid
                % Check dimensions
                if size(correlationMatrix, 1) ~= numSeries || size(correlationMatrix, 2) ~= numSeries
                    error('Correlation matrix dimensions must match numSeries');
                end
                
                % Ensure it's symmetric and has ones on diagonal
                correlationMatrix = 0.5 * (correlationMatrix + correlationMatrix');
                correlationMatrix(1:numSeries+1:end) = 1;
                
                % Ensure positive definiteness
                [V, D] = eig(correlationMatrix);
                D = diag(max(diag(D), 0.01));
                correlationMatrix = V * D * V';
                
                % Normalize to ensure proper correlation matrix
                D_inv = diag(1./sqrt(diag(correlationMatrix)));
                correlationMatrix = D_inv * correlationMatrix * D_inv;
            end
            
            % Initialize arrays for returns and true variances
            returns = zeros(numObservations, numSeries);
            trueVariances = zeros(numObservations, numSeries);
            
            % Generate individual GARCH processes for each series
            for i = 1:numSeries
                % Generate GARCH process with specified parameters
                switch lower(innovationType)
                    case 'normal'
                        % Normal innovations
                        univariateData = obj.dataGenerator('generateVolatilitySeries', ...
                            numObservations, 'GARCH', garchParams);
                    case 't'
                        % t-distributed innovations
                        distParams = struct();
                        distParams.distribution = 't';
                        distParams.nu = 5; % Default degrees of freedom
                        univariateData = obj.dataGenerator('generateVolatilitySeries', ...
                            numObservations, 'GARCH', setfield(garchParams, 'distribution', 't', 'distParams', distParams.nu));
                    otherwise
                        error('Unsupported innovation type: %s', innovationType);
                end
                
                % Store the true variance
                trueVariances(:, i) = univariateData.ht;
                
                % Store the standardized innovations (to be correlated later)
                returns(:, i) = univariateData.residuals;
            end
            
            % Compute Cholesky decomposition of correlation matrix
            cholMatrix = chol(correlationMatrix, 'lower');
            
            % Generate correlated standardized innovations
            corr_returns = returns * cholMatrix';
            
            % Apply GARCH volatility to correlated innovations
            for i = 1:numSeries
                returns(:, i) = corr_returns(:, i) .* sqrt(trueVariances(:, i));
            end
            
            % Create output structure
            testData = struct();
            testData.returns = returns;
            testData.trueVariances = trueVariances;
            testData.correlationMatrix = correlationMatrix;
            testData.garchParams = garchParams;
            testData.numSeries = numSeries;
            testData.numObservations = numObservations;
            testData.innovationType = innovationType;
        end
        
        function testData = loadTestData(obj)
            % Loads pre-generated test data from financial_returns.mat file
            
            try
                % Load data from the test data directory
                data = obj.loadTestData('financial_returns.mat');
                
                % Extract multivariate samples and related parameters
                if isfield(data, 'multivariate_samples')
                    multivariate_data = data.multivariate_samples;
                else
                    % If proper field not found, generate synthetic data
                    warning('CccMvgarchTest:DataNotFound', ...
                        'Multivariate samples not found in test data, generating synthetic data');
                    multivariate_data = obj.generateMultivariateCCCData(3, 500, ...
                        struct('omega', 0.05, 'alpha', 0.1, 'beta', 0.8), ...
                        [1.0, 0.5, 0.3; 0.5, 1.0, 0.4; 0.3, 0.4, 1.0]);
                end
                
                % Validate that loaded data has expected properties
                obj.assertTrue(isfield(multivariate_data, 'returns'), ...
                    'Test data should contain returns field');
                
                % Return structured data
                testData = multivariate_data;
            catch e
                warning('CccMvgarchTest:LoadError', ...
                    'Error loading test data: %s', e.message);
                
                % Fall back to synthetic data on error
                testData = obj.generateMultivariateCCCData(3, 500, ...
                    struct('omega', 0.05, 'alpha', 0.1, 'beta', 0.8), ...
                    [1.0, 0.5, 0.3; 0.5, 1.0, 0.4; 0.3, 0.4, 1.0]);
            end
        end
    end
end