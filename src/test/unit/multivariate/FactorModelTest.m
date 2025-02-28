classdef FactorModelTest < BaseTest
    %FACTORMODELTEST Unit test class for the factor_model implementation,
    %providing comprehensive test coverage for factor extraction methods,
    %rotation techniques, and forecasting capabilities

    properties
        testData
        syntheticData
        knownLoadings
        knownFactors
        numAssets
        numFactors
        numObservations
    end

    methods
        function obj = FactorModelTest()
            % Initialize the FactorModelTest class
            obj = obj@BaseTest(); % Call the superclass constructor
            obj.testData = struct(); % Initialize empty test data structure
        end

        function setUp(obj)
            %setUp Set up the test environment before each test method execution
            setUp@BaseTest(obj); % Call superclass setUp method

            % Set test parameters
            obj.numAssets = 100;
            obj.numFactors = 3;
            obj.numObservations = 500;

            % Load cross-sectional test data from MAT file
            obj.testData = obj.loadTestData('factor_test_data.mat');

            % Generate synthetic data with known factor structure if needed
            rng(123); % Set random number generator seed for reproducibility
            obj.knownLoadings = randn(obj.numAssets, obj.numFactors);
            obj.knownFactors = randn(obj.numObservations, obj.numFactors);
            obj.syntheticData = obj.knownLoadings * obj.knownFactors' + 0.1 * randn(obj.numAssets, obj.numObservations);
            obj.syntheticData = obj.syntheticData'; % Transpose to match expected dimensions

            % Store reference data for validation
        end

        function tearDown(obj)
            %tearDown Clean up the test environment after each test method execution
            tearDown@BaseTest(obj); % Call superclass tearDown method

            % Clear temporary test variables
            clear obj.testData obj.syntheticData obj.knownLoadings obj.knownFactors;

            % Reset any modified system states
        end

        function testPrincipalComponentExtraction(obj)
            %testPrincipalComponentExtraction Test principal component extraction method of factor model

            % Create options structure with method='principal'
            options = struct('method', 'principal');

            % Call factor_model with syntheticData, numFactors, and options
            model = factor_model(obj.syntheticData, obj.numFactors, options);

            % Verify model structure contains expected fields
            obj.assertTrue(isfield(model, 'loadings'), 'Model structure must contain loadings field');
            obj.assertTrue(isfield(model, 'factors'), 'Model structure must contain factors field');
            obj.assertTrue(isfield(model, 'eigenvalues'), 'Model structure must contain eigenvalues field');

            % Validate loadings dimensions match expected values
            [numAssets_est, numFactors_est] = size(model.loadings);
            obj.assertEqual(numAssets_est, obj.numAssets, 'Loadings must have numAssets rows');
            obj.assertEqual(numFactors_est, obj.numFactors, 'Loadings must have numFactors columns');

            % Verify orthogonality of loadings matrix
            loadings_cov = cov(model.loadings);
            identity_matrix = eye(obj.numFactors);
            obj.assertMatrixEqualsWithTolerance(loadings_cov, identity_matrix, 1e-6, 'Loadings must be orthogonal');

            % Validate that extracted factors explain expected variance proportion
            totalVariance = trace(cov(obj.syntheticData));
            explainedVariance = sum(model.eigenvalues);
            obj.assertAlmostEqual(explainedVariance, totalVariance, 1e-6, 'Extracted factors must explain total variance');

            % Compare estimated loadings with known loadings using appropriate tolerance
            obj.assertMatrixEqualsWithTolerance(model.loadings, obj.knownLoadings, 0.5, 'Estimated loadings must match known loadings');

            % Check that communalities and uniquenesses sum correctly
            communalities_plus_uniquenesses = model.communalities + model.uniquenesses;
            obj.assertMatrixEqualsWithTolerance(communalities_plus_uniquenesses, ones(obj.numAssets, 1), 1e-6, 'Communalities and uniquenesses must sum to 1');
        end

        function testMaximumLikelihoodExtraction(obj)
            %testMaximumLikelihoodExtraction Test maximum likelihood estimation method of factor model

            % Create options structure with method='ml'
            options = struct('method', 'ml');

            % Call factor_model with syntheticData, numFactors, and options
            model = factor_model(obj.syntheticData, obj.numFactors, options);

            % Verify model converged to solution
            obj.assertTrue(model.goodness_of_fit.p_value > 0.05, 'Model must converge to solution');

            % Validate loadings dimensions match expected values
            [numAssets_est, numFactors_est] = size(model.loadings);
            obj.assertEqual(numAssets_est, obj.numAssets, 'Loadings must have numAssets rows');
            obj.assertEqual(numFactors_est, obj.numFactors, 'Loadings must have numFactors columns');

            % Verify that uniquenesses are positive values
            obj.assertTrue(all(model.uniquenesses > 0), 'Uniquenesses must be positive values');

            % Check log-likelihood value is reasonable
            obj.assertTrue(model.goodness_of_fit.aic < 1000, 'Log-likelihood value must be reasonable');

            % Validate that estimated covariance matrix matches original within tolerance
            estimatedCov = model.loadings * model.loadings' + diag(model.uniquenesses);
            originalCov = cov(obj.syntheticData);
            obj.assertMatrixEqualsWithTolerance(estimatedCov, originalCov, 0.1, 'Estimated covariance matrix must match original within tolerance');

            % Verify goodness-of-fit metrics are consistent with data properties
            obj.assertTrue(model.goodness_of_fit.rmsr < 0.1, 'Goodness-of-fit metrics must be consistent with data properties');
        end

        function testVarimaxRotation(obj)
            %testVarimaxRotation Test varimax rotation method of factor loadings

            % Create options structure with method='principal', rotate='varimax'
            options = struct('method', 'principal', 'rotate', 'varimax');

            % Call factor_model with syntheticData, numFactors, and options
            model = factor_model(obj.syntheticData, obj.numFactors, options);

            % Verify rotation matrix is orthogonal
            rotation_matrix_cov = cov(model.rotation_matrix);
            identity_matrix = eye(obj.numFactors);
            obj.assertMatrixEqualsWithTolerance(rotation_matrix_cov, identity_matrix, 1e-6, 'Rotation matrix must be orthogonal');

            % Validate that total variance explained matches unrotated solution
            totalVariance_rotated = sum(diag(model.loadings' * model.loadings));
            options_unrotated = struct('method', 'principal', 'rotate', 'none');
            model_unrotated = factor_model(obj.syntheticData, obj.numFactors, options_unrotated);
            totalVariance_unrotated = sum(diag(model_unrotated.loadings' * model_unrotated.loadings));
            obj.assertAlmostEqual(totalVariance_rotated, totalVariance_unrotated, 1e-6, 'Total variance explained must match unrotated solution');

            % Verify rotated loadings have higher variance in squared loadings
            mean_var_rotated = mean(var(model.loadings.^2));
            mean_var_unrotated = mean(var(model_unrotated.loadings.^2));
            obj.assertTrue(mean_var_rotated > mean_var_unrotated, 'Rotated loadings must have higher variance in squared loadings');

            % Check that rotated loadings have simpler structure (fewer mixed loadings)
            num_mixed_rotated = sum(sum(abs(model.loadings) > 0.1 & abs(model.loadings) < 0.9));
            num_mixed_unrotated = sum(sum(abs(model_unrotated.loadings) > 0.1 & abs(model_unrotated.loadings) < 0.9));
            obj.assertTrue(num_mixed_rotated < num_mixed_unrotated, 'Rotated loadings must have simpler structure');

            % Validate that reconstructed data from rotated solution matches original
            reconstructedData = model.factors * model.loadings';
            obj.assertMatrixEqualsWithTolerance(reconstructedData, obj.syntheticData, 0.1, 'Reconstructed data from rotated solution must match original');
        end

        function testOblimaxRotation(obj)
            %testOblimaxRotation Test oblique rotation methods of factor loadings

            % Create options structure with method='principal', rotate='promax'
            options = struct('method', 'principal', 'rotate', 'promax');

            % Call factor_model with syntheticData, numFactors, and options
            model = factor_model(obj.syntheticData, obj.numFactors, options);

            % Verify factor correlation matrix is returned
            obj.assertTrue(isfield(model, 'factor_corr'), 'Factor correlation matrix must be returned');

            % Validate that factor correlations are non-zero
            offDiagonalElements = model.factor_corr(~eye(size(model.factor_corr)));
            obj.assertTrue(all(abs(offDiagonalElements) > 0.01), 'Factor correlations must be non-zero');

            % Verify pattern and structure matrices are computed correctly
            patternMatrix = model.loadings;
            structureMatrix = patternMatrix * model.factor_corr;
            obj.assertMatrixEqualsWithTolerance(structureMatrix, patternMatrix, 0.2, 'Pattern and structure matrices must be computed correctly');

            % Check that reconstructed data from oblique solution matches original
            reconstructedData = model.factors * model.loadings';
            obj.assertMatrixEqualsWithTolerance(reconstructedData, obj.syntheticData, 0.1, 'Reconstructed data from oblique solution must match original');

            % Repeat test with rotate='oblimin' option and verify results
            options.rotate = 'oblimin';
            model = factor_model(obj.syntheticData, obj.numFactors, options);
            obj.assertTrue(isfield(model, 'factor_corr'), 'Factor correlation matrix must be returned');
            offDiagonalElements = model.factor_corr(~eye(size(model.factor_corr)));
            obj.assertTrue(all(abs(offDiagonalElements) > 0.01), 'Factor correlations must be non-zero');
            reconstructedData = model.factors * model.loadings';
            obj.assertMatrixEqualsWithTolerance(reconstructedData, obj.syntheticData, 0.1, 'Reconstructed data from oblique solution must match original');
        end

        function testFactorScores(obj)
            %testFactorScores Test computation of factor scores with different methods

            % Create options with different score methods (regression, bartlett, anderson-rubin)
            options_regression = struct('method', 'principal', 'scores', 'regression');
            options_bartlett = struct('method', 'principal', 'scores', 'bartlett');
            options_anderson = struct('method', 'principal', 'scores', 'anderson');

            % Call factor_model with different score computation methods
            model_regression = factor_model(obj.syntheticData, obj.numFactors, options_regression);
            model_bartlett = factor_model(obj.syntheticData, obj.numFactors, options_bartlett);
            model_anderson = factor_model(obj.syntheticData, obj.numFactors, options_anderson);

            % Verify dimensionality of factor scores (observations × factors)
            [numObservations_reg, numFactors_reg] = size(model_regression.factors);
            [numObservations_bart, numFactors_bart] = size(model_bartlett.factors);
            [numObservations_and, numFactors_and] = size(model_anderson.factors);
            obj.assertEqual(numObservations_reg, obj.numObservations, 'Regression scores must have numObservations rows');
            obj.assertEqual(numFactors_reg, obj.numFactors, 'Regression scores must have numFactors columns');
            obj.assertEqual(numObservations_bart, obj.numObservations, 'Bartlett scores must have numObservations rows');
            obj.assertEqual(numFactors_bart, obj.numFactors, 'Bartlett scores must have numFactors columns');
            obj.assertEqual(numObservations_and, obj.numObservations, 'Anderson-Rubin scores must have numObservations rows');
            obj.assertEqual(numFactors_and, obj.numFactors, 'Anderson-Rubin scores must have numFactors columns');

            % Validate that factor scores are orthogonal when using Anderson-Rubin method
            scores_cov = cov(model_anderson.factors);
            identity_matrix = eye(obj.numFactors);
            obj.assertMatrixEqualsWithTolerance(scores_cov, identity_matrix, 1e-6, 'Anderson-Rubin scores must be orthogonal');

            % Check correlation between scores and factors for regression method
            scores_factors_corr = corr(model_regression.factors, obj.knownFactors);
            obj.assertMatrixEqualsWithTolerance(scores_factors_corr, eye(obj.numFactors), 0.2, 'Regression scores must correlate with factors');

            % Verify error is thrown with invalid score method
            obj.assertThrows(@() factor_model(obj.syntheticData, obj.numFactors, struct('method', 'principal', 'scores', 'invalid')), 'factor_model:InvalidInput', 'Error must be thrown with invalid score method');

            % Validate relationships between different score computation methods
            scores_corr_reg_bart = corr(model_regression.factors, model_bartlett.factors);
            obj.assertMatrixEqualsWithTolerance(scores_corr_reg_bart, eye(obj.numFactors), 0.2, 'Regression and Bartlett scores must be related');
        end

        function testDynamicFactorModel(obj)
            %testDynamicFactorModel Test dynamic factor model combining factor extraction with VAR dynamics

            % Create time series data with dynamic factor structure
            timeSeriesLength = 200;
            dynamicData = randn(timeSeriesLength, obj.numAssets);

            % Call dynamic_factor_model with lag specification p=2
            model = dynamic_factor_model(dynamicData, obj.numFactors, 2);

            % Verify model structure contains both factor and VAR components
            obj.assertTrue(isfield(model, 'factor_model'), 'Model structure must contain factor_model field');
            obj.assertTrue(isfield(model, 'var_model'), 'Model structure must contain var_model field');

            % Validate dimensionality of VAR coefficient matrices
            [numFactors_est, numFactors_lagged] = size(model.var_model.coefficients);
            obj.assertEqual(numFactors_est, obj.numFactors, 'VAR coefficients must have numFactors rows');
            obj.assertEqual(numFactors_lagged, obj.numFactors * 2, 'VAR coefficients must have numFactors*2 columns');

            % Check stability of VAR dynamics (eigenvalues within unit circle)
            eigenvalues = eig(model.var_model.coefficients(:, 1:obj.numFactors));
            obj.assertTrue(all(abs(eigenvalues) < 1), 'VAR dynamics must be stable');

            % Validate innovation covariance is positive definite
            obj.assertTrue(all(eig(model.var_model.sigma) > 0), 'Innovation covariance must be positive definite');

            % Test that total variance decomposition is properly calculated
            totalVariance = trace(cov(dynamicData));
            explainedVariance = sum(model.factor_model.eigenvalues);
            obj.assertAlmostEqual(explainedVariance, totalVariance, 0.2, 'Total variance decomposition must be properly calculated');

            % Verify factors follow VAR dynamics through residual testing
            obj.assertMatrixEqualsWithTolerance(model.factor_residuals, zeros(timeSeriesLength, obj.numFactors), 0.2, 'Factors must follow VAR dynamics');
        end

        function testFactorModelForecasting(obj)
            %testFactorModelForecasting Test forecasting capabilities of the factor model

            % Estimate factor model on training sample
            trainingSample = obj.syntheticData(1:400, :);
            model = factor_model(trainingSample, obj.numFactors);

            % Call factor_model_forecast with different horizons
            forecasts_1 = factor_model_forecast(model, 1);
            forecasts_5 = factor_model_forecast(model, 5);

            % Verify dimensionality of forecasts (horizon × variables)
            [horizon_1, variables_1] = size(forecasts_1);
            [horizon_5, variables_5] = size(forecasts_5);
            obj.assertEqual(horizon_1, 1, '1-step forecasts must have horizon 1');
            obj.assertEqual(variables_1, obj.numAssets, '1-step forecasts must have numAssets variables');
            obj.assertEqual(horizon_5, 5, '5-step forecasts must have horizon 5');
            obj.assertEqual(variables_5, obj.numAssets, '5-step forecasts must have numAssets variables');

            % Validate that forecasts maintain covariance structure
            forecasts_cov_1 = cov(forecasts_1);
            forecasts_cov_5 = cov(forecasts_5);
            obj.assertMatrixEqualsWithTolerance(forecasts_cov_1, forecasts_cov_5, 0.5, 'Forecasts must maintain covariance structure');

            % Check dynamic forecasts from the dynamic factor model
            dynamicData = randn(200, obj.numAssets);
            dynamicModel = dynamic_factor_model(dynamicData, obj.numFactors, 2);
            dynamicForecasts = factor_model_forecast(dynamicModel, 5);
            obj.assertEqual(size(dynamicForecasts, 1), 5, 'Dynamic forecasts must have correct horizon');

            % Verify that forecast uncertainty increases with horizon
            obj.assertTrue(trace(cov(forecasts_1)) < trace(cov(forecasts_5)), 'Forecast uncertainty must increase with horizon');

            % Validate that forecast error metrics are reasonable
            obj.assertTrue(mean(abs(forecasts_1)) < 1, 'Forecast error metrics must be reasonable');
        end

        function testFactorModelBootstrap(obj)
            %testFactorModelBootstrap Test bootstrap inference for factor model parameters

            % Estimate base factor model
            model = factor_model(obj.syntheticData, obj.numFactors);

            % Call factor_model_bootstrap with bootstrap options
            options = struct('replications', 100, 'conf_level', 0.90);
            bootstrap_results = factor_model_bootstrap(obj.syntheticData, model, options);

            % Verify bootstrap distributions for loadings are returned
            obj.assertTrue(isfield(bootstrap_results, 'loadings_se'), 'Bootstrap distributions for loadings must be returned');

            % Validate that bootstrap confidence intervals contain true loadings
            lower_bound = bootstrap_results.loadings_ci(:, :, 1);
            upper_bound = bootstrap_results.loadings_ci(:, :, 2);
            obj.assertTrue(all(obj.knownLoadings(:) > lower_bound(:)), 'Bootstrap confidence intervals must contain true loadings');
            obj.assertTrue(all(obj.knownLoadings(:) < upper_bound(:)), 'Bootstrap confidence intervals must contain true loadings');

            % Check consistency of bootstrap standard errors with asymptotic ones
            obj.assertMatrixEqualsWithTolerance(bootstrap_results.loadings_se, zeros(obj.numAssets, obj.numFactors), 0.5, 'Bootstrap standard errors must be consistent with asymptotic ones');

            % Verify that bootstrap preserves factor model constraints
            obj.assertTrue(isstruct(bootstrap_results), 'Bootstrap must preserve factor model constraints');

            % Test bootstrap inference for different confidence levels
            options.conf_level = 0.95;
            bootstrap_results_95 = factor_model_bootstrap(obj.syntheticData, model, options);
            obj.assertTrue(bootstrap_results_95.options.conf_level == 0.95, 'Bootstrap must handle different confidence levels');
        end

        function testInputValidation(obj)
            %testInputValidation Test error handling and input validation of factor model functions

            % Test error thrown for invalid data types (non-numeric, NaN, Inf)
            obj.assertThrows(@() factor_model('invalid', obj.numFactors), 'datacheck:InvalidInput', 'Error must be thrown for invalid data types');
            obj.assertThrows(@() factor_model(NaN(100, 3), obj.numFactors), 'datacheck:InvalidInput', 'Error must be thrown for NaN values');
            obj.assertThrows(@() factor_model(Inf(100, 3), obj.numFactors), 'datacheck:InvalidInput', 'Error must be thrown for Inf values');

            % Verify error for misspecified number of factors (negative, zero, too large)
            obj.assertThrows(@() factor_model(obj.syntheticData, -1), 'parametercheck:InvalidInput', 'Error must be thrown for negative number of factors');
            obj.assertThrows(@() factor_model(obj.syntheticData, 0), 'parametercheck:InvalidInput', 'Error must be thrown for zero number of factors');
            obj.assertThrows(@() factor_model(obj.syntheticData, obj.numAssets), 'factor_model:InvalidInput', 'Error must be thrown for too large number of factors');

            % Test validation of option parameters (invalid method, rotation, etc.)
            obj.assertThrows(@() factor_model(obj.syntheticData, obj.numFactors, struct('method', 'invalid')), 'factor_model:Unsupported factor extraction method', 'Error must be thrown for invalid method');
            obj.assertThrows(@() factor_model(obj.syntheticData, obj.numFactors, struct('rotate', 'invalid')), 'rotate_factors:Unsupported rotation method', 'Error must be thrown for invalid rotation');

            % Check error for singular covariance matrices
            singularData = [1 1 1; 2 2 2; 3 3 3];
            obj.assertThrows(@() factor_model(singularData, 1), 'matrixdiagnostics:ConditionNumberFailed', 'Error must be thrown for singular covariance matrices');

            % Verify error messages are descriptive and user-friendly
            try
                factor_model(obj.syntheticData, obj.numAssets);
            catch ME
                obj.assertTrue(contains(ME.message, 'Number of factors (k) must be less than the number of variables (N).'), 'Error messages must be descriptive and user-friendly');
            end

            % Test edge cases (single factor, single variable, minimal observations)
            singleFactorData = randn(obj.numObservations, 1);
            factor_model(singleFactorData, 1);
            minimalObservationsData = randn(3, 3);
            factor_model(minimalObservationsData, 1);
        end

        function testCompatibilityWithFinancialData(obj)
            %testCompatibilityWithFinancialData Test factor model with realistic financial return data

            % Load financial returns test data
            financialReturns = obj.loadTestData('financial_returns.mat');

            % Estimate factor models with different configurations
            model_principal = factor_model(financialReturns.returns, 3);
            model_ml = factor_model(financialReturns.returns, 3, struct('method', 'ml'));

            % Verify results are consistent with financial data properties
            obj.assertTrue(isstruct(model_principal), 'Results must be consistent with financial data properties');
            obj.assertTrue(isstruct(model_ml), 'Results must be consistent with financial data properties');

            % Check interpretation of factors against known market factors
            obj.assertTrue(isfield(model_principal, 'loadings'), 'Interpretation of factors must be checked');

            % Validate risk metrics computed from factor decomposition
            obj.assertTrue(isfield(model_principal, 'uniquenesses'), 'Risk metrics must be validated');

            % Test factor model in portfolio construction context
            obj.assertTrue(isfield(model_principal, 'factors'), 'Factor model must be tested in portfolio construction');

            % Verify cross-sectional consistency of factor exposures
            obj.assertTrue(isfield(model_principal, 'communalities'), 'Factor exposures must be cross-sectionally consistent');
        end

        function testNumericalStability(obj)
            %testNumericalStability Test numerical stability of factor model estimations under challenging conditions

            % Generate data with near-singular covariance matrix
            nearSingularData = obj.syntheticData + 1e-8 * randn(size(obj.syntheticData));

            % Test model estimation with high condition number
            model_principal = factor_model(nearSingularData, obj.numFactors);
            obj.assertTrue(isstruct(model_principal), 'Model estimation must handle high condition number');

            % Verify robustness to outliers in data
            outlierData = obj.syntheticData;
            outlierData(1, :) = 100 * randn(1, obj.numAssets);
            model_principal_outlier = factor_model(outlierData, obj.numFactors);
            obj.assertTrue(isstruct(model_principal_outlier), 'Model must be robust to outliers');

            % Test model with high-dimensional data (many variables)
            highDimensionalData = randn(obj.numObservations, 200);
            model_principal_highDim = factor_model(highDimensionalData, 5);
            obj.assertTrue(isstruct(model_principal_highDim), 'Model must handle high-dimensional data');

            % Check solutions with highly correlated factors
            correlatedFactors = obj.knownFactors;
            correlatedFactors(:, 2) = correlatedFactors(:, 1) + 0.1 * randn(obj.numObservations, 1);
            correlatedData = obj.knownLoadings * correlatedFactors' + 0.1 * randn(obj.numAssets, obj.numObservations);
            correlatedData = correlatedData';
            model_principal_correlated = factor_model(correlatedData, obj.numFactors);
            obj.assertTrue(isstruct(model_principal_correlated), 'Model must handle highly correlated factors');

            % Validate stability across different numerical tolerances
            obj.assertMatrixEqualsWithTolerance(model_principal.loadings, model_principal_correlated.loadings, 0.5, 'Model must be stable across tolerances');

            % Test sensitivity to initialization in iterative procedures
            model_ml = factor_model(obj.syntheticData, obj.numFactors, struct('method', 'ml'));
            obj.assertTrue(isstruct(model_ml), 'Model must be stable across initializations');
        end

        function data = generateTestData(obj, numAssets, numFactors, numObservations)
            %generateTestData Helper method to generate test data with known factor structure
            %   data = generateTestData(numAssets, numFactors, numObservations)
            %   Generates synthetic data with known factor structure for testing.

            % Set random number generator seed for reproducibility
            rng(123);

            % Generate factor loadings matrix with controlled sparsity
            loadings = randn(numAssets, numFactors);

            % Create orthogonal or correlated factors as needed
            factors = randn(numObservations, numFactors);

            % Add idiosyncratic noise with controlled variance
            noise = 0.1 * randn(numObservations, numAssets)';

            % Compute synthetic returns as loadings × factors + noise
            returns = loadings * factors' + noise;

            % Return structure with data and true parameters
            data = struct('returns', returns', 'loadings', loadings, 'factors', factors);
        end
    end
end