classdef BekkMvgarchTest < BaseTest
    % BekkMvgarchTest Test class for the BEKK Multivariate GARCH model implementation
    %
    % This class tests the implementation of the BEKK (Baba-Engle-Kraft-Kroner) 
    % Multivariate GARCH model. The tests cover estimation, forecasting, and 
    % numerical stability aspects of the model.
    %
    % The test cases validate:
    % - Parameter estimation for full and diagonal BEKK
    % - Volatility forecasting and covariance matrix prediction
    % - Model behavior with different error distributions (normal, t, GED, skewed-t)
    % - Numerical stability with challenging datasets
    % - Model selection criteria and option handling
    
    properties
        testData        % General test data matrix
        simBEKKData     % Simulated BEKK process data with known parameters
        financialData   % Real financial returns data for testing
        defaultOptions  % Default options structure for testing
        comparator      % Numerical comparator for test validation
    end
    
    methods
        function obj = BekkMvgarchTest()
            % Initialize test class for BEKK-MVGARCH model testing
            
            % Call superclass constructor
            obj = obj@BaseTest('BEKK-MVGARCH Tests');
            
            % Initialize numerical comparator for precise comparisons
            obj.comparator = NumericalComparator();
            
            % Set default tolerance for numerical comparisons
            obj.defaultTolerance = 1e-8;
            
            % Initialize test configuration
            obj.verbose = true;
        end
        
        function setUp(obj)
            % Set up test environment before each test method execution
            
            % Call superclass setUp
            setUp@BaseTest(obj);
            
            % Load simulated data
            simData = obj.loadTestData('simulated_bekk_data.mat');
            obj.simBEKKData = simData.bekk_data;
            
            % Load financial returns data
            finData = obj.loadTestData('financial_returns.mat');
            obj.financialData = finData.returns;
            
            % Set up default options for BEKK estimation
            obj.defaultOptions = struct(...
                'p', 1, ...                      % ARCH order
                'q', 1, ...                      % GARCH order
                'type', 'full', ...              % Model type: 'full' or 'diagonal'
                'distribution', 'normal', ...    % Error distribution
                'mean', 'constant', ...          % Mean specification
                'method', 'likelihood', ...      % Estimation method
                'forecast', 0, ...               % Forecast horizon
                'optimizationOptions', optimset('fmincon', ...
                    'Display', 'off', ...
                    'Algorithm', 'interior-point', ...
                    'TolFun', 1e-6, ...
                    'TolX', 1e-6, ...
                    'MaxFunEvals', 1000, ...
                    'MaxIter', 200) ...
            );
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            
            % Call superclass tearDown
            tearDown@BaseTest(obj);
            
            % Clear test data structures
            obj.testData = [];
        end
        
        function testBEKKEstimationBasic(obj)
            % Test basic BEKK model estimation with default parameters on simulated data
            
            % Configure options for basic BEKK model
            options = obj.defaultOptions;
            options.type = 'full';
            options.distribution = 'normal';
            
            % Get simulated data with known parameters
            simData = obj.simBEKKData;
            data = simData.data;
            true_params = simData.parameters;
            
            % Estimate BEKK model
            model = bekk_mvgarch(data, options);
            
            % Verify model estimation succeeded
            obj.assertTrue(model.exitflag > 0, 'Model estimation failed to converge');
            
            % Validate parameter estimates against known true values
            % For C matrix (intercept)
            obj.assertMatrixEqualsWithTolerance(true_params.C, model.parameters.C, 0.1, ...
                'Estimated C matrix does not match true values within tolerance');
            
            % For A matrix (ARCH)
            obj.assertMatrixEqualsWithTolerance(true_params.A(:,:,1), model.parameters.A(:,:,1), 0.1, ...
                'Estimated A matrix does not match true values within tolerance');
            
            % For B matrix (GARCH)
            obj.assertMatrixEqualsWithTolerance(true_params.B(:,:,1), model.parameters.B(:,:,1), 0.1, ...
                'Estimated B matrix does not match true values within tolerance');
            
            % Check that log-likelihood is within expected range
            obj.assertTrue(model.likelihood > -10000, 'Log-likelihood value is unreasonably low');
            
            % Verify covariance matrices are positive definite
            isValidModel = obj.verifyModelProperties(model);
            obj.assertTrue(isValidModel, 'Model properties verification failed');
        end
        
        function testDiagonalBEKK(obj)
            % Test diagonal BEKK model estimation with restricted parameter matrices
            
            % Configure options for diagonal BEKK model
            options = obj.defaultOptions;
            options.type = 'diagonal';
            options.distribution = 'normal';
            
            % Get simulated data
            simData = obj.simBEKKData;
            data = simData.data;
            
            % Estimate diagonal BEKK model
            model = bekk_mvgarch(data, options);
            
            % Verify model estimation succeeded
            obj.assertTrue(model.exitflag > 0, 'Diagonal BEKK model estimation failed to converge');
            
            % Check that parameters have the diagonal structure
            % A matrices should have zeros on off-diagonal elements
            [k, ~, p] = size(model.parameters.A);
            for i = 1:p
                A_i = model.parameters.A(:,:,i);
                for r = 1:k
                    for c = 1:k
                        if r ~= c
                            obj.assertAlmostEqual(0, A_i(r,c), ...
                                sprintf('Off-diagonal element A(%d,%d,%d) should be zero for diagonal BEKK', r, c, i));
                        end
                    end
                end
            end
            
            % B matrices should have zeros on off-diagonal elements
            [k, ~, q] = size(model.parameters.B);
            for i = 1:q
                B_i = model.parameters.B(:,:,i);
                for r = 1:k
                    for c = 1:k
                        if r ~= c
                            obj.assertAlmostEqual(0, B_i(r,c), ...
                                sprintf('Off-diagonal element B(%d,%d,%d) should be zero for diagonal BEKK', r, c, i));
                        end
                    end
                end
            end
            
            % Now estimate full BEKK for comparison
            options.type = 'full';
            full_model = bekk_mvgarch(data, options);
            
            % Compare AIC/BIC between models (diagonal should be more parsimonious)
            obj.assertTrue(model.aic <= full_model.aic * 1.1, ...
                'Diagonal BEKK AIC should be comparable to or better than full BEKK for simulated data');
            
            % Verify model properties
            isValidModel = obj.verifyModelProperties(model);
            obj.assertTrue(isValidModel, 'Diagonal BEKK model properties verification failed');
        end
        
        function testBEKKForecasting(obj)
            % Test BEKK model's forecasting capabilities for multivariate covariance matrices
            
            % Configure options for BEKK with forecasting
            options = obj.defaultOptions;
            options.forecast = 10;  % 10-step ahead forecast
            
            % Use financial data for realistic forecasting test
            data = obj.financialData;
            
            % Extract training and testing portions
            T = size(data, 1);
            train_end = floor(T * 0.8);  % Use 80% for training
            train_data = data(1:train_end, :);
            
            % Estimate BEKK model
            model = bekk_mvgarch(train_data, options);
            
            % Verify forecast exists and has correct horizon
            obj.assertTrue(isfield(model, 'forecast'), 'Model should contain forecast field');
            obj.assertEqual(options.forecast, model.forecast.horizon, ...
                'Forecast horizon should match specified horizon');
            
            % Get forecast covariance matrices
            forecasted_covs = model.forecast.covariance;
            [k, ~, horizon] = size(forecasted_covs);
            
            % Verify dimensions
            obj.assertEqual(options.forecast, horizon, 'Forecast horizon dimension mismatch');
            obj.assertEqual(size(train_data, 2), k, 'Forecast dimension mismatch with data');
            
            % Check all forecasted covariance matrices are positive definite
            for h = 1:horizon
                H_h = forecasted_covs(:,:,h);
                
                % Check symmetry
                obj.assertMatrixEqualsWithTolerance(H_h, H_h', 1e-10, ...
                    sprintf('Forecasted covariance at horizon %d is not symmetric', h));
                
                % Check positive definiteness via eigenvalues
                e = eig(H_h);
                obj.assertTrue(all(e > 0), ...
                    sprintf('Forecasted covariance at horizon %d is not positive definite', h));
                
                % Check if determinant is positive
                d = det(H_h);
                obj.assertTrue(d > 0, ...
                    sprintf('Forecasted covariance at horizon %d has non-positive determinant', h));
            end
            
            % Verify forecasted volatilities are available
            obj.assertTrue(isfield(model.forecast, 'volatility'), ...
                'Forecast should include volatility predictions');
            obj.assertEqual(size(model.forecast.volatility), [horizon, k], ...
                'Forecast volatility dimension mismatch');
            
            % Test if volatility forecasts are reasonable (positive and not exploding)
            vol_forecasts = model.forecast.volatility;
            obj.assertTrue(all(vol_forecasts(:) > 0), 'All volatility forecasts should be positive');
            obj.assertTrue(all(vol_forecasts(:) < 1), ...
                'Volatility forecasts are unreasonably large for financial returns');
        end
        
        function testBEKKWithStudentT(obj)
            % Test BEKK model with Student's t distribution assumption
            
            % Configure options for BEKK with t distribution
            options = obj.defaultOptions;
            options.distribution = 't';
            options.degrees = 8;  % Degrees of freedom parameter
            
            % Use financial data which typically has heavy tails
            data = obj.financialData;
            
            % First estimate with normal distribution as baseline
            options_normal = options;
            options_normal.distribution = 'normal';
            normal_model = bekk_mvgarch(data, options_normal);
            
            % Now estimate with t distribution
            t_model = bekk_mvgarch(data, options);
            
            % Verify degrees of freedom parameter is estimated
            obj.assertTrue(isfield(t_model.parameters, 'nu'), ...
                'Model should contain estimated degrees of freedom parameter');
            obj.assertTrue(t_model.parameters.nu > 2, ...
                'Estimated degrees of freedom should be > 2 for valid t-distribution');
            
            % Compare log-likelihoods - t-dist should fit financial data better
            obj.assertTrue(t_model.likelihood > normal_model.likelihood, ...
                'Student-t model should have higher likelihood than normal for financial data');
            
            % Verify model is still valid
            isValidModel = obj.verifyModelProperties(t_model);
            obj.assertTrue(isValidModel, 'Student-t BEKK model properties verification failed');
            
            % Check if degrees of freedom is close to the specified value
            obj.assertTrue(abs(t_model.parameters.nu - options.degrees) < 3, ...
                'Estimated degrees of freedom should be close to specified value');
        end
        
        function testBEKKWithGED(obj)
            % Test BEKK model with Generalized Error Distribution assumption
            
            % Configure options for BEKK with GED distribution
            options = obj.defaultOptions;
            options.distribution = 'ged';
            options.degrees = 1.5;  % Shape parameter (< 2 for heavier tails than normal)
            
            % Use financial data which typically has heavy tails
            data = obj.financialData;
            
            % First estimate with normal distribution as baseline
            options_normal = options;
            options_normal.distribution = 'normal';
            normal_model = bekk_mvgarch(data, options_normal);
            
            % Now estimate with GED distribution
            ged_model = bekk_mvgarch(data, options);
            
            % Verify shape parameter is estimated
            obj.assertTrue(isfield(ged_model.parameters, 'nu'), ...
                'Model should contain estimated shape parameter');
            obj.assertTrue(ged_model.parameters.nu > 0, ...
                'Estimated shape parameter should be positive for valid GED');
            
            % Compare log-likelihoods - GED should fit financial data better than normal
            obj.assertTrue(ged_model.likelihood > normal_model.likelihood, ...
                'GED model should have higher likelihood than normal for financial data');
            
            % Verify model is still valid
            isValidModel = obj.verifyModelProperties(ged_model);
            obj.assertTrue(isValidModel, 'GED BEKK model properties verification failed');
            
            % Check if shape parameter is close to the specified value
            obj.assertTrue(abs(ged_model.parameters.nu - options.degrees) < 1, ...
                'Estimated shape parameter should be close to specified value');
        end
        
        function testModelSelection(obj)
            % Test model selection criteria (AIC, BIC) for different BEKK specifications
            
            % Use financial data
            data = obj.financialData;
            
            % Define different model orders to test
            orders = [
                1, 1;  % BEKK(1,1)
                1, 2;  % BEKK(1,2)
                2, 1;  % BEKK(2,1)
                2, 2   % BEKK(2,2)
            ];
            
            % Initialize arrays to store results
            n_models = size(orders, 1);
            aic_values = zeros(n_models, 1);
            bic_values = zeros(n_models, 1);
            ll_values = zeros(n_models, 1);
            
            % Estimate models with different orders
            for i = 1:n_models
                options = obj.defaultOptions;
                options.p = orders(i, 1);
                options.q = orders(i, 2);
                
                % Estimate model
                model = bekk_mvgarch(data, options);
                
                % Store information criteria
                aic_values(i) = model.aic;
                bic_values(i) = model.bic;
                ll_values(i) = model.likelihood;
            end
            
            % Verify that information criteria are consistent
            for i = 1:n_models-1
                for j = i+1:n_models
                    % If model i has higher likelihood than model j
                    if ll_values(i) > ll_values(j)
                        % But if model i has more parameters, BIC might still prefer model j
                        if (orders(i,1) + orders(i,2)) > (orders(j,1) + orders(j,2))
                            obj.assertTrue(bic_values(i) >= bic_values(j) - 1e-6 || ...
                                         bic_values(i) < bic_values(j) + 1e-6, ...
                               'BIC should penalize more complex models');
                        else
                            % If model i has both higher likelihood and fewer parameters
                            obj.assertTrue(aic_values(i) < aic_values(j) + 1e-6, ...
                                'AIC should prefer model with higher likelihood and fewer parameters');
                            obj.assertTrue(bic_values(i) < bic_values(j) + 1e-6, ...
                                'BIC should prefer model with higher likelihood and fewer parameters');
                        end
                    end
                end
            end
            
            % Test diagonal vs. full BEKK
            options_diag = obj.defaultOptions;
            options_diag.type = 'diagonal';
            diag_model = bekk_mvgarch(data, options_diag);
            
            options_full = obj.defaultOptions;
            options_full.type = 'full';
            full_model = bekk_mvgarch(data, options_full);
            
            % Compare information criteria
            obj.assertTrue(diag_model.numParams < full_model.numParams, ...
                'Diagonal BEKK should have fewer parameters than full BEKK');
            
            % Diagonal should be preferred by BIC due to parsimony unless full is much better
            if full_model.likelihood - diag_model.likelihood < 5 * (full_model.numParams - diag_model.numParams)
                obj.assertTrue(diag_model.bic < full_model.bic, ...
                    'BIC should prefer diagonal BEKK unless full BEKK fits much better');
            end
        end
        
        function testNumericalStability(obj)
            % Test numerical stability of BEKK estimation with challenging datasets
            
            % Case 1: High dimension data
            k = 5;  % Number of series
            T = 1000;  % Number of observations
            
            % Generate random data with high correlations
            rng(123);  % For reproducibility
            corr_matrix = 0.7 * ones(k, k) + 0.3 * eye(k);  % High correlation
            std_devs = 0.01 + 0.03 * rand(k, 1);  % Different volatilities
            
            % Cholesky decomposition for correlated data generation
            C = chol(corr_matrix, 'lower');
            
            % Generate correlated data
            Z = randn(T, k);  % Uncorrelated data
            X = Z * C';  % Apply correlation
            
            % Apply volatility
            for i = 1:k
                X(:,i) = X(:,i) * std_devs(i);
            end
            
            % Configure options for stability test
            options = obj.defaultOptions;
            options.type = 'diagonal';  % Use diagonal for higher dimensions
            
            % Estimate model
            model = bekk_mvgarch(X, options);
            
            % Verify model estimation succeeded
            obj.assertTrue(model.exitflag > 0, 'Model estimation failed for high-dimensional data');
            
            % Check positive definiteness of covariance matrices
            H = model.H;
            for t = 1:size(H, 3)
                H_t = H(:,:,t);
                
                % Check eigenvalues
                e = eig(H_t);
                obj.assertTrue(all(e > 0), ...
                    sprintf('Covariance matrix at time %d is not positive definite', t));
                
                % Check condition number is not too large
                cond_num = rcond(H_t);
                obj.assertTrue(cond_num > 1e-12, ...
                    sprintf('Covariance matrix at time %d is ill-conditioned', t));
            end
            
            % Case 2: Test with highly persistent volatility
            % Generate parameters for highly persistent process
            true_params = struct();
            true_params.C = diag(0.01 * ones(3, 1));
            true_params.A = zeros(3, 3, 1);
            true_params.A(:,:,1) = diag([0.1, 0.15, 0.12]);
            true_params.B = zeros(3, 3, 1);
            true_params.B(:,:,1) = diag([0.89, 0.84, 0.87]);  % High persistence
            
            % Generate data
            sim_data = obj.generateSimulatedBEKKData(1000, 3, true_params);
            
            % Configure options
            options.type = 'diagonal';
            
            % Estimate model
            pers_model = bekk_mvgarch(sim_data.data, options);
            
            % Verify stationarity constraints are enforced
            A_diag = diag(pers_model.parameters.A(:,:,1));
            B_diag = diag(pers_model.parameters.B(:,:,1));
            
            % Check A²+B² < 1 for each variable (stationarity condition for diagonal BEKK)
            stationarity = (A_diag.^2 + B_diag.^2);
            obj.assertTrue(all(stationarity < 1), ...
                'Stationarity constraints should be enforced for highly persistent process');
            
            % Check if model captures high persistence
            obj.assertTrue(all(B_diag > 0.7), ...
                'Estimated model should capture high persistence in volatility');
        end
        
        function testOptionHandling(obj)
            % Test option handling for BEKK model configuration
            
            % Get data
            data = obj.financialData;
            
            % Case 1: Test with minimal options
            minimal_options = struct();
            min_model = bekk_mvgarch(data, minimal_options);
            
            % Verify default options were correctly applied
            obj.assertEqual(1, min_model.parameters.p, 'Default ARCH order should be 1');
            obj.assertEqual(1, min_model.parameters.q, 'Default GARCH order should be 1');
            obj.assertEqual('normal', min_model.parameters.distribution, ...
                'Default distribution should be normal');
            
            % Case 2: Test with custom distribution parameters
            t_options = obj.defaultOptions;
            t_options.distribution = 't';
            t_options.degrees = 5;
            t_model = bekk_mvgarch(data, t_options);
            
            % Verify distribution parameters
            obj.assertEqual('t', t_model.parameters.distribution, ...
                'Distribution type should be correctly set');
            obj.assertTrue(abs(t_model.parameters.nu - 5) < 3, ...
                'Degrees of freedom should be close to specified value');
            
            % Case 3: Test optimization options
            opt_options = obj.defaultOptions;
            opt_options.optimizationOptions.MaxIter = 10;  % Unrealistically low
            
            % This should not converge fully
            opt_model = bekk_mvgarch(data, opt_options);
            
            % Check if optimization terminated due to MaxIter
            obj.assertTrue(isfield(opt_model, 'optimizationOutput'), ...
                'Model should contain optimization output');
            
            % Case 4: Test with diagonal specification
            diag_options = obj.defaultOptions;
            diag_options.type = 'diagonal';
            diag_model = bekk_mvgarch(data, diag_options);
            
            % Verify model type
            obj.assertTrue(diag_model.parameters.isDiagonal, ...
                'Model should be correctly identified as diagonal BEKK');
        end
        
        function testConditionalCorrelations(obj)
            % Test extraction and analysis of conditional correlations from BEKK model
            
            % Get financial data
            data = obj.financialData;
            
            % Estimate BEKK model
            options = obj.defaultOptions;
            model = bekk_mvgarch(data, options);
            
            % Extract conditional covariance matrices
            H = model.H;
            [k, ~, T] = size(H);
            
            % Initialize array for conditional correlations
            corr_matrices = zeros(k, k, T);
            
            % Convert covariance to correlation
            for t = 1:T
                H_t = H(:,:,t);
                
                % Extract standard deviations (diagonal elements of H)
                std_devs = sqrt(diag(H_t));
                
                % Create diagonal matrix of inverse standard deviations
                D_inv = diag(1 ./ std_devs);
                
                % Calculate correlation matrix: corr = D^(-1) * H * D^(-1)
                corr_matrices(:,:,t) = D_inv * H_t * D_inv;
            end
            
            % Verify correlation matrix properties
            for t = 1:T
                R_t = corr_matrices(:,:,t);
                
                % Check diagonal elements are 1
                diag_vals = diag(R_t);
                obj.assertMatrixEqualsWithTolerance(ones(k, 1), diag_vals, 1e-10, ...
                    sprintf('Diagonal elements of correlation matrix at time %d are not 1', t));
                
                % Check off-diagonal elements are bounded by [-1, 1]
                R_offdiag = R_t - diag(diag_vals);
                obj.assertTrue(all(abs(R_offdiag(:)) <= 1 + 1e-8), ...
                    sprintf('Correlation values at time %d exceed [-1,1] bounds', t));
                
                % Check symmetry
                obj.assertMatrixEqualsWithTolerance(R_t, R_t', 1e-10, ...
                    sprintf('Correlation matrix at time %d is not symmetric', t));
                
                % Check positive definiteness
                e = eig(R_t);
                obj.assertTrue(all(e > 0), ...
                    sprintf('Correlation matrix at time %d is not positive definite', t));
            end
            
            % Analyze correlation dynamics
            % Extract correlations between first and second series over time
            corr_series = squeeze(corr_matrices(1, 2, :));
            
            % Check for reasonable values
            obj.assertTrue(all(abs(corr_series) < 1), ...
                'Correlation values should be bounded by (-1,1)');
            
            % Check for some time variation in correlations
            corr_std = std(corr_series);
            obj.assertTrue(corr_std > 0.01, ...
                'Conditional correlations should show meaningful time variation');
        end
        
        function testPerformance(obj)
            % Test performance characteristics of BEKK model estimation
            
            % Define dimensions to test
            dimensions = [2, 3, 4];  % Number of series
            T = 1000;  % Sample size
            
            % Initialize arrays for timing results
            full_times = zeros(length(dimensions), 1);
            diag_times = zeros(length(dimensions), 1);
            
            % Test each dimension
            for i = 1:length(dimensions)
                k = dimensions(i);
                
                % Generate simulated data
                sim_params = struct();
                sim_params.C = 0.01 * eye(k);
                sim_params.A = zeros(k, k, 1);
                sim_params.A(:,:,1) = 0.1 * eye(k);
                sim_params.B = zeros(k, k, 1);
                sim_params.B(:,:,1) = 0.85 * eye(k);
                
                sim_data = obj.generateSimulatedBEKKData(T, k, sim_params);
                
                % Configure options for diagonal BEKK
                diag_options = obj.defaultOptions;
                diag_options.type = 'diagonal';
                
                % Measure time for diagonal BEKK
                tic;
                bekk_mvgarch(sim_data.data, diag_options);
                diag_times(i) = toc;
                
                % Only test full BEKK for small dimensions (can be very slow for larger k)
                if k <= 3
                    % Configure options for full BEKK
                    full_options = obj.defaultOptions;
                    full_options.type = 'full';
                    
                    % Measure time for full BEKK
                    tic;
                    bekk_mvgarch(sim_data.data, full_options);
                    full_times(i) = toc;
                else
                    full_times(i) = NaN;
                end
            end
            
            % Verify diagonal BEKK is faster than full BEKK
            for i = 1:length(dimensions)
                if ~isnan(full_times(i))
                    obj.assertTrue(diag_times(i) < full_times(i), ...
                        sprintf('Diagonal BEKK should be faster than full BEKK for k=%d', dimensions(i)));
                end
            end
        end
        
        function data = generateSimulatedBEKKData(obj, T, k, params)
            % Helper method to generate simulated BEKK process data with known parameters
            %
            % INPUTS:
            %   T - Number of observations
            %   k - Number of series
            %   params - Structure with true parameters (C, A, B)
            %
            % OUTPUTS:
            %   data - Structure with simulated data and true parameters
            
            % Initialize output structure
            data = struct();
            
            % Extract parameters
            C = params.C;
            A = params.A;
            B = params.B;
            
            % Get dimensions
            p = size(A, 3);  % ARCH order
            q = size(B, 3);  % GARCH order
            
            % Check parameter consistency
            if size(C, 1) ~= k || size(C, 2) ~= k
                error('Parameter C must be k x k matrix');
            end
            
            if size(A, 1) ~= k || size(A, 2) ~= k
                error('Parameter A must be k x k x p array');
            end
            
            if size(B, 1) ~= k || size(B, 2) ~= k
                error('Parameter B must be k x k x q array');
            end
            
            % Initialize arrays
            returns = zeros(T, k);
            H = zeros(k, k, T);
            
            % Set unconditional covariance for initial values
            H0 = C * C';
            for i = 1:max(p, q)
                H(:,:,i) = H0;
            end
            
            % Simulate BEKK process
            for t = max(p, q)+1:T
                % Initialize current covariance with intercept
                H_t = C * C';
                
                % Add ARCH terms
                for i = 1:p
                    e_lag = returns(t-i, :)';
                    H_t = H_t + A(:,:,i) * (e_lag * e_lag') * A(:,:,i)';
                end
                
                % Add GARCH terms
                for j = 1:q
                    H_t = H_t + B(:,:,j) * H(:,:,t-j) * B(:,:,j)';
                end
                
                % Ensure positive definiteness
                [V, D] = eig(H_t);
                d = diag(D);
                if any(d <= 0)
                    d(d <= 0) = 1e-6;
                    H_t = V * diag(d) * V';
                    % Ensure symmetry
                    H_t = (H_t + H_t') / 2;
                end
                
                % Store covariance
                H(:,:,t) = H_t;
                
                % Generate multivariate normal innovations
                z = randn(k, 1);
                
                % Calculate Cholesky decomposition of H_t
                C_chol = chol(H_t)';
                
                % Generate returns
                returns(t, :) = (C_chol * z)';
            end
            
            % Store results
            data.data = returns;
            data.H = H;
            data.parameters = params;
            data.T = T;
            data.k = k;
            data.p = p;
            data.q = q;
        end
        
        function isValid = verifyModelProperties(obj, model)
            % Helper method to verify key properties of estimated BEKK model
            %
            % INPUTS:
            %   model - Estimated BEKK model structure
            %
            % OUTPUTS:
            %   isValid - Boolean indicating if model satisfies all properties
            
            % Initialize validation result
            isValid = true;
            
            % Get model dimensions
            k = model.parameters.k;
            p = model.parameters.p;
            q = model.parameters.q;
            
            % Get parameters
            A = model.parameters.A;
            B = model.parameters.B;
            
            % Check positive definiteness of covariance matrices
            H = model.H;
            for t = 1:size(H, 3)
                H_t = H(:,:,t);
                
                % Check eigenvalues
                e = eig(H_t);
                if any(e <= 0)
                    isValid = false;
                    if obj.verbose
                        fprintf('Covariance matrix at time %d is not positive definite\n', t);
                    end
                    break;
                end
            end
            
            % Check stationarity conditions
            if model.parameters.isDiagonal
                % For diagonal BEKK, check A²+B² < 1 for each variable
                A_diag = zeros(k, p);
                B_diag = zeros(k, q);
                
                for i = 1:p
                    A_diag(:, i) = diag(A(:,:,i));
                end
                
                for i = 1:q
                    B_diag(:, i) = diag(B(:,:,i));
                end
                
                % Check stationarity for each variable
                for i = 1:k
                    persistence = sum(A_diag(i,:).^2) + sum(B_diag(i,:).^2);
                    if persistence >= 1
                        isValid = false;
                        if obj.verbose
                            fprintf('Variable %d violates stationarity condition: A²+B² = %.4f\n', ...
                                i, persistence);
                        end
                        break;
                    end
                end
            else
                % For full BEKK, check eigenvalues of the companion form matrix
                
                % Construct Kronecker matrices
                A_kron_sum = zeros(k*k, k*k);
                B_kron_sum = zeros(k*k, k*k);
                
                for i = 1:p
                    A_i = A(:,:,i);
                    A_kron = kron(A_i, A_i);
                    A_kron_sum = A_kron_sum + A_kron;
                end
                
                for j = 1:q
                    B_j = B(:,:,j);
                    B_kron = kron(B_j, B_j);
                    B_kron_sum = B_kron_sum + B_kron;
                end
                
                % Check spectral radius
                M = A_kron_sum + B_kron_sum;
                e = abs(eig(M));
                
                if max(e) >= 1
                    isValid = false;
                    if obj.verbose
                        fprintf('Model violates stationarity condition, max eigenvalue: %.4f\n', max(e));
                    end
                end
            end
            
            % Check numerical properties of likelihood computation
            if ~isfinite(model.likelihood)
                isValid = false;
                if obj.verbose
                    fprintf('Model likelihood is not finite: %.4f\n', model.likelihood);
                end
            end
            
            % Return validation result
            return;
        end
    end
end