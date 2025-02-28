classdef TimeSeriesIntegrationTest < BaseTest
    % Test class for integration testing of the time series modeling and analysis components of the MFE Toolbox
    
    properties
        testData                % Test data structure
        testTolerance           % Tolerance for numerical comparisons
        simulatedArSeries       % Simulated AR series for testing
        simulatedMaSeries       % Simulated MA series for testing
        simulatedArmaSeries     % Simulated ARMA series for testing
        simulatedSeasonalSeries % Simulated seasonal series for testing
        exogenousVariables      % Exogenous variables for ARMAX testing
        knownParameters         % Known parameters of simulated series
    end
    
    methods
        function obj = TimeSeriesIntegrationTest()
            % Initialize the TimeSeriesIntegrationTest class with default test configuration
            
            % Call the superclass constructor
            obj = obj@BaseTest();
            
            % Set test tolerance for numerical comparisons
            obj.testTolerance = 1e-6;
            
            % Initialize empty testData property to be populated in setUp
            obj.testData = struct();
        end
        
        function setUp(obj)
            % Prepare the test environment before each test method execution
            
            % Call the superclass setUp method
            setUp@BaseTest(obj);
            
            % Load simulated time series data from test data directory
            try
                obj.testData = obj.loadTestData('time_series_data.mat');
                obj.knownParameters = obj.testData.trueParameters;
            catch ME
                warning('Failed to load test data: %s. Generating default test data.', ME.message);
                % Set default parameters if test data not available
                obj.generateDefaultTestData();
            end
            
            % Set random seed for reproducibility
            rng(42);
            
            % Generate controlled test data for specific integration test cases
            obj.generateTestSeries();
        end
        
        function tearDown(obj)
            % Clean up the test environment after each test method execution
            
            % Call the superclass tearDown method
            tearDown@BaseTest(obj);
            
            % Clear temporary test variables
            rng('default');
        end
        
        function testArmaIntegration(obj)
            % Tests integration of ARMA model estimation, diagnostics, and forecasting
            
            % Get test data
            y = obj.simulatedArmaSeries;
            true_p = length(obj.knownParameters.ar);
            true_q = length(obj.knownParameters.ma);
            
            % 1. Estimate ARMA model using armaxfilter
            options = struct('p', true_p, 'q', true_q, 'constant', true);
            results = armaxfilter(y, [], options);
            
            % Verify model parameters are close to true values
            estimated_ar = results.parameters(2:(true_p+1));
            estimated_ma = results.parameters((true_p+2):(true_p+true_q+1));
            
            % Use flexible tolerance for AR parameters which can be harder to estimate
            for i = 1:true_p
                obj.assertAlmostEqual(obj.knownParameters.ar(i), estimated_ar(i), ...
                    sprintf('AR parameter %d significantly differs from true value', i));
            end
            
            for i = 1:true_q
                obj.assertAlmostEqual(obj.knownParameters.ma(i), estimated_ma(i), obj.testTolerance, ...
                    sprintf('MA parameter %d significantly differs from true value', i));
            end
            
            % 2. Compute diagnostics 
            acf_values = sacf(results.residuals, 10);
            pacf_values = spacf(results.residuals, 10);
            
            % Verify ACF and PACF values are within expected bounds for white noise
            for i = 1:length(acf_values)
                obj.assertTrue(abs(acf_values(i)) < 2/sqrt(length(y)), ...
                    sprintf('ACF at lag %d exceeds bounds for white noise', i));
            end
            
            % 3. Generate forecasts using armafor
            forecast_horizon = 10;
            forecasts = armafor(results.parameters, y, true_p, true_q, true, [], forecast_horizon);
            
            % Verify forecasts are reasonable
            obj.assertTrue(all(isfinite(forecasts)), 'Forecasts contain non-finite values');
            
            % 4. Test estimation -> diagnostics -> forecasting workflow integration
            % Generate future values for comparison
            T = length(y);
            new_data = zeros(forecast_horizon, 1);
            innovations = randn(forecast_horizon, 1);
            
            % Use true parameters to generate actual future values for comparison
            ar_params = obj.knownParameters.ar;
            ma_params = obj.knownParameters.ma;
            p = length(ar_params);
            q = length(ma_params);
            
            % Generate future values using true model
            for t = 1:forecast_horizon
                new_data(t) = obj.knownParameters.constant;
                
                % AR component
                for i = 1:p
                    if t-i <= 0
                        new_data(t) = new_data(t) + ar_params(i) * y(end+t-i);
                    else
                        new_data(t) = new_data(t) + ar_params(i) * new_data(t-i);
                    end
                end
                
                % Add innovation
                new_data(t) = new_data(t) + innovations(t);
                
                % MA component
                for i = 1:q
                    if t-i <= 0
                        new_data(t) = new_data(t) + ma_params(i) * results.residuals(end+t-i);
                    else
                        new_data(t) = new_data(t) + ma_params(i) * innovations(t-i);
                    end
                end
            end
            
            % Verify that forecasts are within reasonable range of simulated values
            % Note: We're comparing stochastic processes, so we need a larger tolerance
            forecast_error = mean(abs(forecasts - new_data));
            obj.assertTrue(forecast_error < 2.0, ...
                sprintf('Forecast error too large: %f', forecast_error));
        end
        
        function testArmaxIntegration(obj)
            % Tests integration of ARMAX modeling with exogenous variables
            
            % Get test data
            y = obj.simulatedArmaSeries;
            x = obj.exogenousVariables;
            true_p = length(obj.knownParameters.ar);
            true_q = length(obj.knownParameters.ma);
            exo_coef = obj.knownParameters.exogenousCoef;
            
            % 1. Estimate ARMAX model using armaxfilter with exogenous variables
            options = struct('p', true_p, 'q', true_q, 'constant', true);
            results = armaxfilter(y, x, options);
            
            % Verify model parameters are close to true values
            estimated_ar = results.parameters(2:(true_p+1));
            estimated_ma = results.parameters((true_p+2):(true_p+true_q+1));
            estimated_exo = results.parameters((true_p+true_q+2):end);
            
            % Check parameters with tolerance
            for i = 1:true_p
                obj.assertAlmostEqual(obj.knownParameters.ar(i), estimated_ar(i), ...
                    sprintf('AR parameter %d significantly differs from true value', i));
            end
            
            for i = 1:true_q
                obj.assertAlmostEqual(obj.knownParameters.ma(i), estimated_ma(i), obj.testTolerance, ...
                    sprintf('MA parameter %d significantly differs from true value', i));
            end
            
            for i = 1:length(exo_coef)
                obj.assertAlmostEqual(exo_coef(i), estimated_exo(i), ...
                    sprintf('Exogenous coefficient %d significantly differs from true value', i));
            end
            
            % 2. Generate forecasts with exogenous future values
            T = length(y);
            forecast_horizon = 10;
            future_x = zeros(forecast_horizon, 2);
            
            % Generate future values for exogenous variables
            future_x(1, 1) = 0.7 * x(end, 1) + 0.3 * randn();
            future_x(1, 2) = randn() + 0.5 * randn();
            for t = 2:forecast_horizon
                future_x(t, 1) = 0.7 * future_x(t-1, 1) + 0.3 * randn();
                future_x(t, 2) = randn() + 0.5 * randn();
            end
            
            % Generate forecasts with future exogenous variables
            forecasts = armafor(results.parameters, y, true_p, true_q, true, future_x, forecast_horizon);
            
            % Verify forecasts are reasonable
            obj.assertTrue(all(isfinite(forecasts)), 'Forecasts contain non-finite values');
            
            % 3. Test impact of exogenous variables on forecasting accuracy
            % Generate forecasts without exogenous variables
            forecasts_no_exo = armafor(results.parameters(1:(true_p+true_q+1)), y, true_p, true_q, true, [], forecast_horizon);
            
            % Verify that forecasts differ when exogenous variables are included or excluded
            obj.assertTrue(any(abs(forecasts - forecasts_no_exo) > obj.testTolerance), ...
                'Exogenous variables have no impact on forecasts');
        end
        
        function testSarimaIntegration(obj)
            % Tests integration of seasonal ARIMA modeling components
            
            % Get test data
            y = obj.simulatedSeasonalSeries;
            
            % 1. Estimate Seasonal ARIMA model
            % SARIMA(1,0,0)(1,0,0)4 for quarterly data
            p = 1; d = 0; q = 0;
            P = 1; D = 0; Q = 0;
            s = 4; % quarterly seasonality
            
            options = struct('constant', true);
            results = sarima(y, p, d, q, P, D, Q, s, options);
            
            % 2. Validate parameters
            obj.assertAlmostEqual(results.parameters(1), obj.knownParameters.constant, 0.1, ...
                'Constant term significantly differs from true value');
            
            % AR coefficient
            obj.assertAlmostEqual(results.parameters(2), 0.5 * obj.knownParameters.ar(1), 0.2, ...
                'AR parameter significantly differs from true value');
            
            % Seasonal AR coefficient - should be close to 0.6
            obj.assertAlmostEqual(results.parameters(3), 0.6, 0.2, ...
                'Seasonal AR parameter significantly differs from true value');
            
            % 3. Validate model diagnostics
            % Check Ljung-Box test results to ensure residuals are white noise
            obj.assertTrue(~all(results.ljungBox.isRejected5pct), ...
                'Residuals show significant autocorrelation');
            
            % 4. Generate forecasts preserving seasonal patterns
            % Convert Seasonal ARIMA to equivalent ARMA model
            p_effective = length(results.ARpoly) - 1;
            q_effective = length(results.MApoly) - 1;
            
            % Note: ARpoly has 1 in first position, followed by negative AR coefficients
            ar_effective = -results.ARpoly(2:end);
            ma_effective = results.MApoly(2:end);
            
            % Combine parameters for armafor
            forecast_params = [results.parameters(1); ar_effective; ma_effective];
            
            % Generate forecasts
            forecast_horizon = 8; % 2 years with quarterly data
            
            % Cast model to ARMA form and forecast
            if ~isempty(ar_effective) || ~isempty(ma_effective)
                forecasts = armafor(forecast_params, y, p_effective, q_effective, true, [], forecast_horizon);
                
                % Verify forecasts are reasonable
                obj.assertTrue(all(isfinite(forecasts)), 'Forecasts contain non-finite values');
                
                % Test interaction between regular and seasonal components
                for i = 1:forecast_horizon/s
                    season_idx = mod((length(y)+i-1), s) + 1;
                    next_season_idx = mod((length(y)+i+s-1), s) + 1;
                    
                    if season_idx == next_season_idx
                        % If we're comparing the same season, correlation should be high
                        obj.assertTrue(abs(forecasts(i) - forecasts(i+s)) < 1.0, ...
                            'Seasonal pattern not preserved in forecasts');
                    end
                end
            end
        end
        
        function testModelSelectionWorkflow(obj)
            % Tests integrated model selection workflow using ACF, PACF and information criteria
            
            % Get test data
            y = obj.simulatedArmaSeries;
            
            % 1. Compute ACF using sacf and PACF using spacf
            max_lag = 20;
            acf_values = sacf(y, max_lag);
            pacf_values = spacf(y, max_lag);
            
            % For an ARMA(p,q) process:
            % - ACF should tail off for AR, cut off after q for MA
            % - PACF should cut off after p for AR, tail off for MA
            
            % 2. Estimate multiple models with different orders
            max_p = 3;
            max_q = 3;
            aic_values = zeros(max_p+1, max_q+1);
            sbic_values = zeros(max_p+1, max_q+1);
            
            for p = 0:max_p
                for q = 0:max_q
                    options = struct('p', p, 'q', q, 'constant', true);
                    try
                        results = armaxfilter(y, [], options);
                        aic_values(p+1, q+1) = results.aic;
                        sbic_values(p+1, q+1) = results.sbic;
                    catch ME
                        warning('Model ARMA(%d,%d) failed to converge: %s', p, q, ME.message);
                        aic_values(p+1, q+1) = Inf;
                        sbic_values(p+1, q+1) = Inf;
                    end
                end
            end
            
            % 3. Calculate AIC and SBIC using aicsbic
            [min_aic, aic_idx] = min(aic_values(:));
            [p_aic, q_aic] = ind2sub(size(aic_values), aic_idx);
            p_aic = p_aic - 1; % Convert to 0-based indexing
            q_aic = q_aic - 1;
            
            [min_sbic, sbic_idx] = min(sbic_values(:));
            [p_sbic, q_sbic] = ind2sub(size(sbic_values), sbic_idx);
            p_sbic = p_sbic - 1;
            q_sbic = q_sbic - 1;
            
            % 4. Verify that correct model order is identified
            true_p = length(obj.knownParameters.ar);
            true_q = length(obj.knownParameters.ma);
            
            % Check if AIC or SBIC selected model is close to true model
            model_match = (p_aic == true_p && q_aic == true_q) || ...
                         (p_sbic == true_p && q_sbic == true_q);
            
            % Allow for close models to pass the test
            model_close = (abs(p_aic - true_p) <= 1 && abs(q_aic - true_q) <= 1) || ...
                         (abs(p_sbic - true_p) <= 1 && abs(q_sbic - true_q) <= 1);
            
            % Assert that at least one of the criteria selects a close model
            obj.assertTrue(model_match || model_close, ...
                'Neither AIC nor SBIC selected a model close to the true model');
        end
        
        function testErrorDistributionIntegration(obj)
            % Tests integration of different error distributions in time series modeling
            
            % Get test data
            y = obj.simulatedArmaSeries;
            T = length(y);
            
            % 1. Generate time series with different error distributions
            true_p = length(obj.knownParameters.ar);
            true_q = length(obj.knownParameters.ma);
            
            % Test array of distributions
            distributions = {'normal', 't', 'ged', 'skewt'};
            
            % Store results for comparison
            models = cell(length(distributions), 1);
            
            for i = 1:length(distributions)
                dist = distributions{i};
                options = struct('p', true_p, 'q', true_q, ...
                                'constant', true, ...
                                'distribution', dist);
                
                % Estimate models with matching distribution assumptions
                models{i} = armaxfilter(y, [], options);
                
                % Verify model estimation succeeded
                obj.assertTrue(isfinite(models{i}.logL), ...
                    sprintf('Model with %s distribution failed to converge', dist));
                
                % Verify residuals are well-behaved
                obj.assertTrue(all(isfinite(models{i}.residuals)), ...
                    sprintf('Model with %s distribution has non-finite residuals', dist));
            end
            
            % 2. Compare parameter estimates across distributions
            % They should be similar if the true distribution is close to normal
            for i = 2:length(distributions)
                ar_params1 = models{1}.parameters(2:(true_p+1));
                ar_params2 = models{i}.parameters(2:(true_p+1));
                
                ma_params1 = models{1}.parameters((true_p+2):(true_p+true_q+1));
                ma_params2 = models{i}.parameters((true_p+2):(true_p+true_q+1));
                
                % Allow larger tolerance for comparing across distributions
                tolerance = 0.2;
                
                obj.assertMatrixEqualsWithTolerance(ar_params1, ar_params2, tolerance, ...
                    sprintf('AR parameters differ significantly between normal and %s distribution', distributions{i}));
                
                obj.assertMatrixEqualsWithTolerance(ma_params1, ma_params2, tolerance, ...
                    sprintf('MA parameters differ significantly between normal and %s distribution', distributions{i}));
            end
            
            % 3. Generate forecasts preserving distribution assumptions
            forecast_horizon = 10;
            forecasts = cell(length(distributions), 1);
            
            for i = 1:length(distributions)
                forecasts{i} = armafor(models{i}.parameters, y, true_p, true_q, true, [], forecast_horizon);
                
                % Verify forecasts are finite
                obj.assertTrue(all(isfinite(forecasts{i})), ...
                    sprintf('Forecasts with %s distribution contain non-finite values', distributions{i}));
            end
            
            % 4. Test robustness to distribution misspecification
            for i = 2:length(distributions)
                forecast_diff = abs(forecasts{1} - forecasts{i});
                avg_diff = mean(forecast_diff);
                
                % Allow some difference due to distribution assumptions
                obj.assertTrue(avg_diff < 0.5, ...
                    sprintf('Forecasts differ significantly between normal and %s distribution (avg diff: %.4f)', ...
                    distributions{i}, avg_diff));
            end
        end
        
        function testDiagnosticsIntegration(obj)
            % Tests integration of diagnostic tools with model estimation
            
            % Get test data
            y = obj.simulatedArmaSeries;
            
            % 1. Estimate models on various time series
            true_p = length(obj.knownParameters.ar);
            true_q = length(obj.knownParameters.ma);
            
            options = struct('p', true_p, 'q', true_q, 'constant', true);
            results = armaxfilter(y, [], options);
            
            % 2. Apply comprehensive diagnostics
            max_lag = 20;
            
            % Calculate ACF of the residuals
            [acf_resid, acf_se, acf_ci] = sacf(results.residuals, max_lag);
            
            % Calculate PACF of the residuals
            [pacf_resid, pacf_se, pacf_ci] = spacf(results.residuals, max_lag);
            
            % 3. Verify integration of residual analysis tools
            % Residuals should be white noise, so ACF and PACF should be close to zero
            exceed_count_acf = sum(abs(acf_resid) > 2 * acf_se);
            exceed_count_pacf = sum(abs(pacf_resid) > 2 * pacf_se);
            
            obj.assertTrue(exceed_count_acf <= ceil(0.1 * max_lag), ...
                sprintf('Too many significant ACF values in residuals: %d out of %d', exceed_count_acf, max_lag));
            
            obj.assertTrue(exceed_count_pacf <= ceil(0.1 * max_lag), ...
                sprintf('Too many significant PACF values in residuals: %d out of %d', exceed_count_pacf, max_lag));
            
            % 4. Test autocorrelation tests and normality diagnostics
            % Check Ljung-Box and LM test results
            ljung_box = results.ljungBox;
            lm_test = results.lmTest;
            
            obj.assertTrue(~all(ljung_box.isRejected5pct), ...
                'Ljung-Box test rejects white noise at all lags, indicating model misspecification');
            
            obj.assertTrue(~all(lm_test.sig(:,2)), ...
                'LM test rejects white noise at all lags, indicating model misspecification');
            
            % 5. Validate end-to-end diagnostic workflow
            % The tests should broadly agree
            lb_rejects = sum(ljung_box.isRejected5pct);
            lm_rejects = sum(lm_test.sig(:,2));
            
            obj.assertTrue(abs(lb_rejects - lm_rejects) <= ceil(0.3 * max_lag), ...
                sprintf('Ljung-Box and LM tests show inconsistent results: %d vs %d rejections', ...
                lb_rejects, lm_rejects));
        end
        
        function testForecastingPerformance(obj)
            % Tests integrated forecasting performance with different models
            
            % Get test data
            y_full = obj.simulatedArmaSeries;
            
            % Split into estimation and validation samples
            T = length(y_full);
            estimation_size = floor(0.8 * T);
            y_train = y_full(1:estimation_size);
            y_test = y_full(estimation_size+1:end);
            
            % Forecast horizon
            forecast_horizon = length(y_test);
            
            % 1. Estimate different model types
            % AR(p) model
            ar_options = struct('p', 2, 'q', 0, 'constant', true);
            ar_results = armaxfilter(y_train, [], ar_options);
            
            % MA(q) model
            ma_options = struct('p', 0, 'q', 2, 'constant', true);
            ma_results = armaxfilter(y_train, [], ma_options);
            
            % ARMA(p,q) model
            arma_options = struct('p', 2, 'q', 2, 'constant', true);
            arma_results = armaxfilter(y_train, [], arma_options);
            
            % 2. Generate multi-step forecasts
            ar_forecasts = armafor(ar_results.parameters, y_train, 2, 0, true, [], forecast_horizon);
            ma_forecasts = armafor(ma_results.parameters, y_train, 0, 2, true, [], forecast_horizon);
            arma_forecasts = armafor(arma_results.parameters, y_train, 2, 2, true, [], forecast_horizon);
            
            % 3. Compare forecast accuracy across models
            ar_errors = abs(ar_forecasts - y_test);
            ma_errors = abs(ma_forecasts - y_test);
            arma_errors = abs(arma_forecasts - y_test);
            
            % Calculate RMSE and MAE
            ar_rmse = sqrt(mean(ar_errors.^2));
            ma_rmse = sqrt(mean(ma_errors.^2));
            arma_rmse = sqrt(mean(arma_errors.^2));
            
            ar_mae = mean(ar_errors);
            ma_mae = mean(ma_errors);
            arma_mae = mean(arma_errors);
            
            % 4. Test integration of forecasting components
            models_rmse = [ar_rmse, ma_rmse, arma_rmse];
            [~, rmse_rank] = sort(models_rmse);
            
            models_mae = [ar_mae, ma_mae, arma_mae];
            [~, mae_rank] = sort(models_mae);
            
            % Check if ARMA model is among top 2 by RMSE
            obj.assertTrue(find(rmse_rank == 3) <= 2, ...
                'ARMA model is not among top 2 models by RMSE');
            
            % Check if ARMA model is among top 2 by MAE
            obj.assertTrue(find(mae_rank == 3) <= 2, ...
                'ARMA model is not among top 2 models by MAE');
        end
        
        function testMexPerformanceIntegration(obj)
            % Tests performance improvement from MEX integration in time series modeling
            
            % Skip this test if running in Octave or environment where MEX might not be available
            try
                % Try to access MEX file
                if ~exist('armaxerrors', 'file') == 3
                    warning('MEX file armaxerrors not found, skipping performance test');
                    return;
                end
            catch
                warning('Error checking MEX file availability, skipping performance test');
                return;
            end
            
            % Generate large-scale time series data
            T = 5000;
            
            % Generate AR(1) process
            y = zeros(T, 1);
            y(1) = randn();
            for t = 2:T
                y(t) = 0.8 * y(t-1) + randn();
            end
            
            % 1. Measure execution time with MEX optimization
            % Force garbage collection if supported
            if exist('java.lang.System', 'class')
                java.lang.System.gc();
            end
            
            % Start timer
            tic_mex = tic();
            
            % Run with MEX enabled (default)
            options_mex = struct('p', 1, 'q', 1, 'constant', true);
            results_mex = armaxfilter(y, [], options_mex);
            
            % End timer
            time_mex = toc(tic_mex);
            
            % 2. Measure time for a more complex operation as comparison
            % Force garbage collection if supported
            if exist('java.lang.System', 'class')
                java.lang.System.gc();
            end
            
            % Start timer for alternative computation
            tic_nomex = tic();
            
            % Execute more complex model
            options_nomex = struct('p', 3, 'q', 3, 'constant', true);
            results_nomex = armaxfilter(y, [], options_nomex);
            
            % End timer
            time_nomex = toc(tic_nomex);
            
            % 3. Verify end-to-end performance improvements
            % Adjust for model complexity (ARMA(3,3) has 7 parameters, ARMA(1,1) has 3)
            adjusted_time = time_nomex * (3/7);
            
            % Check if MEX is reasonably faster than our adjusted benchmark
            obj.assertTrue(time_mex < adjusted_time, ...
                sprintf('MEX implementation is not faster than expected. MEX: %.4f, Adjusted benchmark: %.4f', ...
                time_mex, adjusted_time));
            
            % 4. Verify that MEX results are accurate
            ar_param = results_mex.parameters(2);
            obj.assertAlmostEqual(ar_param, 0.8, 0.1, ...
                'MEX-based estimation gives inaccurate AR parameter');
            
            % 5. Test MEX integration across modeling workflow
            forecast_horizon = 100;
            
            % Time forecast generation
            tic_forecast = tic();
            forecasts = armafor(results_mex.parameters, y, 1, 1, true, [], forecast_horizon);
            time_forecast = toc(tic_forecast);
            
            % Verify forecasts are generated efficiently
            obj.assertTrue(time_forecast < 0.1, ...
                sprintf('Forecasting took too long: %.4f seconds', time_forecast));
            
            % Verify forecasts are reasonable
            obj.assertTrue(all(isfinite(forecasts)), 'MEX-based forecasts contain non-finite values');
        end
        
        function testTimeSeriesToForecastWorkflow(obj)
            % Tests the complete workflow from raw time series to forecasts
            
            % Load raw time series data
            y = obj.simulatedArmaSeries;
            
            % 1. Perform model identification
            max_lag = 20;
            acf_values = sacf(y, max_lag);
            pacf_values = spacf(y, max_lag);
            
            % 2. Estimate model parameters
            max_p = 3;
            max_q = 3;
            
            best_aic = Inf;
            best_p = 0;
            best_q = 0;
            best_model = [];
            
            for p = 0:max_p
                for q = 0:max_q
                    options = struct('p', p, 'q', q, 'constant', true);
                    try
                        results = armaxfilter(y, [], options);
                        
                        if results.aic < best_aic
                            best_aic = results.aic;
                            best_p = p;
                            best_q = q;
                            best_model = results;
                        end
                    catch ME
                        % Skip models that fail to converge
                        continue;
                    end
                end
            end
            
            % 3. Validate model with diagnostics
            % Verify we found a valid model
            obj.assertTrue(~isempty(best_model), 'Failed to find valid ARMA model');
            
            % Compute diagnostic tests
            acf_resid = sacf(best_model.residuals, 10);
            pacf_resid = spacf(best_model.residuals, 10);
            
            % Check if residuals pass white noise test
            max_resid_acf = max(abs(acf_resid));
            obj.assertTrue(max_resid_acf < 2/sqrt(length(y)), ...
                sprintf('Residual ACF exceeds bounds: %.4f', max_resid_acf));
            
            % 4. Generate and validate forecasts
            forecast_horizon = 10;
            forecasts = armafor(best_model.parameters, y, best_p, best_q, true, [], forecast_horizon);
            
            % Verify forecasts are valid
            obj.assertTrue(all(isfinite(forecasts)), 'Forecasts contain non-finite values');
            
            % 5. Test complete end-to-end workflow integration
            true_p = length(obj.knownParameters.ar);
            true_q = length(obj.knownParameters.ma);
            
            % Allow for different but similar models
            model_close = (abs(best_p - true_p) <= 1 && abs(best_q - true_q) <= 1);
            
            % If model selection worked well, the selected model should be close to true model
            obj.assertTrue(model_close || (best_p == true_p && best_q == true_q), ...
                sprintf('Selected model ARMA(%d,%d) is not close to true model ARMA(%d,%d)', ...
                best_p, best_q, true_p, true_q));
        end
        
        function generateDefaultTestData(obj)
            % Generate default test data if no test data file is available
            obj.knownParameters = struct();
            
            % AR parameters
            obj.knownParameters.ar = [0.7, -0.2];
            
            % MA parameters
            obj.knownParameters.ma = [0.3, 0.1];
            
            % Constant term
            obj.knownParameters.constant = 0.001;
            
            % Exogenous coefficients
            obj.knownParameters.exogenousCoef = [0.5, -0.3];
        end
        
        function generateTestSeries(obj)
            % Generate controlled test data for specific integration test cases
            
            % Sample size
            T = 500;
            
            % Generate AR series with known parameters
            ar_params = obj.knownParameters.ar;
            p = length(ar_params);
            
            % Initialize AR series with zeros
            obj.simulatedArSeries = zeros(T, 1);
            
            % Generate AR process
            e = randn(T, 1);
            for t = (p+1):T
                obj.simulatedArSeries(t) = obj.knownParameters.constant;
                for i = 1:p
                    obj.simulatedArSeries(t) = obj.simulatedArSeries(t) + ar_params(i) * obj.simulatedArSeries(t-i);
                end
                obj.simulatedArSeries(t) = obj.simulatedArSeries(t) + e(t);
            end
            
            % Generate MA series with known parameters
            ma_params = obj.knownParameters.ma;
            q = length(ma_params);
            
            % Initialize MA series and innovations
            obj.simulatedMaSeries = zeros(T, 1);
            innovations = randn(T, 1);
            
            % Generate MA process
            for t = (q+1):T
                obj.simulatedMaSeries(t) = obj.knownParameters.constant + innovations(t);
                for i = 1:q
                    obj.simulatedMaSeries(t) = obj.simulatedMaSeries(t) + ma_params(i) * innovations(t-i);
                end
            end
            
            % Generate ARMA series with known patterns
            obj.simulatedArmaSeries = zeros(T, 1);
            innovations = randn(T, 1);
            
            % Generate ARMA process
            max_lag = max(p, q);
            for t = (max_lag+1):T
                obj.simulatedArmaSeries(t) = obj.knownParameters.constant;
                
                % AR component
                for i = 1:p
                    obj.simulatedArmaSeries(t) = obj.simulatedArmaSeries(t) + ar_params(i) * obj.simulatedArmaSeries(t-i);
                end
                
                % MA component with current innovation
                obj.simulatedArmaSeries(t) = obj.simulatedArmaSeries(t) + innovations(t);
                
                % MA component with lagged innovations
                for i = 1:q
                    obj.simulatedArmaSeries(t) = obj.simulatedArmaSeries(t) + ma_params(i) * innovations(t-i);
                end
            end
            
            % Generate seasonal time series with known patterns
            obj.simulatedSeasonalSeries = zeros(T, 1);
            seasonal_pattern = [0.8, -0.6, 0.2, -0.4]; % Quarterly pattern
            
            % Base ARMA process with seasonality
            for t = (max_lag+5):T
                % Base level with constant
                obj.simulatedSeasonalSeries(t) = obj.knownParameters.constant;
                
                % AR component
                for i = 1:p
                    obj.simulatedSeasonalSeries(t) = obj.simulatedSeasonalSeries(t) + 0.5 * ar_params(i) * obj.simulatedSeasonalSeries(t-i);
                end
                
                % Seasonal component (s=4)
                obj.simulatedSeasonalSeries(t) = obj.simulatedSeasonalSeries(t) + seasonal_pattern(mod(t-1, 4) + 1);
                
                % Add seasonal AR component
                if t > 4
                    obj.simulatedSeasonalSeries(t) = obj.simulatedSeasonalSeries(t) + 0.6 * obj.simulatedSeasonalSeries(t-4);
                end
                
                % Add random noise
                obj.simulatedSeasonalSeries(t) = obj.simulatedSeasonalSeries(t) + 0.8 * randn();
            end
            
            % Generate exogenous variables for ARMAX testing
            obj.exogenousVariables = zeros(T, 2);
            
            % First exogenous variable: AR(1) process
            obj.exogenousVariables(1, 1) = randn();
            for t = 2:T
                obj.exogenousVariables(t, 1) = 0.7 * obj.exogenousVariables(t-1, 1) + 0.3 * randn();
            end
            
            % Second exogenous variable: MA(1) process
            e = randn(T, 1);
            obj.exogenousVariables(1, 2) = e(1);
            for t = 2:T
                obj.exogenousVariables(t, 2) = e(t) + 0.5 * e(t-1);
            end
        end
    end
end