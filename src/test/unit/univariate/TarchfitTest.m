classdef TarchfitTest < BaseTest
    % Test class for the tarchfit function which implements Threshold ARCH (TARCH) volatility model estimation
    
    properties
        testData         % Matrix of test data
        testTolerance    % Tolerance for numerical comparisons
        defaultOptions   % Default options structure for tarchfit
    end
    
    methods
        function obj = TarchfitTest()
            % Initialize TarchfitTest with test data and default options
            
            % Call superclass constructor
            obj@BaseTest();
            
            % Set tolerance for numerical comparisons
            obj.testTolerance = 1e-6;
            
            % Initialize default options structure
            obj.defaultOptions = struct();
            obj.defaultOptions.p = 1;                   % ARCH order
            obj.defaultOptions.q = 1;                   % GARCH order
            obj.defaultOptions.distribution = 'NORMAL'; % Error distribution
            obj.defaultOptions.useMEX = true;           % Use MEX acceleration if available
            
            % Set optimization options for faster testing
            obj.defaultOptions.optimoptions = struct();
            obj.defaultOptions.optimoptions.Display = 'off';
            obj.defaultOptions.optimoptions.MaxIter = 500;
            obj.defaultOptions.optimoptions.MaxFunEvals = 500;
            obj.defaultOptions.optimoptions.TolFun = 1e-6;
            obj.defaultOptions.optimoptions.TolX = 1e-6;
        end
        
        function setUp(obj)
            % Prepare test environment before each test execution
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Load financial returns test data
            data = obj.loadTestData('financial_returns.mat');
            
            % Select a single asset return series for testing
            obj.testData = data.returns(:,1);
            
            % Set random number generator seed for reproducibility
            rng(1);
        end
        
        function tearDown(obj)
            % Clean up test environment after each test execution
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
        end
        
        function testBasicTarchFit(obj)
            % Test basic TARCH(1,1) model estimation with normal errors
            
            % Set up options for TARCH(1,1) with normal errors
            options = obj.defaultOptions;
            
            % Fit TARCH model to test data
            params = tarchfit(obj.testData, options);
            
            % Verify parameters structure was returned
            obj.assertTrue(isstruct(params), 'Parameter output should be a structure');
            
            % Verify basic TARCH parameters are present
            obj.assertTrue(isfield(params, 'omega'), 'parameters.omega should exist');
            obj.assertTrue(isfield(params, 'alpha'), 'parameters.alpha should exist');
            obj.assertTrue(isfield(params, 'gamma'), 'parameters.gamma should exist');
            obj.assertTrue(isfield(params, 'beta'), 'parameters.beta should exist');
            
            % Verify parameter constraints are satisfied
            obj.assertTrue(params.omega > 0, 'omega should be positive');
            obj.assertTrue(all(params.alpha >= 0), 'alpha should be non-negative');
            obj.assertTrue(all(params.gamma >= 0), 'gamma should be non-negative');
            obj.assertTrue(all(params.beta >= 0), 'beta should be non-negative');
            
            % Verify stability constraint is satisfied
            obj.assertTrue(sum(params.alpha) + 0.5*sum(params.gamma) + sum(params.beta) < 1, ...
                'Stability constraint should be satisfied');
            
            % Verify likelihood and diagnostics are computed
            obj.assertTrue(isfield(params, 'likelihood'), 'Likelihood should be computed');
            obj.assertTrue(isfield(params, 'ht'), 'Conditional variances should be returned');
            obj.assertTrue(isfield(params, 'aic'), 'AIC should be computed');
            obj.assertTrue(isfield(params, 'bic'), 'BIC should be computed');
            obj.assertTrue(isfield(params, 'LjungBox'), 'Ljung-Box test results should be computed');
        end
        
        function testParameterConstraints(obj)
            % Test that parameter constraints are enforced during estimation
            
            % Set up options with constraints
            options = obj.defaultOptions;
            
            % Create parameter bounds
            options.LB = [0.001; 0.05; 0.1; 0.7]; % Lower bounds [omega, alpha, gamma, beta]
            options.UB = [0.05; 0.2; 0.3; 0.9];   % Upper bounds [omega, alpha, gamma, beta]
            
            % Fit TARCH model with constraints
            params = tarchfit(obj.testData, options);
            
            % Verify parameters respect the provided bounds
            obj.assertTrue(params.omega >= options.LB(1) && params.omega <= options.UB(1), ...
                'omega should respect bounds');
            obj.assertTrue(params.alpha >= options.LB(2) && params.alpha <= options.UB(2), ...
                'alpha should respect bounds');
            obj.assertTrue(params.gamma >= options.LB(3) && params.gamma <= options.UB(3), ...
                'gamma should respect bounds');
            obj.assertTrue(params.beta >= options.LB(4) && params.beta <= options.UB(4), ...
                'beta should respect bounds');
            
            % Verify TARCH-specific constraints are satisfied
            obj.assertTrue(params.omega > 0, 'omega should be positive');
            obj.assertTrue(params.alpha + 0.5*params.gamma + params.beta < 1, ...
                'Stability constraint should be satisfied');
        end
        
        function testStudentTDistribution(obj)
            % Test TARCH model with Student's t error distribution
            
            % Set up options for t-distribution
            options = obj.defaultOptions;
            options.distribution = 'T';
            
            % Fit TARCH model with t-distribution
            params = tarchfit(obj.testData, options);
            
            % Verify t-distribution parameter (nu) is present
            obj.assertTrue(isfield(params, 'nu'), 'parameters.nu should exist for t-distribution');
            
            % Verify nu is greater than 2 (required for finite variance)
            obj.assertTrue(params.nu > 2, 'nu should be greater than 2 for t-distribution');
            
            % Verify likelihood is reasonable
            obj.assertTrue(isfinite(params.likelihood), 'Likelihood should be finite');
            
            % Verify model information criteria are computed
            obj.assertTrue(isfield(params, 'aic'), 'AIC should be computed');
            obj.assertTrue(isfield(params, 'bic'), 'BIC should be computed');
        end
        
        function testGEDDistribution(obj)
            % Test TARCH model with Generalized Error Distribution (GED)
            
            % Set up options for GED
            options = obj.defaultOptions;
            options.distribution = 'GED';
            
            % Fit TARCH model with GED
            params = tarchfit(obj.testData, options);
            
            % Verify GED parameter (nu) is present
            obj.assertTrue(isfield(params, 'nu'), 'parameters.nu should exist for GED');
            
            % Verify nu is positive (required for GED)
            obj.assertTrue(params.nu > 0, 'nu should be positive for GED');
            
            % Verify likelihood is reasonable
            obj.assertTrue(isfinite(params.likelihood), 'Likelihood should be finite');
            
            % Verify model information criteria are computed
            obj.assertTrue(isfield(params, 'aic'), 'AIC should be computed');
            obj.assertTrue(isfield(params, 'bic'), 'BIC should be computed');
        end
        
        function testSkewedTDistribution(obj)
            % Test TARCH model with Hansen's Skewed t-distribution
            
            % Set up options for skewed t-distribution
            options = obj.defaultOptions;
            options.distribution = 'SKEWT';
            
            % Fit TARCH model with skewed t-distribution
            params = tarchfit(obj.testData, options);
            
            % Verify skewed t-distribution parameters are present
            obj.assertTrue(isfield(params, 'nu'), 'parameters.nu should exist for skewed t-distribution');
            obj.assertTrue(isfield(params, 'lambda'), 'parameters.lambda should exist for skewed t-distribution');
            
            % Verify parameters respect constraints
            obj.assertTrue(params.nu > 2, 'nu should be greater than 2 for skewed t-distribution');
            obj.assertTrue(abs(params.lambda) < 1, 'lambda should be between -1 and 1');
            
            % Verify likelihood is reasonable
            obj.assertTrue(isfinite(params.likelihood), 'Likelihood should be finite');
            
            % Verify model information criteria are computed
            obj.assertTrue(isfield(params, 'aic'), 'AIC should be computed');
            obj.assertTrue(isfield(params, 'bic'), 'BIC should be computed');
        end
        
        function testMEXAcceleration(obj)
            % Test MEX acceleration for TARCH model estimation
            
            % Skip test if the test data is too small for meaningful comparison
            if length(obj.testData) < 500
                warning('Test data is too small for meaningful MEX acceleration test.');
                return;
            end
            
            % Set up options with MEX acceleration
            optionsWithMEX = obj.defaultOptions;
            optionsWithMEX.useMEX = true;
            
            % Set up options without MEX acceleration
            optionsWithoutMEX = obj.defaultOptions;
            optionsWithoutMEX.useMEX = false;
            
            % Measure execution time with MEX acceleration
            timeMEX = obj.measureExecutionTime(@() tarchfit(obj.testData, optionsWithMEX));
            
            % Measure execution time without MEX acceleration
            timeNoMEX = obj.measureExecutionTime(@() tarchfit(obj.testData, optionsWithoutMEX));
            
            % Report execution times
            fprintf('MEX time: %.4f seconds, No MEX time: %.4f seconds\n', timeMEX, timeNoMEX);
            
            % Fit models for parameter comparison
            paramsMEX = tarchfit(obj.testData, optionsWithMEX);
            paramsNoMEX = tarchfit(obj.testData, optionsWithoutMEX);
            
            % Compare parameter estimates between MEX and non-MEX implementations
            obj.assertMatrixEqualsWithTolerance(paramsMEX.omega, paramsNoMEX.omega, 1e-4, ...
                'omega should be similar with and without MEX');
            obj.assertMatrixEqualsWithTolerance(paramsMEX.alpha, paramsNoMEX.alpha, 1e-4, ...
                'alpha should be similar with and without MEX');
            obj.assertMatrixEqualsWithTolerance(paramsMEX.gamma, paramsNoMEX.gamma, 1e-4, ...
                'gamma should be similar with and without MEX');
            obj.assertMatrixEqualsWithTolerance(paramsMEX.beta, paramsNoMEX.beta, 1e-4, ...
                'beta should be similar with and without MEX');
        end
        
        function testInvalidInputs(obj)
            % Test error handling for invalid inputs
            
            % Test with empty data
            obj.assertThrows(@() tarchfit([], obj.defaultOptions), ...
                'MATLAB:columncheck:notEmpty', 'Should throw error for empty data');
            
            % Test with non-numeric data
            obj.assertThrows(@() tarchfit({'string'}, obj.defaultOptions), ...
                'MATLAB:columncheck:notNumeric', 'Should throw error for non-numeric data');
            
            % Test with NaN values
            invalidData = obj.testData;
            invalidData(10) = NaN;
            obj.assertThrows(@() tarchfit(invalidData, obj.defaultOptions), ...
                'MATLAB:datacheck:containsNaN', 'Should throw error for data with NaN values');
            
            % Test with invalid p (negative ARCH order)
            invalidOptions = obj.defaultOptions;
            invalidOptions.p = -1;
            obj.assertThrows(@() tarchfit(obj.testData, invalidOptions), ...
                'MATLAB:parametercheck:isPositive', 'Should throw error for negative p');
            
            % Test with invalid q (negative GARCH order)
            invalidOptions = obj.defaultOptions;
            invalidOptions.q = -1;
            obj.assertThrows(@() tarchfit(obj.testData, invalidOptions), ...
                'MATLAB:parametercheck:isPositive', 'Should throw error for negative q');
            
            % Test with invalid distribution type
            invalidOptions = obj.defaultOptions;
            invalidOptions.distribution = 'INVALID';
            obj.assertThrows(@() tarchfit(obj.testData, invalidOptions), ...
                'MATLAB:tarchfit:InvalidDistribution', 'Should throw error for invalid distribution type');
        end
        
        function testFixedParameters(obj)
            % Test estimation with fixed parameter values
            
            % First get baseline parameter estimates
            baselineParams = tarchfit(obj.testData, obj.defaultOptions);
            
            % Set up options with fixed parameters
            options = obj.defaultOptions;
            options.fixed = struct();
            options.fixed.omega = 0.02; % Fix omega at 0.02
            
            % Fit TARCH model with fixed omega
            params = tarchfit(obj.testData, options);
            
            % Verify omega remains fixed at specified value
            obj.assertAlmostEqual(params.omega, 0.02, ...
                'omega should remain fixed at specified value');
            
            % Verify other parameters are estimated
            obj.assertFalse(abs(params.alpha - baselineParams.alpha) < 1e-10, ...
                'alpha should be estimated (not fixed)');
        end
        
        function testCustomStartValues(obj)
            % Test estimation with custom starting values
            
            % First get baseline parameter estimates
            baselineParams = tarchfit(obj.testData, obj.defaultOptions);
            
            % Set up options with custom starting values
            options = obj.defaultOptions;
            options.startingvals = [0.05; 0.15; 0.1; 0.7]; % [omega, alpha, gamma, beta]
            
            % Fit TARCH model with custom starting values
            params = tarchfit(obj.testData, options);
            
            % Verify model converges to similar parameter values despite different starting point
            obj.assertMatrixEqualsWithTolerance(params.omega, baselineParams.omega, 1e-4, ...
                'omega should converge to similar value regardless of starting point');
            obj.assertMatrixEqualsWithTolerance(params.alpha, baselineParams.alpha, 1e-4, ...
                'alpha should converge to similar value regardless of starting point');
            obj.assertMatrixEqualsWithTolerance(params.beta, baselineParams.beta, 1e-4, ...
                'beta should converge to similar value regardless of starting point');
        end
        
        function testDiagnostics(obj)
            % Test diagnostic statistics for TARCH model fit
            
            % Fit basic TARCH model
            params = tarchfit(obj.testData, obj.defaultOptions);
            
            % Verify Ljung-Box statistics on standardized residuals
            obj.assertTrue(isfield(params, 'LjungBox'), 'Ljung-Box statistics should exist');
            obj.assertTrue(isstruct(params.LjungBox), 'Ljung-Box should be a structure');
            obj.assertTrue(isfield(params.LjungBox, 'stats'), 'Ljung-Box stats should exist');
            obj.assertTrue(isfield(params.LjungBox, 'pvals'), 'Ljung-Box p-values should exist');
            
            % Verify Ljung-Box statistics on squared standardized residuals
            obj.assertTrue(isfield(params, 'LBsquared'), 'Ljung-Box for squared residuals should exist');
            
            % Verify information criteria
            obj.assertTrue(isfield(params, 'aic'), 'AIC should exist');
            obj.assertTrue(isfield(params, 'bic'), 'BIC should exist');
            obj.assertTrue(isfinite(params.aic), 'AIC should be finite');
            obj.assertTrue(isfinite(params.bic), 'BIC should be finite');
            
            % Verify log-likelihood
            obj.assertTrue(isfield(params, 'likelihood'), 'Likelihood should exist');
            obj.assertTrue(isfinite(params.likelihood), 'Likelihood should be finite');
            
            % Verify standardized residuals
            obj.assertTrue(isfield(params, 'stdresid'), 'Standardized residuals should exist');
            obj.assertTrue(abs(mean(params.stdresid)) < 0.1, 'Mean of standardized residuals should be close to 0');
            obj.assertTrue(abs(var(params.stdresid) - 1) < 0.1, 'Variance of standardized residuals should be close to 1');
            
            % Verify parameter standard errors
            obj.assertTrue(isfield(params, 'stderrors'), 'Standard errors should exist');
            obj.assertTrue(all(params.stderrors > 0), 'Standard errors should be positive');
            
            % Verify covariance matrix
            obj.assertTrue(isfield(params, 'vcv'), 'Parameter covariance matrix should exist');
            obj.assertTrue(size(params.vcv, 1) == size(params.vcv, 2), 'VCV should be square');
        end
        
        function testNumericalStability(obj)
            % Test numerical stability of TARCH estimation
            
            % Generate synthetic TARCH data with known parameters
            T = 2000;
            omega = 0.05;
            alpha = 0.1;
            gamma = 0.08;
            beta = 0.8;
            
            % Generate TARCH process
            data = obj.generateTarchData(T, omega, alpha, gamma, beta, 'NORMAL');
            
            % Add different levels of noise to test robustness
            noiseScale = [0, 0.001, 0.01, 0.05];
            
            % Initialize storage for parameter estimates
            omegaEst = zeros(length(noiseScale), 1);
            alphaEst = zeros(length(noiseScale), 1);
            gammaEst = zeros(length(noiseScale), 1);
            betaEst = zeros(length(noiseScale), 1);
            
            % Fit TARCH model to data with different noise levels
            for i = 1:length(noiseScale)
                % Add noise
                noiseData = data.returns + noiseScale(i) * randn(T, 1);
                
                % Fit model
                params = tarchfit(noiseData, obj.defaultOptions);
                
                % Store parameter estimates
                omegaEst(i) = params.omega;
                alphaEst(i) = params.alpha;
                gammaEst(i) = params.gamma;
                betaEst(i) = params.beta;
            end
            
            % Verify parameter estimates are stable across noise levels
            for i = 2:length(noiseScale)
                obj.assertTrue(abs(omegaEst(i) - omegaEst(1)) < 0.1, ...
                    'omega estimate should be stable with noise');
                obj.assertTrue(abs(alphaEst(i) - alphaEst(1)) < 0.1, ...
                    'alpha estimate should be stable with noise');
                obj.assertTrue(abs(gammaEst(i) - gammaEst(1)) < 0.1, ...
                    'gamma estimate should be stable with noise');
                obj.assertTrue(abs(betaEst(i) - betaEst(1)) < 0.1, ...
                    'beta estimate should be stable with noise');
            end
            
            % Also test with data at different scales
            scaleFactors = [0.1, 1, 10];
            
            % Initialize storage for scaled parameter estimates
            omegaEstScaled = zeros(length(scaleFactors), 1);
            alphaEstScaled = zeros(length(scaleFactors), 1);
            gammaEstScaled = zeros(length(scaleFactors), 1);
            betaEstScaled = zeros(length(scaleFactors), 1);
            
            % Fit TARCH model to data at different scales
            for i = 1:length(scaleFactors)
                % Scale data
                scaledData = data.returns * scaleFactors(i);
                
                % Fit model
                params = tarchfit(scaledData, obj.defaultOptions);
                
                % Store parameter estimates
                omegaEstScaled(i) = params.omega / (scaleFactors(i)^2); % Scale back
                alphaEstScaled(i) = params.alpha;
                gammaEstScaled(i) = params.gamma;
                betaEstScaled(i) = params.beta;
            end
            
            % Verify parameter estimates scale appropriately
            for i = 2:length(scaleFactors)
                obj.assertTrue(abs(omegaEstScaled(i) - omegaEstScaled(1)) < 0.1, ...
                    'Scaled omega estimate should be stable');
                obj.assertTrue(abs(alphaEstScaled(i) - alphaEstScaled(1)) < 0.1, ...
                    'alpha estimate should be scale-invariant');
                obj.assertTrue(abs(gammaEstScaled(i) - gammaEstScaled(1)) < 0.1, ...
                    'gamma estimate should be scale-invariant');
                obj.assertTrue(abs(betaEstScaled(i) - betaEstScaled(1)) < 0.1, ...
                    'beta estimate should be scale-invariant');
            end
        end
        
        function data = generateTarchData(obj, T, omega, alpha, gamma, beta, errorDist)
            % Helper method to generate synthetic TARCH process data
            %
            % INPUTS:
            %   T          - Number of observations to generate
            %   omega      - Constant term
            %   alpha      - ARCH coefficient
            %   gamma      - Threshold/asymmetric coefficient
            %   beta       - GARCH coefficient
            %   errorDist  - Error distribution ('NORMAL', 'T', 'GED', 'SKEWT')
            %
            % OUTPUTS:
            %   data       - Structure with fields:
            %                .returns - Generated returns series
            %                .ht      - Conditional variances
            %                .params  - True parameter values
            
            % Validate input parameters
            if sum(alpha) + 0.5*sum(gamma) + sum(beta) >= 1
                error('Parameters do not satisfy stability constraint: alpha + 0.5*gamma + beta < 1');
            end
            
            % Initialize arrays
            ht = zeros(T, 1);
            returns = zeros(T, 1);
            
            % Set seed for reproducibility
            rng(1);
            
            % Initialize ht with unconditional variance
            uncond_var = omega / (1 - alpha - 0.5*gamma - beta);
            ht(1) = uncond_var;
            
            % Generate innovations based on specified distribution
            switch upper(errorDist)
                case 'NORMAL'
                    z = randn(T, 1);
                case 'T'
                    nu = 5; % Degrees of freedom
                    z = trnd(nu, T, 1) / sqrt(nu/(nu-2)); % Standardized t
                case 'GED'
                    nu = 1.5; % Shape parameter
                    z = randn(T, 1); % Placeholder (should use real GED random numbers)
                case 'SKEWT'
                    nu = 5; % Degrees of freedom
                    lambda = 0.2; % Skewness parameter
                    z = randn(T, 1); % Placeholder (should use real skewed t random numbers)
                otherwise
                    error('Unknown error distribution: %s', errorDist);
            end
            
            % Generate TARCH process
            for t = 2:T
                % Previous shock (squared return)
                e2 = returns(t-1)^2;
                
                % TARCH variance equation
                ht(t) = omega + alpha*e2;
                
                % Add asymmetric effect if previous return was negative
                if returns(t-1) < 0
                    ht(t) = ht(t) + gamma*e2;
                end
                
                % Add GARCH component
                ht(t) = ht(t) + beta*ht(t-1);
                
                % Generate return
                returns(t) = sqrt(ht(t)) * z(t);
            end
            
            % Create and return data structure
            data = struct();
            data.returns = returns;
            data.ht = ht;
            data.params = struct('omega', omega, 'alpha', alpha, 'gamma', gamma, 'beta', beta);
        end
    end
end