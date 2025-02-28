classdef DistributionIntegrationTest < BaseTest
    properties
        % Test configuration
        dataGenerator;  % TestDataGenerator instance
        comparator;     % NumericalComparator instance
        defaultTolerance; % Default tolerance for numeric comparisons
        
        % Test data
        testData;  % Structure for test data
        sampleSize; % Default sample size for tests
        
        % Distribution configurations for testing
        distributions;  % Structure with distribution configurations
    end
    
    methods
        function obj = DistributionIntegrationTest()
            % Call superclass constructor
            obj = obj@BaseTest('DistributionIntegrationTest');
            
            % Initialize the testData structure for storing test samples
            obj.testData = struct();
            
            % Create a TestDataGenerator instance for generating distribution samples
            obj.dataGenerator = TestDataGenerator();
            
            % Create a NumericalComparator instance for floating-point comparisons
            obj.comparator = NumericalComparator();
            
            % Set default tolerance for numeric comparisons
            obj.defaultTolerance = 1e-10;
            
            % Set default sample size for distribution tests
            obj.sampleSize = 10000;
            
            % Initialize distributions structure with test configurations for each distribution type
            obj.distributions = struct();
        end
        
        function setUp(obj)
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Load reference test data from known_distributions.mat
            obj.testData = obj.loadTestData('known_distributions.mat');
            
            % Set the dataGenerator to reproducible mode with fixed seed
            
            % Initialize test data structures for each distribution type
            
            % GED distribution configurations - different shape parameters
            obj.distributions.ged = struct();
            obj.distributions.ged.shapes = [1, 1.5, 2, 4]; % Various shape parameters including normal (2)
            
            % Skewed t configurations - various degrees of freedom and skewness
            obj.distributions.skewt = struct();
            obj.distributions.skewt.dofs = [5, 10, 20]; % Degrees of freedom
            obj.distributions.skewt.skews = [-0.5, 0, 0.5]; % Skewness parameters
            
            % Standardized t configurations - various degrees of freedom
            obj.distributions.stdt = struct();
            obj.distributions.stdt.dofs = [3, 5, 10, 30]; % Degrees of freedom
        end
        
        function tearDown(obj)
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
            
            % Clear any temporary test data to free memory
            clear temp;
        end
        
        function testGEDFunctionConsistency(obj)
            % Tests the consistency between GED distribution functions (PDF, CDF, inverse CDF)
            
            % For each GED shape parameter in test configuration:
            for i = 1:length(obj.distributions.ged.shapes)
                nu = obj.distributions.ged.shapes(i);
                
                % Generate test points across the distribution range
                testPoints = linspace(-5, 5, 100)';
                
                % Compute PDF values using gedpdf
                pdfValues = gedpdf(testPoints, nu);
                
                % Compute CDF values using gedcdf
                cdfValues = gedcdf(testPoints, nu);
                
                % Test probabilities
                testProbs = linspace(0.01, 0.99, 100)';
                invValues = gedinv(testProbs, nu);
                
                % Verify that CDF(inv(p)) ≈ p for probability values
                cdfOfInv = gedcdf(invValues, nu);
                obj.assertMatrixEqualsWithTolerance(testProbs, cdfOfInv, obj.defaultTolerance, ...
                    sprintf('CDF(inv(p)) should equal p for GED with nu=%.1f', nu));
                
                % Verify that inv(CDF(x)) ≈ x for value points
                invOfCdf = gedinv(cdfValues, nu);
                obj.assertMatrixEqualsWithTolerance(testPoints, invOfCdf, obj.defaultTolerance * 10, ...
                    sprintf('inv(CDF(x)) should equal x for GED with nu=%.1f', nu));
                
                % Validate that PDF integrates to 1 (approximately) using numerical integration
                xRange = linspace(-10, 10, 1000)';
                integrationPdf = gedpdf(xRange, nu);
                integral = trapz(xRange, integrationPdf);
                obj.assertAlmostEqual(1.0, integral, ...
                    sprintf('PDF should integrate to 1 for GED with nu=%.1f', nu));
                
                % Verify symmetry properties of the distribution
                negPoints = -testPoints;
                pdfNeg = gedpdf(negPoints, nu);
                obj.assertMatrixEqualsWithTolerance(pdfValues, pdfNeg, obj.defaultTolerance, ...
                    sprintf('PDF should be symmetric for GED with nu=%.1f', nu));
                
                % Verify mathematical relationships between PDF and CDF
                cdfNeg = gedcdf(negPoints, nu);
                obj.assertMatrixEqualsWithTolerance(1 - cdfValues, cdfNeg, obj.defaultTolerance, ...
                    sprintf('CDF should satisfy CDF(-x) = 1 - CDF(x) for GED with nu=%.1f', nu));
            end
        end
        
        function testSkewtFunctionConsistency(obj)
            % Tests the consistency between skewed-t distribution functions (PDF, CDF, inverse CDF)
            
            % For each skewed-t parameter combination (nu, lambda) in test configuration:
            for i = 1:length(obj.distributions.skewt.dofs)
                nu = obj.distributions.skewt.dofs(i);
                
                for j = 1:length(obj.distributions.skewt.skews)
                    lambda = obj.distributions.skewt.skews(j);
                    
                    % Generate test points across the distribution range
                    testPoints = linspace(-5, 5, 100)';
                    
                    % Compute PDF values using skewtpdf
                    pdfValues = skewtpdf(testPoints, nu, lambda);
                    
                    % Compute CDF values using skewtcdf
                    cdfValues = skewtcdf(testPoints, nu, lambda);
                    
                    % Test probabilities
                    testProbs = linspace(0.01, 0.99, 100)';
                    invValues = skewtinv(testProbs, nu, lambda);
                    
                    % Verify that CDF(inv(p)) ≈ p for probability values
                    cdfOfInv = skewtcdf(invValues, nu, lambda);
                    obj.assertMatrixEqualsWithTolerance(testProbs, cdfOfInv, obj.defaultTolerance * 10, ...
                        sprintf('CDF(inv(p)) should equal p for skew-t with nu=%.1f, lambda=%.1f', nu, lambda));
                    
                    % Verify that inv(CDF(x)) ≈ x for value points
                    invOfCdf = skewtinv(cdfValues, nu, lambda);
                    obj.assertMatrixEqualsWithTolerance(testPoints, invOfCdf, obj.defaultTolerance * 20, ...
                        sprintf('inv(CDF(x)) should equal x for skew-t with nu=%.1f, lambda=%.1f', nu, lambda));
                    
                    % Validate that PDF integrates to 1 (approximately) using numerical integration
                    xRange = linspace(-10, 10, 1000)';
                    integrationPdf = skewtpdf(xRange, nu, lambda);
                    integral = trapz(xRange, integrationPdf);
                    obj.assertAlmostEqual(1.0, integral, ...
                        sprintf('PDF should integrate to 1 for skew-t with nu=%.1f, lambda=%.1f', nu, lambda));
                    
                    % Verify asymmetry properties for lambda ≠ 0
                    if abs(lambda) > 1e-10
                        % For asymmetric distributions, PDF should not be symmetric
                        negPoints = -testPoints;
                        pdfNeg = skewtpdf(negPoints, nu, lambda);
                        pdfDiff = max(abs(pdfValues - pdfNeg));
                        obj.assertTrue(pdfDiff > obj.defaultTolerance, ...
                            sprintf('PDF should be asymmetric for skew-t with nu=%.1f, lambda=%.1f', nu, lambda));
                        
                        % And CDF should not satisfy CDF(-x) = 1 - CDF(x)
                        cdfNeg = skewtcdf(negPoints, nu, lambda);
                        cdfDiff = max(abs(1 - cdfValues - cdfNeg));
                        obj.assertTrue(cdfDiff > obj.defaultTolerance, ...
                            sprintf('CDF should be asymmetric for skew-t with nu=%.1f, lambda=%.1f', nu, lambda));
                    else
                        % For lambda = 0, skew-t reduces to standardized t, should be symmetric
                        negPoints = -testPoints;
                        pdfNeg = skewtpdf(negPoints, nu, lambda);
                        obj.assertMatrixEqualsWithTolerance(pdfValues, pdfNeg, obj.defaultTolerance * 10, ...
                            sprintf('PDF should be symmetric for skew-t with nu=%.1f, lambda=0', nu));
                        
                        cdfNeg = skewtcdf(negPoints, nu, lambda);
                        obj.assertMatrixEqualsWithTolerance(1 - cdfValues, cdfNeg, obj.defaultTolerance * 10, ...
                            sprintf('CDF should satisfy CDF(-x) = 1 - CDF(x) for skew-t with nu=%.1f, lambda=0', nu));
                    end
                end
            end
        end
        
        function testStdtFunctionConsistency(obj)
            % Tests the consistency between standardized t-distribution functions (PDF, CDF, inverse CDF)
            
            % For each standardized t degrees of freedom in test configuration:
            for i = 1:length(obj.distributions.stdt.dofs)
                nu = obj.distributions.stdt.dofs(i);
                
                % Generate test points across the distribution range
                testPoints = linspace(-5, 5, 100)';
                
                % Compute PDF values using stdtpdf
                pdfValues = stdtpdf(testPoints, nu);
                
                % Compute CDF values using stdtcdf
                cdfValues = stdtcdf(testPoints, nu);
                
                % Test probabilities
                testProbs = linspace(0.01, 0.99, 100)';
                invValues = stdtinv(testProbs, nu);
                
                % Verify that CDF(inv(p)) ≈ p for probability values
                cdfOfInv = stdtcdf(invValues, nu);
                obj.assertMatrixEqualsWithTolerance(testProbs, cdfOfInv, obj.defaultTolerance, ...
                    sprintf('CDF(inv(p)) should equal p for std-t with nu=%.1f', nu));
                
                % Verify that inv(CDF(x)) ≈ x for value points
                invOfCdf = stdtinv(cdfValues, nu);
                obj.assertMatrixEqualsWithTolerance(testPoints, invOfCdf, obj.defaultTolerance * 10, ...
                    sprintf('inv(CDF(x)) should equal x for std-t with nu=%.1f', nu));
                
                % Validate that PDF integrates to 1 (approximately) using numerical integration
                xRange = linspace(-10, 10, 1000)';
                integrationPdf = stdtpdf(xRange, nu);
                integral = trapz(xRange, integrationPdf);
                obj.assertAlmostEqual(1.0, integral, ...
                    sprintf('PDF should integrate to 1 for std-t with nu=%.1f', nu));
                
                % Verify symmetry properties of the distribution
                negPoints = -testPoints;
                pdfNeg = stdtpdf(negPoints, nu);
                obj.assertMatrixEqualsWithTolerance(pdfValues, pdfNeg, obj.defaultTolerance, ...
                    sprintf('PDF should be symmetric for std-t with nu=%.1f', nu));
                
                % Verify mathematical relationships between PDF and CDF
                cdfNeg = stdtcdf(negPoints, nu);
                obj.assertMatrixEqualsWithTolerance(1 - cdfValues, cdfNeg, obj.defaultTolerance, ...
                    sprintf('CDF should satisfy CDF(-x) = 1 - CDF(x) for std-t with nu=%.1f', nu));
            end
        end
        
        function testGEDParameterEstimation(obj)
            % Tests the accuracy of GED parameter estimation from random samples
            
            % For each GED shape parameter in test configuration:
            for i = 1:length(obj.distributions.ged.shapes)
                nu_true = obj.distributions.ged.shapes(i);
                
                % Generate random samples using gedrnd with known parameters
                data = gedrnd(nu_true, obj.sampleSize, 1);
                
                % Estimate distribution parameters using gedfit
                estimatedParams = gedfit(data);
                
                % Compare estimated parameters with true parameters
                nu_est = estimatedParams.nu;
                
                % Verify parameter estimation accuracy improves with sample size
                obj.assertAlmostEqual(nu_true, nu_est, ...
                    sprintf('GED shape parameter estimation for nu=%.1f', nu_true), 0.5);
                
                % Check that mu is close to 0 (GED has mean 0)
                obj.assertAlmostEqual(0, estimatedParams.mu, ...
                    sprintf('GED mean should be close to 0 for nu=%.1f', nu_true), 0.1);
                
                % Check that sigma is close to 1 (GED should have unit scale)
                obj.assertAlmostEqual(1, estimatedParams.sigma, ...
                    sprintf('GED scale should be close to 1 for nu=%.1f', nu_true), 0.1);
                
                % Verify goodness-of-fit between estimated distribution and sample data
                x_sorted = sort(data);
                empirical_cdf = (1:obj.sampleSize)' / obj.sampleSize;
                theoretical_cdf = gedcdf(x_sorted, nu_est, estimatedParams.mu, estimatedParams.sigma);
                
                % K-S statistic should be small if fit is good
                ks_stat = max(abs(empirical_cdf - theoretical_cdf));
                obj.assertTrue(ks_stat < 0.03, ...
                    sprintf('KS test for GED fit with nu=%.1f should pass', nu_true));
                
                % Validate standard errors and confidence intervals for parameter estimates
                logL_true = gedloglik(data, nu_true, 0, 1);
                logL_est = estimatedParams.loglik;
                
                % Estimated parameters should give better (or similar) log-likelihood
                obj.assertTrue(logL_est >= logL_true - 1, ...
                    sprintf('Estimated parameters should give similar or better log-likelihood for GED with nu=%.1f', nu_true));
            end
        end
        
        function testSkewtParameterEstimation(obj)
            % Tests the accuracy of skewed-t parameter estimation from random samples
            
            % For each skewed-t parameter combination in test configuration:
            for i = 1:length(obj.distributions.skewt.dofs)
                nu_true = obj.distributions.skewt.dofs(i);
                
                for j = 1:length(obj.distributions.skewt.skews)
                    lambda_true = obj.distributions.skewt.skews(j);
                    
                    % Generate random samples using skewtrnd with known parameters
                    data = skewtrnd(nu_true, lambda_true, obj.sampleSize, 1);
                    
                    % Estimate distribution parameters using skewtfit
                    estimatedParams = skewtfit(data);
                    
                    % Compare estimated parameters with true parameters
                    nu_est = estimatedParams.nu;
                    lambda_est = estimatedParams.lambda;
                    
                    % Verify parameter estimation accuracy improves with sample size
                    obj.assertAlmostEqual(nu_true, nu_est, ...
                        sprintf('Skew-t df estimation for nu=%.1f, lambda=%.1f', nu_true, lambda_true), 2.0);
                    
                    % For skewness parameter
                    obj.assertAlmostEqual(lambda_true, lambda_est, ...
                        sprintf('Skew-t skewness estimation for nu=%.1f, lambda=%.1f', nu_true, lambda_true), 0.2);
                    
                    % Check that mu is close to 0 
                    obj.assertAlmostEqual(0, estimatedParams.mu, ...
                        sprintf('Skew-t mean should be close to 0 for nu=%.1f, lambda=%.1f', nu_true, lambda_true), 0.1);
                    
                    % Check that sigma is close to 1
                    obj.assertAlmostEqual(1, estimatedParams.sigma, ...
                        sprintf('Skew-t scale should be close to 1 for nu=%.1f, lambda=%.1f', nu_true, lambda_true), 0.2);
                    
                    % Verify goodness-of-fit between estimated distribution and sample data
                    x_sorted = sort(data);
                    empirical_cdf = (1:obj.sampleSize)' / obj.sampleSize;
                    theoretical_cdf = skewtcdf(x_sorted, nu_est, lambda_est);
                    
                    % K-S statistic should be small if fit is good
                    ks_stat = max(abs(empirical_cdf - theoretical_cdf));
                    obj.assertTrue(ks_stat < 0.03, ...
                        sprintf('KS test for skew-t fit with nu=%.1f, lambda=%.1f should pass', nu_true, lambda_true));
                    
                    % Validate standard errors and confidence intervals for parameter estimates
                    [nlogL_true, ~] = skewtloglik(data, [nu_true, lambda_true, 0, 1]);
                    logL_true = -nlogL_true;
                    logL_est = estimatedParams.logL;
                    
                    % Estimated parameters should give better (or similar) log-likelihood
                    obj.assertTrue(logL_est >= logL_true - 1, ...
                        sprintf('Estimated parameters should give similar or better log-likelihood for skew-t with nu=%.1f, lambda=%.1f', nu_true, lambda_true));
                    
                    % Test robustness with different degrees of skewness
                end
            end
        end
        
        function testStdtParameterEstimation(obj)
            % Tests the accuracy of standardized t parameter estimation from random samples
            
            % For each standardized t degrees of freedom in test configuration:
            for i = 1:length(obj.distributions.stdt.dofs)
                nu_true = obj.distributions.stdt.dofs(i);
                
                % Generate random samples using stdtrnd with known parameters
                data = stdtrnd([obj.sampleSize, 1], nu_true);
                
                % Estimate distribution parameters using stdtfit
                estimatedParams = stdtfit(data);
                
                % Compare estimated degrees of freedom with true value
                nu_est = estimatedParams.nu;
                
                % Verify parameter estimation accuracy improves with sample size
                obj.assertAlmostEqual(nu_true, nu_est, ...
                    sprintf('std-t df estimation for nu=%.1f', nu_true), 3.0);
                
                % Check that mu is close to 0 (standardized t has mean 0)
                obj.assertAlmostEqual(0, estimatedParams.mu, ...
                    sprintf('std-t mean should be close to 0 for nu=%.1f', nu_true), 0.1);
                
                % Check that sigma is close to 1 (standardized t has unit variance)
                obj.assertAlmostEqual(1, estimatedParams.sigma, ...
                    sprintf('std-t scale should be close to 1 for nu=%.1f', nu_true), 0.1);
                
                % Verify goodness-of-fit between estimated distribution and sample data
                x_sorted = sort(data);
                empirical_cdf = (1:obj.sampleSize)' / obj.sampleSize;
                theoretical_cdf = stdtcdf(x_sorted, nu_est);
                
                % K-S statistic should be small if fit is good
                ks_stat = max(abs(empirical_cdf - theoretical_cdf));
                obj.assertTrue(ks_stat < 0.03, ...
                    sprintf('KS test for std-t fit with nu=%.1f should pass', nu_true));
                
                % Validate standard errors and confidence intervals for parameter estimates
                logL_true = -stdtloglik(data, nu_true, 0, 1);
                logL_est = estimatedParams.logL;
                
                % Estimated parameters should give better (or similar) log-likelihood
                obj.assertTrue(logL_est >= logL_true - 1, ...
                    sprintf('Estimated parameters should give similar or better log-likelihood for std-t with nu=%.1f', nu_true));
            end
        end
        
        function testRandomNumberGeneration(obj)
            % Tests the statistical properties of random number generators for all distributions
            
            % For each distribution type and parameter combination:
            
            % Test GED random number generation
            for i = 1:length(obj.distributions.ged.shapes)
                nu = obj.distributions.ged.shapes(i);
                
                % Generate large samples using respective random number generators
                samples = gedrnd(nu, obj.sampleSize * 2, 1);
                
                % Compute sample statistics (mean, variance, skewness, kurtosis)
                sampleMean = mean(samples);
                sampleVar = var(samples);
                sampleSkewness = skewness(samples);
                sampleKurtosis = kurtosis(samples);
                
                % Compare sample statistics with theoretical values
                obj.assertAlmostEqual(0, sampleMean, ...
                    sprintf('GED random sample mean should be close to 0 for nu=%.1f', nu), 0.05);
                
                obj.assertAlmostEqual(1, sampleVar, ...
                    sprintf('GED random sample variance should be close to 1 for nu=%.1f', nu), 0.1);
                
                % Theoretical kurtosis for GED distribution
                theoreticalKurtosis = 3 * gamma(5/nu) * gamma(1/nu) / (gamma(3/nu)^2);
                
                % Kurtosis is harder to estimate precisely, so we use a wider tolerance
                obj.assertAlmostEqual(theoreticalKurtosis, sampleKurtosis, ...
                    sprintf('GED random sample kurtosis should be close to theoretical for nu=%.1f', nu), ...
                    max(1, theoreticalKurtosis * 0.2)); % 20% tolerance
                
                % GED is symmetric, so skewness should be near 0
                obj.assertAlmostEqual(0, sampleSkewness, ...
                    sprintf('GED random sample skewness should be close to 0 for nu=%.1f', nu), 0.1);
                
                % Perform Kolmogorov-Smirnov test for distribution fit
                [h, ~, ksstat] = kstest((samples - mean(samples))/std(samples), ...
                    @(x) gedcdf(x, nu));
                obj.assertFalse(h, sprintf('K-S test should not reject GED fit for nu=%.1f (ksstat=%.4f)', nu, ksstat));
            end
            
            % Test skewed-t random number generation
            for i = 1:length(obj.distributions.skewt.dofs)
                nu = obj.distributions.skewt.dofs(i);
                
                for j = 1:length(obj.distributions.skewt.skews)
                    lambda = obj.distributions.skewt.skews(j);
                    
                    % Generate large sample
                    samples = skewtrnd(nu, lambda, obj.sampleSize * 2, 1);
                    
                    % Compute sample statistics
                    sampleMean = mean(samples);
                    sampleVar = var(samples);
                    sampleSkewness = skewness(samples);
                    
                    % Verify mean and variance
                    obj.assertAlmostEqual(0, sampleMean, ...
                        sprintf('Skew-t random sample mean should be close to 0 for nu=%.1f, lambda=%.1f', nu, lambda), 0.1);
                    
                    obj.assertAlmostEqual(1, sampleVar, ...
                        sprintf('Skew-t random sample variance should be close to 1 for nu=%.1f, lambda=%.1f', nu, lambda), 0.2);
                    
                    % Check skewness direction matches lambda sign
                    if lambda > 0.1
                        obj.assertTrue(sampleSkewness > 0, ...
                            sprintf('Skew-t random sample should have positive skewness for lambda=%.1f', lambda));
                    elseif lambda < -0.1
                        obj.assertTrue(sampleSkewness < 0, ...
                            sprintf('Skew-t random sample should have negative skewness for lambda=%.1f', lambda));
                    end
                    
                    % Check distribution fit using K-S test
                    stdSamples = (samples - mean(samples))/std(samples);
                    [h, ~, ksstat] = kstest(stdSamples, ...
                        @(x) skewtcdf(x, nu, lambda));
                    obj.assertFalse(h, sprintf('K-S test should not reject skew-t fit for nu=%.1f, lambda=%.1f (ksstat=%.4f)', ...
                        nu, lambda, ksstat));
                end
            end
            
            % Test standardized t random number generation
            for i = 1:length(obj.distributions.stdt.dofs)
                nu = obj.distributions.stdt.dofs(i);
                
                % Generate large sample
                samples = stdtrnd([obj.sampleSize * 2, 1], nu);
                
                % Compute sample statistics
                sampleMean = mean(samples);
                sampleVar = var(samples);
                sampleSkewness = skewness(samples);
                sampleKurtosis = kurtosis(samples);
                
                % Verify sample statistics
                obj.assertAlmostEqual(0, sampleMean, ...
                    sprintf('Std-t random sample mean should be close to 0 for nu=%.1f', nu), 0.05);
                
                obj.assertAlmostEqual(1, sampleVar, ...
                    sprintf('Std-t random sample variance should be close to 1 for nu=%.1f', nu), 0.1);
                
                % Standardized t is symmetric, skewness should be near 0
                obj.assertAlmostEqual(0, sampleSkewness, ...
                    sprintf('Std-t random sample skewness should be close to 0 for nu=%.1f', nu), 0.1);
                
                % Check theoretical kurtosis if nu > 4
                if nu > 4
                    theoreticalKurtosis = 3 + 6/(nu-4);
                    
                    % Kurtosis is harder to estimate precisely, so we use a wider tolerance
                    obj.assertAlmostEqual(theoreticalKurtosis, sampleKurtosis, ...
                        sprintf('Std-t random sample kurtosis should be close to theoretical for nu=%.1f', nu), ...
                        max(1, theoreticalKurtosis * 0.2)); % 20% tolerance
                end
                
                % Check distribution fit using K-S test
                [h, ~, ksstat] = kstest((samples - mean(samples))/std(samples), ...
                    @(x) stdtcdf(x, nu));
                obj.assertFalse(h, sprintf('K-S test should not reject std-t fit for nu=%.1f (ksstat=%.4f)', nu, ksstat));
            end
        end
        
        function testCrossDistributionRelationships(obj)
            % Tests the relationships between different distribution types under special parameter values
            
            % Test relationship between skewed-t with lambda=0 and standardized t-distribution
            nu = 5; % Degrees of freedom
            lambda = 0; % No skewness
            
            % Test points
            x = linspace(-5, 5, 100)';
            
            % Compute PDFs from both distributions
            pdf_skewt = skewtpdf(x, nu, lambda);
            pdf_stdt = stdtpdf(x, nu);
            
            % They should be almost identical
            obj.assertMatrixEqualsWithTolerance(pdf_skewt, pdf_stdt, obj.defaultTolerance * 10, ...
                'Skew-t with lambda=0 should equal std-t distribution (PDF)');
            
            % Also compare CDFs
            cdf_skewt = skewtcdf(x, nu, lambda);
            cdf_stdt = stdtcdf(x, nu);
            
            obj.assertMatrixEqualsWithTolerance(cdf_skewt, cdf_stdt, obj.defaultTolerance * 10, ...
                'Skew-t with lambda=0 should equal std-t distribution (CDF)');
            
            % Test relationship between GED with nu=2 and normal distribution
            nu = 2; % Shape parameter for normal
            
            % Compute PDFs
            pdf_ged = gedpdf(x, nu);
            pdf_norm = normpdf(x, 0, 1);
            
            obj.assertMatrixEqualsWithTolerance(pdf_ged, pdf_norm, obj.defaultTolerance * 10, ...
                'GED with nu=2 should equal normal distribution (PDF)');
            
            % Also compare CDFs
            cdf_ged = gedcdf(x, nu);
            cdf_norm = normcdf(x, 0, 1);
            
            obj.assertMatrixEqualsWithTolerance(cdf_ged, cdf_norm, obj.defaultTolerance * 10, ...
                'GED with nu=2 should equal normal distribution (CDF)');
            
            % Test relationship between GED with nu=1 and Laplace distribution
            nu = 1; % Shape parameter for Laplace
            
            % Compute PDFs
            pdf_ged = gedpdf(x, nu);
            
            % Manual Laplace PDF calculation
            lambda = sqrt(gamma(3/nu) / gamma(1/nu)); % ≈ sqrt(2) for nu=1
            pdf_laplace = (1/(2*lambda)) * exp(-abs(x/lambda));
            
            obj.assertMatrixEqualsWithTolerance(pdf_ged, pdf_laplace, obj.defaultTolerance * 10, ...
                'GED with nu=1 should closely match Laplace distribution (PDF)');
            
            % Test relationship between standardized t with large df and normal distribution
            nu = 100; % Very large degrees of freedom
            
            % Compute PDFs
            pdf_stdt = stdtpdf(x, nu);
            pdf_norm = normpdf(x, 0, 1);
            
            % With large df, should be very close to normal
            obj.assertMatrixEqualsWithTolerance(pdf_stdt, pdf_norm, 0.02, ...
                'Std-t with large nu should approach normal distribution (PDF)');
            
            % Also compare CDFs
            cdf_stdt = stdtcdf(x, nu);
            cdf_norm = normcdf(x, 0, 1);
            
            obj.assertMatrixEqualsWithTolerance(cdf_stdt, cdf_norm, 0.02, ...
                'Std-t with large nu should approach normal distribution (CDF)');
        end
        
        function testDistributionLimits(obj)
            % Tests distribution behavior at parameter boundaries and extreme values
            
            % Test GED with very small and very large nu values
            % Very small nu (heavy-tailed)
            nu_small = 0.5;
            x_center = linspace(-0.5, 0.5, 50)'; % Focus on center
            
            % PDF should be defined and have a peak at zero
            pdf_small = gedpdf(x_center, nu_small);
            [max_val, max_idx] = max(pdf_small);
            obj.assertTrue(max_val > 0, 'GED PDF with small nu should have positive peak');
            obj.assertTrue(abs(x_center(max_idx)) < 0.1, 'GED PDF with small nu should peak near zero');
            
            % Very large nu (thin-tailed)
            nu_large = 10;
            x = linspace(-3, 3, 100)';
            
            % PDF should approach uniform distribution over a bounded range as nu increases
            pdf_large = gedpdf(x, nu_large);
            
            % Check that PDF is relatively flat in the middle and drops sharply at edges
            middle_idx = (x >= -1) & (x <= 1);
            edge_idx = (x < -2) | (x > 2);
            
            middle_std = std(pdf_large(middle_idx));
            middle_mean = mean(pdf_large(middle_idx));
            edge_max = max(pdf_large(edge_idx));
            
            obj.assertTrue(middle_std / middle_mean < 0.5, ...
                'GED PDF with large nu should be relatively flat in the middle');
            obj.assertTrue(edge_max < 0.1 * middle_mean, ...
                'GED PDF with large nu should drop sharply at edges');
            
            % Test skewed-t with extreme skewness (lambda near ±1)
            nu = 5; % Fixed degrees of freedom
            lambda_extreme = 0.99; % Almost maximum skewness
            
            % Test points
            x = linspace(-5, 5, 100)';
            
            % Compute PDF and CDF
            pdf_skew = skewtpdf(x, nu, lambda_extreme);
            cdf_skew = skewtcdf(x, nu, lambda_extreme);
            
            % With extreme positive skewness, PDF should be shifted left
            % Find the mode (peak of PDF)
            [~, mode_idx] = max(pdf_skew);
            mode_x = x(mode_idx);
            
            obj.assertTrue(mode_x < -1, ...
                'Skew-t PDF with extreme positive lambda should have mode shifted left');
            
            % CDF should rise more slowly on the right
            % Check where CDF crosses 0.5
            cross_idx = find(cdf_skew >= 0.5, 1, 'first');
            median_x = x(cross_idx);
            
            obj.assertTrue(median_x < -1, ...
                'Skew-t median with extreme positive lambda should be shifted left');
            
            % Test with opposite skewness
            pdf_neg_skew = skewtpdf(x, nu, -lambda_extreme);
            
            % Test standardized t with very small and very large degrees of freedom
            % Very small df (very heavy tails)
            nu_small = 2.1; % Just above 2 to ensure finite variance
            x = linspace(-5, 5, 100)';
            
            % PDF should have a sharp peak at zero and heavy tails
            pdf_small_t = stdtpdf(x, nu_small);
            [max_val, max_idx] = max(pdf_small_t);
            obj.assertTrue(max_val > normpdf(0, 0, 1), ...
                'Std-t PDF with small nu should have higher peak than normal');
            obj.assertTrue(abs(x(max_idx)) < 0.1, ...
                'Std-t PDF with small nu should peak near zero');
            
            % Check heavy tails - t distribution should have higher density in tails than normal
            tail_idx = abs(x) > 3;
            pdf_norm_tail = normpdf(x(tail_idx), 0, 1);
            pdf_t_tail = stdtpdf(x(tail_idx), nu_small);
            
            obj.assertTrue(all(pdf_t_tail > pdf_norm_tail), ...
                'Std-t with small nu should have heavier tails than normal');
        end
        
        function testFinancialReturnsModeling(obj)
            % Tests distribution fitting with real financial returns data
            
            % Load financial returns test data
            returns = obj.testData.t.returns;
            
            % Standardize returns
            std_returns = (returns - mean(returns)) / std(returns);
            
            % Fit GED, skewed-t, and standardized t-distributions to the data
            ged_params = gedfit(std_returns);
            skewt_params = skewtfit(std_returns);
            stdt_params = stdtfit(std_returns);
            
            % Compare goodness-of-fit measures across distributions
            logL_ged = ged_params.loglik;
            logL_skewt = skewt_params.logL;
            logL_stdt = stdt_params.logL;
            
            % Compare AIC values (lower is better)
            aic_ged = -2 * logL_ged + 2 * 3; % 3 parameters for GED
            aic_skewt = -2 * logL_skewt + 2 * 4; % 4 parameters for skew-t
            aic_stdt = -2 * logL_stdt + 2 * 3; % 3 parameters for std-t
            
            % Create a benchmark with normal distribution for comparison
            norm_logL = sum(log(normpdf(std_returns, 0, 1)));
            aic_norm = -2 * norm_logL + 2 * 2; % 2 parameters for normal
            
            % Verify capturing of stylized facts (fat tails, skewness)
            % At least one model should beat normal distribution
            obj.assertTrue(min([aic_ged, aic_skewt, aic_stdt]) < aic_norm, ...
                'At least one heavy-tailed model should fit financial returns better than normal');
            
            % GED shape parameter typically < 2 for financial returns (leptokurtic)
            obj.assertTrue(ged_params.nu < 2, ...
                'GED shape parameter for financial returns should indicate heavy tails (nu < 2)');
            
            % t degrees of freedom typically < 10 for financial returns
            obj.assertTrue(stdt_params.nu < 10, ...
                'Student-t degrees of freedom for financial returns should be < 10');
            
            % Perform K-S tests to verify the fits
            % GED fit
            [~, ~, ksstat_ged] = kstest(std_returns, @(x) gedcdf(x, ged_params.nu, 0, 1));
            
            % Skew-t fit
            [~, ~, ksstat_skewt] = kstest(std_returns, @(x) skewtcdf(x, skewt_params.nu, skewt_params.lambda));
            
            % Std-t fit
            [~, ~, ksstat_stdt] = kstest(std_returns, @(x) stdtcdf(x, stdt_params.nu));
            
            % Normal fit (benchmark)
            [~, ~, ksstat_norm] = kstest(std_returns);
            
            % At least one model should have a better K-S statistic than normal
            min_ks = min([ksstat_ged, ksstat_skewt, ksstat_stdt]);
            obj.assertTrue(min_ks < ksstat_norm, ...
                'At least one heavy-tailed model should have better K-S fit than normal');
        end
        
        function testNumericalStability(obj)
            % Tests numerical stability of distribution functions with challenging inputs
            
            % Test with extreme parameter values for all distributions
            % Very small shape parameter for GED
            nu_small = 0.1;
            
            % Check if PDF is defined for small nu and doesn't produce NaNs or Infs
            x_test = linspace(-1, 1, 10)';
            pdf_vals = gedpdf(x_test, nu_small);
            obj.assertTrue(all(isfinite(pdf_vals)), ...
                'GED PDF should give finite values for small nu');
            
            % Very large shape parameter for GED
            nu_large = 100;
            pdf_vals = gedpdf(x_test, nu_large);
            obj.assertTrue(all(isfinite(pdf_vals)), ...
                'GED PDF should give finite values for large nu');
            
            % Verify behavior with values far in the distribution tails
            x_extreme = [-1e10; -1e5; -20; 20; 1e5; 1e10];
            
            % GED
            nu = 1.5;
            pdf_extreme = gedpdf(x_extreme, nu);
            cdf_extreme = gedcdf(x_extreme, nu);
            
            % Check that PDF approaches 0 in extreme tails
            obj.assertTrue(all(pdf_extreme(abs(x_extreme) > 1e5) < 1e-10), ...
                'GED PDF should approach 0 in extreme tails');
            
            % Check that CDF approaches 0/1 in extreme tails
            obj.assertAlmostEqual(0, cdf_extreme(1), 'GED CDF should approach 0 in left tail');
            obj.assertAlmostEqual(1, cdf_extreme(end), 'GED CDF should approach 1 in right tail');
            
            % Skewed t
            nu = 5;
            lambda = 0.3;
            pdf_extreme = skewtpdf(x_extreme, nu, lambda);
            cdf_extreme = skewtcdf(x_extreme, nu, lambda);
            
            % Check that PDF approaches 0 in extreme tails
            obj.assertTrue(all(pdf_extreme(abs(x_extreme) > 1e5) < 1e-10), ...
                'Skew-t PDF should approach 0 in extreme tails');
            
            % Check that CDF approaches 0/1 in extreme tails
            obj.assertAlmostEqual(0, cdf_extreme(1), 'Skew-t CDF should approach 0 in left tail');
            obj.assertAlmostEqual(1, cdf_extreme(end), 'Skew-t CDF should approach 1 in right tail');
            
            % Test with very large and very small inputs
            % Very small probabilities for inverse CDF functions
            small_probs = [0; 1e-10; 1e-5; 1-1e-5; 1-1e-10; 1];
            
            % GED
            nu = 1.5;
            inv_vals = gedinv(small_probs, nu);
            
            % Check that extreme probabilities give appropriate quantiles
            obj.assertTrue(inv_vals(1) == -Inf, 'GED inverse CDF should return -Inf for p=0');
            obj.assertTrue(inv_vals(end) == Inf, 'GED inverse CDF should return Inf for p=1');
            obj.assertTrue(all(isfinite(inv_vals(2:end-1))), ...
                'GED inverse CDF should return finite values for p in (0,1)');
            
            % Validate behavior with NaN, Inf, and near-zero inputs
            % Edge cases for parameter values
            nu_edge = 0.01;
            x_test = 0;
            pdf_val = gedpdf(x_test, nu_edge);
            obj.assertTrue(isfinite(pdf_val) && pdf_val > 0, ...
                'GED PDF should be positive at x=0 even for very small nu');
        end
        
        % Helper methods
        
        function fitResults = verifyDistributionFit(obj, trueParams, estimatedParams, tolerance)
            % Helper method to verify the accuracy of distribution parameter estimation
            
            % Extract true and estimated parameters
            if isstruct(trueParams)
                trueParamFields = fieldnames(trueParams);
                trueParamValues = zeros(length(trueParamFields), 1);
                for i = 1:length(trueParamFields)
                    trueParamValues(i) = trueParams.(trueParamFields{i});
                end
            else
                trueParamValues = trueParams;
                trueParamFields = {};
            end
            
            if isstruct(estimatedParams)
                estParamFields = fieldnames(estimatedParams);
                estParamValues = zeros(length(estParamFields), 1);
                for i = 1:length(estParamFields)
                    estParamValues(i) = estimatedParams.(estParamFields{i});
                end
            else
                estParamValues = estimatedParams;
                estParamFields = {};
            end
            
            % Calculate parameter estimation errors
            if length(trueParamValues) == length(estParamValues)
                errors = trueParamValues - estParamValues;
                mse = mean(errors.^2);
                rmse = sqrt(mse);
                
                % Verify parameters are within specified tolerance
                withinTolerance = rmse <= tolerance;
                
                % Check for parameter bias
                bias = mean(errors);
                
                % Compute log-likelihood and information criteria
                fitResults = struct();
                fitResults.errors = errors;
                fitResults.mse = mse;
                fitResults.rmse = rmse;
                fitResults.bias = bias;
                fitResults.withinTolerance = withinTolerance;
                
                % Add parameter names if available
                if ~isempty(trueParamFields) && ~isempty(estParamFields)
                    fitResults.paramNames = trueParamFields;
                end
            else
                error('Number of parameters does not match between true and estimated.');
            end
        end
        
        function isVerified = verifyPdfCdfRelationship(obj, pdfFunc, cdfFunc, points, params)
            % Helper method to verify the relationship between PDF and CDF functions
            
            % Compute PDF values at test points
            pdfValues = pdfFunc(points, params{:});
            
            % Compute CDF values at test points
            cdfValues = cdfFunc(points, params{:});
            
            % Compute numerical derivative of CDF
            h = 1e-4; % Small step for numerical derivative
            pointsPlus = points + h;
            pointsMinus = points - h;
            
            cdfPlus = cdfFunc(pointsPlus, params{:});
            cdfMinus = cdfFunc(pointsMinus, params{:});
            
            % Central difference approximation of derivative
            numDerivative = (cdfPlus - cdfMinus) / (2 * h);
            
            % Compare numerical derivative with PDF values
            validIdx = abs(points) < 5; % Exclude extreme points where numerical issues may occur
            maxDiff = max(abs(numDerivative(validIdx) - pdfValues(validIdx)));
            
            % Verify that PDF integrates to 1
            wideRange = linspace(-20, 20, 1000)';
            widePdf = pdfFunc(wideRange, params{:});
            integral = trapz(wideRange, widePdf);
            
            % Return whether all tests pass within tolerance
            derivativePass = maxDiff < 0.01; % Tolerance for derivative comparison
            integralPass = abs(integral - 1) < 0.01; % Tolerance for integral to 1
            
            isVerified = derivativePass && integralPass;
        end
        
        function validation = validateRandomSample(obj, sample, theoreticalStats, cdfFunc, params)
            % Helper method to validate statistical properties of random samples
            
            % Calculate sample statistics (mean, variance, skewness, kurtosis)
            sampleStats = struct();
            sampleStats.mean = mean(sample);
            sampleStats.variance = var(sample);
            sampleStats.skewness = skewness(sample);
            sampleStats.kurtosis = kurtosis(sample);
            
            % Compare sample statistics with theoretical values
            statDiffs = struct();
            statDiffs.meanDiff = abs(sampleStats.mean - theoreticalStats.mean);
            statDiffs.varianceDiff = abs(sampleStats.variance - theoreticalStats.variance);
            statDiffs.skewnessDiff = abs(sampleStats.skewness - theoreticalStats.skewness);
            statDiffs.kurtosisDiff = abs(sampleStats.kurtosis - theoreticalStats.kurtosis);
            
            % Perform Kolmogorov-Smirnov test against theoretical distribution
            stdSample = (sample - sampleStats.mean) / sqrt(sampleStats.variance);
            [h, p, ksstat] = kstest(stdSample, @(x) cdfFunc(x, params{:}));
            
            % Apply probability integral transform and test uniformity
            u = cdfFunc(sample, params{:});
            [h_uniform, p_uniform] = kstest(u, 'CDF', [0:0.01:1; 0:0.01:1]');
            
            % Check independence properties of the sample
            validation = struct();
            validation.sampleStats = sampleStats;
            validation.theoreticalStats = theoreticalStats;
            validation.statDiffs = statDiffs;
            validation.kstest = struct('h', h, 'p', p, 'ksstat', ksstat);
            validation.uniformTest = struct('h', h_uniform, 'p', p_uniform);
            
            % Return structure with validation results
            validation.statsWithinTolerance = (statDiffs.meanDiff < 0.1) && ...
                                             (statDiffs.varianceDiff < 0.1) && ...
                                             (statDiffs.skewnessDiff < 0.5) && ...
                                             (statDiffs.kurtosisDiff < max(1, 0.2 * theoreticalStats.kurtosis));
            validation.distributionTestPass = ~h;
            validation.uniformityTestPass = ~h_uniform;
            validation.overallValid = validation.statsWithinTolerance && validation.distributionTestPass;
        end
    end
end