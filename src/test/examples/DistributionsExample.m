% DistributionsExample.m
% This script demonstrates the usage of statistical distribution functions in the MFE Toolbox
% including Generalized Error Distribution (GED), Hansen's skewed t-distribution, and 
% standardized Student's t-distribution with examples of parameter estimation, probability
% computation, and random number generation.

% Set random number generator seed for reproducibility
rng('default')

%% Demonstrate GED (Generalized Error Distribution)
demonstrate_ged_distribution();

%% Demonstrate Hansen's skewed t-distribution
demonstrate_skewt_distribution();

%% Demonstrate standardized Student's t-distribution
demonstrate_stdt_distribution();

%% Compare all three distributions
compare_distributions();

%% Apply to real financial data
financial_data_example();

%% ------------------------------------------------------------------------
function demonstrate_ged_distribution()
    % This function demonstrates the use of Generalized Error Distribution functions
    
    % Display section header
    disp('--------------------------------------------------------------');
    disp('GENERALIZED ERROR DISTRIBUTION (GED) DEMONSTRATION');
    disp('--------------------------------------------------------------');
    
    % Define different shape parameters for comparison
    % nu = 1: Laplace (double exponential) distribution
    % nu = 2: Normal distribution
    % nu = 5: Thin-tailed distribution
    nu_values = [1, 2, 5];
    
    % Create grid of x values for function evaluation
    x = linspace(-5, 5, 1000)';
    
    % Calculate and plot PDF values for different shape parameters
    figure('Name', 'GED Distribution - PDF');
    pdf_values = zeros(length(x), length(nu_values));
    
    for i = 1:length(nu_values)
        pdf_values(:, i) = gedpdf(x, nu_values(i));
    end
    
    plot(x, pdf_values);
    title('Generalized Error Distribution PDF');
    xlabel('x');
    ylabel('f(x)');
    legend({sprintf('nu = %.1f (Laplace)', nu_values(1)), ...
            sprintf('nu = %.1f (Normal)', nu_values(2)), ...
            sprintf('nu = %.1f (Thin-tailed)', nu_values(3))}, ...
            'Location', 'best');
    grid on;
    
    % Calculate and plot CDF values for different shape parameters
    figure('Name', 'GED Distribution - CDF');
    cdf_values = zeros(length(x), length(nu_values));
    
    for i = 1:length(nu_values)
        cdf_values(:, i) = gedcdf(x, nu_values(i));
    end
    
    plot(x, cdf_values);
    title('Generalized Error Distribution CDF');
    xlabel('x');
    ylabel('F(x)');
    legend({sprintf('nu = %.1f (Laplace)', nu_values(1)), ...
            sprintf('nu = %.1f (Normal)', nu_values(2)), ...
            sprintf('nu = %.1f (Thin-tailed)', nu_values(3))}, ...
            'Location', 'best');
    grid on;
    
    % Generate and plot random samples for a selected shape parameter
    selected_nu = 1.5;  % Using an intermediate value
    n_samples = 5000;
    random_samples = gedrnd(selected_nu, n_samples, 1);
    
    figure('Name', 'GED Distribution - Random Samples');
    subplot(2, 1, 1);
    hist(random_samples, 50);
    title(sprintf('Histogram of %d Random Samples from GED (nu = %.1f)', n_samples, selected_nu));
    xlabel('Value');
    ylabel('Frequency');
    
    subplot(2, 1, 2);
    % Calculate theoretical PDF for comparison
    x_range = linspace(min(random_samples), max(random_samples), 1000)';
    pdf_theoretical = gedpdf(x_range, selected_nu);
    
    % Plot histogram with PDF overlay
    [counts, edges] = histcounts(random_samples, 50, 'Normalization', 'pdf');
    centers = (edges(1:end-1) + edges(2:end)) / 2;
    bar(centers, counts, 'FaceAlpha', 0.3);
    hold on;
    plot(x_range, pdf_theoretical, 'r-', 'LineWidth', 2);
    hold off;
    title('Normalized Histogram with Theoretical PDF Overlay');
    xlabel('Value');
    ylabel('Density');
    legend('Empirical', 'Theoretical');
    
    % Parameter estimation example
    disp('Parameter Estimation Example for GED:');
    disp('-----------------------------------');
    
    % Generate data with known parameters
    true_nu = 1.3;
    true_mu = 0.5;
    true_sigma = 1.2;
    
    n_samples = 2000;
    sample_data = gedrnd(true_nu, n_samples, 1, true_mu, true_sigma);
    
    % Set options for parameter estimation
    options.display = 'off';
    
    % Estimate parameters
    tic;
    estimated_params = gedfit(sample_data, options);
    estimation_time = toc;
    
    % Display results
    disp('True parameters:');
    fprintf('nu = %.4f, mu = %.4f, sigma = %.4f\n', true_nu, true_mu, true_sigma);
    
    disp('Estimated parameters:');
    fprintf('nu = %.4f (SE: %.4f)\n', estimated_params.nu, estimated_params.stderrors(1));
    fprintf('mu = %.4f (SE: %.4f)\n', estimated_params.mu, estimated_params.stderrors(2));
    fprintf('sigma = %.4f (SE: %.4f)\n', estimated_params.sigma, estimated_params.stderrors(3));
    fprintf('Log-likelihood: %.4f\n', estimated_params.loglik);
    fprintf('Estimation time: %.4f seconds\n', estimation_time);
    
    % Q-Q plot to assess goodness of fit
    figure('Name', 'GED Distribution - Q-Q Plot');
    
    % Sort the sample data
    sorted_data = sort(sample_data);
    
    % Generate theoretical quantiles
    p_values = ((1:n_samples) - 0.5) / n_samples;
    theoretical_quantiles = gedinv(p_values, estimated_params.nu, estimated_params.mu, estimated_params.sigma);
    
    % Create Q-Q plot
    plot(theoretical_quantiles, sorted_data, 'o');
    hold on;
    
    % Add reference line
    min_val = min(min(theoretical_quantiles), min(sorted_data));
    max_val = max(max(theoretical_quantiles), max(sorted_data));
    plot([min_val, max_val], [min_val, max_val], 'r--');
    hold off;
    
    title('Q-Q Plot for GED Fit');
    xlabel('Theoretical Quantiles');
    ylabel('Sample Quantiles');
    grid on;
    
    % Demonstration of log-likelihood function
    disp('Log-likelihood variation with shape parameter:');
    nu_test = linspace(0.8, 3.0, 20);
    ll_values = zeros(size(nu_test));
    
    for i = 1:length(nu_test)
        ll_values(i) = gedloglik(sample_data, nu_test(i), estimated_params.mu, estimated_params.sigma);
    end
    
    figure('Name', 'GED Distribution - Log-Likelihood');
    plot(nu_test, ll_values, 'o-');
    hold on;
    plot([true_nu, true_nu], [min(ll_values), max(ll_values)], 'r--');
    plot([estimated_params.nu, estimated_params.nu], [min(ll_values), max(ll_values)], 'g--');
    hold off;
    
    title('Log-Likelihood vs. Shape Parameter (nu)');
    xlabel('Shape Parameter (nu)');
    ylabel('Log-Likelihood');
    legend('Log-likelihood', 'True value', 'Estimated value');
    grid on;
end

function demonstrate_skewt_distribution()
    % This function demonstrates the use of Hansen's skewed t-distribution functions
    
    % Display section header
    disp('--------------------------------------------------------------');
    disp('HANSEN''S SKEWED T-DISTRIBUTION DEMONSTRATION');
    disp('--------------------------------------------------------------');
    
    % Define different parameters for comparison
    nu = 5;  % Degrees of freedom
    lambda_values = [-0.5, 0, 0.5];  % Skewness parameter (0 = symmetric)
    
    % Create grid of x values for function evaluation
    x = linspace(-5, 5, 1000)';
    
    % Calculate and plot PDF values for different skewness parameters
    figure('Name', 'Skewed t-Distribution - PDF');
    pdf_values = zeros(length(x), length(lambda_values));
    
    for i = 1:length(lambda_values)
        pdf_values(:, i) = skewtpdf(x, nu, lambda_values(i));
    end
    
    plot(x, pdf_values);
    title(sprintf('Hansen''s Skewed t-Distribution PDF (nu = %d)', nu));
    xlabel('x');
    ylabel('f(x)');
    legend({sprintf('lambda = %.1f (negative skew)', lambda_values(1)), ...
            sprintf('lambda = %.1f (symmetric)', lambda_values(2)), ...
            sprintf('lambda = %.1f (positive skew)', lambda_values(3))}, ...
            'Location', 'best');
    grid on;
    
    % Calculate and plot CDF values for different skewness parameters
    figure('Name', 'Skewed t-Distribution - CDF');
    cdf_values = zeros(length(x), length(lambda_values));
    
    for i = 1:length(lambda_values)
        cdf_values(:, i) = skewtcdf(x, nu, lambda_values(i));
    end
    
    plot(x, cdf_values);
    title(sprintf('Hansen''s Skewed t-Distribution CDF (nu = %d)', nu));
    xlabel('x');
    ylabel('F(x)');
    legend({sprintf('lambda = %.1f (negative skew)', lambda_values(1)), ...
            sprintf('lambda = %.1f (symmetric)', lambda_values(2)), ...
            sprintf('lambda = %.1f (positive skew)', lambda_values(3))}, ...
            'Location', 'best');
    grid on;
    
    % Generate and plot random samples
    selected_nu = 5;
    selected_lambda = 0.3;  % Moderate positive skewness
    n_samples = 5000;
    random_samples = skewtrnd(selected_nu, selected_lambda, n_samples, 1);
    
    figure('Name', 'Skewed t-Distribution - Random Samples');
    subplot(2, 1, 1);
    hist(random_samples, 50);
    title(sprintf('Histogram of %d Random Samples (nu = %d, lambda = %.1f)', ...
                  n_samples, selected_nu, selected_lambda));
    xlabel('Value');
    ylabel('Frequency');
    
    subplot(2, 1, 2);
    % Calculate theoretical PDF for comparison
    x_range = linspace(min(random_samples), max(random_samples), 1000)';
    pdf_theoretical = skewtpdf(x_range, selected_nu, selected_lambda);
    
    % Plot histogram with PDF overlay
    [counts, edges] = histcounts(random_samples, 50, 'Normalization', 'pdf');
    centers = (edges(1:end-1) + edges(2:end)) / 2;
    bar(centers, counts, 'FaceAlpha', 0.3);
    hold on;
    plot(x_range, pdf_theoretical, 'r-', 'LineWidth', 2);
    hold off;
    title('Normalized Histogram with Theoretical PDF Overlay');
    xlabel('Value');
    ylabel('Density');
    legend('Empirical', 'Theoretical');
    
    % Parameter estimation example
    disp('Parameter Estimation Example for Skewed t-Distribution:');
    disp('-----------------------------------');
    
    % Generate data with known parameters
    true_nu = 6;
    true_lambda = 0.3;
    true_mu = 0.2;
    true_sigma = 1.5;
    
    n_samples = 2000;
    
    % Generate samples
    sample_data = skewtrnd(true_nu, true_lambda, n_samples, 1);
    
    % Apply location-scale transformation manually
    sample_data = true_sigma * sample_data + true_mu;
    
    % Set options for parameter estimation
    options.display = 'off';
    options.startingVals = [true_nu, true_lambda, true_mu, true_sigma];  % Good starting values
    
    % Estimate parameters
    tic;
    estimated_params = skewtfit(sample_data, options);
    estimation_time = toc;
    
    % Display results
    disp('True parameters:');
    fprintf('nu = %.4f, lambda = %.4f, mu = %.4f, sigma = %.4f\n', ...
           true_nu, true_lambda, true_mu, true_sigma);
    
    disp('Estimated parameters:');
    fprintf('nu = %.4f (SE: %.4f)\n', estimated_params.nu, estimated_params.nuSE);
    fprintf('lambda = %.4f (SE: %.4f)\n', estimated_params.lambda, estimated_params.lambdaSE);
    fprintf('mu = %.4f (SE: %.4f)\n', estimated_params.mu, estimated_params.muSE);
    fprintf('sigma = %.4f (SE: %.4f)\n', estimated_params.sigma, estimated_params.sigmaSE);
    fprintf('Log-likelihood: %.4f\n', estimated_params.logL);
    fprintf('Estimation time: %.4f seconds\n', estimation_time);
    
    % Q-Q plot to assess goodness of fit
    figure('Name', 'Skewed t-Distribution - Q-Q Plot');
    
    % Sort the sample data
    sorted_data = sort(sample_data);
    
    % Generate theoretical quantiles
    p_values = ((1:n_samples) - 0.5) / n_samples;
    
    % Get theoretical quantiles based on the standardized distribution
    theoretical_quantiles = skewtinv(p_values, estimated_params.nu, estimated_params.lambda);
    
    % Convert back to original scale
    theoretical_quantiles = estimated_params.mu + estimated_params.sigma * theoretical_quantiles;
    
    % Create Q-Q plot
    plot(theoretical_quantiles, sorted_data, 'o');
    hold on;
    
    % Add reference line
    min_val = min(min(theoretical_quantiles), min(sorted_data));
    max_val = max(max(theoretical_quantiles), max(sorted_data));
    plot([min_val, max_val], [min_val, max_val], 'r--');
    hold off;
    
    title('Q-Q Plot for Skewed t-Distribution Fit');
    xlabel('Theoretical Quantiles');
    ylabel('Sample Quantiles');
    grid on;
    
    % Demonstration of log-likelihood sensitivity to parameters
    % Let's vary the skewness parameter and see how log-likelihood changes
    lambda_test = linspace(-0.8, 0.8, 20);
    ll_values = zeros(size(lambda_test));
    
    for i = 1:length(lambda_test)
        parameters = [estimated_params.nu, lambda_test(i), estimated_params.mu, estimated_params.sigma];
        [neg_ll, ~] = skewtloglik(sample_data, parameters);
        ll_values(i) = -neg_ll;  % Convert negative log-likelihood to log-likelihood
    end
    
    figure('Name', 'Skewed t-Distribution - Log-Likelihood');
    plot(lambda_test, ll_values, 'o-');
    hold on;
    plot([true_lambda, true_lambda], [min(ll_values), max(ll_values)], 'r--');
    plot([estimated_params.lambda, estimated_params.lambda], [min(ll_values), max(ll_values)], 'g--');
    hold off;
    
    title('Log-Likelihood vs. Skewness Parameter (lambda)');
    xlabel('Skewness Parameter (lambda)');
    ylabel('Log-Likelihood');
    legend('Log-likelihood', 'True value', 'Estimated value');
    grid on;
end

function demonstrate_stdt_distribution()
    % This function demonstrates the use of standardized Student's t-distribution functions
    
    % Display section header
    disp('--------------------------------------------------------------');
    disp('STANDARDIZED STUDENT''S T-DISTRIBUTION DEMONSTRATION');
    disp('--------------------------------------------------------------');
    
    % Define different degrees of freedom for comparison
    nu_values = [4, 6, 30];
    
    % Create grid of x values for function evaluation
    x = linspace(-5, 5, 1000)';
    
    % Calculate and plot PDF values for different degrees of freedom
    figure('Name', 'Standardized t-Distribution - PDF');
    pdf_values = zeros(length(x), length(nu_values));
    
    for i = 1:length(nu_values)
        pdf_values(:, i) = stdtpdf(x, nu_values(i));
    end
    
    % Add normal distribution for comparison
    normal_pdf = normpdf(x, 0, 1);
    
    plot(x, [pdf_values, normal_pdf]);
    title('Standardized Student''s t-Distribution PDF');
    xlabel('x');
    ylabel('f(x)');
    legend({sprintf('nu = %d (heavy tails)', nu_values(1)), ...
            sprintf('nu = %d (moderate tails)', nu_values(2)), ...
            sprintf('nu = %d (near normal)', nu_values(3)), ...
            'Normal (Gaussian)'}, ...
            'Location', 'best');
    grid on;
    
    % Calculate and plot CDF values for different degrees of freedom
    figure('Name', 'Standardized t-Distribution - CDF');
    cdf_values = zeros(length(x), length(nu_values));
    
    for i = 1:length(nu_values)
        cdf_values(:, i) = stdtcdf(x, nu_values(i));
    end
    
    % Add normal distribution for comparison
    normal_cdf = normcdf(x, 0, 1);
    
    plot(x, [cdf_values, normal_cdf]);
    title('Standardized Student''s t-Distribution CDF');
    xlabel('x');
    ylabel('F(x)');
    legend({sprintf('nu = %d (heavy tails)', nu_values(1)), ...
            sprintf('nu = %d (moderate tails)', nu_values(2)), ...
            sprintf('nu = %d (near normal)', nu_values(3)), ...
            'Normal (Gaussian)'}, ...
            'Location', 'best');
    grid on;
    
    % Generate and plot random samples
    selected_nu = 5;  % Moderate heavy tails
    n_samples = 5000;
    random_samples = stdtrnd([n_samples, 1], selected_nu);
    
    figure('Name', 'Standardized t-Distribution - Random Samples');
    subplot(2, 1, 1);
    hist(random_samples, 50);
    title(sprintf('Histogram of %d Random Samples from Std. t (nu = %d)', n_samples, selected_nu));
    xlabel('Value');
    ylabel('Frequency');
    
    subplot(2, 1, 2);
    % Calculate theoretical PDF for comparison
    x_range = linspace(min(random_samples), max(random_samples), 1000)';
    pdf_theoretical = stdtpdf(x_range, selected_nu);
    
    % Plot histogram with PDF overlay
    [counts, edges] = histcounts(random_samples, 50, 'Normalization', 'pdf');
    centers = (edges(1:end-1) + edges(2:end)) / 2;
    bar(centers, counts, 'FaceAlpha', 0.3);
    hold on;
    plot(x_range, pdf_theoretical, 'r-', 'LineWidth', 2);
    
    % Add normal distribution for comparison
    normal_pdf_comp = normpdf(x_range, 0, 1);
    plot(x_range, normal_pdf_comp, 'g--', 'LineWidth', 1.5);
    
    hold off;
    title('Normalized Histogram with Theoretical PDF Overlay');
    xlabel('Value');
    ylabel('Density');
    legend('Empirical', 'Theoretical t', 'Normal');
    
    % Parameter estimation example
    disp('Parameter Estimation Example for Standardized t-Distribution:');
    disp('-----------------------------------');
    
    % Generate data with known parameters
    true_nu = 5;
    true_mu = 0.5;
    true_sigma = 1.2;
    
    n_samples = 2000;
    
    % Generate samples
    % For standardized t, we need to manually apply the location-scale transformation
    sample_data = stdtrnd([n_samples, 1], true_nu);
    sample_data = true_sigma * sample_data + true_mu;
    
    % Set options for parameter estimation
    options.display = 'off';
    options.startingVal = true_nu;  % Provide good starting value
    
    % Estimate parameters
    tic;
    estimated_params = stdtfit(sample_data, options);
    estimation_time = toc;
    
    % Display results
    disp('True parameters:');
    fprintf('nu = %.4f, mu = %.4f, sigma = %.4f\n', true_nu, true_mu, true_sigma);
    
    disp('Estimated parameters:');
    fprintf('nu = %.4f (SE: %.4f)\n', estimated_params.nu, estimated_params.nuSE);
    fprintf('mu = %.4f (SE: %.4f)\n', estimated_params.mu, estimated_params.muSE);
    fprintf('sigma = %.4f (SE: %.4f)\n', estimated_params.sigma, estimated_params.sigmaSE);
    fprintf('Log-likelihood: %.4f\n', estimated_params.logL);
    fprintf('AIC: %.4f, BIC: %.4f\n', estimated_params.AIC, estimated_params.BIC);
    fprintf('Estimation time: %.4f seconds\n', estimation_time);
    
    % Q-Q plot to assess goodness of fit
    figure('Name', 'Standardized t-Distribution - Q-Q Plot');
    
    % Sort the sample data
    sorted_data = sort(sample_data);
    
    % Generate theoretical quantiles
    p_values = ((1:n_samples) - 0.5) / n_samples;
    
    % For standardized t, we need to manually apply the location-scale transformation
    theoretical_quantiles = stdtinv(p_values, estimated_params.nu);
    theoretical_quantiles = estimated_params.mu + estimated_params.sigma * theoretical_quantiles;
    
    % Create Q-Q plot
    plot(theoretical_quantiles, sorted_data, 'o');
    hold on;
    
    % Add reference line
    min_val = min(min(theoretical_quantiles), min(sorted_data));
    max_val = max(max(theoretical_quantiles), max(sorted_data));
    plot([min_val, max_val], [min_val, max_val], 'r--');
    hold off;
    
    title('Q-Q Plot for Standardized t-Distribution Fit');
    xlabel('Theoretical Quantiles');
    ylabel('Sample Quantiles');
    grid on;
    
    % Demonstration of log-likelihood sensitivity to degrees of freedom
    nu_test = linspace(3, 15, 20);
    ll_values = zeros(size(nu_test));
    
    for i = 1:length(nu_test)
        ll_values(i) = -stdtloglik(sample_data, nu_test(i), estimated_params.mu, estimated_params.sigma);
    end
    
    figure('Name', 'Standardized t-Distribution - Log-Likelihood');
    plot(nu_test, ll_values, 'o-');
    hold on;
    plot([true_nu, true_nu], [min(ll_values), max(ll_values)], 'r--');
    plot([estimated_params.nu, estimated_params.nu], [min(ll_values), max(ll_values)], 'g--');
    hold off;
    
    title('Log-Likelihood vs. Degrees of Freedom (nu)');
    xlabel('Degrees of Freedom (nu)');
    ylabel('Log-Likelihood');
    legend('Log-likelihood', 'True value', 'Estimated value');
    grid on;
end

function compare_distributions()
    % This function creates comparative visualizations of all three distributions
    
    % Display section header
    disp('--------------------------------------------------------------');
    disp('COMPARATIVE ANALYSIS OF DISTRIBUTIONS');
    disp('--------------------------------------------------------------');
    
    % Define sets of parameters for each distribution that highlight different characteristics
    % We'll aim for similar variances to make comparison meaningful
    
    % Parameters for distributions
    ged_nu = 1.5;         % Heavier tails than normal
    skewt_nu = 5;         % Moderate heavy tails
    skewt_lambda = 0.3;   % Moderate positive skewness
    stdt_nu = 5;          % Moderate heavy tails
    
    % Create grid of x values for function evaluation
    x = linspace(-5, 5, 1000)';
    
    % Calculate PDF values for each distribution
    ged_pdf = gedpdf(x, ged_nu);
    skewt_pdf = skewtpdf(x, skewt_nu, skewt_lambda);
    stdt_pdf = stdtpdf(x, stdt_nu);
    normal_pdf = normpdf(x, 0, 1);  % Standard normal for reference
    
    % Calculate CDF values for each distribution
    ged_cdf = gedcdf(x, ged_nu);
    skewt_cdf = skewtcdf(x, skewt_nu, skewt_lambda);
    stdt_cdf = stdtcdf(x, stdt_nu);
    normal_cdf = normcdf(x, 0, 1);  % Standard normal for reference
    
    % Create subplot layout for comparative visualization
    figure('Name', 'Distribution Comparison', 'Position', [100, 100, 1000, 800]);
    
    % Plot 1: PDF comparison
    subplot(2, 2, 1);
    plot(x, [ged_pdf, skewt_pdf, stdt_pdf, normal_pdf]);
    title('Probability Density Functions');
    xlabel('x');
    ylabel('f(x)');
    legend({'GED (nu=1.5)', 'Skewed t (nu=5, lambda=0.3)', ...
            'Standardized t (nu=5)', 'Normal'}, ...
            'Location', 'best');
    grid on;
    
    % Plot 2: CDF comparison
    subplot(2, 2, 2);
    plot(x, [ged_cdf, skewt_cdf, stdt_cdf, normal_cdf]);
    title('Cumulative Distribution Functions');
    xlabel('x');
    ylabel('F(x)');
    legend({'GED (nu=1.5)', 'Skewed t (nu=5, lambda=0.3)', ...
            'Standardized t (nu=5)', 'Normal'}, ...
            'Location', 'best');
    grid on;
    
    % Plot 3: Tail behavior comparison (left tail)
    subplot(2, 2, 3);
    x_tail = linspace(-5, -1, 500)';
    ged_pdf_tail = gedpdf(x_tail, ged_nu);
    skewt_pdf_tail = skewtpdf(x_tail, skewt_nu, skewt_lambda);
    stdt_pdf_tail = stdtpdf(x_tail, stdt_nu);
    normal_pdf_tail = normpdf(x_tail, 0, 1);
    
    plot(x_tail, [ged_pdf_tail, skewt_pdf_tail, stdt_pdf_tail, normal_pdf_tail]);
    title('Left Tail Behavior');
    xlabel('x');
    ylabel('f(x)');
    legend({'GED (nu=1.5)', 'Skewed t (nu=5, lambda=0.3)', ...
            'Standardized t (nu=5)', 'Normal'}, ...
            'Location', 'best');
    grid on;
    
    % Plot 4: Tail behavior comparison (right tail)
    subplot(2, 2, 4);
    x_tail = linspace(1, 5, 500)';
    ged_pdf_tail = gedpdf(x_tail, ged_nu);
    skewt_pdf_tail = skewtpdf(x_tail, skewt_nu, skewt_lambda);
    stdt_pdf_tail = stdtpdf(x_tail, stdt_nu);
    normal_pdf_tail = normpdf(x_tail, 0, 1);
    
    plot(x_tail, [ged_pdf_tail, skewt_pdf_tail, stdt_pdf_tail, normal_pdf_tail]);
    title('Right Tail Behavior');
    xlabel('x');
    ylabel('f(x)');
    legend({'GED (nu=1.5)', 'Skewed t (nu=5, lambda=0.3)', ...
            'Standardized t (nu=5)', 'Normal'}, ...
            'Location', 'best');
    grid on;
    
    % Generate random samples from each distribution
    n_samples = 10000;
    ged_samples = gedrnd(ged_nu, n_samples, 1);
    skewt_samples = skewtrnd(skewt_nu, skewt_lambda, n_samples, 1);
    stdt_samples = stdtrnd([n_samples, 1], stdt_nu);
    normal_samples = randn(n_samples, 1);
    
    % Create figure for random sample comparison
    figure('Name', 'Random Sample Comparison', 'Position', [100, 100, 1000, 600]);
    
    % Create histograms with kernel density estimates
    subplot(2, 2, 1);
    hist(ged_samples, 50);
    title(sprintf('GED (nu = %.1f)', ged_nu));
    xlabel('Value');
    ylabel('Frequency');
    
    subplot(2, 2, 2);
    hist(skewt_samples, 50);
    title(sprintf('Skewed t (nu = %d, lambda = %.1f)', skewt_nu, skewt_lambda));
    xlabel('Value');
    ylabel('Frequency');
    
    subplot(2, 2, 3);
    hist(stdt_samples, 50);
    title(sprintf('Standardized t (nu = %d)', stdt_nu));
    xlabel('Value');
    ylabel('Frequency');
    
    subplot(2, 2, 4);
    hist(normal_samples, 50);
    title('Normal Distribution');
    xlabel('Value');
    ylabel('Frequency');
    
    % Calculate and display descriptive statistics
    disp('Descriptive Statistics for Random Samples:');
    disp('----------------------------------------');
    
    % Function to calculate statistics
    function stats = calculate_stats(data)
        stats.mean = mean(data);
        stats.std = std(data);
        stats.skewness = skewness(data);
        stats.kurtosis = kurtosis(data);
        stats.min = min(data);
        stats.q1 = quantile(data, 0.25);
        stats.median = median(data);
        stats.q3 = quantile(data, 0.75);
        stats.max = max(data);
    end
    
    % Calculate statistics for each distribution
    ged_stats = calculate_stats(ged_samples);
    skewt_stats = calculate_stats(skewt_samples);
    stdt_stats = calculate_stats(stdt_samples);
    normal_stats = calculate_stats(normal_samples);
    
    % Display statistics
    fprintf('GED (nu = %.1f):\n', ged_nu);
    fprintf('  Mean: %.4f, Std Dev: %.4f\n', ged_stats.mean, ged_stats.std);
    fprintf('  Skewness: %.4f, Kurtosis: %.4f\n', ged_stats.skewness, ged_stats.kurtosis);
    fprintf('  Min: %.4f, Q1: %.4f, Median: %.4f, Q3: %.4f, Max: %.4f\n', ...
            ged_stats.min, ged_stats.q1, ged_stats.median, ged_stats.q3, ged_stats.max);
    
    fprintf('\nSkewed t (nu = %d, lambda = %.1f):\n', skewt_nu, skewt_lambda);
    fprintf('  Mean: %.4f, Std Dev: %.4f\n', skewt_stats.mean, skewt_stats.std);
    fprintf('  Skewness: %.4f, Kurtosis: %.4f\n', skewt_stats.skewness, skewt_stats.kurtosis);
    fprintf('  Min: %.4f, Q1: %.4f, Median: %.4f, Q3: %.4f, Max: %.4f\n', ...
            skewt_stats.min, skewt_stats.q1, skewt_stats.median, skewt_stats.q3, skewt_stats.max);
    
    fprintf('\nStandardized t (nu = %d):\n', stdt_nu);
    fprintf('  Mean: %.4f, Std Dev: %.4f\n', stdt_stats.mean, stdt_stats.std);
    fprintf('  Skewness: %.4f, Kurtosis: %.4f\n', stdt_stats.skewness, stdt_stats.kurtosis);
    fprintf('  Min: %.4f, Q1: %.4f, Median: %.4f, Q3: %.4f, Max: %.4f\n', ...
            stdt_stats.min, stdt_stats.q1, stdt_stats.median, stdt_stats.q3, stdt_stats.max);
    
    fprintf('\nNormal Distribution:\n');
    fprintf('  Mean: %.4f, Std Dev: %.4f\n', normal_stats.mean, normal_stats.std);
    fprintf('  Skewness: %.4f, Kurtosis: %.4f\n', normal_stats.skewness, normal_stats.kurtosis);
    fprintf('  Min: %.4f, Q1: %.4f, Median: %.4f, Q3: %.4f, Max: %.4f\n', ...
            normal_stats.min, normal_stats.q1, normal_stats.median, normal_stats.q3, normal_stats.max);
    
    % Display key behavioral differences
    disp('Key Behavioral Differences:');
    disp('-------------------------');
    disp('1. GED: Provides flexibility in modeling tail thickness. With nu < 2, has heavier');
    disp('   tails than normal; with nu = 1 corresponds to Laplace distribution; with nu = 2');
    disp('   is equivalent to normal; with nu > 2 has thinner tails than normal.');
    disp('');
    disp('2. Skewed t: Allows modeling of both heavy tails and asymmetry. The lambda parameter');
    disp('   controls skewness: negative values create negative skew, positive values create');
    disp('   positive skew. When lambda = 0, it reduces to standardized t-distribution.');
    disp('');
    disp('3. Standardized t: Models symmetric heavy tails. The nu parameter controls tail');
    disp('   thickness, with smaller values giving heavier tails. As nu increases, it');
    disp('   approaches the normal distribution.');
    disp('');
    disp('4. When analyzing financial returns:');
    disp('   - GED is useful when returns show symmetric fat tails');
    disp('   - Skewed t is best when returns show both asymmetry and fat tails');
    disp('   - Standardized t is appropriate for symmetric fat-tailed returns');
    disp('   - Normal distribution generally underestimates tail risk in financial data');
end

function financial_data_example()
    % This function demonstrates fitting distributions to real financial return data
    
    % Display section header
    disp('--------------------------------------------------------------');
    disp('FINANCIAL DATA DISTRIBUTION FITTING EXAMPLE');
    disp('--------------------------------------------------------------');
    
    % Load financial returns data
    load('returns');
    
    % Select a single asset for analysis (first column)
    data = returns(:, 1);
    
    % Display basic statistics
    fprintf('Financial Return Data Summary:\n');
    fprintf('Number of observations: %d\n', length(data));
    fprintf('Mean: %.6f\n', mean(data));
    fprintf('Standard Deviation: %.6f\n', std(data));
    fprintf('Skewness: %.6f\n', skewness(data));
    fprintf('Excess Kurtosis: %.6f\n', kurtosis(data) - 3);
    fprintf('Min: %.6f, Max: %.6f\n', min(data), max(data));
    
    % Plot histogram of returns
    figure('Name', 'Financial Returns', 'Position', [100, 100, 800, 400]);
    subplot(1, 2, 1);
    hist(data, 50);
    title('Histogram of Financial Returns');
    xlabel('Return');
    ylabel('Frequency');
    
    subplot(1, 2, 2);
    qqplot(data);
    title('Normal Q-Q Plot of Financial Returns');
    
    % Fit distributions
    disp('Fitting Distributions to Financial Returns...');
    
    % Common options
    options.display = 'off';
    
    % 1. Fit GED
    tic;
    ged_params = gedfit(data, options);
    ged_time = toc;
    
    % 2. Fit skewed t
    tic;
    options.startingVals = [5, 0, mean(data), std(data)];
    skewt_params = skewtfit(data, options);
    skewt_time = toc;
    
    % 3. Fit standardized t
    tic;
    stdt_params = stdtfit(data, options);
    stdt_time = toc;
    
    % Display parameter estimates
    disp('Parameter Estimates:');
    disp('------------------');
    
    fprintf('GED (fitting time: %.4f sec):\n', ged_time);
    fprintf('  nu = %.4f (SE: %.4f)\n', ged_params.nu, ged_params.stderrors(1));
    fprintf('  mu = %.6f (SE: %.6f)\n', ged_params.mu, ged_params.stderrors(2));
    fprintf('  sigma = %.6f (SE: %.6f)\n', ged_params.sigma, ged_params.stderrors(3));
    fprintf('  Log-likelihood: %.2f\n', ged_params.loglik);
    
    fprintf('\nSkewed t (fitting time: %.4f sec):\n', skewt_time);
    fprintf('  nu = %.4f (SE: %.4f)\n', skewt_params.nu, skewt_params.nuSE);
    fprintf('  lambda = %.4f (SE: %.4f)\n', skewt_params.lambda, skewt_params.lambdaSE);
    fprintf('  mu = %.6f (SE: %.6f)\n', skewt_params.mu, skewt_params.muSE);
    fprintf('  sigma = %.6f (SE: %.6f)\n', skewt_params.sigma, skewt_params.sigmaSE);
    fprintf('  Log-likelihood: %.2f\n', skewt_params.logL);
    
    fprintf('\nStandardized t (fitting time: %.4f sec):\n', stdt_time);
    fprintf('  nu = %.4f (SE: %.4f)\n', stdt_params.nu, stdt_params.nuSE);
    fprintf('  mu = %.6f (SE: %.6f)\n', stdt_params.mu, stdt_params.muSE);
    fprintf('  sigma = %.6f (SE: %.6f)\n', stdt_params.sigma, stdt_params.sigmaSE);
    fprintf('  Log-likelihood: %.2f\n', stdt_params.logL);
    fprintf('  AIC: %.2f, BIC: %.2f\n', stdt_params.AIC, stdt_params.BIC);
    
    % Compare log-likelihoods and information criteria
    ged_AIC = -2 * ged_params.loglik + 2 * 3;  % 3 parameters
    ged_BIC = -2 * ged_params.loglik + log(length(data)) * 3;
    
    skewt_AIC = -2 * skewt_params.logL + 2 * 4;  % 4 parameters
    skewt_BIC = -2 * skewt_params.logL + log(length(data)) * 4;
    
    disp('Model Selection Criteria:');
    disp('---------------------');
    fprintf('GED:          AIC = %.2f, BIC = %.2f\n', ged_AIC, ged_BIC);
    fprintf('Skewed t:     AIC = %.2f, BIC = %.2f\n', skewt_AIC, skewt_BIC);
    fprintf('Standardized t: AIC = %.2f, BIC = %.2f\n', stdt_params.AIC, stdt_params.BIC);
    
    % Create x grid for PDF plotting
    x_grid = linspace(min(data) - 0.01, max(data) + 0.01, 1000)';
    
    % Calculate PDF values for each fitted distribution
    ged_pdf = gedpdf((x_grid - ged_params.mu) / ged_params.sigma, ged_params.nu) / ged_params.sigma;
    
    skewt_std = (x_grid - skewt_params.mu) / skewt_params.sigma;
    skewt_pdf = skewtpdf(skewt_std, skewt_params.nu, skewt_params.lambda) / skewt_params.sigma;
    
    stdt_std = (x_grid - stdt_params.mu) / stdt_params.sigma;
    stdt_pdf = stdtpdf(stdt_std, stdt_params.nu) / stdt_params.sigma;
    
    % Calculate normal PDF for comparison
    normal_pdf = normpdf(x_grid, mean(data), std(data));
    
    % Plot returns histogram with fitted PDFs
    figure('Name', 'Fitted Distributions', 'Position', [100, 100, 800, 600]);
    
    % Get histogram data
    [counts, edges] = histcounts(data, 50, 'Normalization', 'pdf');
    centers = (edges(1:end-1) + edges(2:end)) / 2;
    
    % Plot histogram
    bar(centers, counts, 'FaceAlpha', 0.3);
    hold on;
    
    % Plot fitted distributions
    plot(x_grid, ged_pdf, 'r-', 'LineWidth', 2);
    plot(x_grid, skewt_pdf, 'g-', 'LineWidth', 2);
    plot(x_grid, stdt_pdf, 'b-', 'LineWidth', 2);
    plot(x_grid, normal_pdf, 'k--', 'LineWidth', 1.5);
    
    hold off;
    title('Financial Returns with Fitted Distributions');
    xlabel('Return');
    ylabel('Density');
    legend({'Empirical', 'GED', 'Skewed t', 'Standardized t', 'Normal'}, ...
           'Location', 'best');
    grid on;
    
    % Calculate VaR (Value-at-Risk) at different confidence levels
    confidence_levels = [0.01, 0.05];
    var_labels = {'1% VaR', '5% VaR'};
    
    disp('Value-at-Risk Estimates:');
    for i = 1:length(confidence_levels)
        alpha = confidence_levels(i);
        
        % Calculate VaR for each distribution
        ged_var = -(ged_params.mu + ged_params.sigma * gedinv(alpha, ged_params.nu));
        skewt_var = -(skewt_params.mu + skewt_params.sigma * ...
                     skewtinv(alpha, skewt_params.nu, skewt_params.lambda));
        stdt_var = -(stdt_params.mu + stdt_params.sigma * stdtinv(alpha, stdt_params.nu));
        normal_var = -(mean(data) + std(data) * norminv(alpha));
        
        % Display results
        fprintf('%s:\n', var_labels{i});
        fprintf('  GED:           %.6f\n', ged_var);
        fprintf('  Skewed t:      %.6f\n', skewt_var);
        fprintf('  Standardized t: %.6f\n', stdt_var);
        fprintf('  Normal:        %.6f\n', normal_var);
    end
end

% Utility functions for statistics calculation
function s = skewness(x)
    % Calculate skewness of a sample
    x = x - mean(x);
    n = length(x);
    s = sum(x.^3)/(n*std(x)^3);
end

function k = kurtosis(x)
    % Calculate kurtosis of a sample
    x = x - mean(x);
    n = length(x);
    k = sum(x.^4)/(n*std(x)^4);
end