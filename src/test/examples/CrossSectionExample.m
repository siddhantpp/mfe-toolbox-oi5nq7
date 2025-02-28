function run_cross_section_example()
% RUN_CROSS_SECTION_EXAMPLE Main example function that demonstrates the cross-sectional analysis 
% capabilities of the MFE Toolbox including data preprocessing, statistical analysis, and 
% regression modeling.
%
% This comprehensive example illustrates how to use the MFE Toolbox for cross-sectional 
% financial data analysis, covering data preprocessing, statistical testing, regression 
% analysis, and portfolio construction.
%
% The example demonstrates practical applications for empirical asset pricing research,
% factor model estimation, portfolio analysis, and cross-sectional statistical inference.
%
% USAGE:
%   run_cross_section_example()
%
% OUTPUTS:
%   This function produces both console output and visualization figures
%   demonstrating key aspects of cross-sectional analysis.
%
% See also: analyze_cross_section, filter_cross_section, cross_section_regression

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

%% Introduction
disp('==================================================================');
disp('               CROSS-SECTIONAL ANALYSIS EXAMPLE                   ');
disp('==================================================================');
disp(' ');
disp('This example demonstrates comprehensive cross-sectional analysis');
disp('capabilities of the MFE Toolbox, including:');
disp(' ');
disp('1. Data preprocessing and filtering');
disp('2. Descriptive statistics and correlation analysis');
disp('3. Cross-sectional regression analysis');
disp('4. Portfolio analysis based on characteristics');
disp('5. Robust diagnostics and visualization');
disp(' ');
disp('==================================================================');
disp(' ');

%% Load example data
% In a real application, you would load your own data
% Here we'll load example data provided with the toolbox
try
    load('example_cross_sectional_data.mat');
    disp('Loaded example cross-sectional data.');
catch
    error('Example data file not found. Please ensure example_cross_sectional_data.mat is in your MATLAB path.');
end

% The example data contains:
% - returns: T×N matrix of asset returns
% - characteristics: T×K matrix of asset characteristics
% - factors: T×F matrix of risk factors
% - dates: T×1 vector of dates

% Print basic information about the dataset
[T, N] = size(returns);
[~, K] = size(characteristics);
[~, F] = size(factors);

disp(['Dataset contains ' num2str(T) ' time periods and ' num2str(N) ' assets.']);
disp(['Number of characteristics: ' num2str(K)]);
disp(['Number of risk factors: ' num2str(F)]);
disp(' ');

%% Basic data exploration
disp('==================================================================');
disp('                     BASIC DATA EXPLORATION                       ');
disp('==================================================================');
disp(' ');

% Calculate simple statistics for returns
mean_returns = mean(returns);
std_returns = std(returns);

disp('Return Statistics:');
disp(['  Mean return range: [' num2str(min(mean_returns)) ', ' num2str(max(mean_returns)) ']']);
disp(['  Std dev range: [' num2str(min(std_returns)) ', ' num2str(max(std_returns)) ']']);
disp(' ');

% Display characteristic names (assuming they exist)
if exist('characteristic_names', 'var')
    disp('Available Characteristics:');
    for i = 1:length(characteristic_names)
        disp(['  ' num2str(i) '. ' characteristic_names{i}]);
    end
else
    characteristic_names = cell(K, 1);
    for i = 1:K
        characteristic_names{i} = ['Characteristic ' num2str(i)];
    end
    disp('Characteristic names not provided. Using generic names.');
end
disp(' ');

%% Data Preprocessing
disp('==================================================================');
disp('                     DATA PREPROCESSING                           ');
disp('==================================================================');
disp(' ');

disp('Preprocessing data to handle missing values and outliers...');

% Create options for filtering
filter_options = struct();
filter_options.missing_handling = 'remove';  % Remove observations with missing values
filter_options.outlier_detection = 'iqr';    % Detect outliers using IQR method
filter_options.outlier_handling = 'winsorize'; % Winsorize outliers
filter_options.winsor_percentiles = [0.01 0.99]; % Winsorize at 1% and 99%

% Apply filtering to returns
returns_filtered = filter_cross_section(returns, filter_options);

disp('Filtering summary:');
disp(['  Original observations: ' num2str(size(returns, 1))]);
disp(['  Filtered observations: ' num2str(size(returns_filtered.data, 1))]);
disp(['  Missing values detected: ' num2str(returns_filtered.missing.total_missing)]);
disp(['  Outliers detected: ' num2str(returns_filtered.outliers.total_outliers) ...
      ' (' num2str(returns_filtered.outliers.outlier_percent) '%)']);
disp(' ');

% Update returns with filtered data
returns_clean = returns_filtered.data;

%% Comprehensive Cross-Sectional Analysis
disp('==================================================================');
disp('                COMPREHENSIVE CROSS-SECTIONAL ANALYSIS            ');
disp('==================================================================');
disp(' ');

% Set up analysis options
analysis_options = struct();
analysis_options.descriptive = true;
analysis_options.correlations = true;
analysis_options.distribution_tests = true;

% Perform analysis on returns
disp('Analyzing cross-sectional return characteristics...');
results = analyze_cross_section(returns_clean, analysis_options);

% Display key results
display_analysis_results(results);

%% Cross-Sectional Regression Analysis
disp('==================================================================');
disp('                CROSS-SECTIONAL REGRESSION ANALYSIS               ');
disp('==================================================================');
disp(' ');

% We'll demonstrate Fama-MacBeth style cross-sectional regressions
% In each period, we regress returns on characteristics

% Initialize storage for time-series of coefficient estimates
coef_ts = zeros(T, K+1);  % +1 for intercept
tstat_ts = zeros(T, K+1);
r2_ts = zeros(T, 1);

% Loop through each time period
disp('Running cross-sectional regressions for each time period...');

for t = 1:T
    % Get data for this period
    y = returns_clean(t, :)';  % Cross-section of returns
    X = characteristics(t, :)'; % Cross-section of characteristics
    
    % Skip periods with too few observations
    if length(y) < K+5
        continue;
    end
    
    % Set regression options
    reg_options = struct();
    reg_options.method = 'ols';
    reg_options.se_type = 'robust';  % Use heteroskedasticity-robust standard errors
    
    % Run regression
    reg_results = cross_section_regression(y, X, reg_options);
    
    % Store results
    coef_ts(t, :) = reg_results.beta';
    tstat_ts(t, :) = reg_results.tstat';
    r2_ts(t) = reg_results.r2;
end

% Calculate time-series averages (Fama-MacBeth coefficients)
avg_coef = mean(coef_ts, 'omitnan');
avg_tstat = mean(tstat_ts, 'omitnan') / sqrt(sum(~isnan(coef_ts(:,1))));
avg_r2 = mean(r2_ts, 'omitnan');

% Display Fama-MacBeth results
disp('Fama-MacBeth Regression Results:');
disp('--------------------------------');
disp('Variable      Coefficient    t-statistic');
disp('--------------------------------');
disp(['Intercept    ' num2str(avg_coef(1), '%10.4f') '    ' num2str(avg_tstat(1), '%10.2f')]);

for k = 1:K
    var_name = characteristic_names{k};
    if length(var_name) > 10
        var_name = [var_name(1:7) '...'];
    end
    disp([var_name '    ' num2str(avg_coef(k+1), '%10.4f') '    ' num2str(avg_tstat(k+1), '%10.2f')]);
end

disp('--------------------------------');
disp(['Average R-squared: ' num2str(avg_r2, '%6.4f')]);
disp(' ');

%% Visualizing Cross-Sectional Relationships
disp('==================================================================');
disp('               VISUALIZING CROSS-SECTIONAL RELATIONSHIPS          ');
disp('==================================================================');
disp(' ');

% Create a figure for visualization
figure;

% Select a characteristic to visualize (e.g., first characteristic)
char_idx = 1;
char_name = characteristic_names{char_idx};

% Calculate time-series average of characteristics and returns
avg_char = mean(characteristics(:,:), 'omitnan');
avg_ret = mean(returns, 'omitnan');

% Plot scatter of characteristic vs. returns
subplot(2,2,1);
scatter(avg_char, avg_ret, 50, 'filled');
xlabel(['Average ' char_name]);
ylabel('Average Return');
title('Returns vs. Characteristic');
grid on;

% Add a simple trend line
hold on;
coef = polyfit(avg_char, avg_ret, 1);
x_range = linspace(min(avg_char), max(avg_char), 100);
y_fit = polyval(coef, x_range);
plot(x_range, y_fit, 'r-', 'LineWidth', 2);
hold off;

% Plot histogram of a characteristic
subplot(2,2,2);
histogram(avg_char, 20);
xlabel(char_name);
title('Distribution of Characteristic');
grid on;

% Plot regression coefficients over time
subplot(2,2,3);
plot(coef_ts(:,char_idx+1), 'b-', 'LineWidth', 1.5);
hold on;
plot(zeros(T,1), 'r--');
hold off;
xlabel('Time Period');
ylabel('Coefficient');
title(['Time-Series of ' char_name ' Coefficient']);
grid on;

% Plot distribution of t-statistics
subplot(2,2,4);
histogram(tstat_ts(:,char_idx+1), 15);
xlabel('t-statistic');
title(['Distribution of ' char_name ' t-statistics']);
grid on;

%% Portfolio Analysis Based on Characteristics
disp('==================================================================');
disp('               PORTFOLIO ANALYSIS BASED ON CHARACTERISTICS        ');
disp('==================================================================');
disp(' ');

% We'll form portfolios based on characteristics and analyze their performance
disp('Forming portfolios based on characteristics...');

% Use the demonstrate_portfolio_analysis function to perform the analysis
portfolio_results = demonstrate_portfolio_analysis(returns_clean, characteristics, characteristic_names);

% Display key portfolio results
disp('Portfolio Performance Summary:');
disp('--------------------------------');
for i = 1:length(portfolio_results.characteristics)
    char_name = portfolio_results.characteristics{i};
    ls_return = portfolio_results.long_short_returns(i);
    ls_tstat = portfolio_results.long_short_tstats(i);
    
    disp([char_name ' L-S portfolio: Return = ' num2str(ls_return*100, '%6.2f') ...
          '%, t-stat = ' num2str(ls_tstat, '%5.2f')]);
end
disp(' ');

%% Final Summary
disp('==================================================================');
disp('                           SUMMARY                                ');
disp('==================================================================');
disp(' ');
disp('This example demonstrated key cross-sectional analysis capabilities:');
disp(' ');
disp('1. Data preprocessing to handle missing values and outliers');
disp('2. Descriptive statistics and distribution analysis');
disp('3. Cross-sectional regression using robust methods');
disp('4. Visualization of financial relationships');
disp('5. Portfolio formation based on characteristics');
disp(' ');
disp('For more advanced analysis, explore the MFE Toolbox documentation.');
disp('==================================================================');

end

function display_analysis_results(results)
% DISPLAY_ANALYSIS_RESULTS Helper function to format and display cross-sectional analysis results in a readable format.
%
% INPUTS:
%   results - Structure containing analysis results from analyze_cross_section
%
% OUTPUTS:
%   Formatted results displayed to console

% Display descriptive statistics summary
if isfield(results, 'descriptive')
    disp('Descriptive Statistics Summary:');
    disp('--------------------------------');
    
    % Display key statistics
    stats = results.descriptive;
    disp(['Mean (min/max): ' num2str(min(stats.mean), '%8.4f') ' / ' num2str(max(stats.mean), '%8.4f')]);
    disp(['Std Dev (min/max): ' num2str(min(stats.std), '%8.4f') ' / ' num2str(max(stats.std), '%8.4f')]);
    disp(['Skewness (min/max): ' num2str(min(stats.skewness), '%8.4f') ' / ' num2str(max(stats.skewness), '%8.4f')]);
    disp(['Kurtosis (min/max): ' num2str(min(stats.kurtosis), '%8.4f') ' / ' num2str(max(stats.kurtosis), '%8.4f')]);
    disp(' ');
end

% Display correlation analysis
if isfield(results, 'correlations')
    disp('Correlation Analysis:');
    disp('--------------------------------');
    
    % Get correlation matrix and find key patterns
    corr_matrix = results.correlations.pearson;
    corr_matrix(logical(eye(size(corr_matrix)))) = NaN; % Remove diagonal
    
    % Find strongest correlations
    [max_corr, max_idx] = max(abs(corr_matrix(:)));
    [row, col] = ind2sub(size(corr_matrix), max_idx);
    
    disp(['Strongest correlation: ' num2str(corr_matrix(row, col), '%6.4f') ...
          ' between variables ' num2str(row) ' and ' num2str(col)]);
    
    % Average absolute correlation
    avg_abs_corr = mean(abs(corr_matrix(:)), 'omitnan');
    disp(['Average absolute correlation: ' num2str(avg_abs_corr, '%6.4f')]);
    disp(' ');
end

% Display distribution test results
if isfield(results, 'distributions') && isfield(results.distributions, 'normality')
    disp('Normality Test Results:');
    disp('--------------------------------');
    
    % Count variables that reject normality
    normality = results.distributions.normality;
    reject_count = sum(~normality.is_normal);
    total_count = length(normality.is_normal);
    
    disp([num2str(reject_count) ' out of ' num2str(total_count) ...
          ' variables reject normality (' num2str(100*reject_count/total_count, '%5.1f') '%)']);
    disp(' ');
end
end

function plot_regression_diagnostics(results)
% PLOT_REGRESSION_DIAGNOSTICS Creates diagnostic plots for cross-sectional regression analysis.
%
% INPUTS:
%   results - Structure containing regression results from cross_section_regression
%
% OUTPUTS:
%   Diagnostic plots displayed in a figure window

% Create figure with 2x2 layout
figure;

% Plot 1: Residuals vs. fitted values
subplot(2,2,1);
scatter(results.fitted, results.residuals, 'filled');
hold on;
plot([min(results.fitted), max(results.fitted)], [0, 0], 'r--');
hold off;
xlabel('Fitted Values');
ylabel('Residuals');
title('Residuals vs. Fitted');
grid on;

% Plot 2: QQ plot of residuals
subplot(2,2,2);
std_resid = results.residuals / std(results.residuals);
n = length(std_resid);
p = ((1:n)' - 0.5) / n;
theoretical_q = norminv(p);
[sorted_resid, ~] = sort(std_resid);

plot(theoretical_q, sorted_resid, 'bo');
hold on;
plot(theoretical_q, theoretical_q, 'r-');
hold off;
xlabel('Theoretical Quantiles');
ylabel('Sample Quantiles');
title('Q-Q Plot of Residuals');
grid on;

% Plot 3: Histogram of residuals
subplot(2,2,3);
histogram(results.residuals, 20, 'Normalization', 'pdf');
hold on;
x = linspace(min(results.residuals), max(results.residuals), 100);
y = normpdf(x, 0, std(results.residuals));
plot(x, y, 'r-', 'LineWidth', 1.5);
hold off;
xlabel('Residuals');
ylabel('Density');
title('Histogram of Residuals');
grid on;

% Plot 4: Leverage vs. Standardized Residuals
subplot(2,2,4);
leverage = results.diagnostics.influence.leverage;
std_resid = results.diagnostics.influence.std_residuals;
scatter(leverage, std_resid, 'filled');
hold on;
plot([0, max(leverage)*1.1], [0, 0], 'r--');
plot([0, max(leverage)*1.1], [2, 2], 'r:');
plot([0, max(leverage)*1.1], [-2, -2], 'r:');
hold off;
xlabel('Leverage');
ylabel('Standardized Residuals');
title('Influence Diagnostics');
grid on;
end

function portfolio_results = demonstrate_portfolio_analysis(returns, characteristics, characteristic_names)
% DEMONSTRATE_PORTFOLIO_ANALYSIS Demonstrates portfolio formation and analysis based on cross-sectional characteristics.
%
% INPUTS:
%   returns            - T by N matrix of asset returns
%   characteristics    - T by K matrix of asset characteristics
%   characteristic_names - Cell array of characteristic names
%
% OUTPUTS:
%   portfolio_results  - Structure containing portfolio analysis results

% Get dimensions
[T, N] = size(returns);
[~, K] = size(characteristics);

% Initialize storage for results
portfolio_returns = zeros(T, K, 5);  % 5 portfolios per characteristic
long_short_returns = zeros(T, K);    % Long-short portfolio returns
avg_returns = zeros(K, 5);           % Average returns by portfolio
long_short_avg = zeros(K, 1);        % Average long-short returns
long_short_tstat = zeros(K, 1);      % t-statistics for long-short returns

% Loop through each characteristic
for k = 1:K
    % Get this characteristic
    char_k = characteristics(:, k);
    
    % Form portfolios for each time period
    for t = 1:T
        % Get data for this time period
        char_t = char_k(t, :)';
        ret_t = returns(t, :)';
        
        % Skip periods with insufficient data
        if sum(~isnan(char_t) & ~isnan(ret_t)) < 10
            continue;
        end
        
        % Get valid data
        valid_idx = ~isnan(char_t) & ~isnan(ret_t);
        valid_char = char_t(valid_idx);
        valid_ret = ret_t(valid_idx);
        
        % Sort by characteristic and divide into quintiles
        [~, sort_idx] = sort(valid_char);
        n_assets = length(sort_idx);
        quintile_size = floor(n_assets / 5);
        
        % Calculate portfolio returns
        for q = 1:5
            if q < 5
                portfolio_idx = sort_idx((q-1)*quintile_size+1:q*quintile_size);
            else
                portfolio_idx = sort_idx((q-1)*quintile_size+1:end);
            end
            
            % Equal-weighted portfolio return
            portfolio_returns(t, k, q) = mean(valid_ret(portfolio_idx));
        end
        
        % Calculate long-short portfolio return (high minus low)
        long_short_returns(t, k) = portfolio_returns(t, k, 5) - portfolio_returns(t, k, 1);
    end
    
    % Calculate average returns for each portfolio
    for q = 1:5
        avg_returns(k, q) = mean(portfolio_returns(:, k, q), 'omitnan');
    end
    
    % Calculate average long-short return and t-statistic
    ls_returns = long_short_returns(:, k);
    ls_returns = ls_returns(~isnan(ls_returns));
    long_short_avg(k) = mean(ls_returns);
    long_short_tstat(k) = mean(ls_returns) / (std(ls_returns) / sqrt(length(ls_returns)));
end

% Compile results
portfolio_results = struct();
portfolio_results.portfolio_returns = portfolio_returns;
portfolio_results.long_short_returns = long_short_avg;
portfolio_results.long_short_tstats = long_short_tstat;
portfolio_results.average_returns = avg_returns;
portfolio_results.characteristics = characteristic_names;

% Plot performance of portfolios for the first characteristic
figure;

% Plot average returns by quintile
subplot(2,1,1);
bar(avg_returns(1,:));
xlabel('Portfolio (1=Low, 5=High)');
ylabel('Average Return');
title(['Portfolios Sorted by ' characteristic_names{1}]);
grid on;

% Plot cumulative returns over time
subplot(2,1,2);
cum_returns = cumsum(long_short_returns(:,1:min(3,K)), 'omitnan');
plot(cum_returns, 'LineWidth', 1.5);
xlabel('Time Period');
ylabel('Cumulative Return');
title('Cumulative Long-Short Portfolio Returns');
legend(characteristic_names(1:min(3,K)), 'Location', 'Best');
grid on;
end