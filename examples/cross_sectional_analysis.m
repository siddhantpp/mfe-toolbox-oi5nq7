% Cross-sectional Analysis Example
% MFE Toolbox Version 4.0
%
% This example demonstrates cross-sectional analysis techniques using the MFE Toolbox,
% including data filtering, regression analysis, and comprehensive statistical testing.
%
% The example uses sample financial data with asset returns and characteristics
% to illustrate various econometric analysis methods.

% Clear workspace and add toolbox path if needed
clear;
clc;
disp('MATLAB Financial Econometrics Toolbox');
disp('Cross-sectional Analysis Example');
disp('---------------------------------');

% Load example cross-sectional financial data
disp('Loading example cross-sectional data...');
load('examples/data/example_cross_sectional_data.mat');

% Display basic information about the dataset
fprintf('Loaded dataset with %d assets and %d time periods\n', size(asset_returns, 2), size(asset_returns, 1));
fprintf('Asset characteristics: %d variables\n', size(asset_characteristics, 2));
fprintf('Factor loadings: %d factors\n', size(factor_loadings, 2));

% Basic data exploration
disp('\nBasic Data Exploration:');
disp('---------------------------------');

% Compute descriptive statistics for asset characteristics
descriptive_stats = compute_descriptive_statistics(asset_characteristics);

% Display summary statistics for first few characteristics
disp('Summary statistics for selected characteristics:');
disp(descriptive_stats.summary(1:min(5,end),:));

% Create histogram of a key characteristic
figure;
histogram(asset_characteristics(:,1));
title(['Distribution of ' characteristic_names{1}]);
xlabel('Value');
ylabel('Frequency');

% Analyze correlations between characteristics
corr_results = analyze_correlations(asset_characteristics);
disp('Correlation matrix between characteristics:');
disp(corr_results.pearson(1:min(5,end),1:min(5,end)));

% Data filtering demonstration
disp('\nData Filtering:');
disp('---------------------------------');

% Configure filtering options
filter_options = struct('method', 'winsorize', ...
                       'percentile', [0.01, 0.99], ...
                       'transform', 'none');

% Apply filtering to asset characteristics
disp('Applying filters to handle outliers...');
filtered_characteristics = filter_cross_section(asset_characteristics, filter_options);

% Compare original and filtered data statistics
original_stats = compute_descriptive_statistics(asset_characteristics);
filtered_stats = compute_descriptive_statistics(filtered_characteristics);

% Display comparison of key statistics before and after filtering
disp('Before filtering (min, max, mean, std):');
disp([original_stats.min(1:3); original_stats.max(1:3); original_stats.mean(1:3); original_stats.std(1:3)]');
disp('After filtering (min, max, mean, std):');
disp([filtered_stats.min(1:3); filtered_stats.max(1:3); filtered_stats.mean(1:3); filtered_stats.std(1:3)]');

% Prepare filtered data structure for further analysis
filtered_data = struct('returns', asset_returns, ...
                       'characteristics', filtered_characteristics, ...
                       'factor_loadings', factor_loadings, ...
                       'asset_names', {asset_names}, ...
                       'characteristic_names', {characteristic_names});

% Cross-sectional regression analysis
disp('\nRegression Analysis:');
disp('---------------------------------');

% Set up regression variables
y = asset_returns(:,1); % Using first asset's returns as dependent variable
X = filtered_characteristics(:,1:3); % Using first three characteristics as predictors

% Configure regression options
regression_options = struct('method', 'OLS', ...
                           'intercept', true, ...
                           'se_type', 'robust');

% Perform cross-sectional regression
disp('Performing cross-sectional regression...');
regression_results = cross_section_regression(y, X, regression_options);

% Display regression results
disp('Regression coefficient estimates:');
disp(regression_results.beta);
disp('t-statistics:');
disp(regression_results.tstat);
disp('p-values:');
disp(regression_results.pvalues);
disp(['R-squared: ', num2str(regression_results.R2)]);

% Test for heteroskedasticity using White's test
disp('\nPerforming White test for heteroskedasticity...');
white_results = white_test(regression_results.residuals, X);
disp(['White test statistic: ', num2str(white_results.statistic)]);
disp(['p-value: ', num2str(white_results.pValue)]);
if white_results.pValue < 0.05
    disp('Result: Reject null hypothesis of homoskedasticity at 5% level');
else
    disp('Result: Fail to reject null hypothesis of homoskedasticity at 5% level');
end

% Plot regression diagnostics
plot_regression_diagnostics(regression_results);

% Comprehensive cross-sectional analysis
disp('\nComprehensive Analysis:');
disp('---------------------------------');

% Configure analysis options
analysis_options = struct('regression', true, ...
                         'distributional_tests', true, ...
                         'correlations', true, ...
                         'portfolio_statistics', true);

% Perform comprehensive cross-sectional analysis
disp('Performing comprehensive cross-sectional analysis...');
analysis_results = analyze_cross_section(filtered_data.characteristics, analysis_options);

% Display selected analysis results
disp('Distributional test results:');
disp(analysis_results.distributional_tests);

disp('\nHighest correlation pairs:');
disp(analysis_results.correlations.highest_pairs(1:3,:));

% Calculate portfolio statistics using equal weights
n_assets = size(asset_returns, 2);
equal_weights = ones(n_assets, 1) / n_assets;
portfolio_stats = analyze_portfolio_statistics(asset_returns, equal_weights);

disp('\nPortfolio statistics with equal weights:');
disp(['Expected return: ', num2str(portfolio_stats.expected_return)]);
disp(['Portfolio volatility: ', num2str(portfolio_stats.volatility)]);
disp(['Sharpe ratio: ', num2str(portfolio_stats.sharpe_ratio)]);

% Create visualization of key analysis results
figure;
bar(analysis_results.regression.beta);
title('Regression Coefficients');
xlabel('Characteristic');
ylabel('Coefficient');
xticks(1:length(analysis_results.regression.beta));
xticklabels(filtered_data.characteristic_names(1:length(analysis_results.regression.beta)));
xtickangle(45);

% Conclusion
disp('\nConclusion:');
disp('---------------------------------');
disp('This example demonstrated various cross-sectional analysis techniques using the MFE Toolbox:');
disp('1. Data exploration and preprocessing');
disp('2. Outlier handling and data filtering');
disp('3. Cross-sectional regression analysis');
disp('4. Heteroskedasticity testing');
disp('5. Comprehensive statistical analysis');
disp('6. Portfolio statistics computation');

disp('\nFor more details on specific functions:');
disp('  help cross_section_filters');
disp('  help cross_section_regression');
disp('  help cross_section_analysis');
disp('  help white_test');