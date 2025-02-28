function [results] = analyze_cross_section(data, options)
% ANALYZE_CROSS_SECTION Comprehensive cross-sectional analysis of financial data
%
% USAGE:
%   RESULTS = analyze_cross_section(DATA)
%   RESULTS = analyze_cross_section(DATA, OPTIONS)
%
% INPUTS:
%   DATA      - T by K matrix of cross-sectional financial variables
%   OPTIONS   - [Optional] Structure with analysis parameters:
%               .preprocess     - Apply data preprocessing [default: false]
%               .filter_options - Options for filter_cross_section
%               .descriptive    - Compute descriptive statistics [default: true]
%               .correlations   - Analyze correlations [default: true]
%               .correlations_options - Options for correlation analysis:
%                  .type         - 'pearson', 'spearman', or 'both' [default: 'both']
%                  .test_significance - Test significance [default: true]
%                  .alpha        - Significance level [default: 0.05]
%               .distribution_tests - Perform distribution tests [default: true]
%               .distribution_options - Options for distribution tests:
%                  .normality    - Test normality [default: true]
%                  .heterogeneity - Test for heterogeneity [default: true]
%                  .groups       - Group identifiers for heterogeneity tests
%               .regression     - Perform regression analysis [default: false]
%               .regression_options - Options for cross_section_regression
%               .portfolio      - Compute portfolio statistics [default: false]
%               .portfolio_options - Options for portfolio analysis:
%                  .weights      - Portfolio weights [default: equal-weighted]
%                  .risk_free    - Risk-free rate [default: 0]
%                  .target_return - Target return for efficient frontier
%               .bootstrap      - Perform bootstrap analysis [default: false]
%               .bootstrap_options - Options for bootstrap_confidence_intervals
%               .report        - Generate analysis report [default: false]
%               .report_options - Options for report formatting
%
% OUTPUTS:
%   RESULTS   - Structure with fields:
%               .descriptive  - Descriptive statistics results
%               .correlations - Correlation analysis results
%               .distributions - Distribution test results
%               .regression   - Regression analysis results (if requested)
%               .portfolio    - Portfolio statistics (if requested)
%               .bootstrap    - Bootstrap results (if requested)
%               .report       - Formatted report (if requested)
%               .options_used - Options used in analysis
%
% COMMENTS:
%   This function provides a comprehensive suite of cross-sectional analysis
%   tools for financial data. It integrates descriptive statistics, correlation
%   analysis, distribution testing, regression analysis, portfolio statistics,
%   and bootstrap inference in a unified framework.
%
%   The function is designed for financial applications including factor analysis,
%   asset pricing tests, portfolio construction, and risk assessment. It
%   implements robust numerical methods to ensure stability in financial
%   calculations.
%
% EXAMPLES:
%   % Basic analysis with default options
%   results = analyze_cross_section(returns_data);
%
%   % Analysis with preprocessing and regression
%   options = struct('preprocess', true, 'regression', true);
%   options.regression_options.method = 'robust';
%   results = analyze_cross_section(returns_data, options);
%
%   % Portfolio analysis with custom weights
%   options = struct('portfolio', true);
%   options.portfolio_options.weights = portfolio_weights;
%   results = analyze_cross_section(returns_data, options);
%
% See also: compute_descriptive_statistics, analyze_correlations,
%          test_distributional_properties, cross_section_regression,
%          analyze_portfolio_statistics, bootstrap_confidence_intervals

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

%% Input validation
data = datacheck(data, 'data');

% Set default options if not provided
if nargin < 2 || isempty(options)
    options = struct();
end

% Set default fields if not provided
if ~isfield(options, 'preprocess') || isempty(options.preprocess)
    options.preprocess = false;
end

if ~isfield(options, 'descriptive') || isempty(options.descriptive)
    options.descriptive = true;
end

if ~isfield(options, 'correlations') || isempty(options.correlations)
    options.correlations = true;
end

if ~isfield(options, 'distribution_tests') || isempty(options.distribution_tests)
    options.distribution_tests = true;
end

if ~isfield(options, 'regression') || isempty(options.regression)
    options.regression = false;
end

if ~isfield(options, 'portfolio') || isempty(options.portfolio)
    options.portfolio = false;
end

if ~isfield(options, 'bootstrap') || isempty(options.bootstrap)
    options.bootstrap = false;
end

if ~isfield(options, 'report') || isempty(options.report)
    options.report = false;
end

% Initialize results structure
results = struct();
results.options_used = options;

%% Step 1: Data preprocessing (if requested)
if options.preprocess
    if ~isfield(options, 'filter_options')
        options.filter_options = struct();
    end
    filter_result = filter_cross_section(data, options.filter_options);
    data = filter_result.data;
    results.preprocessing = filter_result;
end

%% Step 2: Descriptive statistics
if options.descriptive
    if ~isfield(options, 'descriptive_options')
        options.descriptive_options = struct();
    end
    results.descriptive = compute_descriptive_statistics(data, options.descriptive_options);
end

%% Step 3: Correlation analysis
if options.correlations
    if ~isfield(options, 'correlations_options')
        options.correlations_options = struct();
    end
    results.correlations = analyze_correlations(data, options.correlations_options);
end

%% Step 4: Distribution tests
if options.distribution_tests
    if ~isfield(options, 'distribution_options')
        options.distribution_options = struct();
    end
    results.distributions = test_distributional_properties(data, options.distribution_options);
end

%% Step 5: Regression analysis (if requested)
if options.regression
    if ~isfield(options, 'regression_options')
        options.regression_options = struct();
    end
    
    % Check if dependent variable and regressors are specified
    if ~isfield(options.regression_options, 'dependent') || isempty(options.regression_options.dependent)
        error('For regression analysis, OPTIONS.regression_options.dependent must be specified.');
    end
    
    if ~isfield(options.regression_options, 'regressors') || isempty(options.regression_options.regressors)
        error('For regression analysis, OPTIONS.regression_options.regressors must be specified.');
    end
    
    % Extract dependent variable and regressors
    dependent_idx = options.regression_options.dependent;
    regressors_idx = options.regression_options.regressors;
    
    % Validate indices
    [T, K] = size(data);
    if ~isnumeric(dependent_idx) || any(dependent_idx > K) || any(dependent_idx < 1)
        error('Invalid dependent variable index.');
    end
    
    if ~isnumeric(regressors_idx) || any(regressors_idx > K) || any(regressors_idx < 1)
        error('Invalid regressors indices.');
    end
    
    % Check for overlap between dependent and regressors
    if any(ismember(dependent_idx, regressors_idx))
        error('Dependent variable and regressors cannot overlap.');
    end
    
    y = data(:, dependent_idx);
    X = data(:, regressors_idx);
    
    % Perform regression
    reg_results = cross_section_regression(y, X, options.regression_options);
    results.regression = reg_results;
end

%% Step 6: Portfolio analysis (if requested)
if options.portfolio
    if ~isfield(options, 'portfolio_options')
        options.portfolio_options = struct();
    end
    
    % If portfolio weights not provided, use equal weighting
    if ~isfield(options.portfolio_options, 'weights') || isempty(options.portfolio_options.weights)
        [T, K] = size(data);
        options.portfolio_options.weights = ones(K, 1) / K;
    end
    
    results.portfolio = analyze_portfolio_statistics(data, options.portfolio_options.weights, options.portfolio_options);
end

%% Step 7: Bootstrap analysis (if requested)
if options.bootstrap
    if ~isfield(options, 'bootstrap_options')
        options.bootstrap_options = struct();
    end
    
    if ~isfield(options.bootstrap_options, 'statistic_fn')
        % Default to mean if no statistic function provided
        options.bootstrap_options.statistic_fn = @mean;
    end
    
    results.bootstrap = generate_bootstrap_statistics(data, options.bootstrap_options.statistic_fn, options.bootstrap_options);
end

%% Step 8: Generate report (if requested)
if options.report
    if ~isfield(options, 'report_options')
        options.report_options = struct();
    end
    
    results.report = create_analysis_report(results, options.report_options);
end

end

function [stats] = compute_descriptive_statistics(data, options)
% COMPUTE_DESCRIPTIVE_STATISTICS Calculates comprehensive descriptive statistics for cross-sectional data
%
% USAGE:
%   STATS = compute_descriptive_statistics(DATA)
%   STATS = compute_descriptive_statistics(DATA, OPTIONS)
%
% INPUTS:
%   DATA      - T by K matrix of cross-sectional data
%   OPTIONS   - [Optional] Structure with options:
%               .percentiles - Percentiles to compute [default: [0.01 0.05 0.10 0.25 0.50 0.75 0.90 0.95 0.99]]
%               .moments     - Compute higher moments (skewness, kurtosis) [default: true]
%               .robust      - Compute robust statistics (median, IQR, MAD) [default: true]
%
% OUTPUTS:
%   STATS     - Structure with fields:
%               .mean       - Mean values for each variable
%               .median     - Median values for each variable
%               .std        - Standard deviations
%               .var        - Variances
%               .min        - Minimum values
%               .max        - Maximum values
%               .range      - Range (max - min)
%               .percentiles - Specified percentiles
%               .skewness   - Skewness (if moments=true)
%               .kurtosis   - Kurtosis (if moments=true)
%               .excess_kurtosis - Kurtosis - 3 (if moments=true)
%               .iqr        - Interquartile range (if robust=true)
%               .mad        - Median absolute deviation (if robust=true)
%               .summary    - Table of key statistics
%
% See also: mean, median, std, var, skewness, kurtosis, prctile

% Validate input data
data = datacheck(data, 'data');

% Get dimensions
[T, K] = size(data);

% Set default options if not provided
if nargin < 2 || isempty(options)
    options = struct();
end

% Default percentiles
if ~isfield(options, 'percentiles') || isempty(options.percentiles)
    options.percentiles = [0.01 0.05 0.10 0.25 0.50 0.75 0.90 0.95 0.99];
end

% Default for higher moments
if ~isfield(options, 'moments') || isempty(options.moments)
    options.moments = true;
end

% Default for robust statistics
if ~isfield(options, 'robust') || isempty(options.robust)
    options.robust = true;
end

% Initialize output structure
stats = struct();

% Basic descriptive statistics
stats.mean = mean(data, 'omitnan');
stats.std = std(data, 'omitnan');
stats.var = var(data, 'omitnan');
stats.min = min(data, [], 'omitnan');
stats.max = max(data, [], 'omitnan');
stats.range = stats.max - stats.min;

% Compute percentiles
n_percentiles = length(options.percentiles);
percentile_values = zeros(n_percentiles, K);
for k = 1:K
    percentile_values(:, k) = prctile(data(:, k), options.percentiles * 100);
end
stats.percentiles = struct('levels', options.percentiles, 'values', percentile_values);

% Median is the 50th percentile
stats.median = zeros(1, K);
median_idx = find(options.percentiles == 0.5);
if ~isempty(median_idx)
    stats.median = percentile_values(median_idx, :);
else
    stats.median = median(data, 'omitnan');
end

% Higher moments (skewness, kurtosis)
if options.moments
    stats.skewness = skewness(data, 0, 'omitnan');
    stats.kurtosis = kurtosis(data, 0, 'omitnan');
    stats.excess_kurtosis = stats.kurtosis - 3;
end

% Robust statistics
if options.robust
    % Interquartile range
    q1_idx = find(options.percentiles == 0.25);
    q3_idx = find(options.percentiles == 0.75);
    
    if ~isempty(q1_idx) && ~isempty(q3_idx)
        q1 = percentile_values(q1_idx, :);
        q3 = percentile_values(q3_idx, :);
        stats.iqr = q3 - q1;
    else
        % Calculate IQR directly if 25% and 75% aren't in the percentiles
        q1 = prctile(data, 25);
        q3 = prctile(data, 75);
        stats.iqr = q3 - q1;
    end
    
    % Median absolute deviation
    stats.mad = zeros(1, K);
    for k = 1:K
        col_data = data(:, k);
        col_median = stats.median(k);
        stats.mad(k) = median(abs(col_data - col_median), 'omitnan');
    end
end

% Create summary table
summary_rows = {'Mean', 'Median', 'Std Dev', 'Variance', 'Min', 'Max', 'Range'};
summary_data = [stats.mean; stats.median; stats.std; stats.var; stats.min; stats.max; stats.range];

if options.moments
    summary_rows = [summary_rows, 'Skewness', 'Kurtosis', 'Excess Kurtosis'];
    summary_data = [summary_data; stats.skewness; stats.kurtosis; stats.excess_kurtosis];
end

if options.robust
    summary_rows = [summary_rows, 'IQR', 'MAD'];
    summary_data = [summary_data; stats.iqr; stats.mad];
end

stats.summary = struct('rowNames', {summary_rows}, 'data', summary_data);

end

function [results] = analyze_correlations(data, options)
% ANALYZE_CORRELATIONS Analyzes correlations between cross-sectional variables
%
% USAGE:
%   RESULTS = analyze_correlations(DATA)
%   RESULTS = analyze_correlations(DATA, OPTIONS)
%
% INPUTS:
%   DATA      - T by K matrix of cross-sectional data
%   OPTIONS   - [Optional] Structure with options:
%               .type               - 'pearson', 'spearman', or 'both' [default: 'both']
%               .test_significance  - Test significance of correlations [default: true]
%               .alpha             - Significance level [default: 0.05]
%               .heatmap           - Generate heatmap data [default: true]
%
% OUTPUTS:
%   RESULTS   - Structure with fields:
%               .pearson   - Pearson correlation matrix
%               .spearman  - Spearman correlation matrix (if requested)
%               .p_values  - P-values for correlation significance tests
%               .significant - Logical matrix of significant correlations
%               .heatmap   - Data for correlation heatmap visualization
%
% See also: corr, corrcoef, tcdf

% Validate input data
data = datacheck(data, 'data');

% Get dimensions
[T, K] = size(data);

% Set default options if not provided
if nargin < 2 || isempty(options)
    options = struct();
end

% Default correlation type
if ~isfield(options, 'type') || isempty(options.type)
    options.type = 'both';
elseif ~ismember(options.type, {'pearson', 'spearman', 'both'})
    error('OPTIONS.type must be one of: ''pearson'', ''spearman'', or ''both''');
end

% Default for significance testing
if ~isfield(options, 'test_significance') || isempty(options.test_significance)
    options.test_significance = true;
end

% Default significance level
if ~isfield(options, 'alpha') || isempty(options.alpha)
    options.alpha = 0.05;
else
    % Validate alpha
    option_struct = struct('isscalar', true, 'lowerBound', 0, 'upperBound', 1);
    options.alpha = parametercheck(options.alpha, 'alpha', option_struct);
end

% Default for heatmap
if ~isfield(options, 'heatmap') || isempty(options.heatmap)
    options.heatmap = true;
end

% Initialize output structure
results = struct();

% Calculate Pearson correlations
[pearson_corr, pearson_pval] = corr(data, 'type', 'Pearson', 'rows', 'pairwise');
results.pearson = pearson_corr;

% Calculate Spearman correlations if requested
if strcmp(options.type, 'spearman') || strcmp(options.type, 'both')
    [spearman_corr, spearman_pval] = corr(data, 'type', 'Spearman', 'rows', 'pairwise');
    results.spearman = spearman_corr;
    
    % Store p-values for Spearman if that's the only type requested
    if strcmp(options.type, 'spearman')
        results.p_values = spearman_pval;
    end
else
    % Store p-values for Pearson
    results.p_values = pearson_pval;
end

% Store both types of p-values if both correlation types are requested
if strcmp(options.type, 'both')
    results.pearson_p_values = pearson_pval;
    results.spearman_p_values = spearman_pval;
    
    % Use Pearson as the default p-values
    results.p_values = pearson_pval;
end

% Identify significant correlations
if options.test_significance
    results.significant = results.p_values < options.alpha;
    
    % For both correlation types, store separate significance matrices
    if strcmp(options.type, 'both')
        results.pearson_significant = pearson_pval < options.alpha;
        results.spearman_significant = spearman_pval < options.alpha;
    end
end

% Create heatmap data if requested
if options.heatmap
    % Determine which correlation matrix to use for heatmap
    if strcmp(options.type, 'spearman')
        heatmap_corr = results.spearman;
    else
        heatmap_corr = results.pearson;
    end
    
    % Prepare heatmap data
    results.heatmap = struct(...
        'correlation', heatmap_corr, ...
        'p_values', results.p_values, ...
        'significant', results.significant, ...
        'colorscale', [-1 0 1]);
    
    % Add information for significant correlations
    if options.test_significance
        significant_pairs = find(results.significant & ~eye(K));
        if ~isempty(significant_pairs)
            [row_indices, col_indices] = ind2sub([K, K], significant_pairs);
            sig_values = heatmap_corr(significant_pairs);
            sig_pvals = results.p_values(significant_pairs);
            
            results.heatmap.significant_pairs = [row_indices, col_indices, sig_values, sig_pvals];
        else
            results.heatmap.significant_pairs = [];
        end
    end
end

end

function [results] = test_distributional_properties(data, options)
% TEST_DISTRIBUTIONAL_PROPERTIES Tests distributional properties of cross-sectional data
%
% USAGE:
%   RESULTS = test_distributional_properties(DATA)
%   RESULTS = test_distributional_properties(DATA, OPTIONS)
%
% INPUTS:
%   DATA      - T by K matrix of cross-sectional data
%   OPTIONS   - [Optional] Structure with options:
%               .normality     - Test normality [default: true]
%               .heterogeneity - Test for heterogeneity [default: false]
%               .groups        - Group identifiers for heterogeneity tests
%               .alpha         - Significance level [default: 0.05]
%
% OUTPUTS:
%   RESULTS   - Structure with fields:
%               .normality     - Normality test results
%               .heterogeneity - Heterogeneity test results (if requested)
%               .group_tests   - Group-based tests (if groups provided)
%
% See also: jarque_bera, white_test

% Validate input data
data = datacheck(data, 'data');

% Get dimensions
[T, K] = size(data);

% Set default options if not provided
if nargin < 2 || isempty(options)
    options = struct();
end

% Default for normality testing
if ~isfield(options, 'normality') || isempty(options.normality)
    options.normality = true;
end

% Default for heterogeneity testing
if ~isfield(options, 'heterogeneity') || isempty(options.heterogeneity)
    options.heterogeneity = false;
end

% Default significance level
if ~isfield(options, 'alpha') || isempty(options.alpha)
    options.alpha = 0.05;
else
    % Validate alpha
    option_struct = struct('isscalar', true, 'lowerBound', 0, 'upperBound', 1);
    options.alpha = parametercheck(options.alpha, 'alpha', option_struct);
end

% Initialize output structure
results = struct();

% Test for normality
if options.normality
    normality_results = struct(...
        'jb_statistic', zeros(1, K), ...
        'jb_pvalue', zeros(1, K), ...
        'is_normal', false(1, K), ...
        'details', cell(1, K));
    
    for k = 1:K
        col_data = data(:, k);
        
        % Skip columns with too few observations
        if sum(~isnan(col_data)) < 4
            normality_results.jb_statistic(k) = NaN;
            normality_results.jb_pvalue(k) = NaN;
            normality_results.is_normal(k) = false;
            normality_results.details{k} = 'Insufficient observations for normality test';
            continue;
        end
        
        % Perform Jarque-Bera test
        jb_result = jarque_bera(col_data);
        
        % Store results
        normality_results.jb_statistic(k) = jb_result.statistic;
        normality_results.jb_pvalue(k) = jb_result.pval;
        normality_results.is_normal(k) = jb_result.pval >= options.alpha;
        normality_results.details{k} = jb_result;
    end
    
    results.normality = normality_results;
end

% Test for heterogeneity if groups are provided
if isfield(options, 'groups') && ~isempty(options.groups) && options.heterogeneity
    % Validate groups
    groups = options.groups;
    groups = columncheck(groups, 'groups');
    
    if length(groups) ~= T
        error('OPTIONS.groups must have the same length as the number of observations in DATA');
    end
    
    % Get unique groups
    unique_groups = unique(groups);
    n_groups = length(unique_groups);
    
    if n_groups < 2
        error('At least 2 distinct groups are required for heterogeneity testing');
    end
    
    % Initialize heterogeneity results
    heterogeneity_results = struct(...
        'variance_equality', struct(...
            'statistic', zeros(1, K), ...
            'pvalue', zeros(1, K), ...
            'equal_variance', false(1, K)), ...
        'mean_equality', struct(...
            'statistic', zeros(1, K), ...
            'pvalue', zeros(1, K), ...
            'equal_means', false(1, K)), ...
        'group_statistics', struct(...
            'group_ids', unique_groups, ...
            'group_means', zeros(n_groups, K), ...
            'group_vars', zeros(n_groups, K), ...
            'group_counts', zeros(n_groups, 1)));
    
    % Compute statistics for each group
    for i = 1:n_groups
        group_idx = (groups == unique_groups(i));
        heterogeneity_results.group_statistics.group_counts(i) = sum(group_idx);
        
        if sum(group_idx) > 0
            group_data = data(group_idx, :);
            heterogeneity_results.group_statistics.group_means(i, :) = mean(group_data, 'omitnan');
            heterogeneity_results.group_statistics.group_vars(i, :) = var(group_data, 0, 'omitnan');
        else
            heterogeneity_results.group_statistics.group_means(i, :) = NaN(1, K);
            heterogeneity_results.group_statistics.group_vars(i, :) = NaN(1, K);
        end
    end
    
    % Test equality of variances across groups (Levene's test)
    for k = 1:K
        % Skip columns with too few observations in any group
        valid_groups = true;
        for i = 1:n_groups
            group_idx = (groups == unique_groups(i));
            col_data = data(group_idx, k);
            if sum(~isnan(col_data)) < 3
                valid_groups = false;
                break;
            end
        end
        
        if ~valid_groups
            heterogeneity_results.variance_equality.statistic(k) = NaN;
            heterogeneity_results.variance_equality.pvalue(k) = NaN;
            heterogeneity_results.variance_equality.equal_variance(k) = false;
            continue;
        end
        
        % Implement Levene's test for variance equality
        % Compute absolute deviations from group means
        abs_dev = zeros(T, 1);
        for i = 1:n_groups
            group_idx = (groups == unique_groups(i));
            group_mean = heterogeneity_results.group_statistics.group_means(i, k);
            abs_dev(group_idx) = abs(data(group_idx, k) - group_mean);
        end
        
        % Run ANOVA on the absolute deviations
        % Compute group means and overall mean of absolute deviations
        grp_mean_abs_dev = zeros(n_groups, 1);
        grp_n = zeros(n_groups, 1);
        
        for i = 1:n_groups
            group_idx = (groups == unique_groups(i));
            grp_abs_dev = abs_dev(group_idx);
            grp_n(i) = sum(~isnan(grp_abs_dev));
            if grp_n(i) > 0
                grp_mean_abs_dev(i) = mean(grp_abs_dev, 'omitnan');
            else
                grp_mean_abs_dev(i) = NaN;
            end
        end
        
        overall_mean_abs_dev = mean(abs_dev, 'omitnan');
        
        % Compute sum of squares
        SSB = sum(grp_n .* (grp_mean_abs_dev - overall_mean_abs_dev).^2, 'omitnan');
        
        SSW = 0;
        for i = 1:n_groups
            group_idx = (groups == unique_groups(i));
            grp_abs_dev = abs_dev(group_idx);
            SSW = SSW + sum((grp_abs_dev - grp_mean_abs_dev(i)).^2, 'omitnan');
        end
        
        % Compute F-statistic
        dfB = n_groups - 1;
        dfW = sum(grp_n) - n_groups;
        
        if dfW > 0 && SSW > 0
            F_stat = (SSB / dfB) / (SSW / dfW);
            p_value = 1 - fcdf(F_stat, dfB, dfW);
            
            heterogeneity_results.variance_equality.statistic(k) = F_stat;
            heterogeneity_results.variance_equality.pvalue(k) = p_value;
            heterogeneity_results.variance_equality.equal_variance(k) = p_value >= options.alpha;
        else
            heterogeneity_results.variance_equality.statistic(k) = NaN;
            heterogeneity_results.variance_equality.pvalue(k) = NaN;
            heterogeneity_results.variance_equality.equal_variance(k) = false;
        end
    end
    
    % Test equality of means across groups (ANOVA)
    for k = 1:K
        % Skip columns with too few observations in any group
        valid_groups = true;
        for i = 1:n_groups
            group_idx = (groups == unique_groups(i));
            col_data = data(group_idx, k);
            if sum(~isnan(col_data)) < 3
                valid_groups = false;
                break;
            end
        end
        
        if ~valid_groups
            heterogeneity_results.mean_equality.statistic(k) = NaN;
            heterogeneity_results.mean_equality.pvalue(k) = NaN;
            heterogeneity_results.mean_equality.equal_means(k) = false;
            continue;
        end
        
        % Compute group means and overall mean
        grp_means = heterogeneity_results.group_statistics.group_means(:, k);
        grp_n = zeros(n_groups, 1);
        
        for i = 1:n_groups
            group_idx = (groups == unique_groups(i));
            grp_data = data(group_idx, k);
            grp_n(i) = sum(~isnan(grp_data));
        end
        
        overall_mean = mean(data(:, k), 'omitnan');
        
        % Compute sum of squares
        SSB = sum(grp_n .* (grp_means - overall_mean).^2, 'omitnan');
        
        SSW = 0;
        for i = 1:n_groups
            group_idx = (groups == unique_groups(i));
            grp_data = data(group_idx, k);
            SSW = SSW + sum((grp_data - grp_means(i)).^2, 'omitnan');
        end
        
        % Compute F-statistic
        dfB = n_groups - 1;
        dfW = sum(grp_n) - n_groups;
        
        if dfW > 0 && SSW > 0
            F_stat = (SSB / dfB) / (SSW / dfW);
            p_value = 1 - fcdf(F_stat, dfB, dfW);
            
            heterogeneity_results.mean_equality.statistic(k) = F_stat;
            heterogeneity_results.mean_equality.pvalue(k) = p_value;
            heterogeneity_results.mean_equality.equal_means(k) = p_value >= options.alpha;
        else
            heterogeneity_results.mean_equality.statistic(k) = NaN;
            heterogeneity_results.mean_equality.pvalue(k) = NaN;
            heterogeneity_results.mean_equality.equal_means(k) = false;
        end
    end
    
    results.heterogeneity = heterogeneity_results;
end

end

function [results] = analyze_portfolio_statistics(returns, weights, options)
% ANALYZE_PORTFOLIO_STATISTICS Calculates portfolio-level statistics for cross-sectional asset data
%
% USAGE:
%   RESULTS = analyze_portfolio_statistics(RETURNS, WEIGHTS)
%   RESULTS = analyze_portfolio_statistics(RETURNS, WEIGHTS, OPTIONS)
%
% INPUTS:
%   RETURNS   - T by K matrix of asset returns
%   WEIGHTS   - K by 1 vector of portfolio weights
%   OPTIONS   - [Optional] Structure with options:
%               .risk_free      - Risk-free rate [default: 0]
%               .annualization  - Annualization factor [default: 1]
%               .method         - Covariance estimation method [default: 'sample']
%                                 Options: 'sample', 'shrinkage', 'robust'
%               .target_return  - Target return for efficient frontier [default: []]
%               .efficient_points - Number of points for efficient frontier [default: 20]
%
% OUTPUTS:
%   RESULTS   - Structure with fields:
%               .expected_return   - Portfolio expected return
%               .volatility        - Portfolio volatility (standard deviation)
%               .sharpe_ratio      - Sharpe ratio (excess return / volatility)
%               .sortino_ratio     - Sortino ratio (excess return / downside deviation)
%               .information_ratio - Information ratio relative to benchmark
%               .diversification   - Diversification metrics
%               .risk_contribution - Risk contribution of each asset
%               .asset_stats       - Individual asset statistics
%               .efficient_frontier - Efficient frontier if requested
%
% See also: mean, cov, std, var

% Validate input data
returns = datacheck(returns, 'returns');
weights = datacheck(weights, 'weights');
weights = columncheck(weights, 'weights');

% Validate dimensions
[T, K] = size(returns);
if length(weights) ~= K
    error('Length of WEIGHTS must match the number of columns in RETURNS');
end

% Validate that weights sum to 1
if abs(sum(weights) - 1) > 1e-10
    warning('Portfolio weights do not sum to 1. Normalizing weights.');
    weights = weights / sum(weights);
end

% Set default options if not provided
if nargin < 3 || isempty(options)
    options = struct();
end

% Default risk-free rate
if ~isfield(options, 'risk_free') || isempty(options.risk_free)
    options.risk_free = 0;
end

% Default annualization factor
if ~isfield(options, 'annualization') || isempty(options.annualization)
    options.annualization = 1;
end

% Default covariance estimation method
if ~isfield(options, 'method') || isempty(options.method)
    options.method = 'sample';
elseif ~ismember(options.method, {'sample', 'shrinkage', 'robust'})
    error('OPTIONS.method must be one of: ''sample'', ''shrinkage'', or ''robust''');
end

% Default efficient frontier points
if ~isfield(options, 'efficient_points') || isempty(options.efficient_points)
    options.efficient_points = 20;
end

% Initialize output structure
results = struct();

% Calculate asset-level statistics
asset_stats = struct();
asset_stats.mean = mean(returns, 'omitnan');
asset_stats.volatility = std(returns, 0, 'omitnan');
asset_stats.sharpe = (asset_stats.mean - options.risk_free) ./ asset_stats.volatility;

% Calculate downside deviations
downside_returns = returns;
downside_returns(returns > 0) = 0;
asset_stats.downside_deviation = sqrt(mean(downside_returns.^2, 'omitnan'));

% Calculate correlation and covariance matrices
asset_stats.correlation = corr(returns, 'rows', 'pairwise');
asset_stats.covariance = cov(returns, 'omitrows');

% Store asset statistics
results.asset_stats = asset_stats;

% Estimate covariance matrix based on selected method
switch options.method
    case 'sample'
        % Standard sample covariance matrix
        cov_matrix = asset_stats.covariance;
        
    case 'shrinkage'
        % Linear shrinkage estimator (Ledoit-Wolf type)
        % Shrink towards a diagonal target (constant correlation)
        sample_cov = asset_stats.covariance;
        
        % Target = average correlation * sqrt(var_i * var_j)
        var_vector = diag(sample_cov);
        avg_corr = mean(asset_stats.correlation(~eye(K)));
        
        % Construct target matrix
        target = zeros(K, K);
        for i = 1:K
            for j = 1:K
                if i == j
                    target(i, j) = var_vector(i);
                else
                    target(i, j) = avg_corr * sqrt(var_vector(i) * var_vector(j));
                end
            end
        end
        
        % Simple shrinkage with fixed intensity
        shrinkage_intensity = 0.2;  % Can be optimized based on Ledoit-Wolf formula
        cov_matrix = (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * target;
        
    case 'robust'
        % Robust covariance estimation using median absolute deviation
        cov_matrix = zeros(K, K);
        
        % Compute median of each variable
        medians = median(returns, 'omitnan');
        
        % Compute MAD for each variable
        mad_values = zeros(1, K);
        for k = 1:K
            mad_values(k) = median(abs(returns(:, k) - medians(k)), 'omitnan');
        end
        
        % Scale MAD to be consistent with standard deviation for normal distribution
        mad_scale = 1.4826;  % Constant for consistency with normal distribution
        scaled_mad = mad_values * mad_scale;
        
        % Diagonal elements (variances)
        for k = 1:K
            cov_matrix(k, k) = scaled_mad(k)^2;
        end
        
        % Off-diagonal elements (covariances) using Spearman correlation
        spearman_corr = corr(returns, 'type', 'Spearman', 'rows', 'pairwise');
        
        for i = 1:K
            for j = (i+1):K
                cov_matrix(i, j) = spearman_corr(i, j) * scaled_mad(i) * scaled_mad(j);
                cov_matrix(j, i) = cov_matrix(i, j);  % Symmetry
            end
        end
end

% Ensure covariance matrix is positive definite
[V, D] = eig(cov_matrix);
D = diag(max(diag(D), 1e-8));
cov_matrix = V * D * V';

% Calculate portfolio expected return
results.expected_return = asset_stats.mean * weights;

% Calculate portfolio variance and volatility
results.variance = weights' * cov_matrix * weights;
results.volatility = sqrt(results.variance);

% Calculate risk-adjusted performance metrics
results.sharpe_ratio = (results.expected_return - options.risk_free) / results.volatility;

% Calculate Sortino ratio
portfolio_returns = returns * weights;
downside_returns = portfolio_returns;
downside_returns(portfolio_returns > 0) = 0;
downside_dev = sqrt(mean(downside_returns.^2, 'omitnan'));
results.sortino_ratio = (results.expected_return - options.risk_free) / downside_dev;

% Calculate diversification metrics
asset_volatility_contribution = weights .* (cov_matrix * weights) / results.variance;
results.diversification = struct(...
    'ratio', sum(weights .* asset_stats.volatility) / results.volatility, ...
    'effective_bets', 1 / sum(asset_volatility_contribution.^2));

% Calculate risk contribution of each asset
results.risk_contribution = struct(...
    'volatility', asset_volatility_contribution, ...
    'percent', 100 * asset_volatility_contribution);

% Calculate efficient frontier if target return provided
if isfield(options, 'target_return') && ~isempty(options.target_return)
    % Number of points on the frontier
    n_points = options.efficient_points;
    
    % Find minimum variance portfolio
    min_var_weights = quadprog(cov_matrix, zeros(K, 1), [], [], ones(1, K), 1, zeros(K, 1), ones(K, 1));
    min_var_return = asset_stats.mean * min_var_weights;
    min_var_risk = sqrt(min_var_weights' * cov_matrix * min_var_weights);
    
    % Find maximum return portfolio (100% in highest return asset)
    [max_return, max_idx] = max(asset_stats.mean);
    max_return_weights = zeros(K, 1);
    max_return_weights(max_idx) = 1;
    max_return_risk = sqrt(max_return_weights' * cov_matrix * max_return_weights);
    
    % Generate efficient frontier
    target_returns = linspace(min_var_return, max_return, n_points);
    frontier_risks = zeros(n_points, 1);
    frontier_weights = zeros(K, n_points);
    
    % For each target return, find minimum risk portfolio
    for i = 1:n_points
        target = target_returns(i);
        
        % Quadratic programming to minimize portfolio variance
        % subject to: weights >= 0, sum(weights) = 1, expected return = target
        A = [ones(1, K); asset_stats.mean];
        b = [1; target];
        
        try
            w = quadprog(cov_matrix, zeros(K, 1), [], [], A, b, zeros(K, 1), ones(K, 1));
            frontier_risks(i) = sqrt(w' * cov_matrix * w);
            frontier_weights(:, i) = w;
        catch
            % Use previous weights if optimization fails
            if i > 1
                frontier_weights(:, i) = frontier_weights(:, i-1);
                frontier_risks(i) = frontier_risks(i-1);
            else
                frontier_weights(:, i) = min_var_weights;
                frontier_risks(i) = min_var_risk;
            end
        end
    end
    
    % Calculate Sharpe ratios along the frontier
    frontier_sharpe = (target_returns - options.risk_free) ./ frontier_risks;
    [max_sharpe, max_sharpe_idx] = max(frontier_sharpe);
    
    % Store efficient frontier results
    results.efficient_frontier = struct(...
        'returns', target_returns, ...
        'risks', frontier_risks, ...
        'sharpe_ratios', frontier_sharpe, ...
        'weights', frontier_weights, ...
        'min_variance', struct('return', min_var_return, 'risk', min_var_risk, 'weights', min_var_weights), ...
        'max_return', struct('return', max_return, 'risk', max_return_risk, 'weights', max_return_weights), ...
        'max_sharpe', struct('return', target_returns(max_sharpe_idx), 'risk', frontier_risks(max_sharpe_idx), ...
        'sharpe', max_sharpe, 'weights', frontier_weights(:, max_sharpe_idx)));
end

% Apply annualization if requested
if options.annualization ~= 1
    annualization_factor = sqrt(options.annualization);
    
    results.expected_return = results.expected_return * options.annualization;
    results.volatility = results.volatility * annualization_factor;
    results.sharpe_ratio = (results.expected_return - options.risk_free * options.annualization) / results.volatility;
    results.sortino_ratio = (results.expected_return - options.risk_free * options.annualization) / (downside_dev * annualization_factor);
    
    % Update asset statistics
    results.asset_stats.mean = results.asset_stats.mean * options.annualization;
    results.asset_stats.volatility = results.asset_stats.volatility * annualization_factor;
    results.asset_stats.sharpe = (results.asset_stats.mean - options.risk_free * options.annualization) ./ results.asset_stats.volatility;
    
    % Update efficient frontier if calculated
    if isfield(results, 'efficient_frontier')
        results.efficient_frontier.returns = results.efficient_frontier.returns * options.annualization;
        results.efficient_frontier.risks = results.efficient_frontier.risks * annualization_factor;
        results.efficient_frontier.sharpe_ratios = (results.efficient_frontier.returns - options.risk_free * options.annualization) ./ results.efficient_frontier.risks;
        results.efficient_frontier.min_variance.return = results.efficient_frontier.min_variance.return * options.annualization;
        results.efficient_frontier.min_variance.risk = results.efficient_frontier.min_variance.risk * annualization_factor;
        results.efficient_frontier.max_return.return = results.efficient_frontier.max_return.return * options.annualization;
        results.efficient_frontier.max_return.risk = results.efficient_frontier.max_return.risk * annualization_factor;
        results.efficient_frontier.max_sharpe.return = results.efficient_frontier.max_sharpe.return * options.annualization;
        results.efficient_frontier.max_sharpe.risk = results.efficient_frontier.max_sharpe.risk * annualization_factor;
    end
end

end

function [results] = generate_bootstrap_statistics(data, statistic_fn, options)
% GENERATE_BOOTSTRAP_STATISTICS Generates bootstrap confidence intervals for cross-sectional statistics
%
% USAGE:
%   RESULTS = generate_bootstrap_statistics(DATA, STATISTIC_FN)
%   RESULTS = generate_bootstrap_statistics(DATA, STATISTIC_FN, OPTIONS)
%
% INPUTS:
%   DATA        - T by K matrix of financial data
%   STATISTIC_FN - Function handle to compute statistic of interest
%   OPTIONS     - [Optional] Structure with options:
%                 (See bootstrap_confidence_intervals documentation)
%
% OUTPUTS:
%   RESULTS     - Structure with bootstrap results:
%                 .original_statistic: Original statistic value
%                 .bootstrap_statistics: Bootstrap replications
%                 .confidence_intervals: Confidence intervals
%                 .standard_error: Bootstrap standard error
%                 .bias: Bootstrap bias estimate
%                 .options_used: Bootstrap options used
%
% See also: bootstrap_confidence_intervals, block_bootstrap, stationary_bootstrap

% Validate input data
data = datacheck(data, 'data');

% Validate statistic function is a function handle
if ~isa(statistic_fn, 'function_handle')
    error('STATISTIC_FN must be a function handle.');
end

% Set default options if not provided
if nargin < 3 || isempty(options)
    options = struct();
end

% Call bootstrap_confidence_intervals function
bs_results = bootstrap_confidence_intervals(data, statistic_fn, options);

% Extract additional statistics
bootstrap_statistics = bs_results.bootstrap_statistics;
original_statistic = bs_results.original_statistic;

% Calculate bootstrap standard error
bootstrap_se = std(bootstrap_statistics);

% Calculate bootstrap bias estimate
bootstrap_bias = mean(bootstrap_statistics) - original_statistic;

% Compile results
results = struct(...
    'original_statistic', original_statistic, ...
    'bootstrap_statistics', bootstrap_statistics, ...
    'confidence_intervals', struct('lower', bs_results.lower, 'upper', bs_results.upper), ...
    'standard_error', bootstrap_se, ...
    'bias', bootstrap_bias, ...
    'bias_corrected_estimate', original_statistic - bootstrap_bias, ...
    'options_used', bs_results.bootstrap_options);

end

function [report] = create_analysis_report(analysis_results, options)
% CREATE_ANALYSIS_REPORT Generates a structured report of cross-sectional analysis results
%
% USAGE:
%   REPORT = create_analysis_report(ANALYSIS_RESULTS)
%   REPORT = create_analysis_report(ANALYSIS_RESULTS, OPTIONS)
%
% INPUTS:
%   ANALYSIS_RESULTS - Structure with analysis results from analyze_cross_section
%   OPTIONS          - [Optional] Structure with reporting options:
%                      .format         - Report format (default: 'detailed')
%                      .include_plots  - Include plot data (default: true)
%                      .sections       - Sections to include (default: all)
%
% OUTPUTS:
%   REPORT           - Structure with formatted report
%
% See also: analyze_cross_section

% Set default options if not provided
if nargin < 2 || isempty(options)
    options = struct();
end

% Default format
if ~isfield(options, 'format') || isempty(options.format)
    options.format = 'detailed';
end

% Default for including plots
if ~isfield(options, 'include_plots') || isempty(options.include_plots)
    options.include_plots = true;
end

% Default sections to include
if ~isfield(options, 'sections') || isempty(options.sections)
    options.sections = {'descriptive', 'correlations', 'distributions', 'regression', 'portfolio', 'bootstrap'};
end

% Initialize report structure
report = struct();
report.title = 'Cross-Sectional Analysis Report';
report.date = datestr(now);
report.sections = struct();

% Include only requested sections
sections_to_process = intersect(options.sections, fieldnames(analysis_results));

% Process each requested section
for i = 1:length(sections_to_process)
    section_name = sections_to_process{i};
    
    % Skip 'options_used' and other non-result fields
    if ismember(section_name, {'options_used', 'preprocessing', 'report'})
        continue;
    end
    
    % Add section to report based on its type
    section_data = analysis_results.(section_name);
    
    % Format and add the section to the report
    formatted_section = struct();
    
    switch section_name
        case 'descriptive'
            formatted_section.title = 'Descriptive Statistics';
            if isfield(section_data, 'summary')
                formatted_section.summary = section_data.summary;
            end
            if isfield(section_data, 'percentiles')
                formatted_section.percentiles = section_data.percentiles;
            end
            
        case 'correlations'
            formatted_section.title = 'Correlation Analysis';
            if isfield(section_data, 'pearson')
                formatted_section.pearson = section_data.pearson;
            end
            if isfield(section_data, 'significant')
                formatted_section.significant = section_data.significant;
            end
            if options.include_plots && isfield(section_data, 'heatmap')
                formatted_section.heatmap = section_data.heatmap;
            end
            
        case 'distributions'
            formatted_section.title = 'Distribution Analysis';
            if isfield(section_data, 'normality')
                formatted_section.normality = section_data.normality;
            end
            if isfield(section_data, 'heterogeneity')
                formatted_section.heterogeneity = section_data.heterogeneity;
            end
            
        case 'regression'
            formatted_section.title = 'Regression Analysis';
            formatted_section = section_data;  % Include all regression output
            
        case 'portfolio'
            formatted_section.title = 'Portfolio Analysis';
            formatted_section = section_data;  % Include all portfolio output
            
        case 'bootstrap'
            formatted_section.title = 'Bootstrap Analysis';
            formatted_section = section_data;  % Include all bootstrap output
    end
    
    report.sections.(section_name) = formatted_section;
end

% Add summary section
report.summary = struct(...
    'title', 'Analysis Summary', ...
    'sections', sections_to_process);

end