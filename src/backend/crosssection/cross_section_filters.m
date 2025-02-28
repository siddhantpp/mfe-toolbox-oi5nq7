function [result] = filter_cross_section(data, options)
% FILTER_CROSS_SECTION Applies comprehensive filtering operations to cross-sectional financial data.
%
% USAGE:
%   [RESULT] = filter_cross_section(DATA)
%   [RESULT] = filter_cross_section(DATA, OPTIONS)
%
% INPUTS:
%   DATA        - A T by K matrix of cross-sectional financial data, where 
%                 T is the number of observations and K is the number of variables.
%
%   OPTIONS     - Optional structure with filtering parameters:
%                 OPTIONS.missing_handling: Method for handling missing values
%                    'none'    - Do not handle missing values (default)
%                    'remove'  - Remove observations with any missing values
%                    'mean'    - Replace missing values with column means
%                    'median'  - Replace missing values with column medians
%                    'mode'    - Replace missing values with column modes
%                    'knn'     - K-nearest neighbors imputation
%
%                 OPTIONS.missing_k: K parameter for KNN imputation [default: 5]
%
%                 OPTIONS.outlier_detection: Method for detecting outliers
%                    'none'    - No outlier detection (default)
%                    'zscore'  - Z-score method with threshold
%                    'iqr'     - Interquartile range method
%                    'mad'     - Median absolute deviation method
%
%                 OPTIONS.outlier_threshold: Threshold for outlier detection
%                    For zscore: Number of standard deviations [default: 3]
%                    For iqr: Multiplier of IQR [default: 1.5]
%                    For mad: Multiplier of MAD [default: 3]
%
%                 OPTIONS.outlier_handling: Method for handling detected outliers
%                    'none'      - No action, just detect (default)
%                    'winsorize' - Cap extreme values at percentiles
%                    'trim'      - Remove observations identified as outliers
%                    'replace'   - Replace outliers with boundary values
%
%                 OPTIONS.winsor_percentiles: [lower upper] percentiles for 
%                                            winsorization [default: [0.01 0.99]]
%
%                 OPTIONS.transform: Data transformation method
%                    'none'    - No transformation (default)
%                    'log'     - Natural logarithm transformation
%                    'sqrt'    - Square root transformation
%                    'boxcox'  - Box-Cox transformation
%                    'yj'      - Yeo-Johnson transformation
%                    'rank'    - Rank transformation
%
%                 OPTIONS.transform_offset: Offset for log/sqrt transforms [default: 0]
%                 OPTIONS.transform_lambda: Lambda parameter for Box-Cox [default: 0]
%                 OPTIONS.transform_estimate: Estimate optimal transformation 
%                                           parameter [default: true]
%
%                 OPTIONS.normalize: Normalization method
%                    'none'        - No normalization (default)
%                    'standardize' - Standardize to mean 0, std 1
%                    'minmax'      - Scale to [0,1] range
%                    'robust'      - Use median and MAD for scaling
%                    'decimal'     - Scale by powers of 10
%                    'custom'      - Use custom scaling factors
%
%                 OPTIONS.normalize_center: Custom centering values for 'custom' method
%                 OPTIONS.normalize_scale: Custom scaling values for 'custom' method
%
% OUTPUTS:
%   RESULT      - Structure containing filtering results with fields:
%                 RESULT.data: Filtered data
%                 RESULT.missing: Statistics about missing values
%                 RESULT.outliers: Information about detected outliers
%                 RESULT.transform: Transformation parameters used
%                 RESULT.normalize: Normalization parameters used
%                 RESULT.original_dims: Original data dimensions [T K]
%                 RESULT.filtered_dims: Final data dimensions after filtering
%                 RESULT.options: The options used for filtering
%
% COMMENTS:
%   This function provides a comprehensive suite of tools for preparing
%   cross-sectional financial data for analysis. It handles common data
%   preprocessing challenges like missing values, outliers, and distributional
%   properties to improve the quality of subsequent statistical analysis.
%
%   The operations are applied in the following sequence:
%   1. Missing value handling
%   2. Outlier detection and handling
%   3. Data transformation
%   4. Normalization
%
%   For missing values or outliers detection, the function can return the
%   indices of problematic values without modifying the data if the handling
%   method is set to 'none'.
%
% EXAMPLES:
%   % Basic usage with default options
%   result = filter_cross_section(returns_data);
%
%   % Remove missing values and winsorize outliers
%   options = struct('missing_handling', 'remove', ...
%                   'outlier_detection', 'zscore', ...
%                   'outlier_handling', 'winsorize');
%   result = filter_cross_section(returns_data, options);
%
%   % Apply log transformation and standardize data
%   options = struct('transform', 'log', ...
%                   'transform_offset', 0.001, ...
%                   'normalize', 'standardize');
%   result = filter_cross_section(returns_data, options);
%
% See also datacheck, handle_missing_values, handle_outliers, transform_data, normalize_data

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Validate input data
data = datacheck(data, 'data');

% Initialize results structure
result = struct();
result.original_dims = size(data);

% Parse options or use defaults
if nargin < 2 || isempty(options)
    options = struct();
end

% Default options
default_options = struct(...
    'missing_handling', 'none', ...
    'missing_k', 5, ...
    'outlier_detection', 'none', ...
    'outlier_threshold', 3, ...
    'outlier_handling', 'none', ...
    'winsor_percentiles', [0.01 0.99], ...
    'transform', 'none', ...
    'transform_offset', 0, ...
    'transform_lambda', 0, ...
    'transform_estimate', true, ...
    'normalize', 'none', ...
    'normalize_center', [], ...
    'normalize_scale', []);

% Merge with defaults for any unspecified options
option_fields = fieldnames(default_options);
for i = 1:length(option_fields)
    field = option_fields{i};
    if ~isfield(options, field)
        options.(field) = default_options.(field);
    end
end

% Store options used in result
result.options = options;

% Step 1: Handle missing values
[data, missing_info] = handle_missing_values(data, options);
result.missing = missing_info;

% Step 2: Handle outliers
[data, outlier_info] = handle_outliers(data, options);
result.outliers = outlier_info;

% Step 3: Transform data
[data, transform_info] = transform_data(data, options);
result.transform = transform_info;

% Step 4: Normalize data
[data, normalize_info] = normalize_data(data, options);
result.normalize = normalize_info;

% Store filtered data and dimensions
result.data = data;
result.filtered_dims = size(data);

end


function [data_out, missing_info] = handle_missing_values(data, options)
% HANDLE_MISSING_VALUES Handles missing values in cross-sectional data
%
% USAGE:
%   [DATA_OUT, MISSING_INFO] = handle_missing_values(DATA, OPTIONS)
%
% INPUTS:
%   DATA     - T by K matrix of cross-sectional data
%   OPTIONS  - Structure with missing value handling options
%
% OUTPUTS:
%   DATA_OUT     - Data with missing values handled
%   MISSING_INFO - Structure with information about missing values

% Initialize output with input data
data_out = data;

% Find missing values
is_missing = isnan(data);
[rows_with_missing, cols_with_missing] = find(is_missing);
missing_positions = unique(rows_with_missing);
missing_count = sum(is_missing(:));
missing_percent = 100 * missing_count / numel(data);

% Store missing value statistics
missing_info = struct(...
    'total_missing', missing_count, ...
    'missing_percent', missing_percent, ...
    'rows_with_missing', missing_positions, ...
    'cols_with_missing', unique(cols_with_missing), ...
    'missing_map', is_missing, ...
    'method', options.missing_handling);

% Early return if no missing values or no handling requested
if missing_count == 0 || strcmpi(options.missing_handling, 'none')
    return;
end

% Apply selected handling method
switch lower(options.missing_handling)
    case 'remove'
        % Remove rows with any missing values
        data_out = data(~any(is_missing, 2), :);
        missing_info.rows_removed = missing_positions;
        
    case 'mean'
        % Replace missing values with column means
        col_means = mean(data, 'omitnan');
        for j = 1:size(data, 2)
            col_missing = is_missing(:, j);
            if any(col_missing)
                data_out(col_missing, j) = col_means(j);
            end
        end
        missing_info.imputation_values = col_means;
        
    case 'median'
        % Replace missing values with column medians
        col_medians = median(data, 'omitnan');
        for j = 1:size(data, 2)
            col_missing = is_missing(:, j);
            if any(col_missing)
                data_out(col_missing, j) = col_medians(j);
            end
        end
        missing_info.imputation_values = col_medians;
        
    case 'mode'
        % Replace missing values with column modes
        col_modes = zeros(1, size(data, 2));
        for j = 1:size(data, 2)
            non_missing_data = data(~is_missing(:, j), j);
            if ~isempty(non_missing_data)
                % Calculate mode (this is a simplified version)
                [counts, values] = hist(non_missing_data, min(10, length(non_missing_data)));
                [~, idx] = max(counts);
                col_modes(j) = values(idx);
                
                % Apply imputation
                col_missing = is_missing(:, j);
                if any(col_missing)
                    data_out(col_missing, j) = col_modes(j);
                end
            end
        end
        missing_info.imputation_values = col_modes;
        
    case 'knn'
        % K-nearest neighbors imputation
        k = options.missing_k;
        
        % Validate k parameter
        k_options = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
        k = parametercheck(k, 'KNN parameter k', k_options);
        
        % For each column with missing values
        for j = find(any(is_missing, 1))
            col_missing_idx = find(is_missing(:, j));
            if isempty(col_missing_idx)
                continue;
            end
            
            % For each missing value in this column
            for i = 1:length(col_missing_idx)
                row_idx = col_missing_idx(i);
                
                % Get the row with missing value
                target_row = data(row_idx, :);
                
                % Calculate distance to all other rows (using only non-missing values)
                % We exclude rows with missing values in the same column as well
                valid_rows = find(~is_missing(:, j));
                
                if length(valid_rows) <= k
                    % Not enough neighbors, use column mean instead
                    data_out(row_idx, j) = mean(data(valid_rows, j), 'omitnan');
                    continue;
                end
                
                distances = zeros(length(valid_rows), 1);
                for vidx = 1:length(valid_rows)
                    v_row = valid_rows(vidx);
                    
                    % Calculate Euclidean distance using only columns where both rows have values
                    common_cols = ~is_missing(row_idx, :) & ~is_missing(v_row, :);
                    if sum(common_cols) > 0
                        diff_vec = target_row(common_cols) - data(v_row, common_cols);
                        distances(vidx) = sqrt(sum(diff_vec.^2));
                    else
                        % No common non-missing columns, use a large distance
                        distances(vidx) = Inf;
                    end
                end
                
                % Sort distances and get k nearest neighbors
                [~, sort_idx] = sort(distances);
                nearest_indices = valid_rows(sort_idx(1:min(k, sum(~isinf(distances)))));
                
                % Impute using mean of k nearest neighbors
                if ~isempty(nearest_indices)
                    data_out(row_idx, j) = mean(data(nearest_indices, j));
                else
                    % Fallback to column mean if no valid neighbors
                    data_out(row_idx, j) = mean(data(~is_missing(:, j), j));
                end
            end
        end
        missing_info.k_parameter = k;
        
    otherwise
        error('Unsupported missing value handling method: %s', options.missing_handling);
end

% Update missing value counts after handling
missing_info.original_rows = size(data, 1);
missing_info.remaining_rows = size(data_out, 1);
missing_info.rows_removed_percent = 100 * (1 - size(data_out, 1) / size(data, 1));

end


function [data_out, outlier_info] = handle_outliers(data, options)
% HANDLE_OUTLIERS Detects and handles outliers in cross-sectional data
%
% USAGE:
%   [DATA_OUT, OUTLIER_INFO] = handle_outliers(DATA, OPTIONS)
%
% INPUTS:
%   DATA     - T by K matrix of cross-sectional data
%   OPTIONS  - Structure with outlier detection and handling options
%
% OUTPUTS:
%   DATA_OUT     - Data with outliers handled
%   OUTLIER_INFO - Structure with information about outliers

% Initialize output with input data
data_out = data;

% Initialize outlier info structure
outlier_info = struct(...
    'method', options.outlier_detection, ...
    'threshold', options.outlier_threshold, ...
    'handling', options.outlier_handling);

% Early return if no outlier detection requested
if strcmpi(options.outlier_detection, 'none')
    outlier_info.outlier_map = false(size(data));
    outlier_info.total_outliers = 0;
    outlier_info.outlier_percent = 0;
    return;
end

% Detect outliers using specified method
switch lower(options.outlier_detection)
    case 'zscore'
        % Z-score outlier detection
        outlier_map = detect_outliers_zscore(data, options.outlier_threshold);
        
    case 'iqr'
        % Interquartile range outlier detection
        outlier_map = detect_outliers_iqr(data, options.outlier_threshold);
        
    case 'mad'
        % Median absolute deviation outlier detection
        outlier_map = detect_outliers_mad(data, options.outlier_threshold);
        
    otherwise
        error('Unsupported outlier detection method: %s', options.outlier_detection);
end

% Count outliers
total_outliers = sum(outlier_map(:));
outlier_percent = 100 * total_outliers / numel(data);

% Record outlier information
outlier_info.outlier_map = outlier_map;
outlier_info.total_outliers = total_outliers;
outlier_info.outlier_percent = outlier_percent;
[outlier_rows, outlier_cols] = find(outlier_map);
outlier_info.outlier_positions = [outlier_rows, outlier_cols];
outlier_info.unique_outlier_rows = unique(outlier_rows);
outlier_info.unique_outlier_cols = unique(outlier_cols);

% Early return if no outliers found or no handling requested
if total_outliers == 0 || strcmpi(options.outlier_handling, 'none')
    return;
end

% Handle outliers using specified method
switch lower(options.outlier_handling)
    case 'winsorize'
        % Winsorize data at specified percentiles
        lower_pct = options.winsor_percentiles(1);
        upper_pct = options.winsor_percentiles(2);
        data_out = winsorize(data, lower_pct, upper_pct);
        outlier_info.winsor_percentiles = [lower_pct, upper_pct];
        
    case 'trim'
        % Remove rows with outliers
        rows_to_keep = ~any(outlier_map, 2);
        data_out = data(rows_to_keep, :);
        outlier_info.rows_removed = find(~rows_to_keep);
        outlier_info.rows_removed_count = sum(~rows_to_keep);
        outlier_info.rows_removed_percent = 100 * outlier_info.rows_removed_count / size(data, 1);
        
    case 'replace'
        % Replace outliers with boundary values
        for j = 1:size(data, 2)
            col_data = data(:, j);
            col_outliers = outlier_map(:, j);
            
            if any(col_outliers)
                % Calculate boundaries based on the detection method
                switch lower(options.outlier_detection)
                    case 'zscore'
                        % Calculate mean and standard deviation
                        mu = mean(col_data, 'omitnan');
                        sigma = std(col_data, 'omitnan');
                        threshold = options.outlier_threshold;
                        
                        % Replace high outliers with upper bound
                        high_outliers = col_data > mu + threshold * sigma & col_outliers;
                        if any(high_outliers)
                            data_out(high_outliers, j) = mu + threshold * sigma;
                        end
                        
                        % Replace low outliers with lower bound
                        low_outliers = col_data < mu - threshold * sigma & col_outliers;
                        if any(low_outliers)
                            data_out(low_outliers, j) = mu - threshold * sigma;
                        end
                        
                    case 'iqr'
                        % Calculate quartiles and IQR
                        q1 = prctile(col_data, 25);
                        q3 = prctile(col_data, 75);
                        iqr_value = q3 - q1;
                        threshold = options.outlier_threshold;
                        
                        % Replace high outliers with upper bound
                        high_outliers = col_data > q3 + threshold * iqr_value & col_outliers;
                        if any(high_outliers)
                            data_out(high_outliers, j) = q3 + threshold * iqr_value;
                        end
                        
                        % Replace low outliers with lower bound
                        low_outliers = col_data < q1 - threshold * iqr_value & col_outliers;
                        if any(low_outliers)
                            data_out(low_outliers, j) = q1 - threshold * iqr_value;
                        end
                        
                    case 'mad'
                        % Calculate median and MAD
                        med_value = median(col_data, 'omitnan');
                        mad_value = median(abs(col_data - med_value), 'omitnan');
                        threshold = options.outlier_threshold;
                        
                        % Replace high outliers with upper bound
                        high_outliers = col_data > med_value + threshold * mad_value & col_outliers;
                        if any(high_outliers)
                            data_out(high_outliers, j) = med_value + threshold * mad_value;
                        end
                        
                        % Replace low outliers with lower bound
                        low_outliers = col_data < med_value - threshold * mad_value & col_outliers;
                        if any(low_outliers)
                            data_out(low_outliers, j) = med_value - threshold * mad_value;
                        end
                end
            end
        end
        
    otherwise
        error('Unsupported outlier handling method: %s', options.outlier_handling);
end

end


function [data_out, transform_info] = transform_data(data, options)
% TRANSFORM_DATA Applies transformations to improve distributional properties of data
%
% USAGE:
%   [DATA_OUT, TRANSFORM_INFO] = transform_data(DATA, OPTIONS)
%
% INPUTS:
%   DATA     - T by K matrix of cross-sectional data
%   OPTIONS  - Structure with transformation options
%
% OUTPUTS:
%   DATA_OUT       - Transformed data
%   TRANSFORM_INFO - Structure with information about transformations

% Initialize output with input data
data_out = data;

% Initialize transform info structure
transform_info = struct(...
    'method', options.transform, ...
    'parameters', struct());

% Early return if no transformation requested
if strcmpi(options.transform, 'none')
    return;
end

% Apply selected transformation method
switch lower(options.transform)
    case 'log'
        % Natural logarithm transformation
        offset = options.transform_offset;
        transform_info.parameters.offset = offset;
        
        % Check for non-positive values if offset is zero
        if offset == 0 && any(data(:) <= 0)
            error(['Log transformation requires positive data. ' ...
                  'Set OPTIONS.transform_offset to a positive value to handle non-positive data.']);
        end
        
        % Apply transformation
        data_out = log(data + offset);
        transform_info.parameters.applied_offset = offset;
        
    case 'sqrt'
        % Square root transformation
        offset = options.transform_offset;
        transform_info.parameters.offset = offset;
        
        % Check for negative values if offset is zero
        if offset == 0 && any(data(:) < 0)
            error(['Square root transformation requires non-negative data. ' ...
                  'Set OPTIONS.transform_offset to a positive value to handle negative data.']);
        end
        
        % Apply transformation
        data_out = sqrt(data + offset);
        transform_info.parameters.applied_offset = offset;
        
    case 'boxcox'
        % Box-Cox transformation
        lambda = options.transform_lambda;
        transform_info.parameters.lambda = lambda;
        
        % Check for non-positive values (Box-Cox requires positive data)
        if any(data(:) <= 0)
            error(['Box-Cox transformation requires positive data. ' ...
                  'Try applying an offset or using Yeo-Johnson transformation instead.']);
        end
        
        % Estimate optimal lambda if requested
        if options.transform_estimate
            % Optimize lambda for each column separately
            lambda_values = zeros(1, size(data, 2));
            for j = 1:size(data, 2)
                lambda_values(j) = optimize_boxcox(data(:, j));
            end
            lambda = lambda_values;
            transform_info.parameters.lambda = lambda_values;
            transform_info.parameters.lambda_estimated = true;
        end
        
        % Apply transformation column by column
        for j = 1:size(data, 2)
            col_lambda = lambda(min(j, length(lambda)));
            
            if abs(col_lambda) < 1e-10  % Close to zero
                % When lambda is close to zero, use logarithm (limiting case)
                data_out(:, j) = log(data(:, j));
            else
                % Standard Box-Cox formula
                data_out(:, j) = (data(:, j).^col_lambda - 1) / col_lambda;
            end
        end
        
    case 'yj'
        % Yeo-Johnson transformation (can handle negative values)
        lambda = options.transform_lambda;
        transform_info.parameters.lambda = lambda;
        
        % Estimate optimal lambda if requested
        if options.transform_estimate
            % Implementing a simple grid search for optimal lambda
            % For a full implementation, a proper optimization would be better
            lambda_grid = -2:0.1:2;
            best_lambdas = zeros(1, size(data, 2));
            
            for j = 1:size(data, 2)
                col_data = data(:, j);
                best_stat = Inf;
                
                for l = 1:length(lambda_grid)
                    test_lambda = lambda_grid(l);
                    transformed = zeros(size(col_data));
                    
                    % Apply Yeo-Johnson transformation
                    pos_idx = col_data >= 0;
                    neg_idx = ~pos_idx;
                    
                    if abs(test_lambda) < 1e-10  % Close to zero
                        % Limiting case when lambda is close to zero
                        if any(pos_idx)
                            transformed(pos_idx) = log(col_data(pos_idx) + 1);
                        end
                        if any(neg_idx)
                            transformed(neg_idx) = -log(-col_data(neg_idx) + 1);
                        end
                    else
                        % Standard case
                        if any(pos_idx)
                            transformed(pos_idx) = ((col_data(pos_idx) + 1).^test_lambda - 1) / test_lambda;
                        end
                        if any(neg_idx)
                            transformed(neg_idx) = -((-col_data(neg_idx) + 1).^(2-test_lambda) - 1) / (2-test_lambda);
                        end
                    end
                    
                    % Test normality using Jarque-Bera test
                    jb_result = jarque_bera(transformed);
                    if jb_result.statistic < best_stat
                        best_stat = jb_result.statistic;
                        best_lambdas(j) = test_lambda;
                    end
                end
            end
            
            lambda = best_lambdas;
            transform_info.parameters.lambda = best_lambdas;
            transform_info.parameters.lambda_estimated = true;
        end
        
        % Apply transformation column by column
        for j = 1:size(data, 2)
            col_data = data(:, j);
            col_lambda = lambda(min(j, length(lambda)));
            transformed = zeros(size(col_data));
            
            % Separate positive and negative values
            pos_idx = col_data >= 0;
            neg_idx = ~pos_idx;
            
            if abs(col_lambda) < 1e-10  % Close to zero
                % Limiting case when lambda is close to zero
                if any(pos_idx)
                    transformed(pos_idx) = log(col_data(pos_idx) + 1);
                end
                if any(neg_idx)
                    transformed(neg_idx) = -log(-col_data(neg_idx) + 1);
                end
            else
                % Standard case
                if any(pos_idx)
                    transformed(pos_idx) = ((col_data(pos_idx) + 1).^col_lambda - 1) / col_lambda;
                end
                if any(neg_idx)
                    transformed(neg_idx) = -((-col_data(neg_idx) + 1).^(2-col_lambda) - 1) / (2-col_lambda);
                end
            end
            
            data_out(:, j) = transformed;
        end
        
    case 'rank'
        % Rank transformation
        for j = 1:size(data, 2)
            col_data = data(:, j);
            [~, sorted_indices] = sort(col_data);
            ranks = zeros(size(col_data));
            ranks(sorted_indices) = 1:length(col_data);
            
            % Scale ranks to [0,1]
            data_out(:, j) = (ranks - 0.5) / length(col_data);
        end
        
    otherwise
        error('Unsupported transformation method: %s', options.transform);
end

% Calculate normality test statistics for before and after transformation
normality_before = zeros(1, size(data, 2));
normality_after = zeros(1, size(data, 2));
pvalues_before = zeros(1, size(data, 2));
pvalues_after = zeros(1, size(data, 2));

for j = 1:size(data, 2)
    % Test normality before transformation
    jb_before = jarque_bera(data(:, j));
    normality_before(j) = jb_before.statistic;
    pvalues_before(j) = jb_before.pval;
    
    % Test normality after transformation
    jb_after = jarque_bera(data_out(:, j));
    normality_after(j) = jb_after.statistic;
    pvalues_after(j) = jb_after.pval;
end

% Store normality results
transform_info.normality = struct(...
    'jb_before', normality_before, ...
    'jb_after', normality_after, ...
    'pvalue_before', pvalues_before, ...
    'pvalue_after', pvalues_after, ...
    'improvement', normality_before - normality_after);

end


function [data_out, normalize_info] = normalize_data(data, options)
% NORMALIZE_DATA Normalizes data to achieve consistent scale across variables
%
% USAGE:
%   [DATA_OUT, NORMALIZE_INFO] = normalize_data(DATA, OPTIONS)
%
% INPUTS:
%   DATA     - T by K matrix of cross-sectional data
%   OPTIONS  - Structure with normalization options
%
% OUTPUTS:
%   DATA_OUT       - Normalized data
%   NORMALIZE_INFO - Structure with information about normalization

% Initialize output with input data
data_out = data;

% Initialize normalize info structure
normalize_info = struct(...
    'method', options.normalize, ...
    'parameters', struct());

% Early return if no normalization requested
if strcmpi(options.normalize, 'none')
    return;
end

% Apply selected normalization method
switch lower(options.normalize)
    case 'standardize'
        % Standardize to mean 0, std 1: (x - mean) / std
        centers = mean(data, 'omitnan');
        scales = std(data, 'omitnan');
        
        % Handle columns with zero standard deviation
        zero_std_cols = (scales == 0 | isnan(scales));
        if any(zero_std_cols)
            warning(['Columns with zero standard deviation found. ' ...
                     'These columns will not be standardized.']);
            scales(zero_std_cols) = 1;
        end
        
        % Apply normalization
        for j = 1:size(data, 2)
            data_out(:, j) = (data(:, j) - centers(j)) / scales(j);
        end
        
        % Store parameters
        normalize_info.parameters.center = centers;
        normalize_info.parameters.scale = scales;
        
    case 'minmax'
        % Min-max scaling to [0,1]: (x - min) / (max - min)
        min_vals = min(data, [], 'omitnan');
        max_vals = max(data, [], 'omitnan');
        ranges = max_vals - min_vals;
        
        % Handle columns with zero range
        zero_range_cols = (ranges == 0 | isnan(ranges));
        if any(zero_range_cols)
            warning(['Columns with zero range found. ' ...
                     'These columns will be set to 0.5.']);
            ranges(zero_range_cols) = 1;
            
            % Set constant columns to 0.5
            for j = find(zero_range_cols)
                data_out(:, j) = 0.5;
            end
        end
        
        % Apply normalization to non-constant columns
        for j = find(~zero_range_cols)
            data_out(:, j) = (data(:, j) - min_vals(j)) / ranges(j);
        end
        
        % Store parameters
        normalize_info.parameters.min = min_vals;
        normalize_info.parameters.max = max_vals;
        normalize_info.parameters.range = ranges;
        
    case 'robust'
        % Robust scaling using median and MAD: (x - median) / MAD
        centers = median(data, 'omitnan');
        
        % Calculate MAD (Median Absolute Deviation)
        scales = zeros(1, size(data, 2));
        for j = 1:size(data, 2)
            scales(j) = median(abs(data(:, j) - centers(j)), 'omitnan');
        end
        
        % Handle columns with zero MAD
        zero_mad_cols = (scales == 0 | isnan(scales));
        if any(zero_mad_cols)
            warning(['Columns with zero MAD found. ' ...
                     'These columns will not be robustly scaled.']);
            scales(zero_mad_cols) = 1;
        end
        
        % Apply normalization
        for j = 1:size(data, 2)
            data_out(:, j) = (data(:, j) - centers(j)) / scales(j);
        end
        
        % Store parameters
        normalize_info.parameters.center = centers;
        normalize_info.parameters.scale = scales;
        
    case 'decimal'
        % Scale by powers of 10 to achieve consistent decimal places
        scales = zeros(1, size(data, 2));
        
        for j = 1:size(data, 2)
            % Find maximum absolute value
            max_abs = max(abs(data(:, j)), [], 'omitnan');
            
            if max_abs == 0
                scales(j) = 1;
            else
                % Calculate appropriate power of 10
                scales(j) = 10^floor(log10(max_abs));
            end
            
            % Apply scaling
            data_out(:, j) = data(:, j) / scales(j);
        end
        
        % Store parameters
        normalize_info.parameters.scale = scales;
        
    case 'custom'
        % Custom scaling using provided center and scale values
        if ~isfield(options, 'normalize_center') || isempty(options.normalize_center)
            centers = zeros(1, size(data, 2));
        else
            centers = options.normalize_center;
            
            % Ensure correct dimensions
            if length(centers) == 1
                centers = repmat(centers, 1, size(data, 2));
            elseif length(centers) ~= size(data, 2)
                error(['Custom normalization centers must have length 1 or ' ...
                       'match the number of columns in the data.']);
            end
        end
        
        if ~isfield(options, 'normalize_scale') || isempty(options.normalize_scale)
            scales = ones(1, size(data, 2));
        else
            scales = options.normalize_scale;
            
            % Ensure correct dimensions
            if length(scales) == 1
                scales = repmat(scales, 1, size(data, 2));
            elseif length(scales) ~= size(data, 2)
                error(['Custom normalization scales must have length 1 or ' ...
                       'match the number of columns in the data.']);
            end
            
            % Check for zero scales
            if any(scales == 0)
                error('Custom scales cannot be zero.');
            end
        end
        
        % Apply normalization
        for j = 1:size(data, 2)
            data_out(:, j) = (data(:, j) - centers(j)) / scales(j);
        end
        
        % Store parameters
        normalize_info.parameters.center = centers;
        normalize_info.parameters.scale = scales;
        
    otherwise
        error('Unsupported normalization method: %s', options.normalize);
end

% Calculate statistics for before and after normalization
stats_before = struct(...
    'mean', mean(data, 'omitnan'), ...
    'std', std(data, 'omitnan'), ...
    'min', min(data, [], 'omitnan'), ...
    'max', max(data, [], 'omitnan'), ...
    'range', max(data, [], 'omitnan') - min(data, [], 'omitnan'));

stats_after = struct(...
    'mean', mean(data_out, 'omitnan'), ...
    'std', std(data_out, 'omitnan'), ...
    'min', min(data_out, [], 'omitnan'), ...
    'max', max(data_out, [], 'omitnan'), ...
    'range', max(data_out, [], 'omitnan') - min(data_out, [], 'omitnan'));

normalize_info.statistics = struct(...
    'before', stats_before, ...
    'after', stats_after);

end


function outlier_map = detect_outliers_zscore(data, threshold)
% DETECT_OUTLIERS_ZSCORE Detects outliers using Z-score method
%
% USAGE:
%   OUTLIER_MAP = detect_outliers_zscore(DATA, THRESHOLD)
%
% INPUTS:
%   DATA      - T by K matrix of cross-sectional data
%   THRESHOLD - Z-score threshold (default: 3)
%
% OUTPUTS:
%   OUTLIER_MAP - T by K logical matrix with TRUE for outliers

if nargin < 2 || isempty(threshold)
    threshold = 3;
end

% Validate threshold
threshold_options = struct('isscalar', true, 'isPositive', true);
threshold = parametercheck(threshold, 'Z-score threshold', threshold_options);

% Initialize output
outlier_map = false(size(data));

% For each column, calculate Z-scores and identify outliers
for j = 1:size(data, 2)
    col_data = data(:, j);
    
    % Skip columns with all NaN values
    if all(isnan(col_data))
        continue;
    end
    
    % Calculate mean and standard deviation, ignoring NaN values
    mu = mean(col_data, 'omitnan');
    sigma = std(col_data, 'omitnan');
    
    % If standard deviation is zero or NaN, skip this column
    if sigma == 0 || isnan(sigma)
        continue;
    end
    
    % Calculate Z-scores
    z_scores = abs((col_data - mu) / sigma);
    
    % Identify outliers as points with absolute Z-scores exceeding threshold
    outlier_map(:, j) = z_scores > threshold;
end

end


function outlier_map = detect_outliers_iqr(data, multiplier)
% DETECT_OUTLIERS_IQR Detects outliers using Interquartile Range (IQR) method
%
% USAGE:
%   OUTLIER_MAP = detect_outliers_iqr(DATA, MULTIPLIER)
%
% INPUTS:
%   DATA       - T by K matrix of cross-sectional data
%   MULTIPLIER - IQR multiplier (default: 1.5)
%
% OUTPUTS:
%   OUTLIER_MAP - T by K logical matrix with TRUE for outliers

if nargin < 2 || isempty(multiplier)
    multiplier = 1.5;
end

% Validate multiplier
multiplier_options = struct('isscalar', true, 'isPositive', true);
multiplier = parametercheck(multiplier, 'IQR multiplier', multiplier_options);

% Initialize output
outlier_map = false(size(data));

% For each column, calculate IQR and identify outliers
for j = 1:size(data, 2)
    col_data = data(:, j);
    
    % Skip columns with all NaN values
    if all(isnan(col_data))
        continue;
    end
    
    % Calculate quartiles, ignoring NaN values
    q1 = prctile(col_data, 25);
    q3 = prctile(col_data, 75);
    iqr_value = q3 - q1;
    
    % If IQR is zero or NaN, skip this column
    if iqr_value == 0 || isnan(iqr_value)
        continue;
    end
    
    % Calculate upper and lower bounds
    lower_bound = q1 - multiplier * iqr_value;
    upper_bound = q3 + multiplier * iqr_value;
    
    % Identify outliers as points outside the bounds
    outlier_map(:, j) = col_data < lower_bound | col_data > upper_bound;
end

end


function outlier_map = detect_outliers_mad(data, multiplier)
% DETECT_OUTLIERS_MAD Detects outliers using Median Absolute Deviation (MAD) method
%
% USAGE:
%   OUTLIER_MAP = detect_outliers_mad(DATA, MULTIPLIER)
%
% INPUTS:
%   DATA       - T by K matrix of cross-sectional data
%   MULTIPLIER - MAD multiplier (default: 3)
%
% OUTPUTS:
%   OUTLIER_MAP - T by K logical matrix with TRUE for outliers

if nargin < 2 || isempty(multiplier)
    multiplier = 3;
end

% Validate multiplier
multiplier_options = struct('isscalar', true, 'isPositive', true);
multiplier = parametercheck(multiplier, 'MAD multiplier', multiplier_options);

% Initialize output
outlier_map = false(size(data));

% For each column, calculate MAD and identify outliers
for j = 1:size(data, 2)
    col_data = data(:, j);
    
    % Skip columns with all NaN values
    if all(isnan(col_data))
        continue;
    end
    
    % Calculate median, ignoring NaN values
    med_value = median(col_data, 'omitnan');
    
    % Calculate MAD (Median Absolute Deviation)
    mad_value = median(abs(col_data - med_value), 'omitnan');
    
    % If MAD is zero or NaN, skip this column
    if mad_value == 0 || isnan(mad_value)
        continue;
    end
    
    % Scale MAD to make it comparable to standard deviation for normal distribution
    % constant = 1.4826
    mad_scaled = 1.4826 * mad_value;
    
    % Calculate deviations
    deviations = abs(col_data - med_value) / mad_scaled;
    
    % Identify outliers as points with deviations exceeding threshold
    outlier_map(:, j) = deviations > multiplier;
end

end


function data_out = winsorize(data, lower_percentile, upper_percentile)
% WINSORIZE Applies winsorization to data, capping extreme values at percentiles
%
% USAGE:
%   DATA_OUT = winsorize(DATA, LOWER_PERCENTILE, UPPER_PERCENTILE)
%
% INPUTS:
%   DATA             - T by K matrix of cross-sectional data
%   LOWER_PERCENTILE - Lower percentile for winsorization (default: 0.01)
%   UPPER_PERCENTILE - Upper percentile for winsorization (default: 0.99)
%
% OUTPUTS:
%   DATA_OUT - Winsorized data

if nargin < 2 || isempty(lower_percentile)
    lower_percentile = 0.01;
end

if nargin < 3 || isempty(upper_percentile)
    upper_percentile = 0.99;
end

% Validate percentiles
lower_options = struct('isscalar', true, 'lowerBound', 0, 'upperBound', 0.5);
lower_percentile = parametercheck(lower_percentile, 'Lower percentile', lower_options);

upper_options = struct('isscalar', true, 'lowerBound', 0.5, 'upperBound', 1);
upper_percentile = parametercheck(upper_percentile, 'Upper percentile', upper_options);

% Initialize output
data_out = data;

% Apply winsorization column by column
for j = 1:size(data, 2)
    col_data = data(:, j);
    
    % Skip columns with all NaN values
    if all(isnan(col_data))
        continue;
    end
    
    % Calculate percentiles, ignoring NaN values
    lower_bound = prctile(col_data, lower_percentile * 100);
    upper_bound = prctile(col_data, upper_percentile * 100);
    
    % Apply winsorization
    data_out(:, j) = min(max(col_data, lower_bound), upper_bound);
end

end


function optimal_lambda = optimize_boxcox(x)
% OPTIMIZE_BOXCOX Optimizes lambda parameter for Box-Cox transformation
%
% USAGE:
%   OPTIMAL_LAMBDA = optimize_boxcox(X)
%
% INPUTS:
%   X - Data vector for optimization
%
% OUTPUTS:
%   OPTIMAL_LAMBDA - Optimal lambda value for Box-Cox transformation

% Ensure input is a column vector
x = columncheck(x, 'data');

% Remove NaN values for optimization
x = x(~isnan(x));

% Check if data is positive (Box-Cox requires positive data)
if any(x <= 0)
    error('Box-Cox transformation requires positive data.');
end

% Define lambda grid
lambda_grid = -2:0.1:2;
num_lambdas = length(lambda_grid);

% Initialize storage for log-likelihood or normality statistic
normality_stats = zeros(num_lambdas, 1);

% For each lambda value, apply transformation and test normality
for i = 1:num_lambdas
    lambda = lambda_grid(i);
    
    % Apply Box-Cox transformation
    if abs(lambda) < 1e-10  % Close to zero
        transformed = log(x);
    else
        transformed = (x.^lambda - 1) / lambda;
    end
    
    % Calculate log-likelihood (simplified) using Jarque-Bera statistic
    % (Lower JB statistic indicates better normality)
    jb_result = jarque_bera(transformed);
    normality_stats(i) = jb_result.statistic;
end

% Find lambda with minimum normality statistic
[~, min_idx] = min(normality_stats);
optimal_lambda = lambda_grid(min_idx);

end