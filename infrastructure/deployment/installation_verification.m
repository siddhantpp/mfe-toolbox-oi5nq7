function results = verify_installation()
% VERIFY_INSTALLATION Validates the MFE Toolbox installation and functionality
%
% USAGE:
%   results = verify_installation()
%
% OUTPUTS:
%   results - Structure containing detailed verification results with fields:
%       .path_config - Results of path configuration verification
%       .directories - Results of directory structure verification
%       .mex_binaries - Results of MEX binary compatibility check
%       .functions - Results of function availability check
%       .basic_tests - Results of basic functionality tests
%       .overall_success - Boolean indicating overall verification status
%
% COMMENTS:
%   This function performs comprehensive verification of the MFE Toolbox
%   installation. It checks path configuration, directory structure, MEX binary
%   compatibility, function availability, and performs basic functionality tests.
%   The function provides detailed diagnostics and recommendations for resolving
%   any identified issues.
%
% EXAMPLES:
%   % Run complete verification
%   verification_results = verify_installation();
%
%   % Verify installation and display results
%   verify_installation();
%
% See also: ADDTOPATH

% Global constants
TOOLBOX_VERSION = '4.0 (28-Oct-2009)';
REQUIRED_DIRECTORIES = {'bootstrap', 'crosssection', 'distributions', 'GUI', ...
                        'multivariate', 'tests', 'timeseries', 'univariate', ...
                        'utility', 'realized', 'mex_source', 'dlls'};
OPTIONAL_DIRECTORIES = {'duplication'};
REQUIRED_MEX_FILES = {'agarch_core', 'armaxerrors', 'composite_likelihood', ...
                     'egarch_core', 'igarch_core', 'tarch_core'};

% Initialize verification status structure
VERIFICATION_STATUS = struct();

% Display welcome message
disp('======================================================');
disp(['MFE Toolbox Installation Verification - Version ' TOOLBOX_VERSION]);
disp('======================================================');

% Check MATLAB version compatibility
matlab_version = ver('MATLAB');
disp(['MATLAB Version: ' matlab_version.Version ' ' matlab_version.Release]);

% Run verification steps
VERIFICATION_STATUS.path_config = check_path_configuration();
VERIFICATION_STATUS.directories = check_required_directories(REQUIRED_DIRECTORIES, OPTIONAL_DIRECTORIES);
VERIFICATION_STATUS.mex_binaries = check_mex_binaries(REQUIRED_MEX_FILES);
VERIFICATION_STATUS.functions = check_function_availability();
VERIFICATION_STATUS.basic_tests = run_basic_tests();

% Set overall verification status
VERIFICATION_STATUS.overall_success = VERIFICATION_STATUS.path_config.success && ...
                                      VERIFICATION_STATUS.directories.all_required_found && ...
                                      VERIFICATION_STATUS.mex_binaries.all_found && ...
                                      VERIFICATION_STATUS.functions.all_available && ...
                                      VERIFICATION_STATUS.basic_tests.all_passed;

% Display comprehensive verification report
display_verification_report(VERIFICATION_STATUS);

% Return results if requested
if nargout > 0
    results = VERIFICATION_STATUS;
end
end

function result = check_path_configuration()
% CHECK_PATH_CONFIGURATION Verifies that the toolbox path is properly configured
% and attempts to fix if needed

result = struct('success', false, 'details', {{}});

try
    % Check if addToPath.m is in the MATLAB path
    add_to_path_exists = (exist('addToPath', 'file') == 2);
    
    if ~add_to_path_exists
        result.details{end+1} = 'WARNING: addToPath.m not found in MATLAB path.';
        
        % Try to find MFE Toolbox root directory
        potential_roots = find_mfe_root();
        
        if ~isempty(potential_roots)
            result.details{end+1} = ['Potential MFE Toolbox root found at: ' potential_roots{1}];
            try
                % Try to add the potential root to the path
                addpath(potential_roots{1});
                
                % Check again for addToPath
                if exist('addToPath', 'file') == 2
                    result.details{end+1} = 'Successfully added MFE Toolbox root to path.';
                    add_to_path_exists = true;
                else
                    % Try to find addToPath.m directly
                    backend_dir = fullfile(potential_roots{1}, 'src', 'backend');
                    if exist(backend_dir, 'dir')
                        addpath(backend_dir);
                        if exist('addToPath', 'file') == 2
                            result.details{end+1} = 'Successfully added backend directory to path.';
                            add_to_path_exists = true;
                        end
                    end
                end
            catch
                result.details{end+1} = 'Failed to add potential MFE Toolbox root to path.';
            end
        else
            result.details{end+1} = 'Could not locate MFE Toolbox root directory.';
        end
    else
        result.details{end+1} = 'addToPath.m found in MATLAB path.';
    end
    
    % Attempt to fix path issues using addToPath if available
    if add_to_path_exists
        result.details{end+1} = 'Running addToPath() to ensure correct path configuration...';
        fix_success = addToPath(false, true);
        
        if fix_success
            result.details{end+1} = 'Path configuration fixed successfully.';
            result.success = true;
        else
            result.details{end+1} = 'Failed to fix path configuration using addToPath().';
        end
    else
        result.details{end+1} = 'Cannot fix path: addToPath.m not available.';
    end
    
catch err
    result.details{end+1} = ['Error checking path configuration: ' err.message];
    result.success = false;
end

% Final verification that key directories are in the path
if result.success
    % Check that a few key directories are in the path as final verification
    key_dirs = {'distributions', 'univariate', 'timeseries'};
    all_in_path = true;
    
    for i = 1:length(key_dirs)
        % Look for a known function in each key directory
        switch key_dirs{i}
            case 'distributions'
                func_to_check = 'stdtpdf';
            case 'univariate'
                func_to_check = 'garchfit';
            case 'timeseries'
                func_to_check = 'armaxfilter';
            otherwise
                func_to_check = '';
        end
        
        if ~isempty(func_to_check) && exist(func_to_check, 'file') ~= 2
            all_in_path = false;
            result.details{end+1} = ['WARNING: Key function ' func_to_check ' not found in path.'];
        end
    end
    
    if ~all_in_path
        result.success = false;
        result.details{end+1} = 'Some key directories are still not properly in the MATLAB path.';
    else
        result.details{end+1} = 'All key directories verified in MATLAB path.';
    end
end
end

function roots = find_mfe_root()
% Helper function to find potential MFE Toolbox root directories

roots = {};
current_dir = pwd;

% Check current directory
if is_mfe_root(current_dir)
    roots{end+1} = current_dir;
    return;
end

% Check parent directories (up to 3 levels)
for i = 1:3
    current_dir = fileparts(current_dir);
    if isempty(current_dir)
        break;
    end
    
    if is_mfe_root(current_dir)
        roots{end+1} = current_dir;
        return;
    end
end

% Check if we're in a subdirectory of the MFE Toolbox
original_dir = pwd;
dir_parts = strsplit(original_dir, filesep);

for i = length(dir_parts):-1:2
    test_dir = fullfile(dir_parts{1:i});
    if is_mfe_root(test_dir)
        roots{end+1} = test_dir;
        return;
    end
end
end

function is_root = is_mfe_root(directory)
% Helper function to check if a directory is the MFE Toolbox root

is_root = false;

% Check for multiple key directories that would indicate MFE Toolbox root
key_dirs = {'distributions', 'univariate', 'timeseries', 'bootstrap'};
count = 0;

for i = 1:length(key_dirs)
    if exist(fullfile(directory, key_dirs{i}), 'dir')
        count = count + 1;
    end
end

% If at least 3 key directories exist, consider it the MFE root
if count >= 3
    is_root = true;
end

% Also check for Contents.m which often exists at the root
if exist(fullfile(directory, 'Contents.m'), 'file')
    contents = fileread(fullfile(directory, 'Contents.m'));
    if contains(contents, 'MFE') && contains(contents, 'Toolbox')
        is_root = true;
    end
end
end

function result = check_required_directories(required_dirs, optional_dirs)
% CHECK_REQUIRED_DIRECTORIES Verifies that all required subdirectories exist

result = struct('all_required_found', true, 'missing_required', {{}}, ...
                'all_optional_found', true, 'missing_optional', {{}});

try
    % Try to determine MFE Toolbox root directory
    mfe_root = '';
    
    % Method 1: If addToPath.m exists, use its location
    if exist('addToPath', 'file') == 2
        add_to_path_info = which('addToPath');
        if ~isempty(add_to_path_info)
            mfe_root = fileparts(add_to_path_info);
            % If we're in src/backend, move up to the root
            if contains(lower(mfe_root), 'backend') || contains(lower(mfe_root), 'src')
                mfe_root = fileparts(fileparts(mfe_root));
            end
        end
    end
    
    % Method 2: Look for Contents.m in MATLAB path
    if isempty(mfe_root)
        contents_info = which('Contents.m');
        if ~isempty(contents_info)
            % Read Contents.m to check if it's MFE Toolbox
            contents = fileread(contents_info);
            if contains(contents, 'MFE') && contains(contents, 'Toolbox')
                mfe_root = fileparts(contents_info);
            end
        end
    end
    
    % Method 3: Look for key directories in potential locations
    if isempty(mfe_root)
        roots = find_mfe_root();
        if ~isempty(roots)
            mfe_root = roots{1};
        end
    end
    
    if isempty(mfe_root)
        result.all_required_found = false;
        result.details = 'Could not determine MFE Toolbox root directory.';
        return;
    end
    
    % Check required directories
    for i = 1:length(required_dirs)
        dir_path = fullfile(mfe_root, required_dirs{i});
        if ~exist(dir_path, 'dir')
            result.all_required_found = false;
            result.missing_required{end+1} = required_dirs{i};
        end
    end
    
    % Check optional directories
    for i = 1:length(optional_dirs)
        dir_path = fullfile(mfe_root, optional_dirs{i});
        if ~exist(dir_path, 'dir')
            result.all_optional_found = false;
            result.missing_optional{end+1} = optional_dirs{i};
        end
    end
    
    % Add details about the root directory found
    result.mfe_root = mfe_root;
    
catch err
    result.all_required_found = false;
    result.details = ['Error checking directories: ' err.message];
end
end

function result = check_mex_binaries(required_mex_files)
% CHECK_MEX_BINARIES Checks for platform-appropriate MEX binaries

result = struct('all_found', true, 'missing_binaries', {{}}, 'platform', '');

try
    % Determine platform for MEX extension
    if ispc()
        mex_ext = '.mexw64';
        result.platform = 'Windows';
    else
        mex_ext = '.mexa64';
        result.platform = 'Unix';
    end
    
    % Get more detailed platform info
    platform_info = computer();
    result.platform_details = platform_info;
    
    % Check each required MEX binary
    for i = 1:length(required_mex_files)
        mex_name = [required_mex_files{i} mex_ext];
        
        % Check if MEX file exists in path
        if exist(mex_name, 'file') ~= 3 % 3 means MEX-file
            result.all_found = false;
            result.missing_binaries{end+1} = mex_name;
        end
    end
    
    % Add additional information about platform compatibility
    if ~result.all_found
        % Try to find the MEX source files
        mex_source_found = false;
        if exist('mex_source', 'dir') == 7
            mex_source_found = true;
        else
            % Try to locate mex_source directory
            sources_path = which('agarch_core.c');
            if ~isempty(sources_path)
                mex_source_found = true;
            end
        end
        
        if mex_source_found
            result.resolution_hint = 'MEX source files found. You might need to compile MEX binaries for your platform.';
        else
            result.resolution_hint = 'MEX source files not found in path. Ensure the complete MFE Toolbox is installed.';
        end
    end
    
catch err
    result.all_found = false;
    result.details = ['Error checking MEX binaries: ' err.message];
end
end

function result = check_function_availability()
% CHECK_FUNCTION_AVAILABILITY Verifies that essential functions from each module are available

result = struct('all_available', true, 'missing_functions', {{}}, 'module_status', struct());

try
    % Define essential functions to check from each module
    modules = struct();
    
    % Core statistical functions
    modules.distributions = {'stdtpdf', 'stdtcdf', 'stdtrnd', 'gedpdf', 'gedcdf', 'skewtrnd'};
    
    % Time series components
    modules.timeseries = {'armaxfilter', 'autocorr', 'sacf', 'spacf'};
    
    % Volatility models
    modules.univariate = {'garchfit', 'aparchfit', 'egarchfit', 'garchinfer'};
    
    % Bootstrap methods
    modules.bootstrap = {'block_bootstrap', 'stationary_bootstrap'};
    
    % Utility functions
    modules.utility = {'columncheck', 'datacheck', 'isscalar'};
    
    % Check each module
    module_names = fieldnames(modules);
    for i = 1:length(module_names)
        module_name = module_names{i};
        result.module_status.(module_name) = struct('all_found', true, 'missing', {{}});
        
        % Check each function in the module
        for j = 1:length(modules.(module_name))
            func_name = modules.(module_name){j};
            
            % Check if function exists in MATLAB path
            if exist(func_name, 'file') ~= 2 % 2 means M-file or function
                result.all_available = false;
                result.module_status.(module_name).all_found = false;
                result.module_status.(module_name).missing{end+1} = func_name;
                result.missing_functions{end+1} = [module_name '/' func_name];
            else
                % Check if function is in the correct path
                func_path = which(func_name);
                if ~isempty(func_path)
                    % Extract directory name
                    [func_dir, ~, ~] = fileparts(func_path);
                    [~, dir_name, ~] = fileparts(func_dir);
                    
                    % Basic check that function is in an expected directory
                    % This is a simple heuristic and might need refinement
                    if ~strcmpi(dir_name, module_name) && ...
                       ~strcmpi(dir_name, 'duplication') && ...
                       ~contains(lower(func_dir), lower(module_name))
                        result.module_status.(module_name).warnings{end+1} = ...
                            [func_name ' found in unexpected location: ' func_dir];
                    end
                end
            end
        end
    end
    
catch err
    result.all_available = false;
    result.details = ['Error checking function availability: ' err.message];
end
end

function result = run_basic_tests()
% RUN_BASIC_TESTS Executes basic functionality tests to verify core components

result = struct('all_passed', true, 'failed_tests', {{}}, 'test_details', struct());

try
    % Initialize test details
    test_categories = {'distributions', 'timeseries', 'estimations', 'utility'};
    for i = 1:length(test_categories)
        result.test_details.(test_categories{i}) = struct('passed', true, 'failures', {{}}, 'details', {{}});
    end
    
    % Test 1: Distribution functions
    try
        % Test standard T distribution functions
        x = linspace(-4, 4, 100);
        nu = 5;
        
        % PDF computation
        pdf_vals = stdtpdf(x, nu);
        if ~all(isfinite(pdf_vals)) || any(pdf_vals < 0)
            throw(MException('MFE:TestFailed', 'stdtpdf returned invalid values'));
        end
        
        % CDF computation
        cdf_vals = stdtcdf(x, nu);
        if ~all(isfinite(cdf_vals)) || any(cdf_vals < 0) || any(cdf_vals > 1)
            throw(MException('MFE:TestFailed', 'stdtcdf returned invalid values'));
        end
        
        % Random number generation
        rnd_vals = stdtrnd(nu, [100, 1]);
        if ~all(isfinite(rnd_vals))
            throw(MException('MFE:TestFailed', 'stdtrnd returned invalid values'));
        end
        
        result.test_details.distributions.details{end+1} = 'Standard T distribution functions tested successfully';
        
    catch err
        result.all_passed = false;
        result.test_details.distributions.passed = false;
        result.test_details.distributions.failures{end+1} = 'Distribution functions';
        result.test_details.distributions.details{end+1} = ['Error: ' err.message];
        result.failed_tests{end+1} = 'Distribution functions';
    end
    
    % Test 2: Time series functions
    try
        % Generate random time series
        T = 100;
        y = randn(T, 1);
        
        % Test ACF computation
        acf_vals = sacf(y, 10);
        if ~all(isfinite(acf_vals))
            throw(MException('MFE:TestFailed', 'sacf returned invalid values'));
        end
        
        % Test PACF computation
        pacf_vals = spacf(y, 10);
        if ~all(isfinite(pacf_vals))
            throw(MException('MFE:TestFailed', 'spacf returned invalid values'));
        end
        
        result.test_details.timeseries.details{end+1} = 'Time series functions tested successfully';
        
    catch err
        result.all_passed = false;
        result.test_details.timeseries.passed = false;
        result.test_details.timeseries.failures{end+1} = 'Time series functions';
        result.test_details.timeseries.details{end+1} = ['Error: ' err.message];
        result.failed_tests{end+1} = 'Time series functions';
    end
    
    % Test 3: Estimation functions
    try
        % Test if we can run a simple ARMA model estimation
        if exist('armaxfilter', 'file') == 2
            % Generate AR(1) process
            T = 200;
            e = randn(T, 1);
            y = zeros(T, 1);
            phi = 0.7;
            
            for t = 2:T
                y(t) = phi * y(t-1) + e(t);
            end
            
            % Fit AR(1) model
            [parameters, ~, ~, ~, ~, ~] = armaxfilter(y, 1, 0);
            
            % Check if estimation is reasonable
            if ~isfinite(parameters) || abs(parameters(1) - phi) > 0.3
                throw(MException('MFE:TestFailed', 'armaxfilter estimation unreliable'));
            end
            
            result.test_details.estimations.details{end+1} = 'Basic ARMA estimation tested successfully';
        else
            result.test_details.estimations.details{end+1} = 'Skipping ARMA estimation test (armaxfilter not found)';
        end
        
    catch err
        result.all_passed = false;
        result.test_details.estimations.passed = false;
        result.test_details.estimations.failures{end+1} = 'Estimation functions';
        result.test_details.estimations.details{end+1} = ['Error: ' err.message];
        result.failed_tests{end+1} = 'Estimation functions';
    end
    
    % Test 4: Utility functions
    try
        % Test columncheck
        if exist('columncheck', 'file') == 2
            x = randn(10, 2);
            [checked_x, k] = columncheck(x);
            
            if k ~= 2 || ~isequal(size(checked_x), size(x))
                throw(MException('MFE:TestFailed', 'columncheck returned unexpected results'));
            end
            
            result.test_details.utility.details{end+1} = 'Utility function columncheck tested successfully';
        else
            result.test_details.utility.details{end+1} = 'Skipping columncheck test (function not found)';
        end
        
        % Test datacheck
        if exist('datacheck', 'file') == 2
            x = randn(10, 2);
            [checked_x, k, T] = datacheck(x, 1);
            
            if k ~= 2 || T ~= 10
                throw(MException('MFE:TestFailed', 'datacheck returned unexpected results'));
            end
            
            result.test_details.utility.details{end+1} = 'Utility function datacheck tested successfully';
        else
            result.test_details.utility.details{end+1} = 'Skipping datacheck test (function not found)';
        end
        
    catch err
        result.all_passed = false;
        result.test_details.utility.passed = false;
        result.test_details.utility.failures{end+1} = 'Utility functions';
        result.test_details.utility.details{end+1} = ['Error: ' err.message];
        result.failed_tests{end+1} = 'Utility functions';
    end
    
catch err
    result.all_passed = false;
    result.details = ['Error running basic tests: ' err.message];
end
end

function display_verification_report(verification_results)
% DISPLAY_VERIFICATION_REPORT Displays a comprehensive verification report

disp('======================================================');
disp('MFE TOOLBOX INSTALLATION VERIFICATION REPORT');
disp('======================================================');

% 1. Path Configuration
disp('');
disp('1. PATH CONFIGURATION');
disp('---------------------');
if verification_results.path_config.success
    disp('STATUS: [PASS] Path configuration correct');
else
    disp('STATUS: [FAIL] Path configuration issues detected');
end

for i = 1:length(verification_results.path_config.details)
    disp(['  - ' verification_results.path_config.details{i}]);
end

% 2. Directory Structure
disp('');
disp('2. DIRECTORY STRUCTURE');
disp('----------------------');
if verification_results.directories.all_required_found
    disp('STATUS: [PASS] All required directories found');
else
    disp('STATUS: [FAIL] Missing required directories');
    
    for i = 1:length(verification_results.directories.missing_required)
        disp(['  - Missing: ' verification_results.directories.missing_required{i}]);
    end
end

if isfield(verification_results.directories, 'mfe_root')
    disp(['  - MFE Toolbox Root: ' verification_results.directories.mfe_root]);
end

if ~verification_results.directories.all_optional_found
    disp('  - Note: Some optional directories are missing:');
    for i = 1:length(verification_results.directories.missing_optional)
        disp(['    * ' verification_results.directories.missing_optional{i}]);
    end
end

% 3. MEX Binaries
disp('');
disp('3. MEX BINARY COMPATIBILITY');
disp('--------------------------');
if verification_results.mex_binaries.all_found
    disp(['STATUS: [PASS] All required MEX binaries found for ' ...
          verification_results.mex_binaries.platform]);
else
    disp(['STATUS: [FAIL] Missing MEX binaries for ' ...
          verification_results.mex_binaries.platform]);
    
    for i = 1:length(verification_results.mex_binaries.missing_binaries)
        disp(['  - Missing: ' verification_results.mex_binaries.missing_binaries{i}]);
    end
    
    if isfield(verification_results.mex_binaries, 'resolution_hint')
        disp(['  - Hint: ' verification_results.mex_binaries.resolution_hint]);
    end
end

if isfield(verification_results.mex_binaries, 'platform_details')
    disp(['  - Platform details: ' verification_results.mex_binaries.platform_details]);
end

% 4. Function Availability
disp('');
disp('4. FUNCTION AVAILABILITY');
disp('-----------------------');
if verification_results.functions.all_available
    disp('STATUS: [PASS] All essential functions are available');
else
    disp('STATUS: [FAIL] Some essential functions are missing');
    
    for i = 1:length(verification_results.functions.missing_functions)
        disp(['  - Missing: ' verification_results.functions.missing_functions{i}]);
    end
end

% Display module-specific warnings if any
module_names = fieldnames(verification_results.functions.module_status);
for i = 1:length(module_names)
    module = module_names{i};
    module_status = verification_results.functions.module_status.(module);
    
    if isfield(module_status, 'warnings') && ~isempty(module_status.warnings)
        disp(['  - Warnings for ' module ' module:']);
        for j = 1:length(module_status.warnings)
            disp(['    * ' module_status.warnings{j}]);
        end
    end
end

% 5. Basic Tests
disp('');
disp('5. BASIC FUNCTIONALITY TESTS');
disp('---------------------------');
if verification_results.basic_tests.all_passed
    disp('STATUS: [PASS] All basic functionality tests passed');
else
    disp('STATUS: [FAIL] Some functionality tests failed');
    
    for i = 1:length(verification_results.basic_tests.failed_tests)
        disp(['  - Failed: ' verification_results.basic_tests.failed_tests{i}]);
    end
end

% Display test details
test_categories = fieldnames(verification_results.basic_tests.test_details);
for i = 1:length(test_categories)
    category = test_categories{i};
    test_status = verification_results.basic_tests.test_details.(category);
    
    if test_status.passed
        disp(['  - ' category ': [PASS]']);
    else
        disp(['  - ' category ': [FAIL]']);
    end
    
    % Show detailed messages
    for j = 1:length(test_status.details)
        disp(['    * ' test_status.details{j}]);
    end
end

% Overall Status
disp('');
disp('======================================================');
disp('OVERALL VERIFICATION STATUS');
disp('======================================================');
if verification_results.overall_success
    disp('STATUS: [PASS] MFE Toolbox installation verified successfully');
else
    disp('STATUS: [FAIL] Issues detected with MFE Toolbox installation');
    
    % Provide recommendations
    disp('');
    disp('RECOMMENDATIONS:');
    
    if ~verification_results.path_config.success
        disp('  1. Run addToPath() to fix path configuration issues');
        disp('     and consider saving the path permanently.');
    end
    
    if ~verification_results.directories.all_required_found
        disp('  2. Ensure you have the complete MFE Toolbox package');
        disp('     with all required directories.');
    end
    
    if ~verification_results.mex_binaries.all_found
        disp(['  3. Compile or obtain the appropriate MEX binaries for your ' ...
              verification_results.mex_binaries.platform ' platform.']);
    end
    
    if ~verification_results.functions.all_available
        disp('  4. Check that all toolbox modules are properly installed');
        disp('     and added to the MATLAB path.');
    end
    
    if ~verification_results.basic_tests.all_passed
        disp('  5. Investigate functionality test failures, which may');
        disp('     indicate installation or compatibility issues.');
    end
end

disp('======================================================');
end