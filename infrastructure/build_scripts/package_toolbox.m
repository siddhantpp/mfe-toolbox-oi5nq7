function package_path = package_toolbox(compile_mex)
% PACKAGE_TOOLBOX Packages the MFE Toolbox into a distributable ZIP archive
%
% USAGE:
%   package_path = package_toolbox()
%   package_path = package_toolbox(compile_mex)
%
% INPUTS:
%   compile_mex    - [optional] Logical indicating whether to compile MEX files
%                    (Default: false)
%
% OUTPUTS:
%   package_path   - Full path to the generated package
%
% DESCRIPTION:
%   This function automates the MFE Toolbox package creation process including:
%   - Creating the proper directory structure
%   - Compiling MEX binaries for the current platform (if requested)
%   - Copying source files to appropriate directories
%   - Creating a ZIP archive for distribution
%
%   Package includes all required directories:
%   - bootstrap
%   - crosssection
%   - distributions
%   - GUI
%   - multivariate
%   - tests
%   - timeseries
%   - univariate
%   - utility
%   - realized
%   - mex_source
%   - dlls (platform-specific MEX binaries)
%
%   And optional directories:
%   - duplication (work-alike functions)
%
% MFE Toolbox v4.0 (28-Oct-2009)

% Set default parameters if not provided
if nargin < 1 || isempty(compile_mex)
    compile_mex = false;
end

% Generate timestamp for build identification
build_timestamp = clock;
timestamp_str = sprintf('%04d-%02d-%02d %02d:%02d:%02d', ...
                      build_timestamp(1), build_timestamp(2), ...
                      build_timestamp(3), build_timestamp(4), ...
                      build_timestamp(5), floor(build_timestamp(6)));
fprintf('Starting MFE Toolbox packaging at %s\n', timestamp_str);

% Package name and version
PACKAGE_NAME = 'MFEToolbox.zip';
TOOLBOX_VERSION = '4.0'; % Extract from Contents.m if needed

% Get script location and determine paths more robustly
script_dir = fileparts(mfilename('fullpath'));
src_dir = fullfile(fileparts(fileparts(script_dir)), 'src');
build_dir = fullfile(script_dir, 'MFEToolbox_build');

% Verify source directory exists
if ~exist(src_dir, 'dir')
    error('Source directory not found: %s\nPlease ensure this script is located in the infrastructure/build_scripts directory.', src_dir);
end

% Create temporary build directory for packaging
if exist(build_dir, 'dir')
    fprintf('Removing existing build directory...\n');
    if ~rmdir(build_dir, 's')
        error('Failed to remove existing build directory: %s', build_dir);
    end
end

try
    % Create directory structure
    fprintf('Creating package directory structure...\n');
    create_package_structure(build_dir);
    
    % Compile MEX files if requested
    if compile_mex
        fprintf('Compiling MEX files for current platform...\n');
        compile_success = compile_mex_files();
        
        if ~compile_success
            warning('Failed to compile MEX files. Package will include existing MEX binaries only.');
        else
            fprintf('MEX compilation successful.\n');
        end
    end
    
    % Copy MATLAB files to package structure
    fprintf('Copying MATLAB source files...\n');
    num_files = copy_matlab_files(build_dir, src_dir);
    fprintf('Copied %d MATLAB source files to package structure.\n', num_files);
    
    % Copy MEX binaries to dlls directory
    fprintf('Copying MEX binaries...\n');
    num_mex = copy_mex_binaries(build_dir, src_dir);
    fprintf('Copied %d MEX binaries to package structure.\n', num_mex);
    
    % Create ZIP archive
    fprintf('Creating final package...\n');
    package_path = create_zip_package(build_dir, script_dir);
    
    % Clean up
    fprintf('Cleaning up temporary files...\n');
    cleanup_success = cleanup_build_directory(build_dir);
    if ~cleanup_success
        warning('Failed to clean up build directory: %s\nYou may need to remove it manually.', build_dir);
    end
    
    % Display completion message
    fprintf('\nMFE Toolbox v%s package created successfully!\n', TOOLBOX_VERSION);
    fprintf('Package location: %s\n', package_path);
    
    % Final timestamp
    end_timestamp = clock;
    elapsed_time = etime(end_timestamp, build_timestamp);
    fprintf('Build completed at %s (%.1f seconds)\n', ...
            datestr(now, 'yyyy-mm-dd HH:MM:SS'), elapsed_time);
    
catch err
    % Ensure we handle errors gracefully
    fprintf('\nError during package creation: %s\n', err.message);
    
    % Try to clean up build directory if it exists
    if exist('build_dir', 'var') && exist(build_dir, 'dir')
        fprintf('Attempting to clean up build directory...\n');
        try
            rmdir(build_dir, 's');
        catch cleanup_err
            warning('Failed to clean up build directory: %s', cleanup_err.message);
        end
    end
    
    % Re-throw the error
    rethrow(err);
end

end

function success = compile_mex_files()
% Compiles MEX files for the current platform using appropriate script

success = false;

try
    % Determine current platform
    if ispc()
        % Windows platform
        script_dir = fileparts(mfilename('fullpath'));
        win_script = fullfile(script_dir, 'compile_mex_windows.bat');
        
        if exist(win_script, 'file')
            fprintf('Executing Windows MEX compilation script: %s\n', win_script);
            [status, result] = system(['"', win_script, '"']);
            if status ~= 0
                fprintf('MEX compilation output:\n%s\n', result);
                warning('MEX compilation failed with status %d', status);
                return;
            end
            fprintf('MEX compilation output:\n%s\n', result);
        else
            warning('Windows MEX compilation script not found: %s', win_script);
            return;
        end
    else
        % Unix platform
        script_dir = fileparts(mfilename('fullpath'));
        unix_script = fullfile(script_dir, 'compile_mex_unix.sh');
        
        if exist(unix_script, 'file')
            fprintf('Executing Unix MEX compilation script: %s\n', unix_script);
            % Make sure the script is executable
            system(['chmod +x "', unix_script, '"']);
            [status, result] = system(['"', unix_script, '"']);
            if status ~= 0
                fprintf('MEX compilation output:\n%s\n', result);
                warning('MEX compilation failed with status %d', status);
                return;
            end
            fprintf('MEX compilation output:\n%s\n', result);
        else
            warning('Unix MEX compilation script not found: %s', unix_script);
            return;
        end
    end
    
    % Check if MEX files were created - this is now done by the compilation scripts
    success = true;
    
catch err
    warning('Error during MEX compilation: %s', err.message);
    success = false;
end
end

function create_package_structure(build_dir)
% Creates the directory structure for the MFE Toolbox package

% Required directories
REQUIRED_DIRECTORIES = {'bootstrap', 'crosssection', 'distributions', 'GUI', ...
                      'multivariate', 'tests', 'timeseries', 'univariate', ...
                      'utility', 'realized', 'mex_source', 'dlls'};

% Create main build directory if it doesn't exist
if ~exist(build_dir, 'dir')
    if ~mkdir(build_dir)
        error('Failed to create build directory: %s', build_dir);
    end
end

% Create all required directories
created_dirs = 0;
for i = 1:length(REQUIRED_DIRECTORIES)
    dir_path = fullfile(build_dir, REQUIRED_DIRECTORIES{i});
    if ~exist(dir_path, 'dir')
        if ~mkdir(dir_path)
            error('Failed to create directory: %s', dir_path);
        end
        created_dirs = created_dirs + 1;
    end
end

fprintf('Created package directory structure with %d directories.\n', created_dirs);
end

function num_files = copy_matlab_files(build_dir, source_dir)
% Copies all MATLAB source files to the package structure

% Required directories
REQUIRED_DIRECTORIES = {'bootstrap', 'crosssection', 'distributions', 'GUI', ...
                      'multivariate', 'tests', 'timeseries', 'univariate', ...
                      'utility', 'realized', 'mex_source'};

% Initialize file counter
num_files = 0;

% Process each directory
for i = 1:length(REQUIRED_DIRECTORIES)
    dir_name = REQUIRED_DIRECTORIES{i};
    source_path = fullfile(source_dir, 'backend', dir_name);
    dest_path = fullfile(build_dir, dir_name);
    
    if exist(source_path, 'dir')
        % Get all .m files in the directory
        m_files = dir(fullfile(source_path, '*.m'));
        
        % Copy each file to the build directory
        for j = 1:length(m_files)
            copyfile(fullfile(source_path, m_files(j).name), dest_path);
            num_files = num_files + 1;
        end
        
        % Also copy .fig files for GUI directory
        if strcmp(dir_name, 'GUI')
            fig_files = dir(fullfile(source_path, '*.fig'));
            for j = 1:length(fig_files)
                copyfile(fullfile(source_path, fig_files(j).name), dest_path);
                num_files = num_files + 1;
            end
        end
        
        % Copy C source and header files for mex_source directory
        if strcmp(dir_name, 'mex_source')
            % Copy header files
            h_files = dir(fullfile(source_path, '*.h'));
            for j = 1:length(h_files)
                copyfile(fullfile(source_path, h_files(j).name), dest_path);
                num_files = num_files + 1;
            end
            
            % Copy C source files
            c_files = dir(fullfile(source_path, '*.c'));
            for j = 1:length(c_files)
                copyfile(fullfile(source_path, c_files(j).name), dest_path);
                num_files = num_files + 1;
            end
        end
    else
        fprintf('Note: Source directory not found: %s\n', source_path);
    end
end

% Copy optional directories if they exist
OPTIONAL_DIRECTORIES = {'duplication'};
for i = 1:length(OPTIONAL_DIRECTORIES)
    dir_name = OPTIONAL_DIRECTORIES{i};
    source_path = fullfile(source_dir, 'backend', dir_name);
    
    if exist(source_path, 'dir')
        % Create directory in build dir
        dest_path = fullfile(build_dir, dir_name);
        if ~exist(dest_path, 'dir')
            mkdir(dest_path);
        end
        
        % Get all .m files in the directory
        m_files = dir(fullfile(source_path, '*.m'));
        
        % Copy each file to the build directory
        for j = 1:length(m_files)
            copyfile(fullfile(source_path, m_files(j).name), dest_path);
            num_files = num_files + 1;
        end
        
        fprintf('Added optional directory: %s\n', dir_name);
    end
end

% Copy core files to the root
core_files = {'addToPath.m', 'Contents.m'};
for i = 1:length(core_files)
    source_file = fullfile(source_dir, 'backend', core_files{i});
    if exist(source_file, 'file')
        copyfile(source_file, build_dir);
        num_files = num_files + 1;
    else
        warning('Core file not found: %s', source_file);
    end
end

end

function num_mex = copy_mex_binaries(build_dir, source_dir)
% Copies compiled MEX binaries to the package structure

% Initialize file counter
num_mex = 0;

% Determine current platform
if ispc()
    % Windows platform - look for .mexw64 files
    mex_dir = fullfile(source_dir, 'backend', 'dlls');
    if exist(mex_dir, 'dir')
        mex_files = dir(fullfile(mex_dir, '*.mexw64'));
        dest_dir = fullfile(build_dir, 'dlls');
        
        % Copy each MEX file to the build directory
        for j = 1:length(mex_files)
            source_file = fullfile(mex_dir, mex_files(j).name);
            dest_file = fullfile(dest_dir, mex_files(j).name);
            copyfile(source_file, dest_file);
            
            % Verify copy was successful
            if exist(dest_file, 'file')
                num_mex = num_mex + 1;
                fprintf('Copied MEX binary: %s\n', mex_files(j).name);
            else
                warning('Failed to copy MEX file: %s', mex_files(j).name);
            end
        end
    else
        fprintf('Note: Windows MEX directory not found: %s\n', mex_dir);
    end
else
    % Unix platform - look for .mexa64 files
    mex_dir = fullfile(source_dir, 'backend', 'dlls');
    if exist(mex_dir, 'dir')
        mex_files = dir(fullfile(mex_dir, '*.mexa64'));
        dest_dir = fullfile(build_dir, 'dlls');
        
        % Copy each MEX file to the build directory
        for j = 1:length(mex_files)
            source_file = fullfile(mex_dir, mex_files(j).name);
            dest_file = fullfile(dest_dir, mex_files(j).name);
            copyfile(source_file, dest_file);
            
            % Verify copy was successful
            if exist(dest_file, 'file')
                num_mex = num_mex + 1;
                fprintf('Copied MEX binary: %s\n', mex_files(j).name);
            else
                warning('Failed to copy MEX file: %s', mex_files(j).name);
            end
        end
    else
        fprintf('Note: Unix MEX directory not found: %s\n', mex_dir);
    end
end

if num_mex == 0
    warning('No MEX binaries found to copy. The package may not function properly.');
end
end

function package_path = create_zip_package(build_dir, output_dir)
% Creates a ZIP archive of the package structure

% Define output filename
package_name = 'MFEToolbox.zip';
package_path = fullfile(output_dir, package_name);

% Remove existing zip file if it exists
if exist(package_path, 'file')
    fprintf('Removing existing package file: %s\n', package_path);
    delete(package_path);
    
    if exist(package_path, 'file')
        error('Failed to remove existing package file: %s', package_path);
    end
end

% Create ZIP archive
fprintf('Creating ZIP archive: %s\n', package_path);

try
    % Get current directory
    current_dir = pwd;
    
    % Change to build directory to make zip file structure cleaner
    cd(build_dir);
    
    % Get list of files and directories to include
    contents = dir('.');
    contents = contents(~ismember({contents.name}, {'.', '..'}));
    files_to_zip = {contents.name};
    
    % Create ZIP file
    zip(package_path, files_to_zip);
    
    % Verify creation of ZIP file
    if exist(package_path, 'file')
        fprintf('Successfully created ZIP archive at: %s\n', package_path);
    else
        error('Failed to create ZIP archive.');
    end
    
    % Return to original directory
    cd(current_dir);
catch err
    % Ensure we return to the original directory even if an error occurs
    if exist('current_dir', 'var')
        cd(current_dir);
    end
    
    rethrow(err);
end
end

function success = cleanup_build_directory(build_dir)
% Removes temporary build directory after packaging
success = false;

try
    if exist(build_dir, 'dir')
        rmdir(build_dir, 's');
        
        % Verify directory was removed
        if exist(build_dir, 'dir')
            warning('Failed to remove build directory: %s', build_dir);
            success = false;
        else
            fprintf('Successfully removed temporary build directory.\n');
            success = true;
        end
    else
        % Directory doesn't exist, so already clean
        success = true;
    end
catch err
    warning('Error cleaning up build directory: %s', err.message);
    success = false;
end
end