function buildZipFile()
%BUILDZIPFILE Automation script for building the MFE Toolbox
%   This function automates the MFE Toolbox build process including:
%   - Workspace cleanup
%   - MEX compilation with -largeArrayDims flag
%   - Platform-specific binary generation (PCWIN64/Unix)
%   - Component packaging
%   - ZIP archive creation
%
%   The script creates a comprehensive distribution package with all required
%   directories and files according to the MFE Toolbox structure.
%
%   MFE Toolbox v4.0 (28-Oct-2009)

% Clear workspace and close all figures for a clean build environment
clear all;
close all;

% Create timestamp for build identification
timestamp = clock;
dateStr = sprintf('%04d-%02d-%02d %02d:%02d:%02d', timestamp(1), timestamp(2), ...
    timestamp(3), timestamp(4), timestamp(5), floor(timestamp(6)));
fprintf('Starting MFE Toolbox build at %s\n', dateStr);

% Define build directory and remove if it exists
buildDir = 'MFEToolbox_build';
if exist(buildDir, 'dir')
    fprintf('Removing existing build directory...\n');
    rmdir(buildDir, 's');
end

% Create build directory structure
fprintf('Creating directory structure...\n');
mkdir(buildDir);

% Required directories
requiredDirs = {'bootstrap', 'crosssection', 'distributions', 'GUI', ...
    'multivariate', 'tests', 'timeseries', 'univariate', 'utility', ...
    'realized', 'mex_source', 'dlls'};

% Create all required directories
for i = 1:length(requiredDirs)
    mkdir(fullfile(buildDir, requiredDirs{i}));
end

% Determine platform for appropriate MEX compilation
platform = computer;
fprintf('Detected platform: %s\n', platform);
isWindows = strcmp(platform, 'PCWIN64');

% Define MEX source files to compile
mexSourceFiles = {'agarch_core.c', 'armaxerrors.c', 'composite_likelihood.c', ...
    'egarch_core.c', 'igarch_core.c', 'tarch_core.c'};

% Additional utility source files that may be required for compilation
utilSourceFiles = {'matrix_operations.c', 'mex_utils.c'};

% Header files that may be required
headerFiles = {'matrix_operations.h', 'mex_utils.h'};

% Save current directory
currentDir = pwd;

% Compile MEX files
fprintf('Compiling MEX files with -largeArrayDims flag...\n');

% Process MEX source directory
if exist('mex_source', 'dir')
    cd('mex_source');
    
    % Copy all source and header files to build directory
    for i = 1:length(mexSourceFiles)
        if exist(mexSourceFiles{i}, 'file')
            copyfile(mexSourceFiles{i}, fullfile('..', buildDir, 'mex_source'));
        else
            fprintf('Warning: Source file %s not found\n', mexSourceFiles{i});
        end
    end
    
    % Copy utility source files if they exist
    for i = 1:length(utilSourceFiles)
        if exist(utilSourceFiles{i}, 'file')
            copyfile(utilSourceFiles{i}, fullfile('..', buildDir, 'mex_source'));
        end
    end
    
    % Copy header files if they exist
    for i = 1:length(headerFiles)
        if exist(headerFiles{i}, 'file')
            copyfile(headerFiles{i}, fullfile('..', buildDir, 'mex_source'));
        end
    end
    
    % Compile each MEX file
    for i = 1:length(mexSourceFiles)
        if exist(mexSourceFiles{i}, 'file')
            fprintf('Compiling %s...\n', mexSourceFiles{i});
            
            % Determine additional source files to include in compilation
            includeFiles = '';
            for j = 1:length(utilSourceFiles)
                if exist(utilSourceFiles{j}, 'file')
                    includeFiles = [includeFiles, ' ', utilSourceFiles{j}];
                end
            end
            
            % Compile with -largeArrayDims flag for large dataset support
            try
                eval(['mex -largeArrayDims ', mexSourceFiles{i}, includeFiles]);
                
                % Determine MEX extension based on platform
                if isWindows
                    mexExt = '.mexw64';
                else
                    mexExt = '.mexa64';
                end
                
                % Get MEX filename (without extension)
                [~, mexName] = fileparts(mexSourceFiles{i});
                mexFile = [mexName, mexExt];
                
                % Copy compiled MEX to build dlls directory
                if exist(mexFile, 'file')
                    copyfile(mexFile, fullfile('..', buildDir, 'dlls'));
                    fprintf('Successfully compiled %s\n', mexFile);
                else
                    fprintf('Warning: Compiled MEX file %s not found\n', mexFile);
                end
            catch ME
                fprintf('Error compiling %s: %s\n', mexSourceFiles{i}, ME.message);
            end
        end
    end
    
    % Return to original directory
    cd(currentDir);
else
    fprintf('Warning: mex_source directory not found\n');
end

% Copy all MATLAB files to their respective directories
fprintf('Copying MATLAB files to build directory...\n');

% Process each required directory
for i = 1:length(requiredDirs)
    dirName = requiredDirs{i};
    
    % Skip dlls and mex_source as they've been handled separately
    if ~strcmp(dirName, 'dlls') && ~strcmp(dirName, 'mex_source')
        if exist(dirName, 'dir')
            % Get all .m files in the directory
            files = dir(fullfile(dirName, '*.m'));
            
            % Copy each file to the build directory
            for j = 1:length(files)
                copyfile(fullfile(dirName, files(j).name), ...
                    fullfile(buildDir, dirName));
            end
            
            % Also copy any .fig files (for GUI components)
            if strcmp(dirName, 'GUI')
                figFiles = dir(fullfile(dirName, '*.fig'));
                for j = 1:length(figFiles)
                    copyfile(fullfile(dirName, figFiles(j).name), ...
                        fullfile(buildDir, dirName));
                end
            end
        else
            fprintf('Warning: Required directory %s not found\n', dirName);
        end
    end
end

% Check for and process optional directories
optionalDirs = {'duplication'};
for i = 1:length(optionalDirs)
    dirName = optionalDirs{i};
    if exist(dirName, 'dir')
        % Create directory in build
        mkdir(fullfile(buildDir, dirName));
        
        % Get all .m files in the directory
        files = dir(fullfile(dirName, '*.m'));
        
        % Copy each file to the build directory
        for j = 1:length(files)
            copyfile(fullfile(dirName, files(j).name), ...
                fullfile(buildDir, dirName));
        end
        
        fprintf('Added optional directory: %s\n', dirName);
    end
end

% Copy core files to the root
coreFiles = {'addToPath.m', 'Contents.m'};
for i = 1:length(coreFiles)
    if exist(coreFiles{i}, 'file')
        copyfile(coreFiles{i}, buildDir);
    else
        fprintf('Warning: Core file %s not found\n', coreFiles{i});
    end
end

% Create ZIP archive
fprintf('Creating ZIP archive...\n');
try
    % Get all files and folders in the build directory
    cd(buildDir);
    allItems = dir('.');
    zipItems = {};
    
    % Filter out . and .. entries
    for i = 1:length(allItems)
        if ~strcmp(allItems(i).name, '.') && ~strcmp(allItems(i).name, '..')
            zipItems{end+1} = allItems(i).name;
        end
    end
    
    % Return to original directory
    cd(currentDir);
    
    % Create zip file - compress all items from the build directory
    zip('MFEToolbox.zip', zipItems, buildDir);
    
    fprintf('Successfully created MFEToolbox.zip\n');
catch ME
    fprintf('Error creating ZIP archive: %s\n', ME.message);
end

% Calculate and display elapsed time
endTime = clock;
elapsedMinutes = (endTime(4)*60 + endTime(5) + endTime(6)/60) - ...
    (timestamp(4)*60 + timestamp(5) + timestamp(6)/60);

% Display completion message
fprintf('Build completed at %s (%.2f minutes)\n', ...
    sprintf('%04d-%02d-%02d %02d:%02d:%02d', endTime(1), endTime(2), ...
    endTime(3), endTime(4), endTime(5), floor(endTime(6))), elapsedMinutes);

fprintf('MFE Toolbox v4.0 package creation successful\n');

end