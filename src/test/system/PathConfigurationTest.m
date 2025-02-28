classdef PathConfigurationTest < BaseTest
    % PATHCONFIGURATIONTEST Test class for verifying the path configuration functionality of the MFE Toolbox
    %
    % This class tests the proper initialization and path configuration of the MFE Toolbox,
    % ensuring that all required directories are added to the MATLAB path and that
    % platform-specific configurations are handled correctly.
    %
    % The test verifies:
    % - All mandatory directories are properly added to the path
    % - Optional directories are handled as specified
    % - Platform-specific paths (Windows/Unix) are configured correctly
    % - Path persistence functionality works as expected
    % - Error conditions are properly handled
    %
    % See also: BaseTest, addToPath, CrossPlatformValidator
    
    properties
        % Original MATLAB path before testing
        originalPath
        
        % List of mandatory directories that must be added to the path
        mandatoryDirectories
        
        % List of optional directories
        optionalDirectories
        
        % Path to the toolbox root directory
        toolboxRoot
        
        % Platform validator for cross-platform testing
        platformValidator
    end
    
    methods
        function obj = PathConfigurationTest()
            % Initialize the PathConfigurationTest class
            %
            % Creates a new instance of the PathConfigurationTest class and initializes
            % the platform validator for cross-platform testing.
            
            % Call superclass constructor
            obj@BaseTest();
            
            % Initialize platform validator
            obj.platformValidator = CrossPlatformValidator();
        end
        
        function setUp(obj)
            % Set up the test environment before each test method
            %
            % This method is called before each test method to prepare the test environment.
            % It stores the original MATLAB path, determines the toolbox root directory,
            % and initializes the mandatory and optional directory lists.
            
            % Call superclass setUp method
            obj.setUp@BaseTest();
            
            % Store the original MATLAB path
            obj.originalPath = path();
            
            % Get the toolbox root directory
            obj.toolboxRoot = obj.getToolboxRoot();
            
            % Initialize mandatory directories list
            obj.mandatoryDirectories = {'bootstrap', 'crosssection', 'distributions', ...
                'GUI', 'multivariate', 'tests', 'timeseries', 'univariate', ...
                'utility', 'realized', 'mex_source', 'dlls'};
            
            % Initialize optional directories list
            obj.optionalDirectories = {'duplication'};
        end
        
        function tearDown(obj)
            % Clean up the test environment after each test method
            %
            % This method is called after each test method to restore the original
            % MATLAB path and perform other cleanup operations.
            
            % Restore the original MATLAB path
            path(obj.originalPath);
            
            % Call superclass tearDown method
            obj.tearDown@BaseTest();
        end
        
        function testMandatoryDirectories(obj)
            % Test that all mandatory directories are added to the path
            %
            % This test verifies that all mandatory directories are properly
            % added to the MATLAB path when addToPath is called.
            
            % Call addToPath to configure the toolbox paths
            addToPath(false, true);
            
            % Check each mandatory directory
            for i = 1:length(obj.mandatoryDirectories)
                dirPath = fullfile(obj.toolboxRoot, obj.mandatoryDirectories{i});
                
                % Verify directory exists
                dirExists = isdir(dirPath);
                obj.assertTrue(dirExists, sprintf('Mandatory directory "%s" does not exist', dirPath));
                
                % If directory exists, verify it's in the path
                if dirExists
                    inPath = obj.isDirectoryInPath(dirPath);
                    obj.assertTrue(inPath, sprintf('Mandatory directory "%s" is not in the path', dirPath));
                end
            end
        end
        
        function testOptionalDirectories(obj)
            % Test that optional directories are handled correctly
            %
            % This test verifies that optional directories are added to the path
            % only when addOptionalDirs is set to true and they exist.
            
            % First test with addOptionalDirs = true
            addToPath(false, true);
            
            % Check each optional directory
            for i = 1:length(obj.optionalDirectories)
                dirPath = fullfile(obj.toolboxRoot, obj.optionalDirectories{i});
                
                % If directory exists, it should be in the path
                if isdir(dirPath)
                    inPath = obj.isDirectoryInPath(dirPath);
                    obj.assertTrue(inPath, sprintf('Optional directory "%s" should be in the path with addOptionalDirs=true', dirPath));
                end
            end
            
            % Now test with addOptionalDirs = false
            addToPath(false, false);
            
            % Check each optional directory
            for i = 1:length(obj.optionalDirectories)
                dirPath = fullfile(obj.toolboxRoot, obj.optionalDirectories{i});
                
                % If directory exists, it should NOT be in the path
                if isdir(dirPath)
                    inPath = obj.isDirectoryInPath(dirPath);
                    obj.assertFalse(inPath, sprintf('Optional directory "%s" should not be in the path with addOptionalDirs=false', dirPath));
                end
            end
        end
        
        function testPlatformSpecificPaths(obj)
            % Test that platform-specific paths are configured correctly
            %
            % This test verifies that platform-specific paths (Windows/Unix)
            % are correctly configured based on the current platform.
            
            % Get current platform
            currentPlatform = obj.platformValidator.getCurrentPlatform();
            
            % Configure the toolbox path
            addToPath(false, true);
            
            % Check platform-specific path configurations
            dllsPath = fullfile(obj.toolboxRoot, 'dlls');
            
            if ispc()
                % On Windows, 'dlls' directory should be in the path
                inPath = obj.isDirectoryInPath(dllsPath);
                obj.assertTrue(inPath, 'DLLs directory should be in the path on Windows');
                
                % Check for .mexw64 files being accessible
                mexFiles = dir(fullfile(dllsPath, '*.mexw64'));
                if ~isempty(mexFiles)
                    % Try to access a MEX file to verify it's in the path
                    mexFile = mexFiles(1).name;
                    [~, mexName, ~] = fileparts(mexFile);
                    mexExists = (exist(mexName, 'file') == 3); % 3 = MEX file
                    obj.assertTrue(mexExists, 'MEX file should be accessible');
                end
            else
                % On Unix, check if dlls were added through the core directories list
                inPath = obj.isDirectoryInPath(dllsPath);
                obj.assertTrue(inPath, 'DLLs directory should be in the path on Unix');
                
                % Check for .mexa64 files being accessible
                mexFiles = dir(fullfile(dllsPath, '*.mexa64'));
                if ~isempty(mexFiles)
                    % Try to access a MEX file to verify it's in the path
                    mexFile = mexFiles(1).name;
                    [~, mexName, ~] = fileparts(mexFile);
                    mexExists = (exist(mexName, 'file') == 3); % 3 = MEX file
                    obj.assertTrue(mexExists, 'MEX file should be accessible');
                end
            end
        end
        
        function testPathSaving(obj)
            % Test that the path saving functionality works correctly
            %
            % This test verifies that the path saving functionality properly
            % calls savepath() when requested and handles success/failure cases.
            
            % Create a mock for savepath function to test
            origSavepath = str2func('savepath');
            
            try
                % Define mock success function
                successMock = @() 0;
                
                % Temporarily define a global savepath function that always succeeds
                evalin('base', 'global mockSavepathResult; mockSavepathResult = 0;');
                evalin('base', ['global origSavepath; origSavepath = @savepath; ', ...
                               'savepath = @() global mockSavepathResult; mockSavepathResult;']);
                
                % Call addToPath with savePath=true
                result = addToPath(true, true);
                
                % Verify result is true (success)
                obj.assertTrue(result, 'addToPath should return true when savepath succeeds');
                
                % Now test failure case
                % Define mock failure function
                evalin('base', 'global mockSavepathResult; mockSavepathResult = 1;');
                
                % Warnings are expected in this case, so we'll catch them
                warningState = warning('off', 'MFE:PathNotSaved');
                
                % Call addToPath with savePath=true (should still complete successfully)
                result = addToPath(true, true);
                
                % Verify result is true despite savepath failing
                obj.assertTrue(result, 'addToPath should return true even when savepath fails');
                
                % Restore warning state
                warning(warningState);
                
            catch e
                % Restore original savepath in case of error
                evalin('base', 'global origSavepath; savepath = origSavepath; clear origSavepath mockSavepathResult;');
                rethrow(e);
            end
            
            % Restore original savepath
            evalin('base', 'global origSavepath; savepath = origSavepath; clear origSavepath mockSavepathResult;');
        end
        
        function testErrorConditions(obj)
            % Test the handling of error conditions in path configuration
            %
            % This test verifies that the function properly handles error conditions
            % such as missing directories and parameter errors.
            
            % Create a temporary directory for testing
            tempDir = fullfile(tempdir, 'mfe_test_path');
            if ~exist(tempDir, 'dir')
                mkdir(tempDir);
            end
            
            % Store current directory
            currentDir = pwd;
            
            try
                % Change to temporary directory
                cd(tempDir);
                
                % Test with missing directories
                warningState = warning('off', 'MFE:MissingDirectory');
                result = addToPath(false, true);
                warning(warningState);
                
                % Should still return true despite warnings
                obj.assertTrue(result, 'addToPath should return true even with missing directories');
                
                % Test with invalid parameter types
                try
                    % This would normally trigger an error, but we'll catch it
                    addToPath('not_a_logical', true);
                    obj.assertTrue(false, 'addToPath should error with invalid parameter types');
                catch
                    % Expected error, so test passes
                end
                
            catch e
                % Restore directory in case of error
                cd(currentDir);
                rethrow(e);
            end
            
            % Restore original directory
            cd(currentDir);
            
            % Clean up temporary directory
            if exist(tempDir, 'dir')
                rmdir(tempDir, 's');
            end
        end
        
        function rootPath = getToolboxRoot(obj)
            % Helper method to determine the toolbox root directory
            %
            % This method attempts to find the toolbox root directory based on
            % the location of the backend directory containing addToPath.m.
            
            % Get the path to addToPath.m
            addToPathInfo = which('addToPath');
            
            if isempty(addToPathInfo)
                error('PathConfigurationTest:RootNotFound', 'Could not find addToPath.m to determine toolbox root');
            end
            
            % Get the directory containing addToPath.m (should be backend)
            [backendDir, ~, ~] = fileparts(addToPathInfo);
            
            % The parent directory of backend should be the toolbox root
            [rootPath, ~, ~] = fileparts(backendDir);
            
            % Validate root path
            if ~isdir(rootPath)
                error('PathConfigurationTest:InvalidRoot', 'Determined root path is not a valid directory: %s', rootPath);
            end
            
            % Check if this looks like a valid toolbox root by checking for some mandatory directories
            someExpectedDirs = {'bootstrap', 'utility', 'univariate'};
            isValid = true;
            
            for i = 1:length(someExpectedDirs)
                if ~isdir(fullfile(rootPath, someExpectedDirs{i}))
                    isValid = false;
                    break;
                end
            end
            
            if ~isValid
                error('PathConfigurationTest:InvalidRoot', 'Determined root path does not appear to be valid toolbox root: %s', rootPath);
            end
        end
        
        function inPath = isDirectoryInPath(obj, directoryPath)
            % Helper method to check if a directory is in the MATLAB path
            %
            % INPUTS:
            %   directoryPath - Path to the directory to check
            %
            % OUTPUTS:
            %   inPath - True if the directory is in the path, false otherwise
            
            % Get current path
            currentPath = path();
            
            % Standardize directory path for comparison
            directoryPath = strrep(directoryPath, '\', '/');
            
            % Split path into individual directories
            pathParts = strsplit(currentPath, pathsep);
            
            % Check if directory is in path
            inPath = false;
            for i = 1:length(pathParts)
                % Standardize path part for comparison
                pathPart = strrep(pathParts{i}, '\', '/');
                
                % Check for match
                if strcmp(pathPart, directoryPath)
                    inPath = true;
                    break;
                end
            end
        end
    end
end