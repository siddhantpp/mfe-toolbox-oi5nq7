function varargout = ARMAX_about(varargin)
% ARMAX_ABOUT About dialog for the ARMAX modeling interface
%   ARMAX_ABOUT displays the About dialog for the ARMAX modeling
%   interface, which is part of the MFE Toolbox. The dialog shows
%   version information, copyright details, and the Oxford University 
%   Press logo.
%
%   This file is part of the MFE Toolbox v4.0 (28-Oct-2009)

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ARMAX_about_OpeningFcn, ...
                   'gui_OutputFcn',  @ARMAX_about_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before ARMAX_about is made visible.
function ARMAX_about_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ARMAX_about (see VARARGIN)

% Choose default command line output for ARMAX_about
handles.output = hObject;

% Set version text
set(handles.VersionText, 'String', 'MFE Toolbox v4.0 (28-Oct-2009)');

% Set copyright text
set(handles.CopyrightText, 'String', '© 2009 All Rights Reserved');

% Make dialog modal
set(hObject, 'WindowStyle', 'modal');

% Center dialog on screen
center_dialog(hObject);

% Update handles structure
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = ARMAX_about_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on button press in OKButton.
function OKButton_Callback(hObject, eventdata, handles)
% hObject    handle to OKButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Close dialog
delete(gcbf);

% --- Helper function to center dialog on screen
function center_dialog(figureHandle)
% Get screen size
screenSize = get(0, 'ScreenSize');
screenWidth = screenSize(3);
screenHeight = screenSize(4);

% Get figure position
figurePosition = get(figureHandle, 'Position');
figureWidth = figurePosition(3);
figureHeight = figurePosition(4);

% Calculate new position
newLeft = (screenWidth - figureWidth) / 2;
newBottom = (screenHeight - figureHeight) / 2;

% Set new position
set(figureHandle, 'Position', [newLeft, newBottom, figureWidth, figureHeight]);