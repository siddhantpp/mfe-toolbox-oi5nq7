function varargout = ARMAX_close_dialog(varargin)
% ARMAX_CLOSE_DIALOG Close confirmation dialog for ARMAX GUI
%
% This function implements a modal confirmation dialog that prompts users 
% to save changes before closing the ARMAX modeling application.
%
% USAGE:
%   choice = ARMAX_close_dialog()
%
% RETURNS:
%   choice - User's selection: 'Yes' or 'No'
%
% Part of the MFE Toolbox 4.0 (28-Oct-2009)

% Initialize and then hide the GUI as it is being constructed
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ARMAX_close_dialog_OpeningFcn, ...
                   'gui_OutputFcn',  @ARMAX_close_dialog_OutputFcn, ...
                   'gui_LayoutFcn',  [], ...
                   'gui_Callback',   []);
               
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

function ARMAX_close_dialog_OpeningFcn(hObject, eventdata, handles, varargin)
% This function runs when the dialog is created
% Initialize the dialog and set its position

% Choose default command line output for ARMAX_close_dialog
handles.output = 'No';  % Default choice is No

% Center dialog on screen
center_dialog(hObject);

% Update handles structure
guidata(hObject, handles);

% Make the dialog modal
set(handles.figure1, 'WindowStyle', 'modal');

% UIWAIT makes the dialog wait for user response
uiwait(handles.figure1);

function varargout = ARMAX_close_dialog_OutputFcn(hObject, eventdata, handles)
% Get output from handles structure and return it
varargout{1} = handles.output;

% The figure can be deleted now
delete(hObject);

function Yes_button_Callback(hObject, eventdata, handles)
% User chose 'Yes' option
handles.output = 'Yes';
guidata(hObject, handles);
uiresume(handles.figure1);

function No_button_Callback(hObject, eventdata, handles)
% User chose 'No' option
handles.output = 'No';
guidata(hObject, handles);
uiresume(handles.figure1);

function center_dialog(figureHandle)
% Center the dialog on the screen
% Get screen size
screen_size = get(0, 'ScreenSize');
screen_width = screen_size(3);
screen_height = screen_size(4);

% Get the figure size
figure_position = get(figureHandle, 'Position');
figure_width = figure_position(3);
figure_height = figure_position(4);

% Calculate center position
position_x = (screen_width - figure_width) / 2;
position_y = (screen_height - figure_height) / 2;

% Set the figure position
set(figureHandle, 'Position', [position_x, position_y, figure_width, figure_height]);