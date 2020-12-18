function [fitresult, gof] = createFits(x, pre1, pre2)
%CREATEFITS(X,PRE1,PRE2)
%  Create fits.
%
%  Data for 'untitled fit 1' fit:
%      X Input : x
%      Y Output: pre1
%  Data for 'untitled fit 2' fit:
%      X Input : x
%      Y Output: pre2
%  Output:
%      fitresult : a cell-array of fit objects representing the fits.
%      gof : structure array with goodness-of fit info.
%
%  ������� FIT, CFIT, SFIT.

%  �� MATLAB �� 30-Oct-2020 15:06:06 �Զ�����

%% Initialization.

% Initialize arrays to store fits and goodness-of-fit.
fitresult = cell( 2, 1 );
gof = struct( 'sse', cell( 2, 1 ), ...
    'rsquare', [], 'dfe', [], 'adjrsquare', [], 'rmse', [] );

%% Fit: 'untitled fit 1'.
[xData, yData] = prepareCurveData( x, pre1 );

% Set up fittype and options.
ft = fittype( 'power1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [0.863908366677412 0.0214778507353072];

% Fit model to data.
[fitresult{1}, gof(1)] = fit( xData, yData, ft, opts );

% Plot fit with data.
figure( 'Name', 'untitled fit 1' );
plot( fitresult{1},'g', xData, yData ,'.g');

% Label axes
%xlabel x
%ylabel pre1
grid on
hold on;
%% Fit: 'untitled fit 2'.
[xData, yData] = prepareCurveData( x, pre2 );

% Set up fittype and options.
ft = fittype( 'power1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [0.82815380040543 0.0240143163684187];

% Fit model to data.
[fitresult{2}, gof(2)] = fit( xData, yData, ft, opts );

% Plot fit with data.
%figure( 'Name', 'untitled fit 2' );
plot( fitresult{2},'b', xData, yData,'.b' );
legend(  '���ֱ�ͼ��ѵ��mAP',  '���ֱ�ͼ��ѵ��mAP����', 'ԭʼͼ��ѵ��mAP',  'ԭʼͼ��ѵ��mAP����',  'Location', 'NorthWest' );
% Label axes
xlabel ѵ������
ylabel mAP
grid on


