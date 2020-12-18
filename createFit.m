function [fitresult, gof] = createFit(x, pre1,pre2)
%CREATEFIT(X,PRE)
%  Create a fit.
%
%  Data for 'untitled fit 1' fit:
%      X Input : x
%      Y Output: pre
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%
%  另请参阅 FIT, CFIT, SFIT.

%  由 MATLAB 于 30-Oct-2020 14:26:36 自动生成


%% Fit: 'untitled fit 1'.
[xData, yData1] = prepareCurveData( x, pre1 );
[xData, yData2] = prepareCurveData( x, pre2 );
% Set up fittype and options.
ft = fittype( 'power1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [0.863908366677412 0.0214778507353072];

% Fit model to data.
[fitresult, gof] = fit( xData, yData1, ft, opts );

% Plot fit with data.
figure( 'Name', 'untitled fit 1' );
h = plot( fitresult, xData, yData1 );
legend( h, 'pre vs. x', 'untitled fit 1', 'Location', 'NorthEast' );
% Label axes
xlabel x
ylabel pre
grid on



