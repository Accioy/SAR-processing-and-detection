function [fitresult, gof] = createFit1(angle, acc)
%CREATEFIT1(ANGLE,ACC)
%  Create a fit.
%
%  Data for 'untitled fit 1' fit:
%      X Input : angle
%      Y Output: acc
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%
%  ������� FIT, CFIT, SFIT.

%  �� MATLAB �� 14-Nov-2020 13:41:35 �Զ�����


%% Fit: 'untitled fit 1'.
[xData, yData] = prepareCurveData( angle, acc );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );
opts = fitoptions( 'Method', 'SmoothingSpline' );
opts.SmoothingParam = 0.000799208505627837;

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );

% Plot fit with data.
figure( 'Name', 'untitled fit 1' );
h = plot( fitresult, xData, yData );
legend( h, 'ʶ����ȷ��',  'Location', 'NorthEast' );
% Label axes
xlabel ѵ������λ�ǲ����Ƕȼ��
ylabel ʶ����ȷ��
grid on


