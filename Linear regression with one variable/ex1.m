%% Machine Learning Online Class - Exercise 1: Linear Regression

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
% x refers to the population size in 10,000s
% y refers to the profit in $10,000s
%

%% Initialization
clear ; close all; clc

%% ==================== Part 1: Basic Function ====================
% Complete warmUpExercise.m
% ���һ��5*5�ĵ�λ��
fprintf('Running warmUpExercise ... \n');
fprintf('5x5 Identity Matrix: \n');

warmUpExercise() % ���ú������ɵ�λ��

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ======================= Part 2: Plotting data =======================
% ���������ɢ��ͼ
fprintf('Plotting Data ...\n')
data = load('ex1data1.txt'); % �������� csv�ļ�

X = data(:, 1); % X �����˿� ��1��
y = data(:, 2); % y �������� ��2��

m = length(y); % ѵ������С

% ���� plotData ������ɢ��ͼ
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Part 3: Gradient descent ===================
fprintf('Running Gradient Descent ...\n')

% ������������ʼ��һЩֵ
X = [ones(m, 1), data(:,1)]; % ��� ones(m, 1) ���� theta0
theta = zeros(2, 1); % theta = [0,0]
iterations = 1500; % ��������
alpha = 0.01; % ѧϰ��

% ��ʼ���(theta=0)�µ� cost
computeCost(X, y, theta);

% �ݶ��½��������ŵ� theta
theta = gradientDescent(X, y, theta, alpha, iterations);

% 1500��ѧϰ��� theta
fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2)); % �ֱ��ӡ��ע��matlab�±��1��ʼ

% �ڱ���ɢ��Ļ����� ���� ����theta ȷ���� ���Է���
hold on; % keep previous plot visible ������ǰ��ɢ��
plot(X(:,2), X * theta, '-'); % �� '-' ��ʾ�㣬��ɢ��ͼ��ʱ���õ��� 'rx'
legend('Training data', 'Linear regression'); % ����ͼ���ֱ��Ӧ 'rx', '-'
hold off; % not overlay any more plots on this figure ���ٻ��κε�

% �� theta Ԥ�⣬�����˿�����35,000 �� 70,000
predict1 = [1, 3.5] * theta;
fprintf('For population = 35,000, we predict a profit of %f\n', predict1*10000);

predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n', predict2*10000);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% ���ü��� J ������
theta0_vals = linspace(-10, 10, 100); % ����Χ��-10��10���ֳ�100��
theta1_vals = linspace(-1, 4, 100); % ����ɢ��ͼԤ�� theta ��Χ

% ��ʼ�� J_vals �洢����100*100
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% �� J_vals��1���(i, j)�㣬ѭ����ִ��1���
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
        t = [theta0_vals(i); theta1_vals(j)]; % ע����������
        J_vals(i,j) = computeCost(X, y, t);
    end
end


J_vals = J_vals'; % surf �ڻ� meshgrid ����Ҫ��ת��z���ֵ

% Surface plot ����ͼ
figure; % Figure 2
surf(theta0_vals, theta1_vals, J_vals); % surface ����
xlabel('\theta_0'); % �������ǩ���� \ ����ת��Ϊ��ĸ
ylabel('\theta_1');

% Contour plot ��ֵ��ͼ
figure; % Figure 3
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20)) % 2D ��ֵ��ͼ
% logspace �� 10^x ����ʽ����һЩ�� J ֵ��x ��[-2, 3]����Ϊ20��
xlabel('\theta_0');
ylabel('\theta_1');

% �ڵ�ֵ��ͼ��������ʾ theta
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
