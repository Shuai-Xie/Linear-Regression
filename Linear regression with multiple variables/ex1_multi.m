%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear regression exercise.
%
%  You will need to complete the following functions in this
%  exericse:
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
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

% Clear and Close Figures
clear ; close all; clc

%% ================ Part 1: Feature Normalization ================
fprintf('Loading data ...\n');

data = load('ex1data2.txt');
X = data(:, 1:2); % 房屋特征2个 (m*2)
y = data(:, 3); % 价格 (m*1)

m = length(y); % 样本数量

% 输出10个样本点
fprintf('First 10 examples from the dataset: \n');

% X 2个特征，y 1个价格
% 因为 fprintf 按列输出矩阵，所以要将矩阵先转置，使得每一列对应一个实例
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
% fprintf(' x = [%.0f %.0f], y = %.0f \n', data(1:10, :)'); % 也可以

fprintf('Program paused. Press enter to continue.\n');
pause;

% 特征缩放
fprintf('Normalizing Features ...\n');
% X是特征缩放后的特征矩阵，mu是平均值矩阵，sigma是方差矩阵
[X, mu, sigma] = featureNormalize(X);
X = [ones(m, 1), X]; % X 补充 theta0

% 格式化输出特征缩放后的矩阵
fprintf(' x = [%.6f %.6f %.6f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha).
%
%               Your task is to first make sure that your functions -
%               computeCost and gradientDescent already work with
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01; % 学习率
alpha1 = 0.03;
alpha2 = 0.1;
num_iters = 400; % 迭代次数

% Init Theta and Run Gradient Descent
theta = zeros(3, 1); % 3个特征
theta1 = zeros(3, 1);
theta2 = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters); % 多变量梯度下降
[theta1, J_history1] = gradientDescentMulti(X, y, theta1, alpha1, num_iters);
[theta2, J_history2] = gradientDescentMulti(X, y, theta2, alpha2, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
hold on;
plot(1:numel(J_history1), J_history1, '-r', 'LineWidth', 2);
plot(1:numel(J_history2), J_history2, '-k', 'LineWidth', 2);
% numel: number of elements 返回元素个数，作为x轴迭代次数
xlabel('Number of iterations');
ylabel('Cost J');
legend('alpha = 0.01', 'alpha = 0.03', 'alpha = 0.1');

% 显示theta值
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta); % 3*1

% ====================== YOUR CODE HERE ======================
% 估计房屋价格 1650 sq-ft, 3 br house
% x0不需要特征缩放为1，将后面的x1，x2特征缩放
% price = 0; % You should change this
price = [1, (1650 - mu(1,1)) / sigma(1,1), (3 - mu(1,2)) / sigma(1,2)] * theta;

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
    '(using gradient descent): $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form
%               solution for linear regression using the normal
%               equations. You should complete the code in
%               normalEqn.m
%
%               After doing so, you should complete this code
%               to predict the price of a 1650 sq-ft, 3 br house.
%

% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);

% Estimate the price of a 1650 sq-ft, 3 br house
% 不需要特征缩放
price = [1, 1650, 3] * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
    '(using normal equations): $%f\n'], price);