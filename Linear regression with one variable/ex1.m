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
% 输出一个5*5的单位阵
fprintf('Running warmUpExercise ... \n');
fprintf('5x5 Identity Matrix: \n');

warmUpExercise() % 调用函数生成单位阵

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ======================= Part 2: Plotting data =======================
% 输出样本点散点图
fprintf('Plotting Data ...\n')
data = load('ex1data1.txt'); % 加载数据 csv文件

X = data(:, 1); % X 城市人口 第1列
y = data(:, 2); % y 城市利润 第2列

m = length(y); % 训练集大小

% 调用 plotData 函数做散点图
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Part 3: Gradient descent ===================
fprintf('Running Gradient Descent ...\n')

% 调整样本并初始化一些值
X = [ones(m, 1), data(:,1)]; % 添加 ones(m, 1) 适配 theta0
theta = zeros(2, 1); % theta = [0,0]
iterations = 1500; % 迭代次数
alpha = 0.01; % 学习率

% 初始情况(theta=0)下的 cost
computeCost(X, y, theta);

% 梯度下降法求最优的 theta
theta = gradientDescent(X, y, theta, alpha, iterations);

% 1500次学习后的 theta
fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2)); % 分别打印，注意matlab下标从1开始

% 在保留散点的基础上 画出 最终theta 确定的 线性方程
hold on; % keep previous plot visible 保留以前的散点
plot(X(:,2), X * theta, '-'); % 用 '-' 表示点，做散点图的时候用的是 'rx'
legend('Training data', 'Linear regression'); % 设置图例分别对应 'rx', '-'
hold off; % not overlay any more plots on this figure 不再画任何点

% 用 theta 预测，城市人口数：35,000 和 70,000
predict1 = [1, 3.5] * theta;
fprintf('For population = 35,000, we predict a profit of %f\n', predict1*10000);

predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n', predict2*10000);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% 设置计算 J 的网格
theta0_vals = linspace(-10, 10, 100); % 网格范围：-10到10，分成100份
theta1_vals = linspace(-1, 4, 100); % 根据散点图预测 theta 范围

% 初始化 J_vals 存储矩阵：100*100
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% 求 J_vals，1万个(i, j)点，循环体执行1万次
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
        t = [theta0_vals(i); theta1_vals(j)]; % 注意是列向量
        J_vals(i,j) = computeCost(X, y, t);
    end
end


J_vals = J_vals'; % surf 在画 meshgrid 是需要先转置z轴的值

% Surface plot 曲面图
figure; % Figure 2
surf(theta0_vals, theta1_vals, J_vals); % surface 画点
xlabel('\theta_0'); % 坐标轴标签，用 \ 可以转义为字母
ylabel('\theta_1');

% Contour plot 等值线图
figure; % Figure 3
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20)) % 2D 等值线图
% logspace 以 10^x 的形式定义一些列 J 值，x 将[-2, 3]均分为20份
xlabel('\theta_0');
ylabel('\theta_1');

% 在等值线图基础上显示 theta
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
