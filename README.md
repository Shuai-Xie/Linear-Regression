[Linear regression with one variable - 简书](http://www.jianshu.com/p/31f9f6aa4b16)

[Linear regression with multiple variables - 简书](http://www.jianshu.com/p/d6b0d0f044e7)

多变量线性回归 预测房价

ex1data2.txt（房屋尺寸，卧室数量，房屋价格）

```
2104,3,399900
1600,3,329900
2400,3,369000
1416,2,232000
3000,4,539900
1985,4,299900
……
```

### Part 1: Feature Normalization
对于多维特征，要使用特征缩放保证其具有相近尺度，这样梯度下降算法会更快收敛。这个例子中，房屋尺寸是卧室数量接近1000倍。

```matlab
%% ================ Part 1: Feature Normalization ================
fprintf('Loading data ...\n');

data = load('ex1data2.txt');
X = data(:, 1:2); % 房屋特征2个 (m*2)
y = data(:, 3); % 价格 (m*1)

m = length(y); % 样本数量

% 输出10个样本点
fprintf('First 10 examples from the dataset: \n');

% X 2个特征，y 1个价格
% 因为fprintf按列输出矩阵，所以要将矩阵先转置，使得每一列对应一个实例
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
```
```
Loading data ...
First 10 examples from the dataset: 
 x = [2104 3], y = 399900 
 x = [1600 3], y = 329900 
 x = [2400 3], y = 369000 
 x = [1416 2], y = 232000 
 x = [3000 4], y = 539900 
 x = [1985 4], y = 299900 
 x = [1534 3], y = 314900 
 x = [1427 3], y = 198999 
 x = [1380 3], y = 212000 
 x = [1494 3], y = 242500 
Program paused. Press enter to continue.
Normalizing Features ...
 x = [1.000000 0.130010 -0.223675], y = 399900 
 x = [1.000000 -0.504190 -0.223675], y = 329900 
 x = [1.000000 0.502476 -0.223675], y = 369000 
 x = [1.000000 -0.735723 -1.537767], y = 232000 
 x = [1.000000 1.257476 1.090417], y = 539900 
 x = [1.000000 -0.019732 1.090417], y = 299900 
 x = [1.000000 -0.587240 -0.223675], y = 314900 
 x = [1.000000 -0.721881 -0.223675], y = 198999 
 x = [1.000000 -0.781023 -0.223675], y = 212000 
 x = [1.000000 -0.637573 -0.223675], y = 242500 
Program paused. Press enter to continue.
```
featureNormalize 函数
```matlab
function [X_norm, mu, sigma] = featureNormalize(X)
% 标准化 X 的特征到[0,1]之间

% 初始化
X_norm = X; % X(47*2)

% 利用matlab自带函数mean，std求平均值和标准差
mu = mean(X); % mu(1*2) a row vector containing the mean value of each column. 
sigma = std(X); % sigma(1*2) 

% size(X, 1) 获得 X 矩阵的行数，即样本数 = 47
% size(X, 2) 获得 X 矩阵的列数，即特征数 = 2
m = size(X, 1);

% 缩放 X 的 2 个特征
for i = 1:m
	X_norm(i, 1) = ( X_norm(i, 1) - mu(1) ) / sigma(1);
	X_norm(i, 2) = ( X_norm(i, 2) - mu(2) ) / sigma(2);
end

end
```

![特征缩放](http://upload-images.jianshu.io/upload_images/1877813-6e885a87f831981d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### Part 2: Gradient Descent
```matlab
%% ================ Part 2: Gradient Descent ================
fprintf('Running gradient descent ...\n');

% Choose some alpha value 3个学习率
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
hold on; % 继续画图
plot(1:numel(J_history1), J_history1, '-r', 'LineWidth', 2);
plot(1:numel(J_history2), J_history2, '-k', 'LineWidth', 2);
% numel: number of elements 返回元素个数，作为x轴迭代次数
xlabel('Number of iterations');
ylabel('Cost J');
legend('alpha = 0.01', 'alpha = 0.03', 'alpha = 0.1'); % 图例

% 显示theta值
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta); % 3*1

% 估计房屋价格 1650 sq-ft, 3 br house
% x1不需要特征缩放为1，将后面的x2，x3特征缩放
% price = 0; % You should change this
price = [1, (1650 - mu(1,1)) / sigma(1,1), (3 - mu(1,2)) / sigma(1,2)] * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
    '(using gradient descent): $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;
```
```
Running gradient descent ...
Theta computed from gradient descent: 
 334302.063993 
 100087.116006 
 3673.548451 
Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): $289314.620338
Program paused. Press enter to continue.
```


![不同学习率下的收敛曲线](http://upload-images.jianshu.io/upload_images/1877813-9dff72d194b0d358.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

gradientDescentMulti 函数
```matlab
function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

% Initialize some useful values
J_history = zeros(num_iters, 1);
m = length(y);

% 迭代
for iter = 1:num_iters
    
    tmp = (X' * ( X * theta - y )); % 向量运算 求和项 (3*m)*(m*1)=(3*1)
    theta = theta - alpha / m * tmp;
    
    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta); % 根据theta求J
    
end
```
computeCostMulti 函数
```matlab
function J = computeCostMulti(X, y, theta)

% Initialize some useful values
m = length(y); % number of training examples

E = X * theta - y; % 偏差矩阵(m*1)
J = (E' * E) / (2 * m); % (E'*E) 方差和

end
```
### Part 3: Normal Equations
正规方程求 ***θ***
```matlab
%% ================ Part 3: Normal Equations ================
fprintf('Solving with normal equations...\n');

% Load Data
data = csvread('ex1data2.txt'); % load也可以
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
```
预测结果与梯度下降法有出入，正规预测的是准确值，梯度下降是递进值。
```
Solving with normal equations...
Theta computed from the normal equations: 
 89597.909544 
 139.210674 
 -8738.019113 
Predicted price of a 1650 sq-ft, 3 br house (using normal equations): $293081.464335
```
normalEqn 函数
```matlab
function [theta] = normalEqn(X, y)

theta = pinv(X' * X) * X' * y; % 调用方程

end
```
