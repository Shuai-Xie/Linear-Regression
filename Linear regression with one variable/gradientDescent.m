function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
% X 样本特征，y 样本实际值，theta 模型参数，alpha 学习率，num_iters 迭代次数

% 初始化一些值
m = length(y);
J_history = zeros(num_iters, 1); % 存储每次迭代的J值

for iter = 1:num_iters % 迭代次数
    
    tmp1 = 0; % 存储求和项
    tmp2 = 0;
    
    % 计算求和项
    for i = 1:m
        tmp1 = tmp1 + (X(i, :) * theta - y(i)); % theta0 对应 X(i, 1) = 1
        tmp2 = tmp2 + (X(i, :) * theta - y(i)) * X(i, 2);
    end
    
    % 更新 theta
    theta(1) = theta(1) - alpha * tmp1 / m;
    theta(2) = theta(2) - alpha * tmp2 / m;
    
    % 保存每次迭代的 J 值
    J_history(iter) = computeCost(X, y, theta);
    
end

fprintf('After the last iteration: J = %f\n', J_history(num_iters));