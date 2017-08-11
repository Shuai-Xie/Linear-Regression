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
