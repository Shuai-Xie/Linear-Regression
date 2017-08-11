function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

% Initialize some useful values
J_history = zeros(num_iters, 1);
m = length(y);

% ����
for iter = 1:num_iters
    
    tmp = (X' * ( X * theta - y )); % �������� ����� (3*m)*(m*1)=(3*1)
    theta = theta - alpha / m * tmp;
    
    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta); % ����theta��J
    
end
