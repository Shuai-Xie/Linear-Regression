function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
% X ����������y ����ʵ��ֵ��theta ģ�Ͳ�����alpha ѧϰ�ʣ�num_iters ��������

% ��ʼ��һЩֵ
m = length(y);
J_history = zeros(num_iters, 1); % �洢ÿ�ε�����Jֵ

for iter = 1:num_iters % ��������
    
    tmp1 = 0; % �洢�����
    tmp2 = 0;
    
    % ���������
    for i = 1:m
        tmp1 = tmp1 + (X(i, :) * theta - y(i)); % theta0 ��Ӧ X(i, 1) = 1
        tmp2 = tmp2 + (X(i, :) * theta - y(i)) * X(i, 2);
    end
    
    % ���� theta
    theta(1) = theta(1) - alpha * tmp1 / m;
    theta(2) = theta(2) - alpha * tmp2 / m;
    
    % ����ÿ�ε����� J ֵ
    J_history(iter) = computeCost(X, y, theta);
    
end

fprintf('After the last iteration: J = %f\n', J_history(num_iters));