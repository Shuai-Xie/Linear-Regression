function [theta] = normalEqn(X, y)
% ���淽����theta

theta = pinv(X' * X) * X' * y; % ���÷���

end
