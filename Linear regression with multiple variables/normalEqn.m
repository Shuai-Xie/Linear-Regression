function [theta] = normalEqn(X, y)
% 正规方程求theta

theta = pinv(X' * X) * X' * y; % 调用方程

end
