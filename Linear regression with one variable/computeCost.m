function J = computeCost(X, y, theta)

% 初始化一些值
m = length(y); % 训练集数量

E = X * theta - y; % 偏差矩阵
J = (E' * E) / (2 * m); % E' * E 平方和

end