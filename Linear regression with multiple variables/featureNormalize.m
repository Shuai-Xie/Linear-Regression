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

% 缩放 X 的2个特征
for i = 1:m
	X_norm(i, 1) = ( X_norm(i, 1) - mu(1) ) / sigma(1);
	X_norm(i, 2) = ( X_norm(i, 2) - mu(2) ) / sigma(2);
end

end
