function [X_norm, mu, sigma] = featureNormalize(X)
% ��׼�� X ��������[0,1]֮��

% ��ʼ��
X_norm = X; % X(47*2)

% ����matlab�Դ�����mean��std��ƽ��ֵ�ͱ�׼��
mu = mean(X); % mu(1*2) a row vector containing the mean value of each column. 
sigma = std(X); % sigma(1*2) 

% size(X, 1) ��� X ������������������� = 47
% size(X, 2) ��� X ������������������� = 2
m = size(X, 1);

% ���� X ��2������
for i = 1:m
	X_norm(i, 1) = ( X_norm(i, 1) - mu(1) ) / sigma(1);
	X_norm(i, 2) = ( X_norm(i, 2) - mu(2) ) / sigma(2);
end

end
