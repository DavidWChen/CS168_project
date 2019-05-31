function testSpiral2()

load('./twoSpirals.mat');

% not displaying the label Y 
plot(X(:,1), X(:,2), '.');  hold on;

% displaying the label Y 
%for i=1:length(X)
%    plot(X(i,1), X(i,2), '.', 'Color', [Y(i)./2  Y(i)./2 1]); hold on;
%end

options.t = 1;

idx = randperm(numel(Y));
X = X(idx,:);
Y = Y(idx,:);

[model] = incrementalLearn(X, Y, options);

for i=1:25
    plot(model.X(i,1), model.X(i,2), 'rx'); hold on;
end