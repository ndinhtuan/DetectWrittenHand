function [J grad] = nnCost(nn_params, X, y, ...
                                     inputLayerSize,  hiddenLayer1Size,  hiddenLayer2Size, numLabels, lambda)

m = size(X, 1);

Theta1 = reshape(nn_params(1:hiddenLayer1Size * (inputLayerSize + 1)), hiddenLayer1Size, inputLayerSize + 1);
Theta2 = reshape(nn_params(hiddenLayer1Size * (inputLayerSize + 1) + 1 : hiddenLayer1Size * (inputLayerSize + 1) + hiddenLayer2Size * (hiddenLayer1Size + 1)), ...
                          hiddenLayer2Size, hiddenLayer1Size + 1);
Theta3 = reshape(nn_params(hiddenLayer1Size * (inputLayerSize + 1) + 1 + hiddenLayer2Size * (hiddenLayer1Size + 1) : end), ...
                           numLabels, hiddenLayer2Size + 1);
                           
% compute cost function
%%forward 
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = [ones(m, 1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = [ones(m, 1) sigmoid(z3)];
z4 = a3 * Theta3';
H = a4 = sigmoid(z4);

J = (-1 / m) * sum(y .* log(H) + (1 - y) .* log(1 - H));
reg = sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2))  % regularization
        + sum(sum(Theta3(:, 2:end).^2));  
J = J + (lambda / (2 * m)) * reg;

%backward
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));
D3 = zeros(size(Theta3));

for t = 1:m,
  a1 = [1 X(t, :)];
  z2 = a1 * Theta1';
  a2 = [1 sigmoid(z2)];
  z3 = a2 * Theta2';
  a3 = [1 sigmoid(z3)];
  z4 = a3 * Theta3';
  H = a4 = sigmoid(z4);
  
  %compute delta for each node.
  delta4 = H - y(t);
  delta3 = Theta3' * delta4 .* a3' .* (1 - a3'); delta3 = delta3(2 : end);
  delta2 = Theta2' * delta3 .* a2' .* (1 - a2'); delta2 = delta2(2:end);
  
  D1 = D1 + delta2 .* a1;
  D2 = D2 + delta3 .* a2;
  D3 = D3 + delta4 .* a3;
  
end
%partial derivative
tmp1 = Theta1; tmp1(:, 1) = zeros(size(Theta1, 1), 1);
grad1 = (1 / m) * D1 + (lambda / m) * tmp1;
tmp2 = Theta2; tmp2(:, 1) = zeros(size(Theta2, 1), 1);
grad2 = (1 / m) * D2 + (lambda / m) * tmp2;
tmp3 = Theta3; tmp3(:, 1) = zeros(size(Theta3, 1), 1);
grad3 = (1 / m) * D3 + (lambda / m) * tmp3;

%unrolling
grad = [grad1(:); grad2(:); grad3(:)];
end