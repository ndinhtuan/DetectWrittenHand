function [J, grad] = NNsCost(allTheta, X, y, ...
                                        unitsInput, sizeHidden1, sizeHidden2, unitsOutput, lambda)
  %reshape Theta with allTheta because we need Theta change after each loop in fminunc, so change allTheta and not reference Theta
  Theta1 = reshape(allTheta(1:sizeHidden1 * (unitsInput + 1)), sizeHidden1, unitsInput + 1);
  Theta2 = reshape(allTheta(sizeHidden1 * (unitsInput + 1) + 1 : sizeHidden1 * (unitsInput + 1) + sizeHidden2 * (sizeHidden1 + 1)), sizeHidden2, ...
                           sizeHidden1 + 1);
  Theta3 = reshape(allTheta(sizeHidden1 * (unitsInput + 1) + sizeHidden2 * (sizeHidden1 + 1) + 1 : end), unitsOutput, sizeHidden2 + 1);
  % comput cost function J
  m = size(X, 1);
  
  a1 = [ones(m, 1) X];
  z2 = a1 * Theta1';
  a2 = [ones(m, 1) sigmoid(z2)];
  z3 = a2 * Theta2';
  a3 = [ones(m, 1) sigmoid(z3)];
  z4 = a3 * Theta3';
  H = a4 = sigmoid(z4);
  
  J = (-1 / m) * sum(y .* log(H) + (1 - y) .* log(1 - H)); % H has 1 output
  J = J + (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end).^2)) + sum(sum(Theta3(:, 2:end).^2)));
  
  %compute grad
  D1 = zeros(size(Theta1));
  D2 = zeros(size(Theta2));
  D3 = zeros(size(Theta3));
  
  for i = 1:m
    %forward
    a1 = [1 X(i, :)];
    z2 = a1 * Theta1';
    a2 = [1 sigmoid(z2)];
    z3 = a2 * Theta2';
    a3 = [1 sigmoid(z3)];
    z4 = a3 * Theta3';
    a4 = sigmoid(z4);
    %backpropagation
    delta4 = a4 - y(i);
    delta3 = Theta3'*delta4' .* a3' .* (1 - a3'); delta3 = delta3(2:end);
    delta2 = Theta2' * delta3 .* a2' .* (1 - a2'); delta2 = delta2(2:end);
    
    D1 = D1 + delta2.*a1;
    D2 = D2 + delta3.*a2;
    D3 = D3 + delta4'.*a3;
  end
  
  temp1 = Theta1; temp1(:, 1) = zeros(size(Theta1, 1), 1);
  grad1 = (1 / m) * (D1 + lambda * temp1);
  temp2 = Theta2; temp2(:, 1) = zeros(size(Theta2, 1), 1);
  grad2 = (1 / m) * (D2 + lambda * temp2);
  temp3 = Theta3; temp3(:, 1) = zeros(size(Theta3, 1), 1);
  grad3 = (1 / m) * (D3 + lambda * temp3);
  
  grad = [grad1(:); grad2(:); grad3(:)];
end