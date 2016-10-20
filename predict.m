function pred = predict(X, Theta1, Theta2, Theta3)

m = size(X, 1);
pred = zeros(m, 1);

a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = [ones(m, 1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = [ones(m, 1) sigmoid(z3)];
z4 = a3 * Theta3';
H = a4 = sigmoid(z4);

for i = 1:m,
  if(H(i) >= 0.5) pred(i) = 1;
  else pred(i) = 0;
end;

end