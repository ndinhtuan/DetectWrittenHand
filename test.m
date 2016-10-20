X = load("test_data.txt");

m = size(X, 1);

Theta1 = load("nnTheta1.txt");
Theta2 = load("nnTheta2.txt");
Theta3 = load("nnTheta3.txt");

H = predict(X, Theta1, Theta2, Theta3);

save result.txt H;