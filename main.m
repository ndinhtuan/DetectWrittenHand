clear; close all; clc;

inputLayerSize = 784;
hiddenLayer1Size = 350;
hiddenLayer2Size = 350;
numLabels = 10;

X = loadMNISTImages("train-images.idx3-ubyte");
X = X'(1:1000, :);
%y = data(:, 401);
y = loadMNISTLabels("train-labels.idx1-ubyte");
y = y(1:1000);
%X = normalize(X);

%init Theta
Theta1 = randInitWeight(inputLayerSize, hiddenLayer1Size);
Theta2 = randInitWeight(hiddenLayer1Size, hiddenLayer2Size);
Theta3 = randInitWeight(hiddenLayer2Size, numLabels);

%unrolling 
init_nn_params = [Theta1(:); Theta2(:); Theta3(:)]

option = optimset('MaxIter', 250);
lambda = 0.8;

%costFunction = @(p) nnCost(p, X, y, inputLayerSize, hiddenLayer1Size, hiddenLayer2Size, numLabels, lambda);
costFunction = @(p) NNsCost(p, X, y, inputLayerSize, hiddenLayer1Size, hiddenLayer2Size, numLabels, lambda);
[nn_params, cost] = fmincg(costFunction, init_nn_params, option);

Theta1 = reshape(nn_params(1:hiddenLayer1Size * (inputLayerSize + 1)), hiddenLayer1Size, inputLayerSize + 1);
save Theta1.txt Theta1;
Theta2 = reshape(nn_params(hiddenLayer1Size * (inputLayerSize + 1) + 1 : hiddenLayer1Size * (inputLayerSize + 1) + hiddenLayer2Size * (hiddenLayer1Size + 1)), ...
                          hiddenLayer2Size, hiddenLayer1Size + 1);
save Theta2.txt Theta2;
Theta3 = reshape(nn_params(hiddenLayer1Size * (inputLayerSize + 1) + 1 + hiddenLayer2Size * (hiddenLayer1Size + 1) : end), ...
                           numLabels, hiddenLayer2Size + 1);
 save Theta3.txt Theta3;

%dataVal = load("dataValidation.txt");
%Xval = dataVal(:, 1:400); yVal = dataVal(:, 401);
%Xval = normalize(Xval);
Xval = loadMNISTImages("t10k-images.idx3-ubyte");
Xval = Xval';
yval = loadMNISTLabels("t10k-labels.idx1-ubyte");

pred = predict(Xval, Theta1, Theta2, Theta3);

printf("Validation accuracy : %f.\n", mean(double(pred == yVal)) * 100);

