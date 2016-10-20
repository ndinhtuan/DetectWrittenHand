function newX = normalize(X)
  average = mean(X);
  sigma = std(X);
  newX = (X - average) ./ sigma;
end