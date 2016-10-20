function theta = randInitWeight(L_in, L_out)
epsilon =  sqrt(6) / sqrt(L_in + L_out);
theta = 2 * epsilon * rand(L_out, L_in + 1) - epsilon;
end