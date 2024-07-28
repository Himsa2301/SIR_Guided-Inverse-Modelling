function [W_1, b] = random_weights

rng(1)

W_1 = 1*(2*rand(500,1) - 1);
b = 1*(2*rand(500,1) - 1);