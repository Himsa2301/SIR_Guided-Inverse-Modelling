function SIR_guided_inverse_modelling
    
    clc
    clear
    rng(1)
    
    %% Data for spread of Influenza virus in a boys boarding school
    t = 0:1:14; % day
    I = [1 3 8  28 75 221 291 255 235 190 126 70 28 12 5]; % number infected

    %% Train the Infected prediction model separately
    rng(1)
    W_1 = 0.2*(2*rand(500,1) - 1);
    b = 0.2*(2*rand(500,1) - 1);
    n = numel(b); % number of neurons in the hidden layer

    % Train the model on all data with the optimized hyperparameters
    X_comb = zeros(numel(t),n);
    for i = 1:numel(t)
        X_comb(i,:) = tanh(W_1*t(i) + b)';
    end

    w_2 = inv(X_comb'*X_comb + 1e-6*eye(n))*X_comb'*I';

    % Augment the infected number dataset with this trained model
    t = 0:0.05:14;
    I = (tanh(W_1*t + b)'*w_2)'; % predicted infected numbers
       
    %% Constructing a neural network with two heads/outputs for S (susceptible) and R (recovered), respectively
    n = 500; % number of hidden neurons in the randomized NN
    [W_1,b] = random_weights;
        
    % Construction of matrices based on the governing system of differential equations
    t_colloc = t; 
    XS = zeros(numel(t_colloc),n);
    XR = zeros(numel(t_colloc),n);
    XS_t = zeros(numel(t_colloc),n);
    XR_t = zeros(numel(t_colloc),n);
    yS = zeros(numel(t_colloc),1);
    for k = 1:numel(t_colloc)
        XS(k,:) = tanh(W_1*t_colloc(k) + b)';
        XR(k,:) = tanh(W_1*t_colloc(k) + b)';
        XS_t(k,:) = ((1- tanh(W_1*t_colloc(k) + b).^2).*(W_1))';
        XR_t(k,:) = ((1- tanh(W_1*t_colloc(k) + b).^2).*(W_1))';
    end
    
    % Get the optimized parameter estimates based on the variable
    % projection approach, and the corresponding optimized output layer
    % weights of the random neural net
    params = fminsearch(@(params) sqerror(params,XS, XR,  XS_t, XR_t, yS, I, n, numel(t)), [1,1])

    %% Plot the final results for S I and R
    alpha = params(1);
    beta = params(2);
    
    m = numel(t);
    X_comb = zeros(3*m + 1, 2*n);
    X_comb(1:m,1:n) = XS_t + beta*XS.*I';
    X_comb(m+1:2*m,n+1:2*n) = XR_t;
    X_comb(2*m+1:3*m,1:n) = XS;
    X_comb(2*m+1:3*m,n+1:2*n) = XR;
    X_comb(3*m+1,n+1:2*n) = 10*XR(1,:);

    y_comb = [yS; alpha*I'; 763*ones(m,1) - I'; 0];

    w_opt = pinv(X_comb)*y_comb;%inv(X'*X + 0.000001*eye(2*n))*X'*y;
    
    w_optS = w_opt(1:n);
    w_optR = w_opt(n+1:2*n);

    figure(1)
    plot(t, tanh(W_1*t + b)'*w_optS)
    title('Susceptible')
    figure(2)
    plot(t, tanh(W_1*t + b)'*w_optR)
    title('Recovered')
    figure(3)
    plot(t, 763 - tanh(W_1*t + b)'*w_optR - tanh(W_1*t + b)'*w_optS)
    title('Infected')

end

function loss = sqerror(params,XS,XR, XS_t, XR_t, yS, I, n, m)

    alpha = params(1);
    beta = params(2);

    X_comb = zeros(3*m + 1, 2*n);
    X_comb(1:m,1:n) = XS_t + beta*XS.*I';
    X_comb(m+1:2*m,n+1:2*n) = XR_t;
    X_comb(2*m+1:3*m,1:2*n) = [XS,XR];
    X_comb(3*m+1,n+1:2*n) = XR(1,:);

    y_comb = [yS; alpha*I'; (763*ones(m,1) - I'); 0];

    w_opt =  pinv(X_comb)*y_comb;

    loss = norm(X_comb*w_opt - y_comb);    
end