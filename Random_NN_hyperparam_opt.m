function Random_NN_hyperparam_opt
    
    clc
    clear

   %% Data for spread of Influenza virus in a boys boarding school
    t = 0:1:14; % day
    I = [1 3 8  28 75 221 291 255 235 190 126 70 28 12 5]; % number infected
    
    %% Hyperparamter tuning of the neural net 
    lamda = [1e-12, 1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1];
    min_error = inf;
    for r = 0.1:0.1:1 % Badwidth of the random parameters in the 0first layer
        for j = 1:13 % index over different lamda values
            val_error = 0; % sum of errors on the validation set
            for k = 1:numel(t) % index of datapoints for leave ount out cross validation
                rng(1)
                W_1 = r*(2*rand(500,1) - 1);
                b = r*(2*rand(500,1) - 1);
                n = numel(b); % number of neurons in the hidden layer
                    
                % Construct data  matrix based on the governing PDE
                t_train = [t(1:k-1),t(k+1:end)]; 
                y_train = [I(1:k-1),I(k+1:end)]; 
                X = zeros(numel(t_train),n);
                for i = 1:numel(t_train)
                    X(i,:) = tanh(W_1*t_train(i) + b)';
                end
            
                w_2 = inv(X'*X + lamda(j)*eye(n))*X'*y_train';

                val_error = val_error + (tanh(W_1*t(k) + b)'*w_2 - I(k))^2;    
            end
            if val_error < min_error
                r_opt = r;
                lamda_opt = lamda(j);
                min_error = val_error;
            end
        end
    end

    r_opt
    lamda_opt

    rng(1)
    W_1 = r_opt*(2*rand(500,1) - 1);
    b = r_opt*(2*rand(500,1) - 1);
    n = numel(b); % number of neurons in the hidden layer

    % Train the model on all data with the optimized hyperparameters
    X = zeros(numel(t),n);
    for i = 1:numel(t)
        X(i,:) = tanh(W_1*t(i) + b)';
    end

    w_2 = inv(X'*X + lamda_opt*eye(n))*X'*I';

    t_plot = 0:0.1:14;
    plot(t_plot, tanh(W_1*t_plot + b)'*w_2)
    hold on
    plot(t,I,'*')

end
