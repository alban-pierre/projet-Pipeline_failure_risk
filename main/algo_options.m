function algo = algo_options()

    % Function that computes options of the prediction

    %algo.kernel = 0; % Linear kernel
    algo.kernel = 1; % Gaussian kernel
    %algo.kernel = 2; % Laplacian kernel

    algo.kernel_hyp = 1; % Hyperparameter of a kernel

    %algo.regression = 0; % Useless random regression
    algo.regression = 1; % Ridge regression
    %algo.regression = 2; % Kernel ridge regression
    %algo.regression = 3; % Neural network regression

    algo.regr_hyp = 0.1; % Hyperparameter of a regression

    algo.error = 0; % AUC error
    %algo.error = 1; % Absolute error
    %algo.error = 2; % Square error

    algo.deep.sizes = [12,200,100,2];
    %algo.deep.sizes = [12,30,20,2];
    %algo.deep.sizes = [83,50,30,10,2]; % Sizes of layers of the neural network
    algo.deep.batchsize = 518; % Batch size for the stochastic gradient descend
    algo.deep.dropout = 0.0; % Dropout (0.8 = keep 20% of units)
    algo.deep.regularization = 0.5; % L2 regularization parameter of the neural network
    algo.deep.uniformbatch = 5; % Set it to 0 if some examples that have a rare output should be used more
    algo.deep.activation_function = 1; % 1 = Sigmoid, 2 = ReLU
    algo.deep.costfunction = 2; % 1 for square, 2 for cross entropy (better) (square does not have dropout or ReLU yet)
    algo.deep.learn_rate = @(k) 3/sqrt(k); % A coefficient, which describe the learning rate
    algo.deep.epoch = 25; % The number of epochs
    algo.deep.show_epoch_err = 1; % Set it to >0 to plot the training and testing error of the network as a function of epoch, the number is the figure number
    
end
