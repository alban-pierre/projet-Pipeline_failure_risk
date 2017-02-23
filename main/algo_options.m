function algo = algo_options()

    % Function that computes options of the prediction

    %algo.kernel = 0; % Linear kernel
    algo.kernel = 1; % Gaussian kernel
    %algo.kernel = 2; % Laplacian kernel

    algo.kernel_hyp = 1; % Hyperparameter of a kernel

    %algo.regression = 0; % Useless random regression
    algo.regression = 1; % Ridge regression
    %algo.regression = 2; % Kernel ridge regression

    algo.regr_hyp = 100000; % Hyperparameter of a regression

    algo.error = 0; % AUC error
    %algo.error = 1; % Absolute error
    %algo.error = 2; % Square error

    algo.deep.sizes = [12,20,10,2];
    algo.deep.batchsize = 1295;
    algo.deep.dropout = 0.7;
	algo.deep.uniformbatch = 1;
	algo.deep.costfunction = 2; % 1 for square, 2 for cross entropy (better)
    
end
