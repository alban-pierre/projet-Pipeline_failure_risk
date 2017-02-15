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

end
