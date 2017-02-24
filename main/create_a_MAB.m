function MAB = create_a_MAB(datainitx, datainity)

    % Function that create a multi armed bandit setup

    % Dimensions :
    % N  : Number of examples
    % Dx : Dimension of examples
    % Dy : Dimension of prediction
    
    % Input :
    % datainitx : (N*Dx) : Input data
    % datainity : (N*Dy) : Output of ml algo

    % Output :
    % MAB : (structure) : Gathers all options of algorithms we want to test, including the data associated to each algo
    
    algo = algo_options();
    MAB.trainsize = 12951;
    MAB.testsize = 6476;
    
    MAB.datax{1} = datainitx(:,2:end);
    MAB.datay{1} = datainity(:,2:end);
    MAB.datax{1} = remove_constant_columns(add_power2_columns(MAB.datax{1}, ones(size(MAB.datax{1},2))));

    MAB.datax{2} = datainitx(:,2:end);
    MAB.datay{2} = datainity(:,2:end);

    %MAB.datax{3} = datainitx(:,2:end);
    MAB.datay{3} = datainity(:,2:end);
    %MAB.datax{3} = remove_constant_columns(add_power2_columns(MAB.datax{3}, ones(size(MAB.datax{3},2))));
    MAB.datax{3} = set_fixed_mean(MAB.datax{1});
    MAB.datax{3} = set_fixed_variance(MAB.datax{3});

    %MAB.datax{4} = datainitx(:,2:end);
    MAB.datay{4} = datainity(:,2:end);
    MAB.datax{4} = set_fixed_mean(MAB.datax{2});
    MAB.datax{4} = set_fixed_variance(MAB.datax{4});

    


    iarm = 1;

    algo.regression = 1;

    algo.regr_hyp = 100000;
    MAB.arm{iarm} = algo;
    MAB.data{iarm} = 1;
    iarm = iarm+1;

    algo.regression = 3;
    algo.deep.learn_rate = @(k) 10/sqrt(k); % best so far
    MAB.arm{iarm} = algo;
    MAB.data{iarm} = 4;
    iarm = iarm+1;

    algo.regression = 3;
    algo.deep.learn_rate = @(k) 5/sqrt(k); % best so far
    MAB.arm{iarm} = algo;
    MAB.data{iarm} = 4;
    iarm = iarm+1;

    algo.deep.learn_rate = @(k) 1;
    MAB.arm{iarm} = algo;
    MAB.data{iarm} = 4;
    iarm = iarm+1;

    algo.deep.learn_rate = @(k) 10;
    MAB.arm{iarm} = algo;
    MAB.data{iarm} = 4;
    iarm = iarm+1;


    if (0)
    algo.regression = 1;
    for i = -9:10
        algo.regr_hyp = 10^i;
        MAB.arm{iarm} = algo;
        MAB.data{iarm} = 1;
        iarm = iarm+1;
    end
    algo.kernel = 1;
    algo.regression = 2;
    for i = -5:0
        for j = 6:10
            algo.regr_hyp = 10^i;
            algo.kernel_hyp = 10^j;
            MAB.arm{iarm} = algo;
            MAB.data{iarm} = 1;
            iarm = iarm+1;
        end
    end
    algo.kernel = 2;
    algo.regression = 2;
    for i = -5:0
        for j = -3:10
            algo.regr_hyp = 10^i;
            algo.kernel_hyp = 10^j;
            MAB.arm{iarm} = algo;
            MAB.data{iarm} = 1;
            iarm = iarm+1;
        end
    end
    
    algo.regression = 1;
    for i = -9:10
        algo.regr_hyp = 10^i;
        MAB.arm{iarm} = algo;
        MAB.data{iarm} = 2;
        iarm = iarm+1;
    end
    algo.kernel = 1;
    algo.regression = 2;
    for i = -9:10
        for j = -9:10
            algo.regr_hyp = 10^i;
            algo.kernel_hyp = 10^j;
            MAB.arm{iarm} = algo;
            MAB.data{iarm} = 2;
            iarm = iarm+1;
        end
    end
    algo.kernel = 2;
    algo.regression = 2;
    for i = -9:10
        for j = -9:10
            algo.regr_hyp = 10^i;
            algo.kernel_hyp = 10^j;
            MAB.arm{iarm} = algo;
            MAB.data{iarm} = 2;
            iarm = iarm+1;
        end
    end
    end
    MAB.nbArms = iarm-1;

    MAB.draws = zeros(1,MAB.nbArms);

end
