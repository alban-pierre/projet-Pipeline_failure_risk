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
    MAB.trainsize = 12951-10000;
    MAB.testsize = 6476-4000;
    
    MAB.datax{1} = datainitx(:,2:end);
    MAB.datay{1} = datainity(:,2:end);
    MAB.datax{1} = remove_constant_columns(add_power2_columns(MAB.datax{1}, ones(size(MAB.datax{1},2))));
    MAB.datax{2} = datainitx(:,2:end);
    MAB.datay{2} = datainity(:,2:end);
    
    iarm = 1;
    algo.regression = 1;
    for i = -9:10
        algo.regr_hyp = 10^i;
        MAB.arm{iarm} = algo;
        MAB.data{iarm} = 1;
        iarm = iarm+1;
    end
    algo.kernel = 1;
    algo.regression = 2;
    for i = -9:10
        for j = -9:10
            algo.regr_hyp = 10^i;
            algo.kernel_hyp = 10^j;
            MAB.arm{iarm} = algo;
            MAB.data{iarm} = 1;
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
    MAB.nbArms = iarm-1;

    MAB.draws = zeros(1,MAB.nbArms);

end
