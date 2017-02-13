function [settrain, settest] = random_train_test_sets(sizetrain, sizetest, nbr, setrand)

    % Computes train and test sets based on a random algorithm, reapeated nbr times

    % Input :
    % sizetrain : (1*1) : The training size
    % sizetest  : (1*1) : The testing size
    % nbr       : (1*1) : The number of training and testing sets
    % setrand   : (1*1) : Used for setting the random generator

    % Output :
    % settrain : {1*nbr} (sizetrain*1) : The indexes of train sets
    % settest  : {1*nbr} (sizetest*1)   : The indexes of test sets

    if (nargin >= 4)
        rand('seed', setrand);
    end
    
    for i=1:nbr
        rr = randperm(sizetrain+sizetest)';
        settest{i} = rr(1:sizetest);
        settrain{i} = rr(sizetest+1:end);
    end
    
end

