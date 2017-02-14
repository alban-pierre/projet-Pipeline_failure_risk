function [settrain, settest] = k_fold_train_test_sets(n, k, nbr, setrand)

    % Computes train and test sets based on a k_fold algorithm, reapeated nbr times

    % Input :
    % n       : (1*1) : The number of examples
    % k       : (1*1) : The k of k_fold, which means the n examples are divided into k groups
    % nbr     : (1*1) : The number of different k_fold divisions
    % setrand : (1*1) : Used for setting the random generator

    % Output :
    % settrain : {1*(nbr*k)} ((n-n/k)*1) : The indexes of train sets
    % settest  : {1*(nbr*k)} ((n/k)*1)   : The indexes of test sets

    if (nargin >= 4)
        if (setrand >= 0)
            rand('seed', setrand);
        end
    end
    
    if ((k>n) || (k<2))
        fprintf(2, 'Warning : invalid k for k_fold, k set to 2');
        k = 2;
    end

    sep = floor((0:n-1)/n*k);
    for j=1:nbr
        rr = randperm(n)';
        for i=1:k
            settest{i+(j-1)*k} = rr(sep == i-1);
            settrain{i+(j-1)*k} = rr(sep ~= i-1);
        end
    end
    
end

