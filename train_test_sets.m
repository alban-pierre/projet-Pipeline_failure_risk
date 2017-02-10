function [settrain, settest] = train_test_sets(n, k, r)

    % Separate the data between k train and test sets, with k-fold (r = 0) or random (r > 0)
    % n is the size of the data
    % If r = 0, k is the number of the testing sets (the k from k-fold)
    % If r > 0, k is the size of the testing set and r is the number of different sets
    
    if (r == 0)
        sep = floor((0:n-1)/n*k);
        rr = randperm(n)';
        for i=1:k
            settest{i} = rr(sep == i-1);
            settrain{i} = rr(sep ~= i-1);
        end
    else
        for i=1:r
            rr = randperm(n)';
            settest{i} = rr(1:k);
            settrain{i} = rr(k+1:end);
        end
    end
end

