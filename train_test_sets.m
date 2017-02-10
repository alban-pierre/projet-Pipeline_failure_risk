function [settrain, settest] = train_test_sets(n, k, r)

    % Separate the data between k train and test sets, with k-fold (r = 0) or random (r > 0)
    % n is the size of the data
    % k is the size of the testing set
    % If r > 0, r is the number of different sets
    
    if (r == 0)
        
    else
        settrain = zeros(n-k,r);
        settest = zeros(k,r);
        for i=1:r
            rr = randperm(n)';
            settest(:,i) = rr(1:k);
            settrain(:,i) = rr(k+1:end);
        end
    end
end

