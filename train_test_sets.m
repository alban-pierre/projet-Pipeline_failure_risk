function [setx, sety] = train_test_sets(n, k, r)

    % Separate the data between k train and test sets, with k-fold (r = 0) or random (r > 0)
    % n is the size of the data
    % k is the size of the testing set
    % If r > 0, r is the number of different sets
    
    if (r == 0)

    else
        setx = zeros(n,k);
        for i=1:k
            sets(:,i) = randperm(n)';
        end
    end
end

