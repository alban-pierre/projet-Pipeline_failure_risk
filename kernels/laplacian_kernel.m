function k = laplacian_kernel(x1, x2, hyp1)

    % Function that computes the laplacian kernel

    % Dimensions :
    % N1 : First size of x1
    % N2 : First size of x2
    % D : Second size of x1 and x2

    % Input :
    % x1   : (N1*D) : Examples 1
    % x2   : (N2*D) : Examples 2
    % hyp1 : (1*1)  : The sigma hyperparameter

    % Output :
    % k : (N1*N2) : The laplacian kernel of x1 and x2

    assert(nargin >= 2, 'Error : gaussian_kernel requires at least 2 arguments : x1 and x2');
    assert(size(x1,2) == size(x2,2), 'To compute the gaussian kernel, examples must have the same dimension');

    if (nargin < 3)
        sigma = 1;
    else
        sigma = hyp1;
    end
    
    d = onedist(x1', x2');
    k = exp(-d./sigma);
    
end
