function res = set_fixed_variance(data, v)

    % Function that set each column of the data to a fixed variance

    % Dimensions :
    % N : number of examples of data
    % D : dimension of data

    % Input :
    % data : (N*D) : The data

    % Optionnal Input :
    % v : (1*D) or (1*1) : The fixed variances wanted
    
    % Output :
    % res : (N*D) : The data with fixed variances

    if (nargin < 2)
        v = 1;
    end

    N = size(data,1);
    
    m = mean(data,1);
    vv = (N/(N-1))*mean((data-m).^2,1);

    vv = vv.*(vv>0) + 1*(vv==0);
    
    res = (data-m).*(v./sqrt(vv)) + m;
        
end
