function res = set_fixed_mean(data, m)

    % Function that set each column of the data to a fixed mean

    % Dimensions :
    % N : number of examples of data
    % D : dimension of data

    % Input :
    % data : (N*D) : The data

    % Optionnal Input :
    % m : (1*D) or (1*1) : The fixed mean wanted
    
    % Output :
    % res : (N*D) : The data with fixed means

    if (nargin < 2)
        m = 0;
    end

    res = data + (m - mean(data,1));
        
end
