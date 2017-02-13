function res = add_new_column(data, v)

    % Function that adds one last columns in data, constant equal to v

    % Dimensions :
    % N : number of examples of data
    % D : dimension of data

    % Input :
    % data : (N*D) : The data
    % v    : (1,1) : The value of the last columns

    % Output :
    % res : (N*(D+1)) : The data with ones constant column at the end

    N = size(data,1);

    res = [data, v*ones(N,1)];

end
