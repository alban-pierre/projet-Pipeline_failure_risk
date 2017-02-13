function res = remove_constant_columns(data)

    % Function that remove the constant columns

    % Dimensions :
    % N : number of examples of data
    % D : dimension of data

    % Input :
    % data : (N*D) : The data
    
    % Output :
    % res : (N*(<=D)) : The data without removed columms

    m = mean(data,1);

    keep = (sum(abs(data-m),1) > 0);

    res = data(:, keep);
    
end
