function res = add_power2_columns(data, m)

    % Function that adds columns to data, these columns are products of existing columns

    % Dimensions :
    % N : number of examples of data
    % D : dimension of data

    % Input :
    % data : (N*D) : The data
    % m    : (D*D) : Matrix containing information about which columns to multiply (1 in (i,j) correspond to a multiplication of column i and column j)

    % Output :
    % res : (N*(D+sum(m))) : The data with added columms

    assert(size(data, 2) == size(m,1), 'Error : The dimension of data and matrix must be the same');
    assert(size(data, 2) == size(m,2), 'Error : The dimension of data and matrix must be the same');

    mat = (m+m')>0;
    [N, D] = size(data);
    newcol = zeros(N, sum(sum(mat)));

    ic = 1;
    for i=1:D
        for j=i:D
            if (mat(i,j))
                newcol(:,ic) = data(:,i).*data(:,j);
                ic = ic + 1;
            end
        end
    end

    res = [data, newcol];

end
