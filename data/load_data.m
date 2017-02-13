function [datax, datay] = load_data(test);

% Function that loads the data (the test data if an argument is given)

    if (nargin > 0)
        a = textread ('../dataVeolia/input_test.csv', "%s");
        datay = 0;
    else
        a = textread ('../dataVeolia/input_train.csv', "%s");
    end
    datax = zeros(size(a,1)-1, 13);
    warning('off');
    for i=1:size(a,1)-1
        [x1,x2,x3,x4,x5,x6,x7,x8] = strread(a{i+1}, "%d %s %s %f %s %f %d %s", 'delimiter', ',',1);
        datax(i,1) = x1;
        datax(i,2) = any(x2{1} == "P");
        datax(i,3) = any(x2{1} == "T");
        datax(i,4) = any(x3{1} == "I");
        datax(i,5) = any(x3{1} == "O");
        datax(i,6) = any(x3{1} == "U");
        datax(i,7) = x4;
        datax(i,8) = any(x5{1} == "C");
        datax(i,9) = any(x5{1} == "D");
        datax(i,10) = any(x5{1} == "M");
        datax(i,11) = x6;
        datax(i,12) = x7;
        if (size(x8{1},2) == 2)
            datax(i,13) = 0;
        else
            datax(i,13) = str2num(x8{1});
        end
        if (mod(i,100) == 0)
            fprintf(2, '.');
        end
    end
    fprintf(2, '\n');   
    warning('default');

    if (nargin <= 0)
        a = textread ('../dataVeolia/output_train.csv', "%s");
        datay = zeros(size(a,1)-1, 3);
        warning('off');
        for i=1:size(a,1)-1
            [x1,x2,x3] = strread(a{i+1}, "%d %d %d", 'delimiter', ';',1);
            datay(i,1) = x1;
            datay(i,2) = x2;
            datay(i,3) = x3;
            if (mod(i,100) == 0)
                fprintf(2, '.');
            end
        end
        fprintf(2, '\n');
        warning('default');
    end
end
