if (0)
[a, b,c,d,e,f,g] = textread ('input_test.csv', "%s %s %s %s %s %s %s", 'delimiter', ',', 1);
data = zeros(size(b-1),8);
data(:,2) = (cell2mat(b(2:end)) == "P")(:,2);
data(:,3) = 2*((cell2mat(c(2:end)) == "O")(:,2)) + (cell2mat(c(2:end)) == "U")(:,2);
data(:,4) = cell2mat(d(2:end));
data(:,5) = 2*((cell2mat(e(2:end)) == "C")(:,2)) + (cell2mat(e(2:end)) == "D")(:,2);
data(:,6) = cell2mat(f(2:end));
data(:,7) = cell2mat(g(2:end));
end





a = textread ('input_train.csv', "%s");
data = zeros(size(a,1)-1, 8);
warning('off');
for i=1:size(a,1)-1
    [x1,x2,x3,x4,x5,x6,x7,x8] = strread(a{i+1}, "%d %s %s %f %s %f %d %s", 'delimiter', ',',1);
    data(i,1) = x1;
    data(i,2) = any(x2{1} == "T");
    data(i,3) = any(x3{1} == "O") + 2*any(x3{1} == "U");
    data(i,4) = x4;
    data(i,5) = any(x5{1} == "D") + 2*any(x5{1} == "M");
    data(i,6) = x6;
    data(i,7) = x7;
    if (size(x8{1},2) == 2)
        data(i,8) = 0;
    else
        data(i,8) = str2num(x8{1});
    end
    if (mod(i,100) == 0)
        fprintf(2, '.');
    end
end
fprintf(2, '\n');   
warning('on');



a = textread ('output_train.csv', "%s");
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
warning('on');
