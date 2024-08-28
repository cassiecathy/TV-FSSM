function plotY(data,islen,len)
if nargin == 1
    islen = 0;
end

nstage = size(data,2);
ntime = zeros(1,nstage);
nsensor = zeros(1,nstage);
subf1 = zeros(1,nstage);
subf2 = zeros(1,nstage);
for s = 1:nstage
    [ntime(s),nsensor(s),nsam] = size(data{s});
    switch nsensor(s)
        case 1
            subf1(s) = 1;subf2(s) = 1;
        case 2
            subf1(s) = 2;subf2(s) = 1;
        case 3
            subf1(s) = 3;subf2(s) = 1;
        case 4
            subf1(s) = 2;subf2(s) = 2;
        case 5
            subf1(s) = 3;subf2(s) = 2;
        case 6
            subf1(s) = 3;subf2(s) = 2;
        case 7
            subf1(s) = 4;subf2(s) = 2;
        case 8
            subf1(s) = 4;subf2(s) = 2;
        case 9
            subf1(s) = 3;subf2(s) = 3;
        case 10
            subf1(s) = 2;subf2(s) = 5;
        case 12
            subf1(s) = 3;subf2(s) = 4;
        case 15
            subf1(s) = 3;subf2(s) = 5;
    end
end

for s = 1:nstage
    figure;
    for j = 1:nsensor(s)
        subplot(subf1(s),subf2(s),j);
        for i = 1:nsam
            plot(data{s}(:,j,i));hold on;
        end
        xlim([1,ntime(s)]);
        ylabel(['y',num2str(j)]);
        if j > nsensor(s)-subf2(s)
            xlabel('Time Order','Fontname', 'Times New Roman','FontSize',10);
        end
    end
    sgtitle(['stage',num2str(s)]);
    if islen == 1
        legend(len);
    end
    set(gca, 'Fontname', 'Times New Roman','FontSize',10);
end
end