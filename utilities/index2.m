function [out_MSE,out_MSE_std] = index2(b_sum,y_sum,means,stds)
nmethod = size(b_sum,1);
b_ex = b_sum(1,:);
y_true = y_sum(1,:);

nstage = size(b_ex,2);
ntime = zeros(1,nstage);
nsensor = zeros(1,nstage);
npc = zeros(1,nstage);
for s = 1:nstage
    [ntime(s),nsensor(s),nsam] = size(y_true{s});
    [~,npc(s)] = size(b_ex{s});
end
MSE = zeros(nsam,nmethod);

% »•πÈ“ªªØ
for i = 1:nmethod+1
    for s = 1:nstage
        y_sum{i,s} = reshape(y_sum{i,s},[],100)';
        y_sum{i,s} = y_sum{i,s}.*stds{s}+means{s};
        y_sum{i,s} = reshape(y_sum{i,s}',ntime(s),nsensor(s),100);
    end
end


%% MSE/MAE
y_error = cell(nmethod,nstage);

for i = 1:nmethod
    for j = 1:nsam
        sq = [];
        for s = 1:nstage
            y_error{i,s} = y_sum{1,s} - y_sum{i+1,s};
            sq = [sq, reshape(y_error{i,s}(:,:,j).^2,1,[])];
        end
        MSE(j,i) = mean(sq);
    end
end
out_MSE = mean(MSE) ;
out_MSE_std = std(MSE);

% y_sumvec = cell(nmethod+1,nstage);
% for s = 1:nstage
%     for i = 1:nmethod+1 
%         y_sumvec{i,s} = reshape(y_sum{i,s},[ntime(s)*nsensor(s),nsam]);
%     end
% end
% 
% for i = 1:nmethod
%     yp_temp = y_sumvec(i+1,:);
%     sq = zeros(1,nsam);
%     ab = zeros(1,nsam);
%     for k = 1:nsam
%         temp1 = 0;
%         temp2 = 0;
%         for s = 1:nstage
%             temp1 = temp1+(norm(y_sumvec{1,s}(:,k)-yp_temp{s}(:,k)))^2;
%             temp2 = temp2+ sum(abs(y_sumvec{1,s}(:,k)-yp_temp{s}(:,k)));
%         end
%         sq(k) = temp1;
%         ab(k) = temp2;
%     end
%     MSE(i) = mean(sq);
%     MSE_std(i) = std(sq);
%     MAE(i) = mean(ab);
%     MAE_std(i) = std(ab);
%     
% end



end