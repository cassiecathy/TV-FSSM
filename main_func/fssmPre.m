function yp = fssmPre(Y,model,isshow)
if nargin == 2
    isshow = 0;
end
B = model.B;
nstage = size(Y,2);
ntime = zeros(1,nstage);
nsensor = zeros(1,nstage);
npc = zeros(1,nstage);
for s = 1:nstage
    [ntime(s),nsensor(s),nsam] = size(Y{s});
    npc(s) = size(B{s},2);
end

[~, ~, ~,xp] = EstepOnline(model, Y);
% xp = Kalmanfilter(model,Y);
% xp = Estep(model, Y);

for s = 1:nstage
    xp{s} = reshape(xp{s},[npc(s),nsensor(s),nsam]);
end

yp = cell(1,nstage);
for s = 1:nstage
    for i = 1:nsam
        yp{s}(:,:,i) = B{s}*xp{s}(:,:,i);
    end
end

if isshow == 1
    % plot
    temp = 1;
    Ysum = cell(1,nstage);
    for s = 1:nstage
        Ysum{s}(:,:,1) = Y{s}(:,:,temp);
        Ysum{s}(:,:,2) = yp{s}(:,:,temp);
    end
    plotY(Ysum,1,{'true','fssm forecast'});
end
end