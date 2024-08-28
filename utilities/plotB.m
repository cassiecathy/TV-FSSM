function plotB(B)
nstage = size(B,2);
npc = zeros(1,nstage);
for s = 1:nstage
    npc(s) = size(B{s},2);
end

for s = 1:nstage
    figure('position',[200*s,200,200,80]);
    for k = 1:npc(s)
        plot(B{s}(:,k));hold on;
    end
    ylabel('Bases');
    set(gca, 'Fontname', 'Times New Roman','FontSize',8);
end


end