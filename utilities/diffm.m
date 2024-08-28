function diffm = diffm(ntime)
diffm = zeros(ntime-1,ntime);
for i = 1:ntime-1
    for j = 1:ntime
        if i == j
            diffm(i,j) = 1;
        elseif i == j-1
            diffm(i,j) = -1;
        end
    end
end
        

end