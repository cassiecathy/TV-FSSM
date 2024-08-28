function [C,A] = krondec(F,dim1,dim2,dim3,dim4)
% F = kron(C,A)
%    nsensor(s+1)*nsensor(s)  npc(s+1)*npc(s)
% C: dim1*dim2; A:dim3*dim4

    newF = [];
    x = 0:dim3:dim1*dim3;
    y = 0:dim4:dim2*dim4;
    for i = 1:dim1
        for j = 1:dim2
            subF = F(x(i)+1:x(i+1),y(j)+1:y(j+1));
            newF = [newF;reshape(subF,[],1)'];
        end
    end
    [U,S,V] = svd(newF);
    C = reshape(sqrt(S(1,1))*U(:,1),dim1,dim2);  
    A = reshape(sqrt(S(1,1))*V(:,1),dim3,dim4);
    
end