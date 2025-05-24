function [boot,pboot,xindex]=resample(x,w,N)
%
%
%

c0=clock;

if nargin<3
    N=length(w);
end

u=sort (rand(1,N));

%u=([0:N-1]+rand(1))/N;

%u=([0:N-1]+0.5)/N;

wc=cumsum(w);

ind= zeros(1,N);
k=1;
for i=1:N
    while(wc(k)<u(i))
        k=k+1;
    end
    ind(i)=k;
end;

boot=x(:,ind);
pboot=ones(1,N)./N;

xindex = ind;
c1=clock;

c1-c0;

