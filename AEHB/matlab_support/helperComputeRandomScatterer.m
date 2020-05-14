function [txang,rxang,g,scatpos] = helperComputeRandomScatterer(txcenter,rxcenter,Nscat)
% This function is only in support of ArrayProcessingForMIMOExample. It may
% be removed in a future release.

%   Copyright 2016 The MathWorks, Inc.

ang = 90*rand(1,Nscat)+45;
ang = (2*(rand(1,numel(ang))>0.5)-1).*ang;

r = 1.5*norm(txcenter-rxcenter);
scatpos = phased.internal.ellipsepts(txcenter(1:2),rxcenter(1:2),r,ang);
scatpos = [scatpos;zeros(1,Nscat)];
g = ones(1,Nscat);

[~,txang] = rangeangle(scatpos,txcenter);
[~,rxang] = rangeangle(scatpos,rxcenter);
