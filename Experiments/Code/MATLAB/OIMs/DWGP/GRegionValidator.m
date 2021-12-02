function mask = GRegionValidator(mask, minimal)
% mask = GRegionValidator(mask, minimal) sweeps off the glimpse regions formed by less than given
% mimimal number of the glimpsing points.
%
% input:
%        mask          original a priori mask with potential glimpsing regions.
%        minimal       minimal number of the glimpsing points forming a reliable glimping region
%
% output:
%        mask          a priori mask with glimpsing regions formed by no less than given mimimal 
%                      number of the glimpsing points.
%
% Author: Yan Tang


if nargin < 2
    minimal =  1;
end

if minimal < 1
    minimal =  1;
end

% convert boolean to signle precision
mask = single(mask);

% limit the recursion time per call due to maximal recursion limitaton on matlab
% default 500
maxrec = get(0,'RecursionLimit') - 100;

[c, l] = find(mask==1,1,'first');
while ~isempty(c)
    candidate = [];
    [candidate, isleft] = exploreGRegion([c,l], mask, candidate, maxrec);
    candNum = length(candidate); % get candidate number
    if (candNum < minimal && ~isleft)
        flag = 0; % if smaller than given minimal, sweep off
    else
        flag = 2; % otherwise mark up
    end
    for idx = 1:candNum
        mask(candidate(idx)) = flag; % do labelling
    end
    [c, l] = find(mask==1,1,'first'); % find next seed
end

% the final valid points are those flaged as 2;
mask = (mask==2);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PRIVATE FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [candidate, isleft] = exploreGRegion(seedidx, mask, candidate, rec)
% [candidate, isleft] = exploreGRegion(seedidx, mask, candidate, rec) recursively finds thoe 
% connected glmpsing points. Those points must be in adjacent frequency channels at the same time 
% frame, or in adjacent time frames of the same frequency channel.

% input:
%        seedidx       location of a seed glimpsing point on the maske
%        mask          a priori maske having glimpsing points
%        candidate     a holder for existing adjacent glmpsing points
%        rec           maximal number of recursion
% output:
%        candidate     the holder for all condidates found when a recursion stops
%        isleft        a boolean indicating if those candidate points are leftover from last
%                      recursion
%
% Author: Yan Tang


[chans, frames] = size(mask);

seed = mask(seedidx(1), seedidx(2)); %retrieve pixel value
candidx = (seedidx(2) - 1) * chans + seedidx(1);
isleft = 0;
if rec > 0
    if sum(candidate==candidx) == 0
        if seed == 1
            % add current seed into candidate holder;
            candidate(length(candidate)+1) = candidx;
            
            % check availability of four neighbours
            next = zeros(4,2);
            
            % the one in previous channel but same frame
            if seedidx(1) > 1
                next(1,:) = [seedidx(1)-1, seedidx(2)];
            end
            
            % the one in next channel but same frame
            if seedidx(1) < chans
                next(2,:) = [seedidx(1)+1, seedidx(2)];
            end
            
            % the one in previous frame but same channel
            if seedidx(2) > 1
                next(3,:) = [seedidx(1), seedidx(2)-1];
            end
            
            % the one in next frame but same channel
            if seedidx(2) < frames
                next(4,:) = [seedidx(1), seedidx(2)+1];
            end
            
            next(next(:,1)==0,:) = []; %clean up unavilable possibility
            
            for idx = 1:size(next,1)
                [candidate, tmp] = exploreGRegion(next(idx,:), mask,candidate, rec-1);
                if tmp
                    isleft = 1;
                end
            end        
        elseif seed == 2
            isleft = 1;
        end
    end
end