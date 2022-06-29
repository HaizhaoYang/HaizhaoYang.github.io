module TestFun

export testfun, testfun2

function testfun(varargin...)
#
# what fucking happening
#
global z
x = varargin[1]; y =varargin[2];
return x, y
end

function testfun2(varargin...)

return 1, 2
end

function testfun3(varargin...)

return 1, 2
end


end