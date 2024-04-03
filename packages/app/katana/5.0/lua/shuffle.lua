local function TableShuffle(size)
    math.randomseed(size)
    local new = {}
    for i=1, size do
        new[i] = i
    end
    for i=size, 1, -1 do
        local rand = math.random(size)
        new[i], new[rand] = new[rand], new[i]
    end
    return new
end

local function ArrayShuffle(size, ratio)
    local intArray = Array('int', {})
    intArray:resize(size)
    if size > 10 then
        local nsize = math.floor(size * ratio)
        local temp  = TableShuffle(size)
        for i=1, nsize do
            intArray:set(temp[i]-1, 1)
        end
    else
        for i=0, size-1 do
            intArray:set(i, 1)
        end
    end
    return intArray
end

return {
    TableShuffle = TableShuffle,
    ArrayShuffle = ArrayShuffle
}
