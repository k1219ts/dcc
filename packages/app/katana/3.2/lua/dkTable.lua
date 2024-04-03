local function initTable(size)
    local new = {}
    for i=1, size do
        new[i] = i
    end
    return new
end


local function shuffleIndex(tbl)
    local size = #tbl
    math.randomseed(size)
    local new = initTable(size)
    for i=size, 1, -1 do
        local rand = math.random(size)
        new[i], new[rand] = new[rand], new[i]
    end
    return new
end


local function shuffleIndexList(tbl, ratio)
    if not ratio then
        return nil
    end
    local size = #tbl
    math.randomseed(size)
    local new = shuffleIndex(tbl)

    local renew = {}
    for i=1, ratio do
        renew[i] = new[i]
    end
    return renew
end

local function shuffleIndexTable(tbl, ratio)
    if not ratio then
        return tbl
    end
    local size = #tbl
    math.randomseed(size)
    local new = shuffleIndex(tbl)
    local renew = {}
    for i=1, size do
        if i <= ratio then
            local index = new[i]
            renew[index] = index
        end
    end
    return renew
end

return {
    shuffleIndex = shuffleIndex,
    shuffleIndexList = shuffleIndexList,
    shuffleIndexTable = shuffleIndexTable
}
