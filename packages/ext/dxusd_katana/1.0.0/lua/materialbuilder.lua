local function has_key(tab, val)
    for k, v in pairs(tab) do
        if k == val then
            return true
        end
    end
    return false
end

local function has_value(tab, val)
    for k, v in pairs(tab) do
        if v == val then
            return true
        end
    end
    return false
end

-- get first index
local function Index(tab, val)
    local id = -1
    for i, v in pairs(tab) do
        if v == val then
            id = i
        end
    end
    return id
end

-- get first index
local function FirstIndex(tab, val)
    for i, v in pairs(tab) do
        if v == val then
            return i
        end
    end
    return -1
end

local function Indices(tab, val)
    local result = {}
    for i, v in pairs(tab) do
        if v == val then
            table.insert(result, i)
        end
    end
    return result
end

local function RealLocation(location)
    local src = pystring.split(location, '/')
    local res = location
    -- modify location if pointInstancer
    if has_value(src, 'Prototypes') then
        for i, v in pairs(Indices(src, 'Prototypes')) do
            local index = i + v
            table.insert(src, index, src[index])
        end
        res = pystring.join('/', src)
    end
    return res
end

local function LocationName(location)
    local elements = pystring.split(location, '/')
    return elements[#elements]
end

-- find parent : instance master, prototype, nslayer
local function LocationSourceName(location)
    local nameAttr = Interface.GetGlobalAttr('modelName', location)
    if nameAttr ~= nil then
        name = nameAttr:getSamples():front():get(0)
        src  = pystring.split(name, '_')
        return src[1]
    end
    local elements = pystring.split(location, '/')
    local id = -1
    -- scene instance
    id = Index(elements, 'Masters')
    if id > -1 then
        return elements[id + 1]
    end
    -- pointInstancer
    id = Index(elements, 'Prototypes')
    if id > -1 then
        return elements[id + 1]
    end
    -- last check
    if has_value(elements, 'Materials') then
        return elements[Index(elements, 'Materials') - 1]
    elseif has_value(elements, 'Looks') then
        return elements[Index(elements, 'Looks') -1]
    end
    return nil
end

-- find parent : reference group
local function LocationGroupName(location)
    local elements = pystring.split(location, '/')
    local id = -1
    -- scene instance
    id = Index(elements, 'Masters')
    if id > -1 then
        return elements[id + 1]
    end
    -- pointInstancer
    id = FirstIndex(elements, 'Prototypes')
    if id > -1 then
        return elements[id - 2]
    end
    -- last check
    if has_value(elements, 'Materials') then
        return elements[Index(elements, 'Materials') - 1]
    elseif has_value(elements, 'Looks') then
        return elements[Index(elements, 'Looks') -1]
    end
    return nil
end

return {
    has_key = has_key,
    has_value = has_value,
    Index = Index,
    FirstIndex = FirstIndex,
    Indices = Indices,
    RealLocation = RealLocation,
    LocationName = LocationName,
    LocationSourceName = LocationSourceName,
    LocationGroupName = LocationGroupName
}
