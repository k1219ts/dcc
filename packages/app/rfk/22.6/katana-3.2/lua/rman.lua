local json = require 'json'

local RMAN_VER = os.getenv('RMAN_VER')

local configDir = os.getenv('REZ_RFK_ROOT') .. '/config'

local function readConfig(filename)
    local filestr = io.open(filename, 'r'):read('*a')
    local data = json.decode(filestr)
    return data
end

local function aovs()
    local aovfile = configDir .. '/aovs.json'
    local aovs = readConfig(aovfile)
    return aovs
end


local function setDisplayChannel(name, data, remap)
    local outCh = 'prmanGlobalStatements.outputChannels.' .. name
    Interface.SetAttr(outCh .. '.name', StringAttribute(name))

    local ctype = data['channelType']
    if ctype ~= nil then
        Interface.SetAttr(outCh .. '.type', StringAttribute('varying ' .. ctype))
    end

    local source = data['channelSource']
    if source ~= nil then
        if source ~= name then
            Interface.SetAttr(outCh .. '.params.source.type', StringAttribute('string'))
            Interface.SetAttr(outCh .. '.params.source.value', StringAttribute(source))
        end
    end

    local filter = data['filter']
    if filter ~= nil then
        Interface.SetAttr(outCh .. '.params.filter.type', StringAttribute('string'))
        Interface.SetAttr(outCh .. '.params.filter.value', StringAttribute(filter))
    end

    local statistics = data['statistics']
    if statistics ~= nil then
        Interface.SetAttr(outCh .. '.params.statistics.type', StringAttribute('string'))
        Interface.SetAttr(outCh .. '.params.statistics.value', StringAttribute(statistics))
    end

    -- Remap
    local channelRemap = data['channelRemap']
    if channelRemap ~= nil and channelRemap == 'true' then
        if remap ~= nil then
            if remap[3] ~= 0.0 then
                Interface.SetAttr(outCh .. '.params.remap.type', StringAttribute('float[3]'))
                Interface.SetAttr(outCh .. '.params.remap.value', FloatAttribute(remap))
            end
        end
    end
end

return {
    RMAN_VER = RMAN_VER,
    version = RMAN_VER,
    configDir = configDir,
    readConfig = readConfig,
    aovs = aovs,
    setDisplayChannel = setDisplayChannel
}
