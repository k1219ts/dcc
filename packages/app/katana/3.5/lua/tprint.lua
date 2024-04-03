local function Print(tbl, indent)
    if not indent then indent = 0 end
    for k, v in pairs(tbl) do
        local formatting = string.rep(" ", indent) .. k .. ": "
        if type(v) == "table" then
            print(formatting)
            tprint(v, indent+1)
        elseif type(v) == 'boolean' then
            print(formatting .. tostring(v))
        else
            print(formatting .. v)
        end
    end
end


local function dump(tbl)
    if type(tbl) == 'table' then
        local s = '{ '
        for k, v in pairs(tbl) do
            if type(k) ~= 'table' then k = '"' .. k .. '"' end
            s = s .. '[' .. k .. ']=' .. dump(v) .. ', '
        end
        return s .. '} '
    else
        return tostring(tbl)
    end
end


return {
    Print = Print,
    dump = dump
}

