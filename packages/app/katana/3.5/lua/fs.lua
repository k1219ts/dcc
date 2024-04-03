-- A portable filesystem API using LuaJIT's FFI
--
local ffi = require("ffi")
local table = require("table")
require("string")
-- Cache needed functions and locals
local C, errno, string = ffi.C, ffi.errno, ffi.string
local concat, insert = table.concat, table.insert

-- "Standard" C99 functions
ffi.cdef[[
char *strerror(int errnum);
]]

local exists, mkdir, listdir, PATH_SEPERATOR
if ffi.os == "Windows" then
    ffi.cdef[[
    struct _finddata_t {
        unsigned attrib;
        __time32_t time_create;
        __time32_t time_access;
        __time32_t time_write;
        _fsize_t size;
        char name[260];
    };
    intptr_t _findfirst(const char *filespec, struct _finddata_t *fileinfo);
    int _findnext(intptr_t handle, struct _finddata_t *fileinfo);
    int _findclose(intptr_t handle);
    bool CreateDirectory(const char *path, void *lpSecurityAttributes);
    int _access(const char *path, int mode);
    ]]
    local _finddata_t = ffi.typeof("struct _finddata_t[]")
    function exists(path)
        assert(type(path) == "string", "path isn't a string")
        local result = C._access(path, 0) -- Check existence
        return result == 0
    end
    function listdir(path)
        local handle
        return function()
            local data = _finddata_t(1)
            if handle == nil then
                local result = C._findfirst(path .. "/*", data)
                if result == -1 then
                    local message = string(C.strerror(errno()))
                    error("Error iterating over directory " .. path .. ": " .. message)
                else
                    handle = result
                    return string(data.name)
                end
            else
                local result = C._findnext(handle, data)
                if result ~= 0 then
                    if errno() == 2 then
                        -- We're done
                        C._findclose(handle)
                        handle = nil
                        return nil
                    else
                        local message = string(C.strerror(errno()))
                        error("Error iterating over directory " .. path .. ": " .. message)
                    end
                else
                    return string(data.name)
                end
            end
        end
    end
    function mkdir(path, _)
        assert(type(path) == "string", "path isn't a string")
        if not C.CreateDirectory(path, nil) then
            local message = string(C.strerror(errno()))
            error("Unable to create directory " .. path .. ": " .. message)
        end
    end
    PATH_SEPERATOR = "\\"
elseif ffi.os == "Linux" or ffi.os == "OSX" then
    ffi.cdef[[
    struct dirent {
        unsigned long int d_ino;
        long int d_off;
        unsigned short d_reclen;
        unsigned char  d_type;
        char name[256];
    };
    typedef struct __dirstream DIR;
    DIR *opendir(const char *name);
    struct dirent *readdir(DIR *dirp);
    int closedir(DIR *dirp);
    int mkdir(const char *path, int mode);
    int access(const char *path, int amode);
    ]]
    function exists(path)
        assert(type(path) == "string", "path isn't a string")
        local result = C.access(path, 0) -- Check existence
        return result == 0
    end
    function listdir(path)
        local dir = C.opendir(path)
        if dir == nil then
            local message = string(C.strerror(errno()))
            error("Error listing directory " .. dir .. ": " .. message)
        end
--        local function nextDir()
--            local entry = C.readdir(dir)
--            if entry ~= nil then
--                local result = string(entry.name)
--                if result == "." or result == ".." then
--                    return nextDir()
--                else
--                    return result
--                end
--            else
--                C.closedir(dir)
--                dir = nil
--                return nil
--            end
--        end
--        return nextDir
        local result = {}
        local entry = C.readdir(dir)
        while(entry ~= nil)
        do
            local n = string(entry.name)
            if(n ~= "." and n ~= "..") then
                table.insert(result, n)
            end
            entry = C.readdir(dir)
        end
        return result
    end
    function mkdir(path, mode)
        assert(type(path) == "string", "path isn't a string")
        local mode = tonumber(mode or "755", 8)
        if C.mkdir(path, mode) ~= 0 then
            local message = string(C.strerror(errno()))
            error("Unable to create directory " .. path .. ": " .. message)
        end
    end
    PATH_SEPERATOR = "/"
else
    error("Unsupported operating system: " .. ffi.os)
end

local function join(...)
    local parts = {}
    for i = 1, select("#", ...) do
        local part = select(i, ...)
        insert(parts, part)
    end
    return concat(parts, PATH_SEPERATOR)
end

local function splitPath(path)
    assert(type(path) == "string", "path isn't a string!")
    local parts = {}
    local lastIndex = 0
    for i = 1, path:len() do
        if path:sub(i, i) == PATH_SEPERATOR then
            insert(parts, path:sub(lastIndex, i - 1))
            lastIndex = i + 1
        end
    end
    insert(parts, path:sub(lastIndex))
    return parts
end

local function mkdirs(path)
    local parts = splitPath(path)
    local currentPath = parts[1]
    for i=2, #parts do
        if not exists(currentPath) then
            mkdir(currentPath)
        end
        -- Note: This isn't suboptimal, since we really do need the intermediate results
        currentPath = currentPath .. PATH_SEPERATOR .. parts[i]
    end
    if not exists(path) then
        mkdir(path)
    end
end

return {
    exists = exists,
    join = join,
    mkdir = mkdir,
    mkdirs = mkdirs,
    splitPath = splitPath,
    listdir = listdir,
    PATH_SEPERATOR = PATH_SEPERATOR
}
