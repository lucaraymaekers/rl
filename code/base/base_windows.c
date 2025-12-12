global u8 LogBuffer[Kilobytes(64)];

void OS_PrintFormat(char *Format, ...)
{
    va_list Args;
    va_start(Args, Format);
    vprintf(Format, Args);
}

str8 OS_ReadEntireFileIntoMemory(char *FileName)
{
    str8 Result = {};
    
    HANDLE FileHandle = CreateFileA(FileName, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, 0, 0);
    if(FileHandle != INVALID_HANDLE_VALUE)
    {
        LARGE_INTEGER FileSize;
        if(GetFileSizeEx(FileHandle, &FileSize))
        {
            u32 FileSize32 = (u32)(FileSize.QuadPart);
            Result.Data = (u8 *)VirtualAlloc(0, FileSize32, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
            if(Result.Data)
            {
                DWORD BytesRead;
                if(ReadFile(FileHandle, Result.Data, FileSize32, &BytesRead, 0) &&
                   (FileSize32 == BytesRead))
                {
                    // NOTE(casey): File read successfully
                    Result.Size = FileSize32;
                }
                else
                {                    
                    // TODO(casey): Logging
                    
                    if(Result.Data)
                    {
                        VirtualFree(Result.Data, 0, MEM_RELEASE);
                    }
                    
                    Result.Data = 0;
                }
            }
            else
            {
                // TODO(casey): Logging
            }
        }
        else
        {
            // TODO(casey): Logging
        }
    }
    
    return Result;
}