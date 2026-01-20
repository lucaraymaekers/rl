/* date = January 20th 2026 10:27 am */

#ifndef MESSAGES_H
#define MESSAGES_H

#include "zc_random.h"

global_variable u16 GlobalVersion = 0;

typedef struct sockaddr_in sockaddr_in;

typedef struct server server;
struct server
{
    int Handle;
    sockaddr_in Address;
};

NO_STRUCT_PADDING_BEGIN

typedef struct service service;
struct service
{
    str8 Name;
};

typedef union u128 u128;
union u128
{
    u8 U8[16];
    u16 U16[8];
    u32 U32[4];
    u64 U64[2];
};

typedef enum m_type m_type;
enum m_type
{
    m_TypeNull = 0,
    m_TypeAnnounce
};

typedef struct m_header m_header;
struct m_header
{
    u16 Version;
    u32 Type;
    u64 MessageID;
};

typedef struct m_announce m_announce;
struct m_announce
{
    m_header Header;
    
    u128 PeerUUID;
    // NOTE(luca): If this is greater than 0, than there are this number of strings following the message.
    u64 ServicesCount;
    s64 Timestamp;
};

NO_STRUCT_PADDING_END

internal u128
GenUUID(random_series *Series)
{
    u128 Result = {0};
    
    Result.U32[0] = RandomNext(Series);
    Result.U32[1] = RandomNext(Series);
    Result.U32[2] = RandomNext(Series);
    Result.U32[3] = RandomNext(Series);
    
    Result.U8[6] = (Result.U8[6] & 0x0F) | 0x40; // Version 4
    Result.U8[8] = (Result.U8[8] & 0x3F) | 0x80; // RFC 4122
    
    return Result;
}

internal str8
UUIDtoStr8(arena *Arena, u128 *UUID)
{
    str8 Result = PushS8(Arena, 37);
    
    snprintf((char *)Result.Data, Result.Size,
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             UUID->U8[0], UUID->U8[1], UUID->U8[2], UUID->U8[3],
             UUID->U8[4], UUID->U8[5],
             UUID->U8[6], UUID->U8[7],
             UUID->U8[8], UUID->U8[9],
             UUID->U8[10], UUID->U8[11], UUID->U8[12], 
             UUID->U8[13], UUID->U8[14], UUID->U8[15]);
    
    return Result;
}


internal void
m_Copy(u8 **CopyPointer, void *Data, umm Size)
{
    MemoryCopy(*CopyPointer, Data, Size);
    *CopyPointer += Size;
}

internal void 
m_Send(server Server, str8 Message)
{
    
    smm BytesSent = sendto(Server.Handle, Message.Data, Message.Size, 
                           0, (struct sockaddr *)&Server.Address, sizeof(Server.Address));
    Assert(BytesSent != -1);
    Assert((umm)BytesSent == Message.Size);
}

internal m_header
m_Header(m_type Type, u64 ID)
{
    m_header Header = {0};
    
    Header.Version = GlobalVersion;
    Header.Type = Type;
    Header.MessageID = ID;
    
    return Header;
}

internal void
m_Announce(server Server,
           str8 Buffer,
           u128 UUID, 
           u64 ServicesCount, 
           service *Services,
           u64 *ID)
{
    m_announce Message = {0};
    
    u64 MessageID = *ID;
    *ID += 1;
    
    Message.Header = m_Header(m_TypeAnnounce, MessageID);
    
    Message.PeerUUID = UUID;
    {
        struct timespec Counter;
        clock_gettime(CLOCK_REALTIME, &Counter);
        Message.Timestamp = LinuxTimeSpecToSeconds(Counter);
    }
    Message.ServicesCount = ServicesCount;
    
    u8 *CopyPointer = Buffer.Data;
    m_Copy(&CopyPointer, &Message, sizeof(Message));
    
    umm TotalSize = sizeof(Message);
    for EachIndex(Idx, ServicesCount)
    {
        service Service = Services[Idx];
        
        m_Copy(&CopyPointer, &Service.Name.Size, sizeof(Service.Name.Size));
        m_Copy(&CopyPointer, Service.Name.Data, Service.Name.Size);
        
        TotalSize += sizeof(Services[Idx].Name.Size);
        TotalSize += Services[Idx].Name.Size;
    }
    
    Assert(TotalSize < Buffer.Size);
    
    m_Send(Server, S8To(Buffer, TotalSize));
}


#endif //MESSAGES_H
