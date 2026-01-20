/* date = January 20th 2026 10:27 am */

#ifndef MESSAGES_H
#define MESSAGES_H

#include "zc_random.h"

typedef struct sockaddr_in sockaddr_in;

typedef struct server server;
struct server
{
    int Handle;
    sockaddr_in Address;
};

void 
m_Send(server Server, str8 Message)
{
    
    smm BytesSent = sendto(Server.Handle, Message.Data, Message.Size, 
                           0, (struct sockaddr *)&Server.Address, sizeof(Server.Address));
    Assert(BytesSent != -1);
    Assert((umm)BytesSent == Message.Size);
    
}

global_variable u16 GlobalVersion = 0;

NO_STRUCT_PADDING_BEGIN

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
    m_Null = 0,
    m_Announce
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
    u32 ServicesCount;
    s64 Timestamp;
};

NO_STRUCT_PADDING_END

internal void
m_Copy(u8 **CopyPointer, void *Data, umm Size)
{
    MemoryCopy(*CopyPointer, Data, Size);
    *CopyPointer += Size;
}

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

void
UUIDtoStr8(str8 Buffer, u128 *UUID)
{
    Assert(Buffer.Size >= 37);
    snprintf((char *)Buffer.Data, Buffer.Size,
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             UUID->U8[0], UUID->U8[1], UUID->U8[2], UUID->U8[3],
             UUID->U8[4], UUID->U8[5],
             UUID->U8[6], UUID->U8[7],
             UUID->U8[8], UUID->U8[9],
             UUID->U8[10], UUID->U8[11], UUID->U8[12], 
             UUID->U8[13], UUID->U8[14], UUID->U8[15]);
}

#endif //MESSAGES_H
