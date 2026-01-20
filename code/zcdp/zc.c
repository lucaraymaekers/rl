#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#define RL_BASE_NO_ENTRYPOINT 1
#include "base/base.h"
#include "base/base.c"

#include "zc.h"

int 
main(int ArgsCount, char **Args)
{
    int MaxConnections = 64;
    u16 Port = 2600;
    arena *Arena = 0;
    str8 Buffer = {0};
    
    Arena = ArenaAlloc();
    Buffer = PushS8(Arena, KB(2));
    
    LinuxSetDebuggerAttached();
    
    server Server = {0};
    
    // Start listening on the socket
    {
        s32 Result = 0;
        Server.Handle = socket(AF_INET, SOCK_DGRAM, 0);
        Assert(Server.Handle > 2);
        
        {
            u32 On = 1;
            Result = setsockopt(Server.Handle, SOL_SOCKET, SO_REUSEADDR, (u8*)&On, sizeof(On));
            AssertErrno(Result == 0);
            Result = setsockopt(Server.Handle, SOL_SOCKET, SO_REUSEPORT, (u8 *)&On, sizeof(On));
            AssertErrno(Result == 0);
        }
        
        Server.Address.sin_family = AF_INET;
#if 1
        Server.Address.sin_addr.s_addr = htonl(INADDR_ANY);  
#else
        Server.Address.sin_addr.s_addr = inet_addr("224.0.0.26");
#endif
        Server.Address.sin_port = htons(Port);
        
        Result = bind(Server.Handle, (const struct sockaddr *)&Server.Address, sizeof(Server.Address));
        AssertErrno(!Result);
        
        struct ip_mreq Group1;
        Group1.imr_multiaddr.s_addr = inet_addr("224.0.0.26");
        // NOTE(luca): All interfaces
        Group1.imr_interface.s_addr = inet_addr("0.0.0.0");
        
        Result = setsockopt(Server.Handle, IPPROTO_IP, IP_ADD_MEMBERSHIP, &Group1, sizeof(Group1));
        AssertErrno(Result == 0);
        
        Log("Listening on :%u\n", Port);
    }
    
    sockaddr_in ClientAddress = {0};
    socklen_t SizeOfClientAddress = sizeof(ClientAddress);
    
    b32 Running = true;
    while(Running)
    {        
        smm BytesReceived = recvfrom(Server.Handle, Buffer.Data, Buffer.Size,
                                     0, (struct sockaddr*)&ClientAddress, &SizeOfClientAddress);
        
        m_announce *Message = (m_announce *)Buffer.Data;
        
        switch(Message->Header.Type)
        {
            case m_Announce:
            {
                
                u8 *Data = Buffer.Data + sizeof(m_announce);
                
                u8 TimestampBuffer[32] = {0};
                {                
                    time_t Seconds = (time_t)(Message->Timestamp / 1000000000LL);
                    
                    // Convert to UTC time structure
                    struct tm *TimeInfo = gmtime(&Seconds);
                    
                    // Format as ISO 8601: "YYYY-MM-DDTHH:MM:SSZ"
                    strftime((char *)TimestampBuffer, sizeof(TimestampBuffer), "%Y-%m-%dT%H:%M:%SZ", TimeInfo);
                }
                str8 UUIDBuffer = PushS8(Arena, 37);
                UUIDtoStr8(UUIDBuffer, &Message->PeerUUID);
                
                Log("Received(%s:%d):\n"
                    " Timestamp: %s\n"
                    " UUID: " S8Fmt "\n"
                    , 
                    inet_ntoa(ClientAddress.sin_addr), ntohs(ClientAddress.sin_port),
                    TimestampBuffer,
                    S8Arg(UUIDBuffer));
                
                for EachIndex(Idx, Message->ServicesCount)
                {
                    if(Idx == 0)
                    {
                        Log(" Services:\n");
                    }
                    
                    str8 Service = {0};
                    MemoryCopy(&Service.Size, Data, sizeof(Service.Size));
                    Data += sizeof(Service.Size);
                    
                    Service.Data = PushArray(Arena, u8, Service.Size);
                    
                    MemoryCopy(Service.Data, Data, Service.Size);
                    Data += Service.Size;
                    
                    Log(" - " S8Fmt "\n", S8Arg(Service));
                }
                
            } break;
            default:
            {
                Log("Unhandled message.\n");
            } break;
        }
        
    }
    
    return 0;
}