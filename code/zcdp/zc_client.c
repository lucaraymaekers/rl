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
    u16 Port = 2600;
    arena *Arena = ArenaAlloc();
    str8 Buffer = PushS8(Arena, KB(1));
    random_series Series;
    RandomSeed(&Series, (u64)OS_GetWallClock());
    
    LinuxSetDebuggerAttached();
    
    server Server = {0};
    
    // Connect to server
    Server.Handle = socket(AF_INET, SOCK_DGRAM, 0);
    
    struct sockaddr_in Address = {0};
    Server.Address.sin_family = AF_INET;
    Server.Address.sin_port = htons(Port);
    Server.Address.sin_addr.s_addr = inet_addr("224.0.0.26");
    
    u128 ClientUUID = GenUUID(&Series);
    u64 MessageID = 0;
    
    str8 Services[3] = {0};
    Services[0] = S8("Messaging");
    Services[1] = S8("FileTransfer");
    Services[2] = S8("AIChat");
    
    m_announce Message = {0};
    str8 MessageBuffer = PushS8(Arena, KB(1));
    
    // Create Message and send
    {    
        Message.Header.Version = GlobalVersion;
        Message.Header.Type = m_Announce;
        Message.Header.MessageID = MessageID;
        Message.PeerUUID = ClientUUID;
        {
            struct timespec Counter;
            clock_gettime(CLOCK_REALTIME, &Counter);
            Message.Timestamp = LinuxTimeSpecToSeconds(Counter);
        }
        
        MessageID += 1;
        
        Message.ServicesCount = ArrayCount(Services);
        
        u8 *CopyPointer = MessageBuffer.Data;
        m_Copy(&CopyPointer, &Message, sizeof(Message));
        
        umm TotalSize = sizeof(Message);
        for EachElement(Idx, Services)
        {
            str8 Service = Services[Idx];
            
            m_Copy(&CopyPointer, &Service.Size, sizeof(Service.Size));
            m_Copy(&CopyPointer, Service.Data, Service.Size);
            
            TotalSize += sizeof(Services[Idx].Size);
            TotalSize += Services[Idx].Size;
        }
        
        m_Send(Server, S8To(MessageBuffer, TotalSize));
    }
    
#if 0
    {
        struct sockaddr_in Address = {0};
        socklen_t SizeOfAddress = sizeof(Address);
        smm BytesReceived = recvfrom(Server.Handle, Buffer.Data, Buffer.Size, 0, (struct sockaddr *)&Address, &SizeOfAddress);
        Buffer.Data[BytesReceived] = 0;
        
        printf("Received(%s:%d): %s\n"
               ,
               inet_ntoa(Address.sin_addr), ntohs(Address.sin_port),
               Buffer.Data);
        
    }
#endif
    
    return 0;
}