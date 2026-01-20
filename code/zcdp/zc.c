#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#define RL_BASE_NO_ENTRYPOINT 1
#include "base/base.h"
#include "base/base.c"

#include "lib/md5.h"

#include "zc.h"

int 
main(int ArgsCount, char **Args)
{
    int MaxConnections = 64;
    u16 Port = 2600;
    
    b32 ServerMode = true;
    
    if(ArgsCount > 1) ServerMode = false;
    
    random_series Series;
    RandomSeed(&Series, (u64)OS_GetWallClock());
    
    arena *Arena = ArenaAlloc();
    str8 MessageBuffer = PushS8(Arena, KB(1));
    
    LinuxSetDebuggerAttached();
    
    u128 ClientUUID = GenUUID(&Series);
    u64 MessageID = 0;
    
    service Services[] = 
    {
        {.Name = S8("Messaging")},
        {.Name = S8("FileTransfer")},
        {.Name = S8("AIChat")},
        {.Name = S8("VideoCall")},
        {.Name = S8("VoiceCall")},
        {.Name = S8("ScreenShare")},
        {.Name = S8("CloudStorage")},
        {.Name = S8("Authentication")},
        {.Name = S8("Payment")},
        {.Name = S8("Notification")},
        {.Name = S8("Analytics")},
        {.Name = S8("LocationTracking")},
        {.Name = S8("ImageProcessing")},
        {.Name = S8("VideoStreaming")},
        {.Name = S8("AudioStreaming")},
        {.Name = S8("DatabaseSync")},
        {.Name = S8("Encryption")},
        {.Name = S8("Compression")},
        {.Name = S8("Translation")},
        {.Name = S8("SpeechToText")},
        {.Name = S8("TextToSpeech")},
        {.Name = S8("FaceRecognition")},
        {.Name = S8("Biometrics")},
        {.Name = S8("QRCodeScanner")},
        {.Name = S8("BarcodeScanner")},
        {.Name = S8("Calendar")},
        {.Name = S8("Contacts")},
        {.Name = S8("Email")},
        {.Name = S8("TaskManager")},
        {.Name = S8("RemoteDesktop")},
        {.Name = S8("BackupRestore")},
        {.Name = S8("SoftwareUpdate")},
        {.Name = S8("Telemetry")},
        {.Name = S8("CrashReporting")},
        {.Name = S8("Logging")},
        {.Name = S8("Caching")},
        {.Name = S8("LoadBalancing")},
        {.Name = S8("RateLimiting")},
        {.Name = S8("SessionManagement")},
        {.Name = S8("UserProfile")},
        {.Name = S8("SearchIndex")},
        {.Name = S8("ContentDelivery")},
        {.Name = S8("MediaTranscoding")},
        {.Name = S8("OCR")},
        {.Name = S8("Blockchain")},
        {.Name = S8("MachineLearning")},
        {.Name = S8("DataMining")},
        {.Name = S8("RealtimeSync")},
        {.Name = S8("WebSocket")},
        {.Name = S8("API Gateway")},
    };
    
    server Server = {0};
    // Start listening on the socket
    {
        s32 Result = 0;
        Server.Handle = socket(AF_INET, SOCK_DGRAM|SOCK_NONBLOCK, 0);
        Assert(Server.Handle > 2);
        
        // Set socket options
        {
            u32 On = 1;
            Result = setsockopt(Server.Handle, SOL_SOCKET, SO_REUSEADDR, &On, sizeof(On));
            AssertErrno(Result == 0);
            Result = setsockopt(Server.Handle, SOL_SOCKET, SO_REUSEPORT, &On, sizeof(On));
            AssertErrno(Result == 0);
#if 0
            Result = setsockopt(Server.Handle, IPPROTO_IP, IP_MULTICAST_LOOP, &On, sizeof(On));
            AssertErrno(Result == 0);
#endif
            
            
            struct ip_mreq Group1;
            Group1.imr_multiaddr.s_addr = inet_addr("224.0.0.26");
            // NOTE(luca): All interfaces
            Group1.imr_interface.s_addr = inet_addr("0.0.0.0");
            
            Result = setsockopt(Server.Handle, IPPROTO_IP, IP_ADD_MEMBERSHIP, &Group1, sizeof(Group1));
            AssertErrno(Result == 0);
        }
        
        Server.Address.sin_family = AF_INET;
        Server.Address.sin_addr.s_addr = htonl(INADDR_ANY);  
        Server.Address.sin_port = htons(Port);
        
        Result = bind(Server.Handle, (const struct sockaddr *)&Server.Address, sizeof(Server.Address));
        AssertErrno(!Result);
        
        Log("Listening on :%u\n", Port);
    }
    
    sockaddr_in ClientAddress = {0};
    socklen_t SizeOfClientAddress = sizeof(ClientAddress);
    
    str8 UUIDBuffer = UUIDtoStr8(Arena, &ClientUUID);
    Log("Started(" S8Fmt ")\n", S8Arg(UUIDBuffer));
    
    b32 Skip = true;
    
    b32 Running = true;
    while(Running)
    { 
        
        smm BytesReceived = 0;
        
        if(ServerMode)
        {        
            BytesReceived = recvfrom(Server.Handle, MessageBuffer.Data, MessageBuffer.Size,
                                     0, (struct sockaddr*)&ClientAddress, &SizeOfClientAddress);
        }
        
        if(BytesReceived > 0)
        {            
            m_announce *Message = (m_announce *)MessageBuffer.Data;
            
            switch(Message->Header.Type)
            {
                case m_TypeAnnounce:
                {
                    u8 *Data = MessageBuffer.Data + sizeof(m_announce);
                    
                    u8 TimestampBuffer[32] = {0};
                    {                
                        time_t Seconds = (time_t)(Message->Timestamp / 1000000000LL);
                        
                        // Convert to UTC time structure
                        struct tm *TimeInfo = gmtime(&Seconds);
                        
                        // Format as ISO 8601: "YYYY-MM-DDTHH:MM:SSZ"
                        strftime((char *)TimestampBuffer, sizeof(TimestampBuffer), "%Y-%m-%dT%H:%M:%SZ", TimeInfo);
                    }
                    
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
        
        // Announce every 10 seconds.
        {
            Log("Announcing...\n");
            
            service *ServiceOffset = Services + RandomChoice(&Series, ArrayCount(Services) - 1);
            m_Announce(Server, MessageBuffer, ClientUUID, 2, ServiceOffset, &MessageID);
            
            OS_Sleep(1000*1000*3);
        }
        
    }
    
    return 0;
}