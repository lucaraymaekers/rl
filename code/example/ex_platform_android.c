#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <byteswap.h>
#include <errno.h>
#include <fcntl.h>

#define RL_BASE_NO_ENTRYPOINT 1
#include "base/base.h"

NO_WARNINGS_BEGIN

#include <GLES3/gl3.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android_native_app_glue.h>
#include <android/sensor.h>

#include "rawdraw/os_generic.h"
#include "rawdraw/CNFGAndroid.h"
#define CNFA_IMPLEMENTATION
#include "cnfa/CNFA.h"
#define CNFG_IMPLEMENTATION
#define CNFG3D
#include "rawdraw/CNFG.h"

NO_WARNINGS_END

// #define EX_SLOW_COMPILE 1
// #include "ex_libs.h"

//~ Types
typedef GLuint gl_handle;
typedef struct android_app android_app;

//~ Macro's
#undef Log
#define Log printf

#define SHOW_DEBUG_INFO 0

//~ Globals 

// Sensors
global_variable ASensorManager *SensorManager;
global_variable const ASensor *AccelerometerSensor;
global_variable bool NoSensorForGyro;
global_variable ASensorEventQueue* AccelerometerEventQueue;
global_variable ALooper *Looper;
// Audio
global_variable const u32 SAMPLE_RATE = 44100;
global_variable const u16 SAMPLE_COUNT = 512;
global_variable u32 StreamOffset;
global_variable u16 AudioFrequency;
// Accumulator
global_variable float AccelerometerX;
global_variable float AccelerometerY;
global_variable float AccelerometerZ;
global_variable int AccelerometerSamples;
// Keys
global_variable int LastButtonX;
global_variable int LastButtonY;
global_variable int LastMotionX;
global_variable int LastMotionY;
global_variable int LastButtonId;
global_variable int LastMask;
global_variable int LastKey;
global_variable int LastKeyDown;

global_variable int KeyboardUp;
global_variable u8 ButtonState[8];

global_variable volatile b32 *GlobalRunning;
global_variable volatile int Suspended;

//~ External

extern android_app *gapp;

extern void AndroidDisplayKeyboard(int pShow);

//~ Types
typedef struct android_context android_context;
struct android_context
{
    umm FrameIdx;
};

//~ Functions
UPDATE_AND_RENDER(UpdateAndRender);

void SetupIMU()
{
	SensorManager = ASensorManager_getInstanceForPackage("gyroscope");
	AccelerometerSensor = ASensorManager_getDefaultSensor(SensorManager, ASENSOR_TYPE_GYROSCOPE);
	NoSensorForGyro = AccelerometerSensor == NULL;
	Looper = ALooper_prepare(ALOOPER_PREPARE_ALLOW_NON_CALLBACKS);
	AccelerometerEventQueue = ASensorManager_createEventQueue(SensorManager, (ALooper*)&Looper, 0, 0, 0); //XXX??!?! This looks wrong.
	if(!NoSensorForGyro) 
    {
		ASensorEventQueue_enableSensor(AccelerometerEventQueue, AccelerometerSensor);
		Log("setEvent Rate: %d\n", ASensorEventQueue_setEventRate(AccelerometerEventQueue, AccelerometerSensor, 10000));
	}
}

void AccCheck()
{
	if(!NoSensorForGyro) 
    {
		ASensorEvent Event;
        
        while(true)
        {
            smm SensorStatus = ASensorEventQueue_getEvents(AccelerometerEventQueue, &Event, 1);
            if(SensorStatus > 0)
            {
                AccelerometerX = Event.vector.v[0];
                AccelerometerY = Event.vector.v[1];
                AccelerometerZ = Event.vector.v[2];
                AccelerometerSamples++;
            }
            else
            {
                break;
            }
        }
    }
}

void HandleKey(int Keycode, int IsDown)
{
	LastKey = Keycode;
	LastKeyDown = IsDown;
	if(Keycode == 10 && !IsDown) 
    { 
        KeyboardUp = 0; 
        AndroidDisplayKeyboard(KeyboardUp);
    }
    
	if(Keycode == 4) 
    { 
        AndroidSendToBack(1);
    } //Handle Physical Back Button.
}

void HandleButton(int X, int Y, int Button, int IsDown)
{
	ButtonState[Button] = (u8)IsDown;
	LastButtonId = Button;
	LastButtonX = X;
	LastButtonY = Y;
    
	if(IsDown) 
    { 
        KeyboardUp = !KeyboardUp; 
        //AndroidDisplayKeyboard(KeyboardUp);
    }
}

void HandleMotion(int X, int Y, int Mask)
{
	LastMask = Mask;
	LastMotionX = X;
	LastMotionY = Y;
}

// NOTE: writes the text to a file to path: storage/emulated/0/Android/data/org.yourorg.cnfgtest/files
void Logger(const char *Format, ...)
{
	const char* Path = AndroidGetExternalFilesDir();
	char Buffer[2048];
	snprintf(Buffer, sizeof(Buffer), "%s/log.txt", Path);
	FILE *File = fopen(Buffer, "w");
	if (File == NULL)
	{
		exit(1);
	}
    
	va_list Arguments;
	va_start(Arguments, Format);
	vsnprintf(Buffer, sizeof(Buffer), Format, Arguments);
	va_end(Arguments);	
    
	fprintf(File, "%s\n", Buffer);
    
	fclose(File);
}

int HandleDestroy()
{
	Log("Destroying\n");
	return 0;
}

void HandleSuspend()
{
	Suspended = 1;
}

void HandleResume()
{
	Suspended = 0;
}

void HandleThisWindowTermination()
{
	Suspended = 1;
}

void AudioCallback(struct CNFADriver *SoundDriver, short *Output, short *Input, int FramesPending, int FramesReceived)
{
	memset(Output, 0, FramesPending*sizeof(u16));
	
    if(!Suspended)
    {
        if(ButtonState[1]) // play audio only if ~touching with two fingers
        {
            AudioFrequency = 440;
            for EachIndexType(smm, Index, FramesPending)
            {
                s16 Sample = INT16_MAX * (s16)(sin(AudioFrequency*(2*M_PI)*(f64)(StreamOffset+Index)/SAMPLE_RATE));
                Output[Index] = Sample;
            }
            StreamOffset += FramesPending;
        }
    }
}

void MakeNotification(const char *channelID, const char *channelName, const char *title, const char *message)
{
	static int id;
	id++;
    
	const struct JNINativeInterface *env = 0;
	const struct JNINativeInterface ** envptr = &env;
	const struct JNIInvokeInterface ** jniiptr = gapp->activity->vm;
	const struct JNIInvokeInterface *jnii = *jniiptr;
    
	jnii->AttachCurrentThread(jniiptr, &envptr, NULL);
	env = (*envptr);
    
	jstring channelIDStr = env->NewStringUTF(ENVCALL channelID);
	jstring channelNameStr = env->NewStringUTF(ENVCALL channelName);
    
	// Runs getSystemService(Context.NOTIFICATION_SERVICE).
	jclass NotificationManagerClass = env->FindClass(ENVCALL "android/app/NotificationManager");
	jclass activityClass = env->GetObjectClass(ENVCALL gapp->activity->clazz);
	jmethodID MethodGetSystemService = env->GetMethodID(ENVCALL activityClass, "getSystemService", "(Ljava/lang/String;)Ljava/lang/Object;");
	jstring notificationServiceName = env->NewStringUTF(ENVCALL "notification");
	jobject notificationServiceObj = env->CallObjectMethod(ENVCALL gapp->activity->clazz, MethodGetSystemService, notificationServiceName);
    
	// create the Notification channel.
	jclass notificationChannelClass = env->FindClass(ENVCALL "android/app/NotificationChannel");
	jmethodID notificationChannelConstructorID = env->GetMethodID(ENVCALL notificationChannelClass, "<init>", "(Ljava/lang/String;Ljava/lang/CharSequence;I)V");
	jobject notificationChannelObj = env->NewObject(ENVCALL notificationChannelClass, notificationChannelConstructorID, channelIDStr, channelNameStr, 3); // IMPORTANCE_DEFAULT
	jmethodID createNotificationChannelID = env->GetMethodID(ENVCALL NotificationManagerClass, "createNotificationChannel", "(Landroid/app/NotificationChannel;)V");
	env->CallVoidMethod(ENVCALL notificationServiceObj, createNotificationChannelID, notificationChannelObj);
    
	env->DeleteLocalRef(ENVCALL channelNameStr);
	env->DeleteLocalRef(ENVCALL notificationChannelObj);
    
	// Create the Notification builder.
	jclass classBuilder = env->FindClass(ENVCALL "android/app/Notification$Builder");
	jstring titleStr = env->NewStringUTF(ENVCALL title);
	jstring messageStr = env->NewStringUTF(ENVCALL message);
	jmethodID eventConstructor = env->GetMethodID(ENVCALL classBuilder, "<init>", "(Landroid/content/Context;Ljava/lang/String;)V");
	jobject eventObj = env->NewObject(ENVCALL classBuilder, eventConstructor, gapp->activity->clazz, channelIDStr);
	jmethodID setContentTitleID = env->GetMethodID(ENVCALL classBuilder, "setContentTitle", "(Ljava/lang/CharSequence;)Landroid/app/Notification$Builder;");
	jmethodID setContentTextID = env->GetMethodID(ENVCALL classBuilder, "setContentText", "(Ljava/lang/CharSequence;)Landroid/app/Notification$Builder;");
	jmethodID setSmallIconID = env->GetMethodID(ENVCALL classBuilder, "setSmallIcon", "(I)Landroid/app/Notification$Builder;");
    
	// You could do things like setPriority, or setContentIntent if you want it to do something when you click it.
    
	env->CallObjectMethod(ENVCALL eventObj, setContentTitleID, titleStr);
	env->CallObjectMethod(ENVCALL eventObj, setContentTextID, messageStr);
	env->CallObjectMethod(ENVCALL eventObj, setSmallIconID, 17301504); // R.drawable.alert_dark_frame
    
	// eventObj.build()
	jmethodID buildID = env->GetMethodID(ENVCALL classBuilder, "build", "()Landroid/app/Notification;");
	jobject notification = env->CallObjectMethod(ENVCALL eventObj, buildID);
    
	// NotificationManager.notify(...)
	jmethodID notifyID = env->GetMethodID(ENVCALL NotificationManagerClass, "notify", "(ILandroid/app/Notification;)V");
	env->CallVoidMethod(ENVCALL notificationServiceObj, notifyID, id, notification);
    
	env->DeleteLocalRef(ENVCALL notification);
	env->DeleteLocalRef(ENVCALL titleStr);
	env->DeleteLocalRef(ENVCALL activityClass);
	env->DeleteLocalRef(ENVCALL messageStr);
	env->DeleteLocalRef(ENVCALL channelIDStr);
	env->DeleteLocalRef(ENVCALL NotificationManagerClass);
	env->DeleteLocalRef(ENVCALL notificationServiceObj);
	env->DeleteLocalRef(ENVCALL notificationServiceName);
}


//~ API

internal P_context 
P_ContextInit(arena *Arena, app_offscreen_buffer *Buffer, b32 *Running)
{
    android_context *Context = PushStruct(Arena, android_context);
    
    GlobalRunning = Running;
    
    CNFGBGColor = 0x000040ff;
    CNFGSetupFullscreen("Handmade Window", 0);
    HandleWindowTermination = HandleThisWindowTermination;
    
    SetupIMU();
    InitCNFAAndroid(AudioCallback, "Handmade AudioCallback", SAMPLE_RATE, 0, 1, 0, SAMPLE_COUNT, 0, 0, 0);
    
    short ScreenWidth, ScreenHeight;
    CNFGGetDimensions(&ScreenWidth, &ScreenHeight);
    Buffer->Width = ScreenWidth;
    Buffer->Height = ScreenHeight;
    Buffer->BytesPerPixel = 4;
    Buffer->Pitch = (Buffer->BytesPerPixel*Buffer->Width);
    // NOTE(luca): Allocate enough memory to handle if the buffer got resized to something bigger.
    Buffer->Pixels = PushArray(Arena, u8, MB(32));
    
    printf("### Setup Completed ###\n");
    
    return (P_context)Context;
}

internal void      
P_UpdateImage(P_context Context, app_offscreen_buffer *Buffer)
{
    CNFGSwapBuffers();
}

internal void      
P_ProcessMessages(P_context Context, app_input *Input, app_offscreen_buffer *Buffer, b32 *Running)
{
    android_context *Android = (android_context *)Context;
    
    Android->FrameIdx += 1;
    
#if 0    
    if(Android->FrameIdx == 200)
    {
        MakeNotification("default", "example_droid alerts", "rldroid", "Hit frame two hundred\nNew Line");
    }
#endif
    
    CNFGHandleInput();
    AccCheck();
    
    Input->MouseX = LastMotionX;
    Input->MouseY = LastMotionY;
    
    short ScreenWidth, ScreenHeight;
    CNFGGetDimensions(&ScreenWidth, &ScreenHeight);
    Buffer->Width = ScreenWidth;
    Buffer->Height = ScreenHeight;
    Buffer->BytesPerPixel = 4;
    Buffer->Pitch = (Buffer->BytesPerPixel*Buffer->Width);
    
    if(Suspended)
    {
        usleep(50000);
    }
    
    CNFGClearFrame();
    CNFGFlushRender();
}

internal void      
P_LoadAppCode(arena *Arena, app_code *Code, app_memory *Memory, s64 *LastWriteTime)
{
    Code->UpdateAndRender = UpdateAndRender;
}

#include "ex_app.c"
