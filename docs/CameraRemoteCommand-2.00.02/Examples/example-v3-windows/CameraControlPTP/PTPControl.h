#pragma once
#include <vector>

#include "DataManager.h"
#include "PTPDef.h"
#include "Wia.h"

extern UINT8 WhitebalanceAB;
extern UINT8 WhitebalanceGM;
extern UINT32 ExposureProgram;

#define MUTEX TEXT("PTPMUTEX")

class PTPControl : public DataManager {
 protected:
 public:
  static PTPControl& getInstance();
  BOOL prepareConnection();
  BOOL LoadDisplayImageFromLiveview(BYTE* buffer, BITMAP* bitmap,
                                    HBITMAP& hBitmap);
  HRESULT SDIOConnect(DWORD param1, DWORD param2, DWORD param3);
  HRESULT SDIOGetExtDeviceInfo(DWORD param);
  HRESULT SDIOControlDevice(DWORD param1, UINT32 value);
  HRESULT SDIOSetContentsTransferMode(DWORD param1, DWORD param2, DWORD param3);
  HRESULT ExectuteDeleteObject(int objectHandle);
  template <typename T>
  HRESULT SDIOSetExtDevicePropValue(DWORD param1, T& value,
                                    DWORD size = sizeof(T)) {
    return SDIOSetExtDevicePropValueImpl(param1, &value, size);
  }
  HRESULT ExecuteGetObject(DWORD objectHandle, BYTE* buffer, DWORD size);
  HRESULT SDIOGetAllExtDevicePropInfo(HWND pHwnd);
  HRESULT CloseSession();
  HRESULT GetObjectInfo(int objectHandle, PTP_GetObjectInfo& info);
  HRESULT GetDeviceInfo();
  HRESULT GetStorageID();
  HRESULT GetNumObjects();
  HRESULT GetObjectHandles(DWORD param1, DWORD param2, DWORD param3);
  HRESULT GetStorageInfo(int objectHandle, PTP_GetStorageInfo& info);
  void closeMutex();

 private:
  PTPControl();
  PTPControl(const PTPControl& other) {}
  PTPControl& operator=(const PTPControl& other) {}

  HRESULT SDIOSetExtDevicePropValueImpl(DWORD param1, void* value, DWORD size);

 public:
};
