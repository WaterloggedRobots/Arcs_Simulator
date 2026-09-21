#include "PTPControl.h"

#include <Sti.h>
#include <tchar.h>

#include <string>

#include "afxdialogex.h"
#include "atlimage.h"

UINT8 WhitebalanceAB;
UINT8 WhitebalanceGM;
UINT8 HostPC;
UINT32 ExposureProgram;

IWiaItemExtras* itemExtra = NULL;
typedef int(__stdcall* PrepareDll)(IWiaItemExtras* itemExtraArray[10]);
typedef BOOL(__stdcall* createBitmapDataForLiveViewDll)(BYTE& buffer,
                                                        BITMAP& bitmap,
                                                        HBITMAP& hBitmap);
typedef BOOL(__stdcall* createBitmapDataForThumbnailDll)(BYTE& buffer,
                                                         BITMAP& bitmap,
                                                         HBITMAP& hBitmap,
                                                         DWORD size);
static HANDLE hMutex;
static DataManager& dMgr = DataManager::getInstance();

struct Liveview_ObjectInfoStr {
  DWORD Offset;
  DWORD Size;
};

PTPControl& PTPControl::getInstance() {
  static PTPControl instance;
  return instance;
}

PTPControl::PTPControl() {}

BOOL PTPControl::prepareConnection() {
  HRESULT hr = 0;
  CComPtr<IWiaDevMgr> pWiaDevMgr;
  CComPtr<IWiaItem> pWiaItemRoot;
  IWiaItemExtras* pWiaItemExtras = NULL;
  BSTR id = NULL;

  hr = pWiaDevMgr.CoCreateInstance(CLSID_WiaDevMgr);

  if (hr != S_OK) {
    return FALSE;
  }

  hr = pWiaDevMgr->SelectDeviceDlg(NULL, StiDeviceTypeDigitalCamera, 0, &id,
                                   &pWiaItemRoot);
  SysFreeString(id);
  if (hr != S_OK) {
    itemExtra = NULL;
    return FALSE;
  }

  hr = pWiaItemRoot->QueryInterface(IID_IWiaItemExtras,
                                    reinterpret_cast<void**>(&pWiaItemExtras));
  if (hr != S_OK) {
    return FALSE;
  }

  itemExtra = pWiaItemExtras;
  return TRUE;
}

BOOL PTPControl::LoadDisplayImageFromLiveview(BYTE* buffer, BITMAP* bitmap,
                                              HBITMAP& hBitmap) {
  HGLOBAL hG = NULL;
  CImage img;
  IStream* stream = NULL;
  Liveview_ObjectInfoStr* objInfo =
      reinterpret_cast<Liveview_ObjectInfoStr*>(buffer);
  hG = GlobalAlloc(GMEM_FIXED | GMEM_ZEROINIT, objInfo->Size);

  if (hG) {
    memcpy_s(hG, objInfo->Size, &buffer[objInfo->Offset], objInfo->Size);
    HRESULT hr = CreateStreamOnHGlobal(hG, TRUE, &stream);
    if (hr != S_OK) {
      return FALSE;
    }
    img.Load(stream);

    CBitmap* bmp = NULL;
    bmp = CBitmap::FromHandle(img);

    if (bmp == NULL) {
      GlobalFree(hG);
      return FALSE;
    } else {
      bmp->GetBitmap(bitmap);
      HBITMAP hBitmapTemp = static_cast<HBITMAP>(bmp->GetSafeHandle());
      BITMAP bmp_info;
      int testsize = GetObject(hBitmapTemp, sizeof(BITMAP), &bmp_info);

      hBitmap = static_cast<HBITMAP>(
          ::CopyImage(hBitmapTemp, IMAGE_BITMAP, 0, 0, LR_COPYRETURNORG));

      GlobalFree(hG);
      return TRUE;
    }
  }
  return FALSE;
}

HRESULT PTPControl::GetObjectInfo(int objectHandle, PTP_GetObjectInfo& info) {
  if (itemExtra == NULL) {
    return S_FALSE;
  }

  WaitForSingleObject(hMutex, INFINITE);

  HRESULT hr = 0;
  PTP_VENDOR_DATA_IN* pDataIn = NULL;
  PTP_VENDOR_DATA_OUT* pDataOut = NULL;
  DWORD dwDataInSize = SIZEOF_REQUIRED_VENDOR_DATA_IN;
  DWORD dwDataOutSize = sizeof(info) + 0x1000;
  DWORD dwActualDataOutSize = 0;

  pDataIn = static_cast<PTP_VENDOR_DATA_IN*>(CoTaskMemAlloc(dwDataInSize));
  if (pDataIn == NULL) {
    return S_FALSE;
  }
  pDataOut = static_cast<PTP_VENDOR_DATA_OUT*>(CoTaskMemAlloc(dwDataOutSize));
  if (pDataOut == NULL) {
    CoTaskMemFree(pDataIn);
    return S_FALSE;
  }

  ZeroMemory(pDataIn, dwDataInSize);
  ZeroMemory(pDataOut, dwDataOutSize);

  pDataIn->OpCode = PTP_OC_GetObjectInfo;
  pDataIn->NextPhase = PTP_NEXTPHASE_READ_DATA;
  pDataIn->Params[0] = objectHandle;
  pDataIn->NumParams = 1;

  hr = itemExtra->Escape(
      ESCAPE_PTP_VENDOR_COMMAND, reinterpret_cast<BYTE*>(pDataIn), dwDataInSize,
      reinterpret_cast<BYTE*>(pDataOut), dwDataOutSize, &dwActualDataOutSize);

  if (SUCCEEDED(hr)) {
    memcpy_s(&info, dwActualDataOutSize, pDataOut->VendorReadData,
             dwActualDataOutSize - sizeof(PTP_VENDOR_DATA_OUT));
    CoTaskMemFree(pDataIn);
  }
  CoTaskMemFree(pDataOut);

  ReleaseMutex(hMutex);

  return hr;
}

HRESULT PTPControl::ExecuteGetObject(DWORD objectHandle, BYTE* buffer,
                                     DWORD size) {
  if (itemExtra == NULL) {
    return S_FALSE;
  }

  WaitForSingleObject(hMutex, INFINITE);

  HRESULT hr = 0;
  PTP_VENDOR_DATA_IN* pDataIn = NULL;
  PTP_VENDOR_DATA_OUT* pDataOut = NULL;
  DWORD dwDataInSize = SIZEOF_REQUIRED_VENDOR_DATA_IN;
  DWORD dwDataOutSize = SIZEOF_REQUIRED_VENDOR_DATA_OUT + size;
  DWORD dwActualDataOutSize = 0;

  pDataIn = static_cast<PTP_VENDOR_DATA_IN*>(CoTaskMemAlloc(dwDataInSize));
  if (pDataIn == NULL) {
    return S_FALSE;
  }
  pDataOut = static_cast<PTP_VENDOR_DATA_OUT*>(CoTaskMemAlloc(dwDataOutSize));
  if (pDataOut == NULL) {
    CoTaskMemFree(pDataIn);
    return S_FALSE;
  }

  ZeroMemory(pDataIn, dwDataInSize);
  if (pDataOut != NULL) {
    ZeroMemory(pDataOut, dwDataOutSize);
  }

  pDataIn->OpCode = PTP_OC_GetObject;
  pDataIn->NextPhase = PTP_NEXTPHASE_READ_DATA;
  pDataIn->Params[0] = objectHandle;
  pDataIn->NumParams = 1;

  hr = itemExtra->Escape(
      ESCAPE_PTP_VENDOR_COMMAND, reinterpret_cast<BYTE*>(pDataIn), dwDataInSize,
      reinterpret_cast<BYTE*>(pDataOut), dwDataOutSize, &dwActualDataOutSize);

  if (SUCCEEDED(hr)) {
    memcpy_s(buffer, size, pDataOut->VendorReadData, size);
  }

  CoTaskMemFree(pDataIn);
  CoTaskMemFree(pDataOut);

  ReleaseMutex(hMutex);

  return hr;
}

HRESULT PTPControl::SDIOConnect(DWORD param1, DWORD param2, DWORD param3) {
  if (itemExtra == NULL) {
    return S_FALSE;
  }
  if (hMutex == NULL) {
    hMutex = CreateMutex(NULL, FALSE, MUTEX);
    if (hMutex == NULL) {
      return S_FALSE;
    }
  }
  WaitForSingleObject(hMutex, INFINITE);

  HRESULT hr = 0;
  PTP_VENDOR_DATA_IN* pDataIn = NULL;
  PTP_VENDOR_DATA_OUT* pDataOut = NULL;
  DWORD dwDataInSize = SIZEOF_REQUIRED_VENDOR_DATA_IN;
  DWORD dwDataOutSize = SIZEOF_REQUIRED_VENDOR_DATA_OUT + 0x0008;
  DWORD dwActualDataOutSize = 0;

  pDataIn = static_cast<PTP_VENDOR_DATA_IN*>(CoTaskMemAlloc(dwDataInSize));
  if (pDataIn == NULL) {
    return S_FALSE;
  }
  pDataOut = static_cast<PTP_VENDOR_DATA_OUT*>(CoTaskMemAlloc(dwDataOutSize));
  if (pDataOut == NULL) {
    CoTaskMemFree(pDataIn);
    return S_FALSE;
  }

  ZeroMemory(pDataIn, dwDataInSize);
  ZeroMemory(pDataOut, dwDataOutSize);

  pDataIn->OpCode = PTP_OC_SDIOConnect;
  pDataIn->NextPhase = PTP_NEXTPHASE_READ_DATA;
  pDataIn->Params[0] = param1;
  pDataIn->Params[1] = param2;
  pDataIn->Params[2] = param3;
  pDataIn->NumParams = 3;

  hr = itemExtra->Escape(
      ESCAPE_PTP_VENDOR_COMMAND, reinterpret_cast<BYTE*>(pDataIn), dwDataInSize,
      reinterpret_cast<BYTE*>(pDataOut), dwDataOutSize, &dwActualDataOutSize);

  CoTaskMemFree(pDataIn);
  CoTaskMemFree(pDataOut);

  ReleaseMutex(hMutex);

  return hr;
}

HRESULT PTPControl::SDIOGetExtDeviceInfo(DWORD param) {
  if (itemExtra == NULL) {
    return S_FALSE;
  }
  WaitForSingleObject(hMutex, INFINITE);

  HRESULT hr = 0;
  PTP_VENDOR_DATA_IN* pDataIn = NULL;
  PTP_VENDOR_DATA_OUT* pDataOut = NULL;
  DWORD dwDataInSize = SIZEOF_REQUIRED_VENDOR_DATA_IN;
  DWORD dwDataOutSize = SIZEOF_REQUIRED_VENDOR_DATA_OUT + 0x1000;
  DWORD dwActualDataOutSize = 0;

  pDataIn = static_cast<PTP_VENDOR_DATA_IN*>(CoTaskMemAlloc(dwDataInSize));
  if (pDataIn == NULL) {
    return S_FALSE;
  }
  pDataOut = static_cast<PTP_VENDOR_DATA_OUT*>(CoTaskMemAlloc(dwDataOutSize));
  if (pDataOut == NULL) {
    CoTaskMemFree(pDataIn);
    return S_FALSE;
  }

  ZeroMemory(pDataIn, dwDataInSize);
  ZeroMemory(pDataOut, dwDataOutSize);

  pDataIn->OpCode = PTP_OC_SDIOGetExtDeviceInfo;
  pDataIn->NextPhase = PTP_NEXTPHASE_READ_DATA;
  pDataIn->Params[0] = param;
  pDataIn->NumParams = 1;

  hr = itemExtra->Escape(
      ESCAPE_PTP_VENDOR_COMMAND, reinterpret_cast<BYTE*>(pDataIn), dwDataInSize,
      reinterpret_cast<BYTE*>(pDataOut), dwDataOutSize, &dwActualDataOutSize);

  setSupportedCommands(pDataOut);
  CoTaskMemFree(pDataIn);

  ReleaseMutex(hMutex);

  return hr;
}

HRESULT PTPControl::SDIOControlDevice(DWORD param1, UINT32 value) {
  if (itemExtra == NULL) {
    return S_FALSE;
  }
  WaitForSingleObject(hMutex, INFINITE);

  HRESULT hr = 0;
  PTP_VENDOR_DATA_IN* pDataIn = NULL;
  PTP_VENDOR_DATA_OUT* pDataOut = NULL;
  DWORD dwDataInSize = SIZEOF_REQUIRED_VENDOR_DATA_IN + sizeof(UINT32);
  DWORD dwDataOutSize = SIZEOF_REQUIRED_VENDOR_DATA_OUT;
  DWORD dwActualDataOutSize = 0;

  pDataIn = static_cast<PTP_VENDOR_DATA_IN*>(CoTaskMemAlloc(dwDataInSize));
  if (pDataIn == NULL) {
    return S_FALSE;
  }
  pDataOut = static_cast<PTP_VENDOR_DATA_OUT*>(CoTaskMemAlloc(dwDataOutSize));
  if (pDataOut == NULL) {
    CoTaskMemFree(pDataIn);
    return S_FALSE;
  }

  ZeroMemory(pDataIn, dwDataInSize);
  ZeroMemory(pDataOut, dwDataOutSize);

  pDataIn->OpCode = PTP_OC_SDIOControlDevice;
  pDataIn->NextPhase = PTP_NEXTPHASE_WRITE_DATA;
  pDataIn->Params[0] = param1;
  pDataIn->NumParams = 1;

  UINT32* tmp = reinterpret_cast<UINT32*>(pDataIn->VendorWriteData);
  *tmp = value;

  hr = itemExtra->Escape(
      ESCAPE_PTP_VENDOR_COMMAND, reinterpret_cast<BYTE*>(pDataIn), dwDataInSize,
      reinterpret_cast<BYTE*>(pDataOut), dwDataOutSize, &dwActualDataOutSize);
  if (hr != S_OK) {
    return hr;
  }

  if (PTP_RC_OK != pDataOut->ResponseCode) {
    hr = E_FAIL;
  }

  CoTaskMemFree(pDataIn);
  CoTaskMemFree(pDataOut);

  ReleaseMutex(hMutex);

  return hr;
}

HRESULT PTPControl::ExectuteDeleteObject(int objectHandle) {
  if (itemExtra == NULL) {
    return S_FALSE;
  }
  WaitForSingleObject(hMutex, INFINITE);

  HRESULT hr = 0;
  PTP_VENDOR_DATA_IN* pDataIn = NULL;
  PTP_VENDOR_DATA_OUT* pDataOut = NULL;
  DWORD dwDataInSize = SIZEOF_REQUIRED_VENDOR_DATA_IN + sizeof(UINT16);
  DWORD dwDataOutSize = SIZEOF_REQUIRED_VENDOR_DATA_OUT;
  DWORD dwActualDataOutSize = 0;

  pDataIn = static_cast<PTP_VENDOR_DATA_IN*>(CoTaskMemAlloc(dwDataInSize));
  if (pDataIn == NULL) {
    return S_FALSE;
  }
  pDataOut = static_cast<PTP_VENDOR_DATA_OUT*>(CoTaskMemAlloc(dwDataOutSize));
  if (pDataOut == NULL) {
    CoTaskMemFree(pDataIn);
    return S_FALSE;
  }

  ZeroMemory(pDataIn, dwDataInSize);
  ZeroMemory(pDataOut, dwDataOutSize);

  pDataIn->OpCode = PTP_OC_DeleteObject;
  pDataIn->NextPhase = PTP_NEXTPHASE_WRITE_DATA;
  pDataIn->Params[0] = objectHandle;
  pDataIn->NumParams = 1;

  hr = itemExtra->Escape(
      ESCAPE_PTP_VENDOR_COMMAND, reinterpret_cast<BYTE*>(pDataIn), dwDataInSize,
      reinterpret_cast<BYTE*>(pDataOut), dwDataOutSize, &dwActualDataOutSize);
  if (hr != S_OK) {
    return hr;
  }

  if (PTP_RC_OK != pDataOut->ResponseCode) {
    hr = E_FAIL;
  }

  CoTaskMemFree(pDataIn);
  CoTaskMemFree(pDataOut);

  ReleaseMutex(hMutex);

  return hr;
}

HRESULT PTPControl::SDIOSetContentsTransferMode(DWORD param1, DWORD param2,
                                                DWORD param3) {
  if (itemExtra == NULL) {
    return S_FALSE;
  }
  if (hMutex == NULL) {
    hMutex = CreateMutex(NULL, FALSE, MUTEX);
    if (hMutex == NULL) {
      return S_FALSE;
    }
  }
  WaitForSingleObject(hMutex, INFINITE);

  HRESULT hr = 0;
  PTP_VENDOR_DATA_IN* pDataIn = NULL;
  PTP_VENDOR_DATA_OUT* pDataOut = NULL;
  DWORD dwDataInSize = SIZEOF_REQUIRED_VENDOR_DATA_IN;
  DWORD dwDataOutSize = SIZEOF_REQUIRED_VENDOR_DATA_OUT;
  DWORD dwActualDataOutSize = 0;

  pDataIn = static_cast<PTP_VENDOR_DATA_IN*>(CoTaskMemAlloc(dwDataInSize));
  if (pDataIn == NULL) {
    return S_FALSE;
  }
  pDataOut = static_cast<PTP_VENDOR_DATA_OUT*>(CoTaskMemAlloc(dwDataOutSize));
  if (pDataOut == NULL) {
    CoTaskMemFree(pDataIn);
    return S_FALSE;
  }

  ZeroMemory(pDataIn, dwDataInSize);
  ZeroMemory(pDataOut, dwDataOutSize);

  pDataIn->OpCode = PTP_OC_SDIOSetContentsTransferMode;
  pDataIn->NextPhase = PTP_NEXTPHASE_READ_DATA;
  pDataIn->Params[0] = param1;
  pDataIn->Params[1] = param2;
  pDataIn->Params[2] = param3;
  pDataIn->NumParams = 3;

  hr = itemExtra->Escape(
      ESCAPE_PTP_VENDOR_COMMAND, reinterpret_cast<BYTE*>(pDataIn), dwDataInSize,
      reinterpret_cast<BYTE*>(pDataOut), dwDataOutSize, &dwActualDataOutSize);

  CoTaskMemFree(pDataIn);
  CoTaskMemFree(pDataOut);

  ReleaseMutex(hMutex);

  return hr;
}

HRESULT PTPControl::SDIOSetExtDevicePropValueImpl(DWORD param1, void* value,
                                                  DWORD size) {
  if (itemExtra == NULL) {
    return S_FALSE;
  }

  WaitForSingleObject(hMutex, INFINITE);

  HRESULT hr = 0;
  PTP_VENDOR_DATA_IN* pDataIn = NULL;
  PTP_VENDOR_DATA_OUT* pDataOut = NULL;
  DWORD dwDataInSize = SIZEOF_REQUIRED_VENDOR_DATA_IN + size;
  DWORD dwDataOutSize = SIZEOF_REQUIRED_VENDOR_DATA_OUT;
  DWORD dwActualDataOutSize = 0;

  pDataIn = static_cast<PTP_VENDOR_DATA_IN*>(CoTaskMemAlloc(dwDataInSize));
  if (pDataIn == NULL) {
    return S_FALSE;
  }
  pDataOut = static_cast<PTP_VENDOR_DATA_OUT*>(CoTaskMemAlloc(dwDataOutSize));
  if (pDataOut == NULL) {
    CoTaskMemFree(pDataIn);
    return S_FALSE;
  }

  ZeroMemory(pDataIn, dwDataInSize);
  ZeroMemory(pDataOut, dwDataOutSize);

  pDataIn->OpCode = PTP_OC_SDIOSetExtDevicePropValue;
  pDataIn->NextPhase = PTP_NEXTPHASE_WRITE_DATA;
  pDataIn->Params[0] = param1;
  pDataIn->NumParams = 1;

  memcpy_s(pDataIn->VendorWriteData, size, value, size);

  hr = itemExtra->Escape(
      ESCAPE_PTP_VENDOR_COMMAND, reinterpret_cast<BYTE*>(pDataIn), dwDataInSize,
      reinterpret_cast<BYTE*>(pDataOut), dwDataOutSize, &dwActualDataOutSize);
  if (hr != S_OK) {
    return hr;
  }

  if (PTP_RC_OK != pDataOut->ResponseCode) {
    hr = E_FAIL;
  }

  CoTaskMemFree(pDataIn);
  CoTaskMemFree(pDataOut);

  ReleaseMutex(hMutex);

  return hr;
}

HRESULT PTPControl::SDIOGetAllExtDevicePropInfo(HWND pHwnd) {
  if (itemExtra == NULL) {
    return S_FALSE;
  }
  WaitForSingleObject(hMutex, INFINITE);

  HRESULT hr = 0;
  PTP_VENDOR_DATA_IN* pDataIn = NULL;
  PTP_VENDOR_DATA_OUT* pDataOut = NULL;
  DWORD dwDataInSize = SIZEOF_REQUIRED_VENDOR_DATA_IN;
  DWORD dwDataOutSize = SIZEOF_REQUIRED_VENDOR_DATA_OUT + 64 * 1024;
  DWORD dwActualDataOutSize = 0;

  pDataIn = static_cast<PTP_VENDOR_DATA_IN*>(CoTaskMemAlloc(dwDataInSize));
  if (pDataIn == NULL) {
    return S_FALSE;
  }
  pDataOut = static_cast<PTP_VENDOR_DATA_OUT*>(CoTaskMemAlloc(dwDataOutSize));
  if (pDataOut == NULL) {
    CoTaskMemFree(pDataIn);
    return S_FALSE;
  }

  ZeroMemory(pDataIn, dwDataInSize);
  ZeroMemory(pDataOut, dwDataOutSize);

  pDataIn->OpCode = PTP_OC_SDIOGetAllExtDeviceInfo;
  pDataIn->NextPhase = PTP_NEXTPHASE_READ_DATA;
  pDataIn->NumParams = 0;

  hr = itemExtra->Escape(
      ESCAPE_PTP_VENDOR_COMMAND, reinterpret_cast<BYTE*>(pDataIn), dwDataInSize,
      reinterpret_cast<BYTE*>(pDataOut), dwDataOutSize, &dwActualDataOutSize);
  setCameraData(pDataOut);

  CoTaskMemFree(pDataIn);

  ReleaseMutex(hMutex);

  return hr;
}

HRESULT PTPControl::CloseSession() {
  if (itemExtra == NULL) {
    return S_FALSE;
  }
  WaitForSingleObject(hMutex, INFINITE);

  HRESULT hr = 0;
  PTP_VENDOR_DATA_IN* pDataIn = NULL;
  PTP_VENDOR_DATA_OUT* pDataOut = NULL;
  DWORD dwDataInSize = SIZEOF_REQUIRED_VENDOR_DATA_IN;
  DWORD dwDataOutSize = SIZEOF_REQUIRED_VENDOR_DATA_OUT + 0x1000;
  DWORD dwActualDataOutSize = 0;

  pDataIn = static_cast<PTP_VENDOR_DATA_IN*>(CoTaskMemAlloc(dwDataInSize));
  if (pDataIn == NULL) {
    return S_FALSE;
  }
  pDataOut = static_cast<PTP_VENDOR_DATA_OUT*>(CoTaskMemAlloc(dwDataOutSize));
  if (pDataOut == NULL) {
    CoTaskMemFree(pDataIn);
    return S_FALSE;
  }

  ZeroMemory(pDataIn, dwDataInSize);
  ZeroMemory(pDataOut, dwDataOutSize);

  pDataIn->OpCode = PTP_OC_CloseSession;
  pDataIn->NextPhase = PTP_NEXTPHASE_READ_DATA;
  pDataIn->NumParams = 0;

  hr = itemExtra->Escape(
      ESCAPE_PTP_VENDOR_COMMAND, reinterpret_cast<BYTE*>(pDataIn), dwDataInSize,
      reinterpret_cast<BYTE*>(pDataOut), dwDataOutSize, &dwActualDataOutSize);
  setCameraData(pDataOut);

  CoTaskMemFree(pDataIn);

  ReleaseMutex(hMutex);

  return hr;
}

HRESULT PTPControl::GetDeviceInfo() {
  if (itemExtra == NULL) {
    return S_FALSE;
  }
  WaitForSingleObject(hMutex, INFINITE);

  HRESULT hr = 0;
  PTP_VENDOR_DATA_IN* pDataIn = NULL;
  PTP_VENDOR_DATA_OUT* pDataOut = NULL;
  DWORD dwDataInSize = SIZEOF_REQUIRED_VENDOR_DATA_IN;
  DWORD dwDataOutSize = SIZEOF_REQUIRED_VENDOR_DATA_OUT + 0x1000;
  DWORD dwActualDataOutSize = 0;

  pDataIn = static_cast<PTP_VENDOR_DATA_IN*>(CoTaskMemAlloc(dwDataInSize));
  if (pDataIn == NULL) {
    return S_FALSE;
  }
  pDataOut = static_cast<PTP_VENDOR_DATA_OUT*>(CoTaskMemAlloc(dwDataOutSize));
  if (pDataOut == NULL) {
    CoTaskMemFree(pDataIn);
    return S_FALSE;
  }

  ZeroMemory(pDataIn, dwDataInSize);
  ZeroMemory(pDataOut, dwDataOutSize);

  pDataIn->OpCode = PTP_OC_GetDeviceInfo;
  pDataIn->NextPhase = PTP_NEXTPHASE_READ_DATA;
  pDataIn->NumParams = 0;

  hr = itemExtra->Escape(
      ESCAPE_PTP_VENDOR_COMMAND, reinterpret_cast<BYTE*>(pDataIn), dwDataInSize,
      reinterpret_cast<BYTE*>(pDataOut), dwDataOutSize, &dwActualDataOutSize);

  dMgr.setDeviceInfoData(pDataOut);
  CoTaskMemFree(pDataIn);

  ReleaseMutex(hMutex);

  return hr;
}

HRESULT PTPControl::GetStorageID() {
  if (itemExtra == NULL) {
    return S_FALSE;
  }
  WaitForSingleObject(hMutex, INFINITE);

  HRESULT hr = 0;
  PTP_VENDOR_DATA_IN* pDataIn = NULL;
  PTP_VENDOR_DATA_OUT* pDataOut = NULL;
  DWORD dwDataInSize = SIZEOF_REQUIRED_VENDOR_DATA_IN;
  DWORD dwDataOutSize = SIZEOF_REQUIRED_VENDOR_DATA_OUT + 0x1000;
  DWORD dwActualDataOutSize = 0;

  pDataIn = static_cast<PTP_VENDOR_DATA_IN*>(CoTaskMemAlloc(dwDataInSize));
  if (pDataIn == NULL) {
    return S_FALSE;
  }
  pDataOut = static_cast<PTP_VENDOR_DATA_OUT*>(CoTaskMemAlloc(dwDataOutSize));
  if (pDataOut == NULL) {
    CoTaskMemFree(pDataIn);
    return S_FALSE;
  }

  ZeroMemory(pDataIn, dwDataInSize);
  ZeroMemory(pDataOut, dwDataOutSize);

  pDataIn->OpCode = PTP_OC_GetStorageID;
  pDataIn->NextPhase = PTP_NEXTPHASE_READ_DATA;
  pDataIn->NumParams = 0;

  hr = itemExtra->Escape(
      ESCAPE_PTP_VENDOR_COMMAND, reinterpret_cast<BYTE*>(pDataIn), dwDataInSize,
      reinterpret_cast<BYTE*>(pDataOut), dwDataOutSize, &dwActualDataOutSize);

  dMgr.setStorageID(pDataOut);
  CoTaskMemFree(pDataIn);

  ReleaseMutex(hMutex);

  return hr;
}

HRESULT PTPControl::GetNumObjects() {
  if (itemExtra == NULL) {
    return S_FALSE;
  }
  WaitForSingleObject(hMutex, INFINITE);

  HRESULT hr = 0;
  PTP_VENDOR_DATA_IN* pDataIn = NULL;
  PTP_VENDOR_DATA_OUT* pDataOut = NULL;
  DWORD dwDataInSize = SIZEOF_REQUIRED_VENDOR_DATA_IN;
  DWORD dwDataOutSize = SIZEOF_REQUIRED_VENDOR_DATA_OUT + 0x1000;
  DWORD dwActualDataOutSize = 0;

  pDataIn = static_cast<PTP_VENDOR_DATA_IN*>(CoTaskMemAlloc(dwDataInSize));
  if (pDataIn == NULL) {
    return S_FALSE;
  }
  pDataOut = static_cast<PTP_VENDOR_DATA_OUT*>(CoTaskMemAlloc(dwDataOutSize));
  if (pDataOut == NULL) {
    CoTaskMemFree(pDataIn);
    return S_FALSE;
  }

  ZeroMemory(pDataIn, dwDataInSize);
  ZeroMemory(pDataOut, dwDataOutSize);

  pDataIn->OpCode = PTP_OC_GetNumObjects;
  pDataIn->NextPhase = PTP_NEXTPHASE_READ_DATA;
  pDataIn->Params[0] = 0x00010001;
  pDataIn->Params[1] = 0;
  pDataIn->Params[2] = 0;
  pDataIn->NumParams = 3;

  hr = itemExtra->Escape(
      ESCAPE_PTP_VENDOR_COMMAND, reinterpret_cast<BYTE*>(pDataIn), dwDataInSize,
      reinterpret_cast<BYTE*>(pDataOut), dwDataOutSize, &dwActualDataOutSize);

  dMgr.setNumObjects(pDataOut);
  CoTaskMemFree(pDataIn);

  ReleaseMutex(hMutex);

  return hr;
}

HRESULT PTPControl::GetObjectHandles(DWORD param1, DWORD param2, DWORD param3) {
  if (itemExtra == NULL) {
    return S_FALSE;
  }
  WaitForSingleObject(hMutex, INFINITE);

  HRESULT hr = 0;
  PTP_VENDOR_DATA_IN* pDataIn = NULL;
  PTP_VENDOR_DATA_OUT* pDataOut = NULL;
  DWORD dwDataInSize = SIZEOF_REQUIRED_VENDOR_DATA_IN;
  DWORD dwDataOutSize = SIZEOF_REQUIRED_VENDOR_DATA_OUT + 0x1000;
  DWORD dwActualDataOutSize = 0;

  pDataIn = static_cast<PTP_VENDOR_DATA_IN*>(CoTaskMemAlloc(dwDataInSize));
  if (pDataIn == NULL) {
    return S_FALSE;
  }
  pDataOut = static_cast<PTP_VENDOR_DATA_OUT*>(CoTaskMemAlloc(dwDataOutSize));
  if (pDataOut == NULL) {
    CoTaskMemFree(pDataIn);
    return S_FALSE;
  }

  ZeroMemory(pDataIn, dwDataInSize);
  ZeroMemory(pDataOut, dwDataOutSize);

  pDataIn->OpCode = PTP_OC_GetObjectHandles;
  pDataIn->NextPhase = PTP_NEXTPHASE_READ_DATA;
  pDataIn->Params[0] = param1;
  pDataIn->Params[1] = param2;
  pDataIn->Params[2] = param3;
  pDataIn->NumParams = 3;

  hr = itemExtra->Escape(
      ESCAPE_PTP_VENDOR_COMMAND, reinterpret_cast<BYTE*>(pDataIn), dwDataInSize,
      reinterpret_cast<BYTE*>(pDataOut), dwDataOutSize, &dwActualDataOutSize);
  dMgr.setObjectHandles(pDataOut);
  CoTaskMemFree(pDataIn);

  ReleaseMutex(hMutex);

  return hr;
}

void PTPControl::closeMutex() {
  if (hMutex) {
    ReleaseMutex(hMutex);
    CloseHandle(hMutex);
  }
}
