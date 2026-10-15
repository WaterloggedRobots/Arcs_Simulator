#include "DataManager.h"

#include <commctrl.h>
#include <tchar.h>
#include <windows.h>

#include "DevicePropItemList.h"
#include "PTPControl.h"
#include "resource.h"

using namespace std;

BOOL hideUnsupportedCombo(ListEntryItems* comboItemList,
                          unsigned char isEnabled);

int createList(ListEntryItems* comboItemList, unsigned short num,
               unsigned long sizeofType, PTP_VENDOR_DATA_OUT* pDataOut,
               unsigned int offset);

void updateCombo(HWND itemHwnd, ListEntryItems* comboItemList,
                 unsigned long long currentValue, int index);

static PTP_VENDOR_DATA_OUT* pDataHolder = NULL;
static PTP_VENDOR_DATA_OUT* pSupportedCommands = NULL;
static PTP_VENDOR_DATA_OUT* pDeviceInfoHolder = NULL;
static PTP_VENDOR_DATA_OUT* pGetStorageIDHolder = NULL;
static PTP_VENDOR_DATA_OUT* pGetNumObjectsHolder = NULL;
static PTP_VENDOR_DATA_OUT* pGetObjectHandlesHolder = NULL;
static PTP_VENDOR_DATA_OUT* pGetObjectHandlesFolderHolder = NULL;
static PTP_VENDOR_DATA_OUT* pGetStorageInfoHolder = NULL;
static HWND captureHWND;
static HWND capture_mtpHWND;
static PTPControl& ptp = PTPControl::getInstance();

BOOL isLiveviewValid = false;
BOOL comboFocusFlag = FALSE;
int start = NULL;
int count_battery = NULL;
int count_fnum = NULL;
int count_shutterspeed = NULL;
int colortempflag = NULL;
UINT16 ColorTemp_before = 0x157C;

DataManager& DataManager::getInstance() {
  static DataManager instance;
  return instance;
}

DataManager::DataManager() {}

DataManager::~DataManager() {}

void DataManager::setCaptureHWND(HWND cHwnd) { captureHWND = cHwnd; }

void DataManager::setIsLiveviewValidFlag(BOOL validFlag) {
  isLiveviewValid = validFlag;
}

BOOL DataManager::getIsLiveviewValidFlag() { return isLiveviewValid; }

void DataManager::setComboFocus(BOOL fucus) { comboFocusFlag = fucus; }

BOOL DataManager::getComboFocus() { return comboFocusFlag; }

void DataManager::setSupportedCommands(PTP_VENDOR_DATA_OUT* pDataOut) {
  pSupportedCommands = pDataOut;
}

BOOL DataManager::checkSupportedCommands(unsigned short code) {
  unsigned int offset = sizeof(unsigned short);
  unsigned short size = *reinterpret_cast<unsigned short*>(
      &pSupportedCommands->VendorReadData[offset]);
  offset = sizeof(unsigned short);

  for (unsigned long long i = 0; i < size; i++) {
    {
      unsigned short propertyCode = *reinterpret_cast<unsigned short*>(
          &pSupportedCommands->VendorReadData[offset]);
      offset += sizeof(unsigned short);
      if (code == propertyCode) {
        CoTaskMemFree(pSupportedCommands);
        return TRUE;
      }
    }
  }
  CoTaskMemFree(pSupportedCommands);
  return FALSE;
}

void DataManager::setCameraData(PTP_VENDOR_DATA_OUT* pDataOut) {
  pDataHolder = pDataOut;
}

void DataManager::setDeviceInfoData(PTP_VENDOR_DATA_OUT* pDataOut) {
  pDeviceInfoHolder = pDataOut;
}

void DataManager::setStorageID(PTP_VENDOR_DATA_OUT* pDataOut) {
  pGetStorageIDHolder = pDataOut;
}

void DataManager::setNumObjects(PTP_VENDOR_DATA_OUT* pDataOut) {
  pGetNumObjectsHolder = pDataOut;
}

void DataManager::setObjectHandles(PTP_VENDOR_DATA_OUT* pDataOut) {
  pGetObjectHandlesHolder = pDataOut;
}

PTP_VENDOR_DATA_OUT* DataManager::getDeviceInfoData() {
  return pDeviceInfoHolder;
}

PTP_VENDOR_DATA_OUT* DataManager::getStorageID() { return pGetStorageIDHolder; }

PTP_VENDOR_DATA_OUT* DataManager::getNumObjects() {
  return pGetNumObjectsHolder;
}

PTP_VENDOR_DATA_OUT* DataManager::getObjectHandles() {
  return pGetObjectHandlesHolder;
}

PTP_VENDOR_DATA_OUT* DataManager::getstorageinfoData() {
  return pGetStorageInfoHolder;
}

void DataManager::readCameraData() {
  if (pDataHolder == NULL) {
    return;
  }
  unsigned long long length =
      *reinterpret_cast<unsigned long long*>(&pDataHolder->VendorReadData[0]);
  unsigned int offset = sizeof(length);

  for (unsigned long long i = 0; i < length; i++) {
    unsigned short propertyCode = *reinterpret_cast<unsigned short*>(
        &pDataHolder->VendorReadData[offset]);
    offset += sizeof(unsigned short);
    unsigned short dataType = *reinterpret_cast<unsigned short*>(
        &pDataHolder->VendorReadData[offset]);
    offset += sizeof(unsigned short);
    unsigned char flagGetSet =
        *static_cast<unsigned char*>(&pDataHolder->VendorReadData[offset]);
    offset += sizeof(unsigned char);
    unsigned char isEnabled =
        *static_cast<unsigned char*>(&pDataHolder->VendorReadData[offset]);
    offset += sizeof(unsigned char);
    unsigned long sizeofType = 0;
    unsigned long long defaultValue = 0;
    unsigned long long currentValue = 0;
    unsigned char numchar_defaultValue = 0;
    unsigned char numchar_currentValue = 0;

    switch (dataType) {
      case PTP_DT_INT8:
      case PTP_DT_UINT8:
        sizeofType = sizeof(char);
        defaultValue =
            *static_cast<unsigned char*>(&pDataHolder->VendorReadData[offset]);
        offset += sizeof(unsigned char);
        currentValue =
            *static_cast<unsigned char*>(&pDataHolder->VendorReadData[offset]);
        offset += sizeof(unsigned char);
        break;
      case PTP_DT_INT16:
      case PTP_DT_UINT16:
        sizeofType = sizeof(short);
        defaultValue = *reinterpret_cast<unsigned short*>(
            &pDataHolder->VendorReadData[offset]);
        offset += sizeof(unsigned short);
        currentValue = *reinterpret_cast<unsigned short*>(
            &pDataHolder->VendorReadData[offset]);
        offset += sizeof(unsigned short);
        break;
      case PTP_DT_INT32:
      case PTP_DT_UINT32:
        sizeofType = sizeof(long);
        defaultValue = *reinterpret_cast<unsigned long*>(
            &pDataHolder->VendorReadData[offset]);
        offset += sizeof(unsigned long);
        currentValue = *reinterpret_cast<unsigned long*>(
            &pDataHolder->VendorReadData[offset]);
        offset += sizeof(unsigned long);
        break;
      case PTP_DT_INT64:
      case PTP_DT_UINT64:
        sizeofType = sizeof(long long);
        defaultValue = *reinterpret_cast<unsigned long long*>(
            &pDataHolder->VendorReadData[offset]);
        offset += sizeof(unsigned long long);
        currentValue = *reinterpret_cast<unsigned long long*>(
            &pDataHolder->VendorReadData[offset]);
        offset += sizeof(unsigned long long);
        break;
      case PTP_DT_STR:
        sizeofType = 0;
        numchar_defaultValue =
            *static_cast<UINT8*>(&pDataHolder->VendorReadData[offset]);
        offset += sizeof(UINT8);
        offset += numchar_defaultValue * sizeof(wchar_t);
        numchar_currentValue =
            *static_cast<UINT8*>(&pDataHolder->VendorReadData[offset]);
        offset += sizeof(UINT8);
        offset += numchar_currentValue * sizeof(wchar_t);
        break;
      case PTP_DT_AINT8:
      case PTP_DT_AUINT8:
        sizeofType = 0;
        defaultValue = *reinterpret_cast<unsigned long*>(
            &pDataHolder->VendorReadData[offset]);
        offset += sizeof(unsigned long);
        offset +=
            static_cast<unsigned int>(sizeof(unsigned char) * defaultValue);
        currentValue = *reinterpret_cast<unsigned long*>(
            &pDataHolder->VendorReadData[offset]);
        offset += sizeof(unsigned long);
        offset +=
            static_cast<unsigned int>(sizeof(unsigned char) * currentValue);
        break;
      case PTP_DT_AINT16:
      case PTP_DT_AUINT16:
        sizeofType = 0;
        defaultValue = *reinterpret_cast<unsigned long*>(
            &pDataHolder->VendorReadData[offset]);
        offset += sizeof(unsigned long);
        offset +=
            static_cast<unsigned int>(sizeof(unsigned short) * defaultValue);
        currentValue = *reinterpret_cast<unsigned long*>(
            &pDataHolder->VendorReadData[offset]);
        offset += sizeof(unsigned long);
        offset +=
            static_cast<unsigned int>(sizeof(unsigned short) * currentValue);
        break;
      case PTP_DT_AINT32:
      case PTP_DT_AUINT32:
        sizeofType = 0;
        defaultValue = *reinterpret_cast<unsigned long*>(
            &pDataHolder->VendorReadData[offset]);
        offset += sizeof(unsigned long);
        offset +=
            static_cast<unsigned int>(sizeof(unsigned long) * defaultValue);
        currentValue = *reinterpret_cast<unsigned long*>(
            &pDataHolder->VendorReadData[offset]);
        offset += sizeof(unsigned long);
        offset +=
            static_cast<unsigned int>(sizeof(unsigned long) * currentValue);
        break;
      case PTP_DT_AINT64:
      case PTP_DT_AUINT64:
        sizeofType = 0;
        defaultValue = *reinterpret_cast<unsigned long*>(
            &pDataHolder->VendorReadData[offset]);
        offset += sizeof(unsigned long);
        offset += static_cast<unsigned int>(sizeof(unsigned long long) *
                                            defaultValue);
        currentValue = *reinterpret_cast<unsigned long*>(
            &pDataHolder->VendorReadData[offset]);
        offset += sizeof(unsigned long);
        offset += static_cast<unsigned int>(sizeof(unsigned long long) *
                                            currentValue);
        break;
      default:
        sizeofType = 0;
        break;
    }

    unsigned char formFlag =
        *static_cast<unsigned char*>(&pDataHolder->VendorReadData[offset]);
    offset += sizeof(unsigned char);

    if (formFlag == 0x01) {
      checkShootingFiles(propertyCode, currentValue, pDataHolder, offset);

      updataRangeDataValue(propertyCode, currentValue);

      offset += (sizeofType * 3);

    }

    else if (formFlag == 0)

    {
      if (propertyCode == DPC_ZOOM_OPTIC)
      {
        updataRangeDataValue(propertyCode, currentValue);
      }
      if (start == 0) {
        setIsEnableStatus(0, isEnabled);
        start = 1;
      }

    }

    else
    {
      unsigned short num = *reinterpret_cast<unsigned short*>(
          &pDataHolder->VendorReadData[offset]);
      offset += sizeof(unsigned short);

      offset += (num * sizeofType);
      num = *reinterpret_cast<unsigned short*>(
          &pDataHolder->VendorReadData[offset]);
      offset += sizeof(unsigned short);

      updataEnumDataValue(propertyCode, currentValue);

      for (int prop = 0;
           prop < sizeof(devicePropItemList) / sizeof(devicePropItemList[0]);
           prop++) {
        if (propertyCode == devicePropItemList[prop].propertyCode) {
          if (!hideUnsupportedCombo(&devicePropItemList[prop], isEnabled)) {
            break;
          }
          setIsEnableStatus(prop, isEnabled);
          int index = createList(&devicePropItemList[prop], num, sizeofType,
                                 pDataHolder, offset);

          if (getComboFocus() == FALSE) {
            updateCombo(
                GetDlgItem(captureHWND, devicePropItemList[prop].resourceID),
                &devicePropItemList[prop], currentValue, index);
          }
        }
      }
      offset += (num * sizeofType);
    }
  }
  CoTaskMemFree(pDataHolder);
  return;
}

void DataManager::setIsEnableStatus(unsigned int listNum,
                                    unsigned char isEnabled) {
  devicePropItemList[listNum].isEnabled = isEnabled;
}

unsigned char DataManager::getIsEnableStatus(unsigned int listNum) {
  return devicePropItemList[listNum].isEnabled;
}

void DataManager::checkShootingFiles(unsigned short propertyCode,
                                     unsigned long long currentValue,
                                     PTP_VENDOR_DATA_OUT* pDataHolder,
                                     unsigned int offset) {
  if (propertyCode == DPC_SHOOTING_FILE_INFOMATION) {
    if ((currentValue & 0x8000) == 0x8000) {
      PTP_GetObjectInfo info = {0};
      HRESULT hr = ptp.GetObjectInfo(SHOT_OBJECT_HANDLE, info);
      if (hr != S_OK) {
        MessageBox(NULL, __T("Error: GetObjectInfo(SHOT_OBJECT_HANDLE, info)"),
                   __T("Error"), MB_OK);
        return;
      }
      BYTE* buffer = new BYTE[info.ObjCompSz];

      hr = ptp.ExecuteGetObject(SHOT_OBJECT_HANDLE, buffer, info.ObjCompSz);
      if (hr != S_OK) {
        MessageBox(NULL,
                   __T("Error: ExecuteGetObject(SHOT_OBJECT_HANDLE, buffer, "
                       "info.ObjCompSz)"),
                   __T("Error"), MB_OK);
        delete[] buffer;
        return;
      }
      hr = saveImage(buffer, info.ObjCompSz, info.FileName, 0);
      delete[] buffer;
    }
  }
}

void DataManager::updataRangeDataValue(unsigned short propertyCode,
                                       unsigned long long currentValue) {
  TCHAR tcText[50];
  TCHAR msg[8];
  int nTestSize = 0;
  UINT16 ColorTemp = 0;
  int vOut = static_cast<int>(currentValue);

  switch (propertyCode) {
    case DPC_ZOOM_OPTIC:
      ZeroMemory(tcText, sizeof(tcText));
      nTestSize = (sizeof(tcText) / 2);

      wchar_t current_value_str[256];

      swprintf_s(current_value_str, L"0x%x", vOut);

      SetDlgItemText(captureHWND, IDC_COMBO_ZOOM_OPTIC, current_value_str);
      break;

    case DPC_SHUTTER_SPEED:
      updateDisplayValue(IDC_STATIC_SHUTTER_SPEED, currentValue,
                         ShutterSpeedList);
      break;

    case DPC_FNUMBER:
      updateDisplayValue(IDC_STATIC_FNUM, currentValue, FnumberList);
      break;

    case DPC_WHITEBALANCE_AB:
      updateDisplayValue(IDC_COMBO_WHITEBALANCE_AB, currentValue,
                         whitebalanceabList);
      ZeroMemory(tcText, sizeof(tcText));
      nTestSize = (sizeof(tcText) / 2);
      wsprintf(msg, TEXT("0x%x"), static_cast<UINT>(currentValue));
      WhitebalanceAB = std::stoi(msg, NULL, 16);
      break;

    case DPC_WHITEBALANCE_GM:
      updateDisplayValue(IDC_COMBO_WHITEBALANCE_GM, currentValue,
                         whitebalancegmList);
      ZeroMemory(tcText, sizeof(tcText));
      nTestSize = (sizeof(tcText) / 2);
      wsprintf(msg, TEXT("0x%x"), static_cast<UINT8>(currentValue));
      WhitebalanceGM = std::stoi(msg, NULL, 16);
      break;

    case DPC_COLOR_TEMP:
      ZeroMemory(tcText, sizeof(tcText));
      nTestSize = (sizeof(tcText) / 2);
      wsprintf(msg, TEXT("0x%x"), static_cast<UINT16>(currentValue));
      ColorTemp = std::stoi(msg, NULL, 16);

      if ((colortempflag == NULL) || (ColorTemp != ColorTemp_before)) {
        Sleep(200);
        HWND hBar = GetDlgItem(captureHWND, IDC_SLIDER_COLOR_TEMP);
        SendMessage(hBar, TBM_SETPOS, TRUE, ColorTemp);
        wsprintf(msg, TEXT("%d K"), static_cast<UINT16>(currentValue));
        SetWindowText(GetDlgItem(captureHWND, IDC_COLOR_TEMP_VALUE),
                      static_cast<LPCWSTR>(msg));
        ColorTemp_before = ColorTemp;
        colortempflag = 1;
      }
      break;
  }
}

void DataManager::updataEnumDataValue(unsigned short propertyCode,
                                      unsigned long long currentValue) {
  HWND cpDlg = NULL;
  switch (propertyCode) {
    case DPC_LIVEVIEW_STATUS:
      setIsLiveviewValidFlag((currentValue == 1));
      break;

    case DPC_ISO:
      updateDisplayValue(IDC_STATIC_ISO, currentValue, ISOItemList);
      break;

    case DPC_EXPOSURE_MODE:
      updateDisplayValue(IDC_COMBO_EXPOSURE_MODE, currentValue,
                         exposureModeList);
      ExposureProgram = static_cast<UINT32>(currentValue);
      break;

    case DPC_ASPECT_RATIO:
      updateDisplayValue(IDC_COMBO_ASPECT_RATIO, currentValue, AspectRatioList);
      break;

    case DPC_EXPOSURE_COMPENSATION:
      updateDisplayValue(IDC_STATIC_EXPOSURE, currentValue, ExpList);
      break;

    case DPC_AELOCK_INDICATION:
      updateDisplayValue(IDC_AELOCK_INDICATION, currentValue, aelList);
      break;

    case DPC_WHITE_BALANCE:
      updateDisplayValue(IDC_COMBO_WHITE_BALANCE, currentValue,
                         whitebalanceList);
      break;

    case DPC_FOCUS_AREA:
      updateDisplayValue(IDC_COMBO_FOCUS_AREA, currentValue, focusareaList);
      break;

    case DPC_BATTERY_LEVEL:
      updateDisplayValue(IDC_BATTERY_LEVEL, currentValue, batterylevelList);
      break;

    case DPC_AF_STATUS:
      updateDisplayValue(IDC_COMBO_AF_STATUS, currentValue, afstatusList);
      break;

    case DPC_LIVEVIEW_MODE:
      updateDisplayValue(DPC_LIVEVIEW_MODE, currentValue, liveviewModeList);
      break;

    case DPC_PICTURE_PROFILE:
      updateDisplayValue(IDC_COMBO_PICTURE_PROFILE, currentValue,
                         pictureprofileList);
      break;

    case DPC_CREATIVE_STYLE:
      updateDisplayValue(IDC_COMBO_CREATIVE_STYLE, currentValue,
                         creativestyleList);
      break;

    case DPC_MOVIE_FORMAT:
      updateDisplayValue(IDC_COMBO_MOVIE_FORMAT, currentValue, movieformatList);
      break;

    case DPC_MOVIE_QUALITY:
      updateDisplayValue(IDC_COMBO_MOVIE_QUALITY, currentValue,
                         moviequalityList);
      break;

    case DPC_SAVE_MEDIA:
      updateDisplayValue(IDC_COMBO_SAVE_MEDIA, currentValue, savemediaList);
      break;

    case DPC_FOCUS_MODE:
      updateDisplayValue(IDC_COMBO_FOCUS_MODE, currentValue, focusmodeList);
      break;

    case DPC_FILE_FORMAT:
      updateDisplayValue(IDC_COMBO_FILE_FORMAT, currentValue,
                         imagefileformatList);
      break;

    case DPC_JPEG_QUALITY:
      updateDisplayValue(IDC_COMBO_JPEG_QUALITY, currentValue, JpegQualityList);
      break;

    case DPC_ZOOM_SETTING:
      updateDisplayValue(IDC_COMBO_ZOOM_SETTING, currentValue, zoomsettingList);
      break;

    case DPC_ZOOM_SCALE:
      updateDisplayValue(IDC_COMBO_ZOOM_SCALE, currentValue, zoomscaleList);
      break;
  }
}

void DataManager::updateDisplayValue(int idc, unsigned long long currentValue,
                                     int listNum) {
  TCHAR tcText[50];
  TCHAR msg[8];
  int nTestSize = 0;

  ZeroMemory(tcText, sizeof(tcText));
  nTestSize = (sizeof(tcText) / 2);

  if (listNum >= 0) {
    ListItem* p = NULL;

    for (UINT8 num = 0; num < devicePropItemList[listNum].size; num++) {
      p = &devicePropItemList[listNum].itemList[num];
      if (p->value == currentValue) {
        GetDlgItemText(captureHWND, idc, tcText, nTestSize);
        if (_tcscmp(p->str, tcText)) {
          SetDlgItemText(captureHWND, idc, p->str);
        }
        break;
      }
      p++;
    }
  } else {
    wsprintf(msg, TEXT("%d"), static_cast<UINT32>(currentValue));
    GetDlgItemText(captureHWND, idc, tcText, nTestSize);
    if (_tcscmp(msg, tcText)) {
      SetDlgItemText(captureHWND, idc, static_cast<LPCWSTR>(msg));
    }
  }
}

BOOL hideUnsupportedCombo(ListEntryItems* comboItemList,
                          unsigned char isEnabled) {
  HWND itemHwnd = NULL;

  if ('\x1' != isEnabled) {
    itemHwnd = GetDlgItem(captureHWND, comboItemList->resourceID);
    EnableWindow(itemHwnd, FALSE);
    SendMessage(itemHwnd, CB_RESETCONTENT, 0, 0);
    return FALSE;
  }
  return TRUE;
}

int createList(ListEntryItems* comboItemList, unsigned short num,
               unsigned long sizeofType, PTP_VENDOR_DATA_OUT* pDataOut,
               unsigned int offset) {
  if ((comboItemList->supportList != NULL) ||
      (comboItemList->supportNum != num)) {
    delete[] comboItemList->supportList;
  }
  comboItemList->supportList = new ListItem*[comboItemList->size * num];

  ListItem* p = &comboItemList->itemList[0];
  UINT16 index = 0;
  for (UINT8 idx = 1; idx <= comboItemList->size; idx++) {
    for (int listNum = 0; listNum < num; listNum++) {
      if (sizeofType == 1) {
        if ((*static_cast<unsigned char*>(
                &pDataOut->VendorReadData[offset + listNum])) == p->value) {
          comboItemList->supportList[index] = p;
          index++;
          break;
        }
      } else if (sizeofType == 2) {
        if ((*reinterpret_cast<unsigned short*>(
                &pDataOut->VendorReadData[offset + (listNum * sizeofType)])) ==
            p->value) {
          comboItemList->supportList[index] = p;
          index++;
          break;
        }
      } else if (sizeofType == 4) {
        if ((*reinterpret_cast<unsigned long*>(
                &pDataOut->VendorReadData[offset + (listNum * sizeofType)])) ==
            p->value) {
          comboItemList->supportList[index] = p;
          index++;
          break;
        }
      }
    }
    p++;
  }

  return index;
}

void updateCombo(HWND itemHwnd, ListEntryItems* comboItemList,
                 unsigned long long currentValue, int index) {
  if (itemHwnd == NULL) {
    return;
  }

  int setCurselIndex = 0;

  if (SendMessage(itemHwnd, CB_GETCOUNT, 0, 0) > 0) {
    if ((comboItemList->previousValue == currentValue) &&
        (comboItemList->legacyList == *comboItemList->supportList) &&
        comboItemList->supportNum == index) {
      for (int num = 0; num < index; num++) {
        if (comboItemList->supportList[num]->value == currentValue) {
          TCHAR strtmp[128] = {NULL};
          SendMessage(itemHwnd, CB_GETLBTEXT,
                      SendMessage(itemHwnd, CB_GETCURSEL, 0L, 0L),
                      reinterpret_cast<LPARAM>(strtmp));
          if (_tcscmp(strtmp, comboItemList->supportList[num]->str) != 0) {
            setCurselIndex = num;
            comboItemList->previousValue = currentValue;
            SendMessage(itemHwnd, CB_SETCURSEL, setCurselIndex, -1);
          }
        }
      }
      return;
    }
  }
  comboItemList->supportNum = index;
  SendMessage(itemHwnd, CB_RESETCONTENT, 0, 0);

  setCurselIndex = 0;

  for (int createCount = 0; createCount < index; createCount++) {
    LPCTSTR str = NULL;
    str = comboItemList->supportList[createCount]->str;
    SendMessage(itemHwnd, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(str));
    SendMessage(itemHwnd, CB_SETITEMDATA, createCount,
                reinterpret_cast<LPARAM>(str));
    if (comboItemList->supportList[createCount]->value == currentValue) {
      setCurselIndex = createCount;
      comboItemList->previousValue = currentValue;
    }
  }
  comboItemList->legacyList = *comboItemList->supportList;
  EnableWindow(itemHwnd, TRUE);
  SendMessage(itemHwnd, CB_SETCURSEL, setCurselIndex, -1);

  return;
}

void DataManager::sendSelectValue(int comboItem, int idx) {
  HRESULT hr = 0;
  DWORD dwDataInSize = 0;

  for (int i = 0;
       i < sizeof(devicePropItemList) / sizeof(devicePropItemList[0]); i++) {
    if (devicePropItemList[i].resourceID == comboItem) {
      if (getIsEnableStatus(i) != '\x1') {
        return;
      }
      switch (devicePropItemList[i].dataType) {
        case PTP_DT_INT8:
        case PTP_DT_UINT8:
          dwDataInSize = sizeof(INT8);
          break;
        case PTP_DT_INT16:
        case PTP_DT_UINT16:
          dwDataInSize = sizeof(INT16);
          break;
        case PTP_DT_INT32:
        case PTP_DT_UINT32:
          dwDataInSize = sizeof(INT32);
          break;
        case PTP_DT_INT64:
        case PTP_DT_UINT64:
          dwDataInSize = sizeof(INT64);
          break;
        case PTP_DT_AINT8:
        case PTP_DT_AUINT8:
          dwDataInSize = sizeof(INT8);
          break;
        case PTP_DT_AINT16:
        case PTP_DT_AUINT16:
          dwDataInSize = sizeof(INT16);
          break;
        case PTP_DT_AINT32:
        case PTP_DT_AUINT32:
          dwDataInSize = sizeof(INT32);
          break;
        case PTP_DT_AINT64:
        case PTP_DT_AUINT64:
          dwDataInSize = sizeof(INT64);
          break;
        default:
          dwDataInSize = sizeof(INT8);
          break;
      }

      hr = ptp.SDIOSetExtDevicePropValue(
          devicePropItemList[i].propertyCode,
          devicePropItemList[i].supportList[idx]->value, dwDataInSize);
    }
  }
}

HRESULT DataManager::saveImage(BYTE* buffer, DWORD fileSize, BYTE* fileName,
                               int index) {
  HANDLE hFile = NULL;
  std::wstring wFileName;
  std::wstring slash = TEXT("\\");
  wFileName = reinterpret_cast<WCHAR*>(fileName);
  std::wstring selectedPath;
  TCHAR szBuf[256];
  GetWindowText(GetDlgItem(captureHWND, IDC_SAVEPATH_EDIT), szBuf,
                _countof(szBuf));

  if (_tcscmp(szBuf, __TEXT("")) == 0) {
    TCHAR exePath[MAX_PATH];
    if (::GetModuleFileName(NULL, exePath,
                            MAX_PATH))
    {
      TCHAR* ptmp = _tcsrchr(exePath, _T('\\'));
      if (ptmp != NULL) {
        ptmp = _tcsinc(ptmp);
        *ptmp = _T('\0');
        selectedPath = exePath + wFileName;
      } else {
        selectedPath = TEXT("C:\\") + wFileName;
      }
    } else {
      selectedPath = TEXT("C:\\") + wFileName;
    }
  } else {
    selectedPath = szBuf + slash + wFileName;
  }

  hFile = CreateFile(selectedPath.c_str(), GENERIC_WRITE, 0, NULL,
                     CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  if (hFile == INVALID_HANDLE_VALUE) {
    return E_FAIL;
  }
  DWORD dwByte = 0;
  BOOL ret = 0;
  ret = WriteFile(hFile, buffer, fileSize, &dwByte, NULL);
  CloseHandle(hFile);
  if (ret == FALSE) {
    ;
    return E_FAIL;
  }
  return S_OK;
}
