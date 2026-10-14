#pragma once

#include <tchar.h>
#include <windows.h>

#include "PTPDef.h"

class DataManager {
 public:
  static DataManager& getInstance();

  DataManager();

  ~DataManager();

  void setCaptureHWND(HWND cHwnd);
  void setIsLiveviewValidFlag(BOOL isLiveviewValid);
  BOOL getIsLiveviewValidFlag();

  BOOL checkSupportedCommands(unsigned short propertyCode);

  void readCameraData(void);

  void setComboFocus(BOOL fucus);

  void sendSelectValue(int comboItem, int idx);

  unsigned char getIsEnableStatus(unsigned int listNum);

  void SetFocusAreaPositionButtonClicked();

  void setDeviceInfoData(PTP_VENDOR_DATA_OUT* pDataOut);
  void setStorageID(PTP_VENDOR_DATA_OUT* pDataOut);
  void setNumObjects(PTP_VENDOR_DATA_OUT* pDataOut);
  void setObjectHandles(PTP_VENDOR_DATA_OUT* pDataOut);
  PTP_VENDOR_DATA_OUT* getDeviceInfoData();
  PTP_VENDOR_DATA_OUT* getStorageID();
  PTP_VENDOR_DATA_OUT* getNumObjects();
  PTP_VENDOR_DATA_OUT* getObjectHandles();
  PTP_VENDOR_DATA_OUT* getstorageinfoData();
  HRESULT saveImage(BYTE* buffer, DWORD fileSize, BYTE* fileName, int index);

 protected:
  void setSupportedCommands(PTP_VENDOR_DATA_OUT* pDataOut);
  void setCameraData(PTP_VENDOR_DATA_OUT* pDataOut);

 private:
  BOOL getComboFocus(void);

  void setIsEnableStatus(unsigned int num, unsigned char isEnabled);

  void checkShootingFiles(unsigned short propertyCode,
                          unsigned long long currentValue,
                          PTP_VENDOR_DATA_OUT* pDataHolder,
                          unsigned int offset);

  void updataRangeDataValue(unsigned short propertyCode,
                            unsigned long long currentValue);

  void updataEnumDataValue(unsigned short propertyCode,
                           unsigned long long currentValue);

  void updateDisplayValue(int idc, unsigned long long currentValue,
                          int listNum);

  const UINT ShutterSpeedList = 0;
  const UINT FnumberList = 2;
  const UINT ISOItemList = 3;
  const UINT ExpList = 4;
  const UINT exposureModeList = 5;
  const UINT flashmodeList = 6;
  const UINT liveviewModeList = 7;
  const UINT focusareaList = 8;
  const UINT drohdrModeList = 9;
  const UINT ImageSizeList = 10;
  const UINT JpegQualityList = 11;
  const UINT imagefileformatList = 12;
  const UINT AspectRatioList = 13;
  const UINT whitebalanceList = 14;
  const UINT DriveModeList = 15;
  const UINT meteringmodeList = 16;
  const UINT pictureeffectsList = 17;
  const UINT aelList = 18;
  const UINT batterylevelList = 19;
  const UINT focusmodeList = 20;
  const UINT whitebalanceabList = 21;
  const UINT whitebalancegmList = 22;
  const UINT flashcompList = 23;
  const UINT viewList = 24;
  const UINT afstatusList = 25;
  const UINT pictureprofileList = 27;
  const UINT creativestyleList = 28;
  const UINT movieformatList = 29;
  const UINT moviequalityList = 30;
  const UINT savemediaList = 31;
  const UINT zoomsettingList = 32;
  const UINT zoomscaleList = 33;
};
