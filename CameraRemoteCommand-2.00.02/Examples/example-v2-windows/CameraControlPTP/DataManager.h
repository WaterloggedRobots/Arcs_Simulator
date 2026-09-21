#pragma once

#include <tchar.h>
#include <windows.h>

#include "PTPDef.h"

class DataManager {
 public:
  static DataManager &getInstance();
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

  void setDeviceInfoData(PTP_VENDOR_DATA_OUT *pDataOut);
  PTP_VENDOR_DATA_OUT *getDeviceInfoData();

 protected:
  void setSupportedCommands(PTP_VENDOR_DATA_OUT *pDataOut);
  void setCameraData(PTP_VENDOR_DATA_OUT *pDataOut);

 private:
  BOOL getComboFocus(void);

  void setIsEnableStatus(unsigned int num, unsigned char isEnabled);

  void checkShootingFiles(unsigned short propertyCode,
                          unsigned long long currentValue,
                          PTP_VENDOR_DATA_OUT *pDataHolder,
                          unsigned int offset);

  void updataRangeDataValue(unsigned short propertyCode,
                            unsigned long long currentValue);

  void updataEnumDataValue(unsigned short propertyCode,
                           unsigned long long currentValue);

  void updateDisplayValue(int idc, unsigned long long currentValue,
                          int listNum);

  HRESULT saveImage(BYTE *buffer, DWORD fileSize, BYTE *fileName, int index);

  const UINT ShutterSpeedList = 0;
  const UINT FnumberList = 1;
  const UINT ISOItemList = 2;
  const UINT ExpList = 3;
  const UINT exposureModeList = 4;
  const UINT flashmodeList = 5;
  const UINT liveviewModeList = 6;
  const UINT focusareaList = 7;
  const UINT drohdrModeList = 8;
  const UINT ImageSizeList = 9;
  const UINT CompressionSettingList = 10;
  const UINT AspectRatioList = 11;
  const UINT SaveMediaList = 12;
  const UINT whitebalanceList = 13;
  const UINT DriveModeList = 14;
  const UINT meteringmodeList = 15;
  const UINT pictureeffectsList = 16;
  const UINT batterylevelList = 17;
  const UINT aelList = 18;
  const UINT aflList = 19;
  const UINT focusmodeList = 20;
  const UINT whitebalanceabList = 21;
  const UINT whitebalancegmList = 22;
  const UINT movierecordingstatusList = 23;
  const UINT nearfarList = 24;
  const UINT flashcompList = 25;
  const UINT viewList = 26;
  const UINT afstatusList = 27;
  const UINT FocusMagnifierPhaseList = 28;
  const UINT mfstatusList = 29;
};
