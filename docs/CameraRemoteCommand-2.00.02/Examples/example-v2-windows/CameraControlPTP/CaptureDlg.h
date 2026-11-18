#pragma once

#include <commctrl.h>
#include <tchar.h>
#include <windows.h>

class CaptureDlg {
 public:
  CaptureDlg();

  ~CaptureDlg();

  void createOnDialog(HWND hDlg);

  static BOOL CALLBACK CaptureDlgWndProc(HWND hDlg, UINT msg, WPARAM wp,
                                         LPARAM lp);

  void getDisplayImageFromLiveview();
  void displayOnDialog();

 private:
  BOOL OnInitializeButtonClicked(HWND hDlg);
  void DisconnectButtonClicked(HWND hWnd);
  void s1ShutterButtonClicked(HWND hWnd);
  void s2ShutterButtonClicked();
  void AELockButtonClicked(HWND hWnd);
  void AFLockButtonClicked(HWND hWnd);
  void AWBLockButtonClicked(HWND hWnd);
  void FocusToggleButtonClicked(HWND hWnd);
  void NormalButtonClicked(HWND hWnd);
  void FocusMagnifyRequestButtonClicked(HWND hWnd);
  void FocusMagnifyResetButtonClicked(HWND hWnd);
  void FocusMagnifyMoveUpButtonClicked(HWND hWnd);
  void FocusMagnifyMoveDownButtonClicked(HWND hWnd);
  void FocusMagnifyMoveLeftButtonClicked(HWND hWnd);
  void FocusMagnifyMoveRightButtonClicked(HWND hWnd);
  void ShutterSpeedUpButtonClicked(HWND hWnd);
  void ShutterSpeedDownButtonClicked(HWND hWnd);
  void ISOUpButtonClicked(HWND hWnd);
  void ISODownButtonClicked(HWND hWnd);
  void FNumberUpButtonClicked(HWND hWnd);
  void FNumberDownButtonClicked(HWND hWnd);
  void ExposureCompUpButtonClicked(HWND hWnd);
  void ExposureCompDownButtonClicked(HWND hWnd);
  void FlashCompUpButtonClicked(HWND hWnd);
  void FlashCompDownButtonClicked(HWND hWnd);
  void Near1ButtonClicked(HWND hWnd);
  void Near2ButtonClicked(HWND hWnd);
  void Near3ButtonClicked(HWND hWnd);
  void Far1ButtonClicked(HWND hWnd);
  void Far2ButtonClicked(HWND hWnd);
  void Far3ButtonClicked(HWND hWnd);
  void MovieRecButtonClicked();
  void DisplayImageFromLiveview(HWND hWnd);
  void getDeviceInfoButtonClicked();
  void setDisplayImageFromByteArray(BYTE* buffer);
  void setSaveFolderPath();
  void WhitebalanceabUpButtonClicked(HWND hWnd);
  void WhitebalanceabDownButtonClicked(HWND hWnd);
  void WhitebalancegmUpButtonClicked(HWND hWnd);
  void WhitebalancegmDownButtonClicked(HWND hWnd);
  void AF_SButtonClicked(HWND hWnd);
  void AF_AButtonClicked(HWND hWnd);
  void AF_CButtonClicked(HWND hWnd);
  void DMFButtonClicked(HWND hWnd);
  void MFButtonClicked(HWND hWnd);

  static const int SDIO_CONNECT_ID = 0x00000000;

  bool memoryAlloc(DWORD bufsize);
  void memoryDispose(void);

  static OPENFILENAME m_ofn;
  static TCHAR m_strFile[];
  static TCHAR m_strCustom[];

  static UINT8* m_buffer;
  static DWORD m_updateFileSize;

  void cCreateLPWSTRImpl(int idc, BYTE* value, UINT8 length);
  void updatemsg(UINT8 status);

  typedef enum {
    UPDATE_STATUS_SYSTEMSTANDBY = 0,
    UPDATE_STATUS_FILE_SENDING,
    UPDATE_STATUS_UPDATING,
    UPDATE_STATUS_UPDATE_FIN,
  } UPDATA_STATUS;
};
