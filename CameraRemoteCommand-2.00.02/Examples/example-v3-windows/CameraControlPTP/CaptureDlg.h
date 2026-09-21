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
  void SetFocusAreaPositionTopLeft();
  void SetFocusAreaPositionTopLeft2();
  void SetFocusAreaPositionTopMid();
  void SetFocusAreaPositionTopRight();
  void SetFocusAreaPositionTop2Left();
  void SetFocusAreaPositionTop2Left2();
  void SetFocusAreaPositionTop2Mid();
  void SetFocusAreaPositionTop2Right2();
  void SetFocusAreaPositionTopRight2();
  void SetFocusAreaPositionBottomMid();
  void SetFocusAreaPositionBottomRight2();
  void SetFocusAreaPositionBottomRight();
  void SetFocusAreaPositionMidMid();
  void SetFocusAreaPositionMidRight2();
  void SetFocusAreaPositionMid2Right();
  void SetFocusAreaPositionTop2Right();
  void SetFocusAreaPositionMidLeft();
  void SetFocusAreaPositionMidLeft2();
  void SetFocusAreaPositionBottom2Right();
  void SetFocusAreaPositionBottomLeft();
  void SetFocusAreaPositionBottomLeft2();
  void SetFocusAreaPositionMidRight();
  void SetFocusAreaPositionBottom2Left();
  void SetFocusAreaPositionBottom2Left2();
  void SetFocusAreaPositionBottom2Mid();
  void SetFocusAreaPositionBottom2Right2();
  void SetFocusAreaPositionTopLeft(HWND hWnd);
  void SetFocusAreaPositionButtonClicked();
  void SetFocusAreaPositionButtonClicked(HWND hWnd);
  void s1ShutterButtonClicked(HWND hWnd);
  void s2ShutterButtonClicked();
  void AELockButtonClicked(HWND hWnd);
  void AFLockButtonClicked(HWND hWnd);
  void AWBLockButtonClicked(HWND hWnd);
  void NormalButtonClicked(HWND hWnd);
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
  void MovieRecButtonClicked();
  void DisplayImageFromLiveview(HWND hWnd);
  void getDeviceInfoButtonClicked();
  void setDisplayImageFromByteArray(BYTE* buffer);
  void setSaveFolderPath();
  void WhitebalanceabUpButtonClicked(HWND hWnd);
  void WhitebalanceabDownButtonClicked(HWND hWnd);
  void WhitebalancegmUpButtonClicked(HWND hWnd);
  void WhitebalancegmDownButtonClicked(HWND hWnd);
  void Near1ButtonClicked(HWND hWnd);
  void Near2ButtonClicked(HWND hWnd);
  void Near3ButtonClicked(HWND hWnd);
  void Far1ButtonClicked(HWND hWnd);
  void Far2ButtonClicked(HWND hWnd);
  void Far3ButtonClicked(HWND hWnd);
  void AF_SButtonClicked(HWND hWnd);
  void AF_AButtonClicked(HWND hWnd);
  void AF_CButtonClicked(HWND hWnd);
  void DMFButtonClicked(HWND hWnd);
  void MFButtonClicked(HWND hWnd);
  void ZoomwideButtonClicked(HWND hWnd);
  void ZoomstopButtonClicked(HWND hWnd);
  void ZoomteleButtonClicked(HWND hWnd);
  void ptpButtonClicked(HWND hWnd);
  void mtpButtonClicked(HWND hWnd);
  void jpegfilelist(HWND hWnd);
  void rawfilelist(HWND hWnd);
  void mp4filelist(HWND hWnd);
  void jpegfiletransfer(HWND hWnd);
  void rawfiletransfer(HWND hWnd);
  void mp4filetransfer(HWND hWnd);
  void mtpreadButtonClicked(HWND hWnd);
  void transferButtonClicked(HWND hWnd);
  void formatButtonClicked(HWND hWnd);

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
