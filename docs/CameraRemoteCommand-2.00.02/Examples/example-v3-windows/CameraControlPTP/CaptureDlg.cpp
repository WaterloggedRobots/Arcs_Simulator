#include "CaptureDlg.h"

#include <commctrl.h>
#include <shlobj.h>
#include <shlwapi.h>
#include <stdio.h>
#include <tchar.h>
#include <windows.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "PTPControl.h"
#include "UserMessage.h"
#include "resource.h"

#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "winmm.lib")

HWND hCaptureDlg = NULL;
static CaptureDlg *captureDlg = NULL;
static PTPControl &ptp = PTPControl::getInstance();
static DataManager &dMgr = DataManager::getInstance();
HWND cpDlg = NULL;
BOOL isS1Started = FALSE;
BOOL isAELockStarted = FALSE;
BOOL isAFLockStarted = FALSE;
BOOL isAWBLockStarted = FALSE;
BOOL isFocusToggletarted = FALSE;
BOOL isPositionKeystarted = FALSE;
BOOL isRedEyeReductionstarted = FALSE;
BOOL isNormalstarted = FALSE;
BOOL isWirelessFlashstarted = FALSE;
BOOL isFocusMagnifystarted = FALSE;
BOOL isZoomAbosoluteStarted = FALSE;
BOOL isMediaFormatStarted = FALSE;
BOOL Teleclicked = FALSE;
BOOL Wideclicked = FALSE;
BOOL MovieRecclicked = FALSE;
BOOL LiveViewclicked = FALSE;
BOOL CameraConnected = FALSE;
const int UP = 0x0001;
const int DOWN = 0x0002;
OPENFILENAME CaptureDlg::m_ofn = {};
TCHAR CaptureDlg::m_strFile[MAX_PATH] = {};
TCHAR CaptureDlg::m_strCustom[MAX_PATH] = TEXT("Before files\0*.*\0\0");
UINT8 *CaptureDlg::m_buffer = NULL;
DWORD CaptureDlg::m_updateFileSize = NULL;
UINT8 SystemStandbyMode = 0x01;
TCHAR update_msg[UCHAR_MAX] = {};
UINT8 WhiteBalanceReset = 0x01;
static PTP_VENDOR_DATA_OUT *pDataHolder = NULL;
HWND Shutter_Speed = NULL;
UINT PreExposureProgram = 0x00000001;
int mtp_button_clicked = NULL;
int ptp_mode = NULL;

#define PB_MIN 0
#define PB_MAX 100
#define PB_STEP 10
int iProg;

CaptureDlg::CaptureDlg() {
  memset(&m_ofn, 0, sizeof(m_ofn));

  m_ofn.lStructSize = sizeof(OPENFILENAME);
  m_ofn.hwndOwner = NULL;
  m_ofn.lpstrFilter = TEXT("All files {*.*}\0*.*\0\0");
  m_ofn.lpstrCustomFilter = m_strCustom;
  m_ofn.nMaxCustFilter = MAX_PATH;
  m_ofn.nFilterIndex = 0;
  m_ofn.lpstrFile = m_strFile;
  m_ofn.nMaxFile = MAX_PATH;
  m_ofn.Flags = OFN_FILEMUSTEXIST;
}

CaptureDlg::~CaptureDlg() {}

void CaptureDlg::createOnDialog(HWND hDlg) {
  HINSTANCE hInst = GetModuleHandle(NULL);

  if (hCaptureDlg != NULL) {
    ShowWindow(hCaptureDlg, SW_HIDE);
  } else {
    hCaptureDlg = CreateDialog(hInst, TEXT("ONDIALOG_CAPTURE"), hDlg,
                               (DLGPROC)CaptureDlgWndProc);
  }
}

void CaptureDlg::displayOnDialog() {
  if (hCaptureDlg != NULL) {
    ShowWindow(hCaptureDlg, SW_SHOW);
  }
}

BOOL CALLBACK CaptureDlg::CaptureDlgWndProc(HWND hDlg, UINT msg, WPARAM wp,
                                            LPARAM lp) {
  HWND hCombo = NULL;
  cpDlg = hDlg;

  HWND hBar = NULL;
  int PosColorTemp = 5500;
  TCHAR str[10];

  UINT16 colortemvalue = 0;

  switch (msg) {
    case WM_INITDIALOG:

      dMgr.setCaptureHWND(hDlg);
      InitCommonControls();

      hBar = GetDlgItem(hDlg, IDC_SLIDER_COLOR_TEMP);

      SendMessage(hBar, TBM_SETRANGE, TRUE,
                  MAKELPARAM(2500, 9900));
      SendMessage(hBar, TBM_SETRANGEMAX, TRUE, 9900);
      SendMessage(hBar, TBM_SETRANGEMIN, TRUE, 2500);
      SendMessage(hBar, TBM_SETTICFREQ, 100, 0);
      SendMessage(hBar, TBM_SETPOS, TRUE, PosColorTemp);
      _stprintf_s(str, 10, TEXT("%d K"), PosColorTemp);
      SetWindowText(GetDlgItem(hDlg, IDC_COLOR_TEMP_VALUE),
                    static_cast<LPCTSTR>(str));
      EnableWindow(GetDlgItem(hDlg, IDC_SD_READ), FALSE);
      EnableWindow(GetDlgItem(hDlg, IDC_TRANSFER), FALSE);

      break;

    case WM_HSCROLL:

      if (GetDlgItem(hDlg, IDC_SLIDER_COLOR_TEMP) ==
          reinterpret_cast<HWND>(lp)) {
        hBar = GetDlgItem(hDlg, IDC_SLIDER_COLOR_TEMP);
        int PosColorTemp =
            SendMessage(hBar, TBM_GETPOS, NULL, NULL);
        int value = PosColorTemp;
        int a[4];

        a[0] = (value % 10);
        value = value / 10;
        a[1] = (value % 10);
        value /= 10;
        a[2] = (value % 10);
        value /= 10;
        a[3] = (value % 10);
        value /= 10;

        if (a[1] < 5) {
          PosColorTemp = a[3] * 1000 + a[2] * 100;
          SendMessage(hBar, TBM_SETPOS, TRUE, PosColorTemp);
        } else {
          if (a[3] == 9) {
            PosColorTemp = a[3] * 1000 + a[2] * 100;
          } else {
            PosColorTemp = a[3] * 1000 + a[2] * 100 + 100;
          }
          SendMessage(hBar, TBM_SETPOS, TRUE, PosColorTemp);
        }

        colortemvalue = PosColorTemp;

        _stprintf_s(str, 10, TEXT("%d K"), PosColorTemp);
        SetWindowText(GetDlgItem(hDlg, IDC_COLOR_TEMP_VALUE),
                      static_cast<LPCTSTR>(str));
        ptp.SDIOSetExtDevicePropValue(DPC_COLOR_TEMP,
                                      colortemvalue);
      }

      break;

    case WM_COMMAND:
      switch (LOWORD(wp)) {
        case IDC_INITIALIZE_BUTTON:

          if (captureDlg->OnInitializeButtonClicked(hDlg) == TRUE) {
            SendMessage(GetParent(hDlg), WM_CAMERA_CONNECTION, 0, 0);
            captureDlg->getDeviceInfoButtonClicked();
            captureDlg->updatemsg(UPDATE_STATUS_UPDATE_FIN);

            hBar = GetDlgItem(hDlg, IDC_SLIDER_COLOR_TEMP);
          }

          break;

        case IDC_DISCONNECT_BUTTON:
          if (CameraConnected) {
            CheckDlgButton(cpDlg, IDC_DISPLAY_LIVEVIEW_CHECK, BST_UNCHECKED);
            Sleep(200);
            InvalidateRect(hDlg, NULL, TRUE);
            UpdateWindow(hDlg);
            captureDlg->DisconnectButtonClicked(hDlg);
            break;
          }
        case IDC_LIVE_VIEW_BUTTON:
          if (CameraConnected) {
            captureDlg->DisplayImageFromLiveview(hDlg);
            break;
          }
        case IDC_MOVIE_REC_BUTTON:
          if (CameraConnected) {
            captureDlg->MovieRecButtonClicked();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_BUTTON:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionButtonClicked();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_TOPLEFT:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionTopLeft();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_TOP2LEFT:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionTop2Left();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_TOPLEFT2:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionTopLeft2();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_TOP2LEFT2:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionTop2Left2();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_TOPMID:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionTopMid();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_TOP2MID:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionTop2Mid();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_TOPRIGHT:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionTopRight();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_TOP2RIGHT:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionTop2Right();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_TOPRIGHT2:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionTopRight2();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_TOP2RIGHT2:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionTop2Right2();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_MIDMID:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionMidMid();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_MIDLEFT:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionMidLeft();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_MIDLEFT2:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionMidLeft2();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_MIDRIGHT:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionMidRight();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_MIDRIGHT2:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionMidRight2();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_BOTTOMLEFT:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionBottomLeft();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_BOTTOM2LEFT:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionBottom2Left();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_BOTTOMLEFT2:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionBottomLeft2();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_BOTTOM2LEFT2:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionBottom2Left2();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_BOTTOMMID:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionBottomMid();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_BOTTOM2MID:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionBottom2Mid();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_BOTTOMRIGHT:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionBottomRight();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_BOTTOMRIGHT2:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionBottomRight2();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_BOTTOM2RIGHT:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionBottom2Right();
            break;
          }
        case IDC_FOCUS_AREA_POSITION_BOTTOM2RIGHT2:
          if (CameraConnected) {
            captureDlg->SetFocusAreaPositionBottom2Right2();
            break;
          }
        case IDC_S1_SHUTTER_BUTTON:
          if (CameraConnected) {
            captureDlg->s1ShutterButtonClicked(hDlg);
            break;
          }
        case IDC_S2_SHUTTER_BUTTON:
          if (CameraConnected) {
            captureDlg->s2ShutterButtonClicked();
            break;
          }
        case IDC_AE_LOCK:
          if (CameraConnected) {
            captureDlg->AELockButtonClicked(hDlg);
          }
          break;
        case IDC_AF_LOCK:
          if (CameraConnected) {
            captureDlg->AFLockButtonClicked(hDlg);
          }
          break;
        case IDC_AWB_LOCK:
          if (CameraConnected) {
            captureDlg->AWBLockButtonClicked(hDlg);
          }
          break;

        case IDC_COMBO_NORMAL:
          if (CameraConnected) {
            captureDlg->NormalButtonClicked(hDlg);
          }
          break;

        case IDC_SHUTTER_SPEED_UP:
          if (CameraConnected) {
            captureDlg->ShutterSpeedUpButtonClicked(hDlg);
          }
          break;
        case IDC_SHUTTER_SPEED_DOWN:
          if (CameraConnected) {
            captureDlg->ShutterSpeedDownButtonClicked(hDlg);
          }
          break;
        case IDC_ISO_UP:
          if (CameraConnected) {
            captureDlg->ISOUpButtonClicked(hDlg);
          }
          break;
        case IDC_ISO_DOWN:
          if (CameraConnected) {
            captureDlg->ISODownButtonClicked(hDlg);
          }
          break;
        case IDC_FNUMBER_UP:
          if (CameraConnected) {
            captureDlg->FNumberUpButtonClicked(hDlg);
          }
          break;
        case IDC_FNUMBER_DOWN:
          if (CameraConnected) {
            captureDlg->FNumberDownButtonClicked(hDlg);
          }
          break;
        case IDC_EXPOSURE_COMP_UP:
          if (CameraConnected) {
            captureDlg->ExposureCompUpButtonClicked(hDlg);
          }
          break;
        case IDC_EXPOSURE_COMP_DOWN:
          if (CameraConnected) {
            captureDlg->ExposureCompDownButtonClicked(hDlg);
          }
          break;
        case IDC_FLASH_COMP_UP:
          if (CameraConnected) {
            captureDlg->FlashCompUpButtonClicked(hDlg);
          }
          break;
        case IDC_FLASH_COMP_DOWN:
          if (CameraConnected) {
            captureDlg->FlashCompDownButtonClicked(hDlg);
          }
          break;
        case IDC_SAVEFOLDER_BUTTON:
          if (CameraConnected) {
            captureDlg->setSaveFolderPath();
            break;
          }
        case IDC_WHITEBALANCE_AB_UP:
          if (CameraConnected) {
            captureDlg->WhitebalanceabUpButtonClicked(hDlg);
          }
          break;
        case IDC_WHITEBALANCE_AB_DOWN:
          if (CameraConnected) {
            captureDlg->WhitebalanceabDownButtonClicked(hDlg);
          }
          break;
        case IDC_WHITEBALANCE_GM_UP:
          if (CameraConnected) {
            captureDlg->WhitebalancegmUpButtonClicked(hDlg);
          }
          break;
        case IDC_WHITEBALANCE_GM_DOWN:
          if (CameraConnected) {
            captureDlg->WhitebalancegmDownButtonClicked(hDlg);
          }
          break;
        case IDC_NEAR1:
          if (CameraConnected) {
            captureDlg->Near1ButtonClicked(hDlg);
          }
          break;

        case IDC_NEAR2:
          if (CameraConnected) {
            captureDlg->Near2ButtonClicked(hDlg);
          }
          break;

        case IDC_NEAR3:
          if (CameraConnected) {
            captureDlg->Near3ButtonClicked(hDlg);
          }
          break;

        case IDC_FAR1:
          if (CameraConnected) {
            captureDlg->Far1ButtonClicked(hDlg);
          }
          break;
        case IDC_FAR2:
          if (CameraConnected) {
            captureDlg->Far2ButtonClicked(hDlg);
          }
          break;
        case IDC_FAR3:
          if (CameraConnected) {
            captureDlg->Far2ButtonClicked(hDlg);
          }
          break;
        case IDC_AF_S:
          if (CameraConnected) {
            captureDlg->AF_SButtonClicked(hDlg);
          }
          break;
        case IDC_AF_A:
          if (CameraConnected) {
            captureDlg->AF_AButtonClicked(hDlg);
          }
          break;
        case IDC_AF_C:
          if (CameraConnected) {
            captureDlg->AF_CButtonClicked(hDlg);
          }
          break;
        case IDC_DMF:
          if (CameraConnected) {
            captureDlg->DMFButtonClicked(hDlg);
          }
          break;
        case IDC_MF:
          if (CameraConnected) {
            captureDlg->MFButtonClicked(hDlg);
          }
          break;
        case IDC_ZOOM_WIDE:
          if (CameraConnected) {
            captureDlg->ZoomwideButtonClicked(hDlg);
          }
          break;
        case IDC_ZOOM_STOP:
          if (CameraConnected) {
            captureDlg->ZoomstopButtonClicked(hDlg);
          }
          break;
        case IDC_ZOOM_TELE:
          if (CameraConnected) {
            captureDlg->ZoomteleButtonClicked(hDlg);
          }
          break;
        case IDC_PTP:
          if (CameraConnected) {
            SendMessage(hDlg, WM_MTP_DISCONNECTION, 0, 0);
            captureDlg->ptpButtonClicked(hDlg);
          }
          break;
        case IDC_MTP:
          if (CameraConnected) {
            captureDlg->mtpButtonClicked(hDlg);
          }
          break;
        case IDC_TRANSFER:
          if (CameraConnected) {
            captureDlg->transferButtonClicked(hDlg);
          }
          break;
        case IDC_SD_READ:
          if (CameraConnected) {
            captureDlg->mtpreadButtonClicked(hDlg);
          }
          break;
        case IDC_MEDIA_FORMAT:
          if (CameraConnected) {
            captureDlg->formatButtonClicked(hDlg);
          }
          break;
        default:
          if (HIWORD(wp) == CBN_SELENDOK) {
            hCombo = GetDlgItem(hDlg, LOWORD(wp));
            int idx = SendMessage(hCombo, CB_GETCURSEL, 0, 0);
            dMgr.sendSelectValue(LOWORD(wp), idx);
            dMgr.setComboFocus(FALSE);

            if (GetDlgItem(hDlg, IDC_COMBO_ASPECT_RATIO) ==
                reinterpret_cast<HWND>(lp)) {
              Sleep(300);
              InvalidateRect(hDlg, NULL, TRUE);
              UpdateWindow(hDlg);
            }

          } else if (HIWORD(wp) == CBN_SETFOCUS || HIWORD(wp) == CBN_DROPDOWN) {
            dMgr.setComboFocus(TRUE);
          } else {
            dMgr.setComboFocus(FALSE);
          }
          break;
      }
    default:
      break;
    case WM_CLOSE:
      return TRUE;
  }
  return FALSE;
}

BOOL CaptureDlg::OnInitializeButtonClicked(HWND hDlg) {
  if (!ptp.prepareConnection()) {
    MessageBox(NULL, __T("Camera Not Detected"), __T("Information"), MB_OK);
    return FALSE;
  }

  HRESULT hr = ptp.SDIOConnect(1, SDIO_CONNECT_ID, SDIO_CONNECT_ID);

  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOConnect(1, 0x00000000, 0x00000000)"),
               __T("Error"), MB_OK);
    return FALSE;
  }
  hr = ptp.SDIOConnect(2, SDIO_CONNECT_ID, SDIO_CONNECT_ID);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOConnect(2, 0x00000000, 0x00000000)"),
               __T("Error"), MB_OK);
    return FALSE;
  }
  hr = ptp.SDIOGetExtDeviceInfo(SDI_Extension_Version);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOGetExtDeviceInfo(SDI_Extension_Version)"),
               __T("Error"), MB_OK);
    return FALSE;
  }
  hr = ptp.SDIOConnect(3, SDIO_CONNECT_ID, SDIO_CONNECT_ID);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOConnect(3, 0x00000000, 0x00000000)"),
               __T("Error"), MB_OK);
    return FALSE;
  }
  hr = S_OK;
  CameraConnected = TRUE;
  Sleep(200);
  UINT8 HOSTPC = 0x01;
  hr = ptp.SDIOSetExtDevicePropValue(DPC_POSITION_KEY, HOSTPC, 0x01);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: HOSTPC KEY"), __T("Error"), MB_OK);
    return FALSE;
  }
  Sleep(200);
  return TRUE;
}

void CaptureDlg::DisconnectButtonClicked(HWND hDlg) {
  int result =
      MessageBox(NULL, __T("Do you really want to terminate the application?"),
                 __T("CameraControlPTP ver1.00"), MB_YESNO);
  LiveViewclicked = FALSE;
  if (result == IDYES) {
    ptp.SDIOControlDevice(DPC_MOVIE_REC, UP);
    ptp.SDIOControlDevice(DPC_S1_BUTTON, UP);
    ptp.CloseSession();
  }
}

void CaptureDlg::SetFocusAreaPositionTopLeft() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x00600055);
}

void CaptureDlg::SetFocusAreaPositionTopLeft2() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x01000055);
}

void CaptureDlg::SetFocusAreaPositionTopMid() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x01500055);
}

void CaptureDlg::SetFocusAreaPositionTopRight2() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x01990055);
}

void CaptureDlg::SetFocusAreaPositionTopRight() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x02350055);
}

void CaptureDlg::SetFocusAreaPositionTop2Left() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x00600095);
}

void CaptureDlg::SetFocusAreaPositionTop2Left2() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x01000095);
}

void CaptureDlg::SetFocusAreaPositionTop2Mid() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x01500095);
}

void CaptureDlg::SetFocusAreaPositionTop2Right2() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x01990095);
}

void CaptureDlg::SetFocusAreaPositionTop2Right() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x02350095);
}

void CaptureDlg::SetFocusAreaPositionMidLeft() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x00600100);
}

void CaptureDlg::SetFocusAreaPositionMidLeft2() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x01000100);
}

void CaptureDlg::SetFocusAreaPositionMidMid() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x01500100);
}

void CaptureDlg::SetFocusAreaPositionMidRight2() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x01990100);
}

void CaptureDlg::SetFocusAreaPositionMidRight() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x02350100);
}

void CaptureDlg::SetFocusAreaPositionBottom2Left() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x00600140);
}

void CaptureDlg::SetFocusAreaPositionBottom2Left2() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x01000140);
}

void CaptureDlg::SetFocusAreaPositionBottom2Mid() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x01500140);
}

void CaptureDlg::SetFocusAreaPositionBottom2Right2() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x01990140);
}

void CaptureDlg::SetFocusAreaPositionBottom2Right() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x02350140);
}

void CaptureDlg::SetFocusAreaPositionBottomLeft() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x00600190);
}

void CaptureDlg::SetFocusAreaPositionBottomLeft2() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x01000190);
}

void CaptureDlg::SetFocusAreaPositionBottomMid() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x01500190);
}

void CaptureDlg::SetFocusAreaPositionBottomRight2() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x01990190);
}

void CaptureDlg::SetFocusAreaPositionBottomRight() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x02350190);
}

void CaptureDlg::SetFocusAreaPositionButtonClicked() {
  HRESULT hr = 0;
  static HWND captureHWND;
  static HWND hDlg;

  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x00600055);
  Sleep(1000);

  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x00600100);
  Sleep(1000);

  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x00600190);
  Sleep(1000);

  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x01500055);
  Sleep(1000);

  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x02350055);
  Sleep(1000);

  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x01500190);
  Sleep(1000);

  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x02350190);
  Sleep(1000);

  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x01500100);
  Sleep(1000);

  hr = ptp.SDIOControlDevice(DPC_FOCUS_AREA_X_Y, 0x02350100);
  Sleep(1000);

  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_FOCUS_AREA_X_Y, )"),
               __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::MovieRecButtonClicked() {
  HRESULT hr = 0;
  if (!MovieRecclicked) {
    hr = ptp.SDIOControlDevice(DPC_MOVIE_REC, DOWN);
    if (hr != S_OK) {
      MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_MOVIE_REC, DOWN)"),
                 __T("Error"), MB_OK);
      return;
    }
    MovieRecclicked = TRUE;
  } else {
    hr = ptp.SDIOControlDevice(DPC_MOVIE_REC, UP);
    if (hr != S_OK) {
      MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_MOVIE_REC, UP)"),
                 __T("Error"), MB_OK);
      return;
    }
    MovieRecclicked = FALSE;
  }
}

void CaptureDlg::s1ShutterButtonClicked(HWND hWnd) {
  HRESULT hr = 0;

  if (!isS1Started) {
    hr = ptp.SDIOControlDevice(DPC_S1_BUTTON, DOWN);
    if (hr != S_OK) {
      MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_S1_BUTTON, DOWN)"),
                 __T("Error"), MB_OK);
      CheckDlgButton(hWnd, IDC_S1_SHUTTER_BUTTON, 0);
      return;
    }
    isS1Started = TRUE;
  } else {
    hr = ptp.SDIOControlDevice(DPC_S1_BUTTON, UP);
    if (hr != S_OK) {
      MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_S1_BUTTON, UP)"),
                 __T("Error"), MB_OK);
      return;
    }
    CheckDlgButton(hWnd, IDC_S1_SHUTTER_BUTTON, 0);
    isS1Started = FALSE;
  }
}

void CaptureDlg::s2ShutterButtonClicked() {
  HRESULT hr = 0;
  hr = ptp.SDIOControlDevice(DPC_S2_BUTTON, DOWN);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_S2_BUTTON, DOWN)"),
               __T("Error"), MB_OK);
    return;
  }
  Sleep(200);
  hr = ptp.SDIOControlDevice(DPC_S2_BUTTON, UP);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_S2_BUTTON, UP)"),
               __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::AELockButtonClicked(HWND hWnd) {
  HRESULT hr = 0;

  if (!isAELockStarted) {
    hr = ptp.SDIOControlDevice(DPC_AE_LOCK, DOWN);
    if (hr != S_OK) {
      MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_AE_LOCK, DOWN)"),
                 __T("Error"), MB_OK);
      CheckDlgButton(hWnd, IDC_AE_LOCK, 0);
      return;
    }
    isAELockStarted = TRUE;
  } else {
    hr = ptp.SDIOControlDevice(DPC_AE_LOCK, UP);
    if (hr != S_OK) {
      MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_AE_LOCK, UP)"),
                 __T("Error"), MB_OK);
      return;
    }
    CheckDlgButton(hWnd, IDC_AE_LOCK, 0);
    isAELockStarted = FALSE;
  }
}

void CaptureDlg::AFLockButtonClicked(HWND hWnd) {
  HRESULT hr = 0;

  if (!isAFLockStarted) {
    hr = ptp.SDIOControlDevice(DPC_AF_LOCK, DOWN);
    if (hr != S_OK) {
      MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_AF_LOCK, DOWN)"),
                 __T("Error"), MB_OK);
      CheckDlgButton(hWnd, IDC_AF_LOCK, 0);
      return;
    }
    isAFLockStarted = TRUE;
  } else {
    hr = ptp.SDIOControlDevice(DPC_AF_LOCK, UP);
    if (hr != S_OK) {
      MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_AF_LOCK, UP)"),
                 __T("Error"), MB_OK);
      return;
    }
    CheckDlgButton(hWnd, IDC_AF_LOCK, 0);
    isAFLockStarted = FALSE;
  }
}

void CaptureDlg::AWBLockButtonClicked(HWND hWnd) {
  HRESULT hr = 0;

  if (!isAWBLockStarted) {
    hr = ptp.SDIOControlDevice(DPC_AWB_LOCK, DOWN);
    if (hr != S_OK) {
      MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_AWB_LOCK, DOWN)"),
                 __T("Error"), MB_OK);
      CheckDlgButton(hWnd, IDC_AWB_LOCK, 0);
      return;
    }
    isAWBLockStarted = TRUE;
  } else {
    hr = ptp.SDIOControlDevice(DPC_AWB_LOCK, UP);
    if (hr != S_OK) {
      MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_AWB_LOCK, UP)"),
                 __T("Error"), MB_OK);
      return;
    }
    CheckDlgButton(hWnd, IDC_AWB_LOCK, 0);
    isAWBLockStarted = FALSE;
  }
}

void CaptureDlg::NormalButtonClicked(HWND hWnd) {
  HRESULT hr = 0;

  if (!isNormalstarted) {
    hr = ptp.SDIOControlDevice(DPC_NORMAL, DOWN);
    if (hr != S_OK) {
      MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_NORMAL, DOWN)"),
                 __T("Error"), MB_OK);
      CheckDlgButton(hWnd, IDC_COMBO_NORMAL, 0);
      return;
    }
    isNormalstarted = TRUE;
  } else {
    hr = ptp.SDIOControlDevice(DPC_NORMAL, UP);
    if (hr != S_OK) {
      MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_NORMAL, UP)"),
                 __T("Error"), MB_OK);
      return;
    }
    CheckDlgButton(hWnd, IDC_COMBO_NORMAL, 0);
    isNormalstarted = FALSE;
  }
}

void CaptureDlg::ShutterSpeedUpButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT32 UP = 0x01;

  hr = ptp.SDIOControlDevice(DPC_SHUTTER_SPEED, UP);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_SHUTTER_SPEED_UP, UP)"),
               __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::ShutterSpeedDownButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT32 DOWN = 0xFF;

  hr = ptp.SDIOControlDevice(DPC_SHUTTER_SPEED, DOWN);
  if (hr != S_OK) {
    MessageBox(NULL,
               __T("Error: SDIOControlDevice(DPC_SHUTTER_SPEED_DOWN, DOWN)"),
               __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::ISOUpButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT32 UP = 0x01;

  hr = ptp.SDIOControlDevice(DPC_ISO, UP);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_ISO, UP)"), __T("Error"),
               MB_OK);
    return;
  }
}

void CaptureDlg::ISODownButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT32 DOWN = 0xFF;

  hr = ptp.SDIOControlDevice(DPC_ISO, DOWN);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_ISO_DOWN, DOWN)"),
               __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::FNumberUpButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT32 UP = 0x01;

  hr = ptp.SDIOControlDevice(DPC_FNUMBER, UP);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_FNUMBER_UP, UP)"),
               __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::FNumberDownButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT32 DOWN = 0xFF;

  hr = ptp.SDIOControlDevice(DPC_FNUMBER, DOWN);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_FNUMBER_DOWN, DOWN)"),
               __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::ExposureCompUpButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT32 UP = 0x01;

  hr = ptp.SDIOControlDevice(DPC_EXPOSURE_COMPENSATION, UP);
  if (hr != S_OK) {
    MessageBox(
        NULL, __T("Error: SDIOControlDevice(DPC_EXPOSURE_COMPENSATION_UP, UP)"),
        __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::ExposureCompDownButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT32 DOWN = 0xF1;

  hr = ptp.SDIOControlDevice(DPC_EXPOSURE_COMPENSATION, DOWN);
  if (hr != S_OK) {
    MessageBox(
        NULL,
        __T("Error: SDIOControlDevice(DPC_EXPOSURE_COMPENSATION_DOWN, DOWN)"),
        __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::FlashCompUpButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT32 UP = 0x01;

  hr = ptp.SDIOControlDevice(DPC_FLASH_COMP, UP);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_FLASH_COMP, UP)"),
               __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::FlashCompDownButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT32 DOWN = 0xF1;

  hr = ptp.SDIOControlDevice(DPC_FLASH_COMP, DOWN);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_FLASH_COMP, DOWN)"),
               __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::WhitebalanceabUpButtonClicked(HWND hWnd) {
  HRESULT hr = 0;

  WhitebalanceAB += 2;

  hr = ptp.SDIOSetExtDevicePropValue(DPC_WHITEBALANCE_AB, WhitebalanceAB, 0x02);
  if (hr != S_OK) {
    MessageBox(NULL,
               __T("Error: SDIOSetExtDevicePropValue(DPC_WHITEBALANCE_AB, "
                   "WhitebalanceAB, 0x02)"),
               __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::WhitebalanceabDownButtonClicked(HWND hWnd) {
  HRESULT hr = 0;

  WhitebalanceAB -= 2;

  hr = ptp.SDIOSetExtDevicePropValue(DPC_WHITEBALANCE_AB, WhitebalanceAB, 0x02);
  if (hr != S_OK) {
    MessageBox(NULL,
               __T("Error: SDIOSetExtDevicePropValue(DPC_WHITEBALANCE_AB, "
                   "WhitebalanceAB, 0x02)"),
               __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::WhitebalancegmUpButtonClicked(HWND hWnd) {
  HRESULT hr = 0;

  WhitebalanceGM += 2;

  hr = ptp.SDIOSetExtDevicePropValue(DPC_WHITEBALANCE_GM, WhitebalanceGM, 0x02);
  if (hr != S_OK) {
    MessageBox(NULL,
               __T("Error: SDIOSetExtDevicePropValue(DPC_WHITEBALANCE_GM, "
                   "WhitebalanceGM, 0x02)"),
               __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::WhitebalancegmDownButtonClicked(HWND hWnd) {
  HRESULT hr = 0;

  WhitebalanceGM -= 2;

  hr = ptp.SDIOSetExtDevicePropValue(DPC_WHITEBALANCE_GM, WhitebalanceGM, 0x02);
  if (hr != S_OK) {
    MessageBox(NULL,
               __T("Error: SDIOSetExtDevicePropValue(DPC_WHITEBALANCE_GM, "
                   "WhitebalanceGM, 0x02)"),
               __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::Near1ButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT16 hexstring = 0xFFFF;

  hr = ptp.SDIOControlDevice(DPC_NEAR_FAR, hexstring);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_NEAR_FAR, 0xFFFF)"),
               __T("Error"), MB_OK);
    return;
  }
  Sleep(200);
}

void CaptureDlg::Near2ButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT16 hexstring = 0xFFFD;

  hr = ptp.SDIOControlDevice(DPC_NEAR_FAR, hexstring);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_NEAR_FAR, 0xFDFF)"),
               __T("Error"), MB_OK);
    return;
  }
  Sleep(200);
}

void CaptureDlg::Near3ButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT16 hexstring = 0xFFF9;

  hr = ptp.SDIOControlDevice(DPC_NEAR_FAR, hexstring);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_NEAR_FAR, 0xF9FF)"),
               __T("Error"), MB_OK);
    return;
  }
  Sleep(200);
}

void CaptureDlg::Far1ButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT16 hexstring = 0x0001;

  hr = ptp.SDIOControlDevice(DPC_NEAR_FAR, hexstring);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_NEAR_FAR, 0x0001)"),
               __T("Error"), MB_OK);
    return;
  }
  Sleep(200);
}

void CaptureDlg::Far2ButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT16 hexstring = 0x0007;

  hr = ptp.SDIOControlDevice(DPC_NEAR_FAR, hexstring);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_NEAR_FAR, 0x0003)"),
               __T("Error"), MB_OK);
    return;
  }
  Sleep(200);
}

void CaptureDlg::Far3ButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT16 hexstring = 0x0007;

  hr = ptp.SDIOControlDevice(DPC_NEAR_FAR, hexstring);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_NEAR_FAR, 0x0007)"),
               __T("Error"), MB_OK);
    return;
  }
  Sleep(200);
}

void CaptureDlg::AF_SButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT16 AF_S = 0x0002;

  hr = ptp.SDIOSetExtDevicePropValue(DPC_FOCUS_MODE, AF_S, 0x04);
  if (hr != S_OK) {
    MessageBox(
        NULL,
        __T("Error: SDIOSetExtDevicePropValue(DPC_FOCUS_MODE, 0x0002, 0x04)"),
        __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::AF_AButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT16 AF_A = 0x8005;

  hr = ptp.SDIOSetExtDevicePropValue(DPC_FOCUS_MODE, AF_A, 0x04);
  if (hr != S_OK) {
    MessageBox(
        NULL,
        __T("Error: SDIOSetExtDevicePropValue(DPC_FOCUS_MODE, 0x8005, 0x04)"),
        __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::AF_CButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT16 AF_C = 0x0004;

  hr = ptp.SDIOSetExtDevicePropValue(DPC_FOCUS_MODE, AF_C, 0x04);
  if (hr != S_OK) {
    MessageBox(
        NULL,
        __T("Error: SDIOSetExtDevicePropValue(DPC_FOCUS_MODE, 0x0004, 0x04)"),
        __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::DMFButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT16 DMF = 0x8006;

  hr = ptp.SDIOSetExtDevicePropValue(DPC_FOCUS_MODE, DMF, 0x04);
  if (hr != S_OK) {
    MessageBox(
        NULL,
        __T("Error: SDIOSetExtDevicePropValue(DPC_FOCUS_MODE, 0x8006, 0x04)"),
        __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::MFButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT16 MF = 0x0001;

  hr = ptp.SDIOSetExtDevicePropValue(DPC_FOCUS_MODE, MF, 0x04);
  if (hr != S_OK) {
    MessageBox(
        NULL,
        __T("Error: SDIOSetExtDevicePropValue(DPC_FOCUS_MODE, 0x0001, 0x04)"),
        __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::ZoomwideButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT32 WIDE = 0xFF;
  UINT32 STOP = 0x00;
  hr = ptp.SDIOControlDevice(DPC_ZOOM, WIDE);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_ZOOM, WIDE)"),
               __T("Error"), MB_OK);
    return;
  }
  Sleep(500);
  hr = ptp.SDIOControlDevice(DPC_ZOOM, STOP);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_ZOOM, STOP)"),
               __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::ZoomstopButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT32 STOP = 0x00;

  hr = ptp.SDIOControlDevice(DPC_ZOOM, STOP);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_ZOOM, STOP)"),
               __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::ZoomteleButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT32 TELE = 0x01;
  UINT32 STOP = 0x00;

  hr = ptp.SDIOControlDevice(DPC_ZOOM, TELE);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_ZOOM, TELE)"),
               __T("Error"), MB_OK);
    return;
  }
  Sleep(500);
  hr = ptp.SDIOControlDevice(DPC_ZOOM, STOP);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_ZOOM, STOP)"),
               __T("Error"), MB_OK);
    return;
  }
}

void CaptureDlg::ptpButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT32 SELECT_ON_REMOTE_DEVICE = 0x00000002;
  UINT32 TRANSFER_MODE_OFF = 0x00000000;
  UINT32 ADD_INFO_NONE = 0x00000000;

  hr = ptp.SDIOSetContentsTransferMode(SELECT_ON_REMOTE_DEVICE,
                                       TRANSFER_MODE_OFF, ADD_INFO_NONE);
  if (hr != S_OK) {
    MessageBox(
        NULL,
        __T("Error: SDIOSetContentsTransferMode(SELECT_ON_REMOTE_DEVICE, "
            "TRANSFER_MODE_OFF, ADD_INFO_NONE)"),
        __T("Error"), MB_OK);
    return;
  }

  if (!LiveViewclicked) {
    CheckDlgButton(cpDlg, IDC_DISPLAY_LIVEVIEW_CHECK, BST_CHECKED);
    LiveViewclicked = TRUE;
  }

  HWND hWnd_mtp_list = GetDlgItem(hWnd, IDC_PHOTO_LIST);
  SendMessage(hWnd_mtp_list, LB_RESETCONTENT, 0, 0);
  SetWindowText(GetDlgItem(hWnd, IDC_MTP_MESSAGE), _TEXT(""));

  EnableWindow(GetDlgItem(hWnd, IDC_SD_READ), FALSE);
  EnableWindow(GetDlgItem(hWnd, IDC_TRANSFER), FALSE);
}

void CaptureDlg::mtpButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT32 SELECT_ON_REMOTE_DEVICE = 0x00000002;
  UINT32 TRANSFER_MODE_ON = 0x00000001;
  UINT32 ADD_INFO_NONE = 0x00000000;
  ptp_mode = NULL;

  if (!mtp_button_clicked && CameraConnected) {
    hr = ptp.SDIOControlDevice(DPC_S1_BUTTON, UP);
    if (hr != S_OK) {
      MessageBox(NULL, __T("Error: SDIOControlDevice(DPC_S1_BUTTON, UP)"),
                 __T("Error"), MB_OK);
      return;
    }
    Sleep(200);
    mtp_button_clicked = 1;
  }
  hr = ptp.SDIOSetContentsTransferMode(SELECT_ON_REMOTE_DEVICE,
                                       TRANSFER_MODE_ON, ADD_INFO_NONE);
  if (hr != S_OK) {
    MessageBox(
        NULL,
        __T("Error: SDIOSetContentsTransferMode(SELECT_ON_REMOTE_DEVICE, "
            "TRANSFER_MODE_ON, ADD_INFO_NONE)"),
        __T("Error"), MB_OK);
    return;
  }

  CheckDlgButton(cpDlg, IDC_DISPLAY_LIVEVIEW_CHECK, BST_UNCHECKED);
  Sleep(1500);
  InvalidateRect(hWnd, NULL, TRUE);
  UpdateWindow(hWnd);
  LiveViewclicked = FALSE;

  HWND hWnd_mtp_list = GetDlgItem(hWnd, IDC_PHOTO_LIST);
  SendMessage(hWnd_mtp_list, LB_RESETCONTENT, 0, 0);
  SetWindowText(GetDlgItem(hWnd, IDC_MTP_MESSAGE), _TEXT(""));
  EnableWindow(GetDlgItem(hWnd, IDC_SD_READ), TRUE);
  EnableWindow(GetDlgItem(hWnd, IDC_TRANSFER), TRUE);
}

void CaptureDlg::jpegfilelist(HWND hWnd) {
  HRESULT hr = 0;
  wchar_t buffer[1024] = {};
  UINT32 ObjectHandles = 0;
  UINT32 StorageID = 0x00010001;
  UINT32 FOLDER = 0x3001;
  UINT32 JPEG = 0x3801;

  hr = ptp.GetObjectHandles(StorageID, FOLDER, 0xFFFFFFFF);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: Escape GetObjectHandles"), __T("Error"),
               MB_OK);
    return;
  }
  Sleep(200);
  PTP_VENDOR_DATA_OUT *pDataOut_2 = dMgr.getObjectHandles();

  INT32 NumElements = ((pDataOut_2->VendorReadData[3]) << 24) |
                      ((pDataOut_2->VendorReadData[2]) << 16) |
                      ((pDataOut_2->VendorReadData[1]) << 8) |
                      (pDataOut_2->VendorReadData[0]);

  for (int i = 0; i < NumElements; i++) {
    ObjectHandles = pDataOut_2->VendorReadData[4 * i + 7] << 24 |
                    pDataOut_2->VendorReadData[4 * i + 6] << 16 |
                    pDataOut_2->VendorReadData[4 * i + 5] << 8 |
                    pDataOut_2->VendorReadData[4 * i + 4];

    hr = ptp.GetObjectHandles(StorageID, JPEG, ObjectHandles);
    if (hr != S_OK) {
      MessageBox(NULL, __T("Error: Escape GetObjectHandlesFolder"),
                 __T("Error"), MB_OK);
      return;
    }
    Sleep(200);

    PTP_VENDOR_DATA_OUT *pDataOut_3 = dMgr.getObjectHandles();

    INT32 NumElements_3 = ((pDataOut_3->VendorReadData[3]) << 24) |
                          ((pDataOut_3->VendorReadData[2]) << 16) |
                          ((pDataOut_3->VendorReadData[1]) << 8) |
                          (pDataOut_3->VendorReadData[0]);

    for (int i = 0; i < NumElements_3; i++) {
      UINT32 ObjectHandlesFolder = pDataOut_3->VendorReadData[4 * i + 7] << 24 |
                                   pDataOut_3->VendorReadData[4 * i + 6] << 16 |
                                   pDataOut_3->VendorReadData[4 * i + 5] << 8 |
                                   pDataOut_3->VendorReadData[4 * i + 4];

      PTP_GetObjectInfo info = {0};
      hr = ptp.GetObjectInfo(ObjectHandlesFolder, info);
      if (hr != S_OK) {
        MessageBox(NULL, __T("Error: Escape GetObjectInfo"), __T("Error"),
                   MB_OK);
        return;
      }

      wsprintf(buffer, L"%s\n", reinterpret_cast<UINT16 *>(&info.FileName));

      HWND hWnd_mtp_list = GetDlgItem(hWnd, IDC_PHOTO_LIST);
      SendMessage(hWnd_mtp_list, LB_ADDSTRING, 0,
                  reinterpret_cast<LPARAM>(buffer));
      UpdateWindow(hWnd_mtp_list);
      DisableProcessWindowsGhosting();
    }
    CoTaskMemFree(pDataOut_3);
  }
  CoTaskMemFree(pDataOut_2);
}

void CaptureDlg::rawfilelist(HWND hWnd) {
  HRESULT hr = 0;
  wchar_t buffer[1024] = {};
  UINT32 ObjectHandles = 0;
  UINT32 StorageID = 0x00010001;
  UINT32 FOLDER = 0x3001;
  UINT32 RAW = 0xB101;

  hr = ptp.GetObjectHandles(StorageID, FOLDER, 0xFFFFFFFF);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: Escape GetObjectHandles"), __T("Error"),
               MB_OK);
    return;
  }
  Sleep(200);
  PTP_VENDOR_DATA_OUT *pDataOut_2 = dMgr.getObjectHandles();

  INT32 NumElements = ((pDataOut_2->VendorReadData[3]) << 24) |
                      ((pDataOut_2->VendorReadData[2]) << 16) |
                      ((pDataOut_2->VendorReadData[1]) << 8) |
                      (pDataOut_2->VendorReadData[0]);

  for (int i = 0; i < NumElements; i++) {
    ObjectHandles = pDataOut_2->VendorReadData[4 * i + 7] << 24 |
                    pDataOut_2->VendorReadData[4 * i + 6] << 16 |
                    pDataOut_2->VendorReadData[4 * i + 5] << 8 |
                    pDataOut_2->VendorReadData[4 * i + 4];

    hr = ptp.GetObjectHandles(StorageID, RAW, ObjectHandles);
    if (hr != S_OK) {
      MessageBox(NULL, __T("Error: Escape GetObjectHandlesFolder"),
                 __T("Error"), MB_OK);
      return;
    }
    Sleep(200);

    PTP_VENDOR_DATA_OUT *pDataOut_3 = dMgr.getObjectHandles();

    INT32 NumElements_3 = ((pDataOut_3->VendorReadData[3]) << 24) |
                          ((pDataOut_3->VendorReadData[2]) << 16) |
                          ((pDataOut_3->VendorReadData[1]) << 8) |
                          (pDataOut_3->VendorReadData[0]);

    for (int i = 0; i < NumElements_3; i++) {
      UINT32 ObjectHandlesFolder = pDataOut_3->VendorReadData[4 * i + 7] << 24 |
                                   pDataOut_3->VendorReadData[4 * i + 6] << 16 |
                                   pDataOut_3->VendorReadData[4 * i + 5] << 8 |
                                   pDataOut_3->VendorReadData[4 * i + 4];

      PTP_GetObjectInfo info = {0};
      hr = ptp.GetObjectInfo(ObjectHandlesFolder, info);
      if (hr != S_OK) {
        MessageBox(NULL, __T("Error: Escape GetObjectInfo"), __T("Error"),
                   MB_OK);
        return;
      }

      wsprintf(buffer, L"%s\n", reinterpret_cast<UINT16 *>(&info.FileName));
      HWND hWnd_mtp_list = GetDlgItem(hWnd, IDC_PHOTO_LIST);
      SendMessage(hWnd_mtp_list, LB_ADDSTRING, 0,
                  reinterpret_cast<LPARAM>(buffer));
      UpdateWindow(hWnd_mtp_list);
      DisableProcessWindowsGhosting();
    }
    CoTaskMemFree(pDataOut_3);
  }
  CoTaskMemFree(pDataOut_2);
}

void CaptureDlg::mp4filelist(HWND hWnd) {
  HRESULT hr = 0;
  wchar_t buffer[1024] = {};
  UINT32 ObjectHandles = 0;
  UINT32 StorageID = 0x00010001;
  UINT32 FOLDER = 0x3001;
  UINT32 MP4 = 0xB982;

  hr = ptp.GetObjectHandles(StorageID, FOLDER, 0xFFFFFFFF);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: Escape GetObjectHandles"), __T("Error"),
               MB_OK);
    return;
  }
  Sleep(200);
  PTP_VENDOR_DATA_OUT *pDataOut_2 = dMgr.getObjectHandles();

  INT32 NumElements = ((pDataOut_2->VendorReadData[3]) << 24) |
                      ((pDataOut_2->VendorReadData[2]) << 16) |
                      ((pDataOut_2->VendorReadData[1]) << 8) |
                      (pDataOut_2->VendorReadData[0]);

  for (int i = 0; i < NumElements; i++) {
    ObjectHandles = pDataOut_2->VendorReadData[4 * i + 7] << 24 |
                    pDataOut_2->VendorReadData[4 * i + 6] << 16 |
                    pDataOut_2->VendorReadData[4 * i + 5] << 8 |
                    pDataOut_2->VendorReadData[4 * i + 4];

    hr = ptp.GetObjectHandles(StorageID, MP4, ObjectHandles);
    if (hr != S_OK) {
      MessageBox(NULL, __T("Error: Escape GetObjectHandlesFolder"),
                 __T("Error"), MB_OK);
      return;
    }
    Sleep(200);

    PTP_VENDOR_DATA_OUT *pDataOut_3 = dMgr.getObjectHandles();

    INT32 NumElements_3 = ((pDataOut_3->VendorReadData[3]) << 24) |
                          ((pDataOut_3->VendorReadData[2]) << 16) |
                          ((pDataOut_3->VendorReadData[1]) << 8) |
                          (pDataOut_3->VendorReadData[0]);

    for (int i = 0; i < NumElements_3; i++) {
      UINT32 ObjectHandlesFolder = pDataOut_3->VendorReadData[4 * i + 7] << 24 |
                                   pDataOut_3->VendorReadData[4 * i + 6] << 16 |
                                   pDataOut_3->VendorReadData[4 * i + 5] << 8 |
                                   pDataOut_3->VendorReadData[4 * i + 4];

      PTP_GetObjectInfo info = {0};
      hr = ptp.GetObjectInfo(ObjectHandlesFolder, info);
      if (hr != S_OK) {
        MessageBox(NULL, __T("Error: Escape GetObjectInfo"), __T("Error"),
                   MB_OK);
        return;
      }

      wsprintf(buffer, L"%s\n", reinterpret_cast<UINT16 *>(&info.FileName));
      HWND hWnd_mtp_list = GetDlgItem(hWnd, IDC_PHOTO_LIST);
      SendMessage(hWnd_mtp_list, LB_ADDSTRING, 0,
                  reinterpret_cast<LPARAM>(buffer));
      UpdateWindow(hWnd_mtp_list);
      DisableProcessWindowsGhosting();
    }
    CoTaskMemFree(pDataOut_3);
  }
  CoTaskMemFree(pDataOut_2);
}

void CaptureDlg::jpegfiletransfer(HWND hWnd) {
  HRESULT hr = 0;
  wchar_t buffer[1024] = {};
  UINT32 ObjectHandles = 0;
  UINT32 StorageID = 0x00010001;
  UINT32 FOLDER = 0x3001;
  UINT32 JPEG = 0x3801;

  hr = ptp.GetObjectHandles(StorageID, FOLDER, 0xFFFFFFFF);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: Escape GetObjectHandles"), __T("Error"),
               MB_OK);
    return;
  }
  Sleep(200);
  PTP_VENDOR_DATA_OUT *pDataOut_2 = dMgr.getObjectHandles();

  INT32 NumElements = ((pDataOut_2->VendorReadData[3]) << 24) |
                      ((pDataOut_2->VendorReadData[2]) << 16) |
                      ((pDataOut_2->VendorReadData[1]) << 8) |
                      (pDataOut_2->VendorReadData[0]);

  for (int i = 0; i < NumElements; i++) {
    ObjectHandles = pDataOut_2->VendorReadData[4 * i + 7] << 24 |
                    pDataOut_2->VendorReadData[4 * i + 6] << 16 |
                    pDataOut_2->VendorReadData[4 * i + 5] << 8 |
                    pDataOut_2->VendorReadData[4 * i + 4];

    hr = ptp.GetObjectHandles(StorageID, JPEG, ObjectHandles);
    if (hr != S_OK) {
      MessageBox(NULL, __T("Error: Escape GetObjectHandlesFolder"),
                 __T("Error"), MB_OK);
      return;
    }
    Sleep(200);

    PTP_VENDOR_DATA_OUT *pDataOut_3 = dMgr.getObjectHandles();

    INT32 NumElements_3 = ((pDataOut_3->VendorReadData[3]) << 24) |
                          ((pDataOut_3->VendorReadData[2]) << 16) |
                          ((pDataOut_3->VendorReadData[1]) << 8) |
                          (pDataOut_3->VendorReadData[0]);

    for (int i = 0; i < NumElements_3; i++) {
      UINT32 ObjectHandlesFolder = pDataOut_3->VendorReadData[4 * i + 7] << 24 |
                                   pDataOut_3->VendorReadData[4 * i + 6] << 16 |
                                   pDataOut_3->VendorReadData[4 * i + 5] << 8 |
                                   pDataOut_3->VendorReadData[4 * i + 4];

      PTP_GetObjectInfo info = {0};
      hr = ptp.GetObjectInfo(ObjectHandlesFolder, info);

      BYTE *buffer_t = new BYTE[info.ObjCompSz];
      hr = ptp.ExecuteGetObject(ObjectHandlesFolder, buffer_t, info.ObjCompSz);
      if (hr != S_OK) {
        MessageBox(NULL,
                   __T("Error: ExecuteGetObject(OBJECT_HANDLE, buffer, "
                       "info.ObjCompSz)"),
                   __T("Error"), MB_OK);
        delete[] buffer_t;
        return;
      }
      hr = dMgr.saveImage(buffer_t, info.ObjCompSz, info.FileName, 0);
      if (hr != S_OK) {
        MessageBox(NULL, __T("Error: saveImage"), __T("Error"), MB_OK);
        delete[] buffer_t;
        return;
      }
      DisableProcessWindowsGhosting();
      delete[] buffer_t;
    }
    CoTaskMemFree(pDataOut_3);
  }
  CoTaskMemFree(pDataOut_2);
}

void CaptureDlg::rawfiletransfer(HWND hWnd) {
  HRESULT hr = 0;
  wchar_t buffer[1024] = {};
  UINT32 ObjectHandles = 0;
  UINT32 StorageID = 0x00010001;
  UINT32 FOLDER = 0x3001;
  UINT32 RAW = 0xB101;

  hr = ptp.GetObjectHandles(StorageID, FOLDER, 0xFFFFFFFF);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: Escape GetObjectHandles"), __T("Error"),
               MB_OK);
    return;
  }
  Sleep(200);
  PTP_VENDOR_DATA_OUT *pDataOut_2 = dMgr.getObjectHandles();

  INT32 NumElements = ((pDataOut_2->VendorReadData[3]) << 24) |
                      ((pDataOut_2->VendorReadData[2]) << 16) |
                      ((pDataOut_2->VendorReadData[1]) << 8) |
                      (pDataOut_2->VendorReadData[0]);

  for (int i = 0; i < NumElements; i++) {
    ObjectHandles = pDataOut_2->VendorReadData[4 * i + 7] << 24 |
                    pDataOut_2->VendorReadData[4 * i + 6] << 16 |
                    pDataOut_2->VendorReadData[4 * i + 5] << 8 |
                    pDataOut_2->VendorReadData[4 * i + 4];

    hr = ptp.GetObjectHandles(StorageID, RAW, ObjectHandles);
    if (hr != S_OK) {
      MessageBox(NULL, __T("Error: Escape GetObjectHandlesFolder"),
                 __T("Error"), MB_OK);
      return;
    }
    Sleep(200);

    PTP_VENDOR_DATA_OUT *pDataOut_3 = dMgr.getObjectHandles();

    INT32 NumElements_3 = ((pDataOut_3->VendorReadData[3]) << 24) |
                          ((pDataOut_3->VendorReadData[2]) << 16) |
                          ((pDataOut_3->VendorReadData[1]) << 8) |
                          (pDataOut_3->VendorReadData[0]);

    for (int i = 0; i < NumElements_3; i++) {
      UINT32 ObjectHandlesFolder = pDataOut_3->VendorReadData[4 * i + 7] << 24 |
                                   pDataOut_3->VendorReadData[4 * i + 6] << 16 |
                                   pDataOut_3->VendorReadData[4 * i + 5] << 8 |
                                   pDataOut_3->VendorReadData[4 * i + 4];

      PTP_GetObjectInfo info = {0};
      hr = ptp.GetObjectInfo(ObjectHandlesFolder, info);

      BYTE *buffer_t = new BYTE[info.ObjCompSz];
      hr = ptp.ExecuteGetObject(ObjectHandlesFolder, buffer_t, info.ObjCompSz);
      if (hr != S_OK) {
        MessageBox(NULL,
                   __T("Error: ExecuteGetObject(OBJECT_HANDLE, buffer, "
                       "info.ObjCompSz)"),
                   __T("Error"), MB_OK);
        delete[] buffer_t;
        return;
      }
      hr = dMgr.saveImage(buffer_t, info.ObjCompSz, info.FileName, 0);
      if (hr != S_OK) {
        MessageBox(NULL, __T("Error: saveImage"), __T("Error"), MB_OK);
        delete[] buffer_t;
        return;
      }
      DisableProcessWindowsGhosting();
      delete[] buffer_t;
    }
    CoTaskMemFree(pDataOut_3);
  }
  CoTaskMemFree(pDataOut_2);
}

void CaptureDlg::mp4filetransfer(HWND hWnd) {
  HRESULT hr = 0;
  wchar_t buffer[1024] = {};
  UINT32 ObjectHandles = 0;
  UINT32 StorageID = 0x00010001;
  UINT32 FOLDER = 0x3001;
  UINT32 MP4 = 0xB982;

  hr = ptp.GetObjectHandles(StorageID, FOLDER, 0xFFFFFFFF);
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: Escape GetObjectHandles"), __T("Error"),
               MB_OK);
    return;
  }
  Sleep(200);
  PTP_VENDOR_DATA_OUT *pDataOut_2 = dMgr.getObjectHandles();

  INT32 NumElements = ((pDataOut_2->VendorReadData[3]) << 24) |
                      ((pDataOut_2->VendorReadData[2]) << 16) |
                      ((pDataOut_2->VendorReadData[1]) << 8) |
                      (pDataOut_2->VendorReadData[0]);

  for (int i = 0; i < NumElements; i++) {
    ObjectHandles = pDataOut_2->VendorReadData[4 * i + 7] << 24 |
                    pDataOut_2->VendorReadData[4 * i + 6] << 16 |
                    pDataOut_2->VendorReadData[4 * i + 5] << 8 |
                    pDataOut_2->VendorReadData[4 * i + 4];

    hr = ptp.GetObjectHandles(StorageID, MP4, ObjectHandles);
    if (hr != S_OK) {
      MessageBox(NULL, __T("Error: Escape GetObjectHandlesFolder"),
                 __T("Error"), MB_OK);
      return;
    }
    Sleep(200);

    PTP_VENDOR_DATA_OUT *pDataOut_3 = dMgr.getObjectHandles();

    INT32 NumElements_3 = ((pDataOut_3->VendorReadData[3]) << 24) |
                          ((pDataOut_3->VendorReadData[2]) << 16) |
                          ((pDataOut_3->VendorReadData[1]) << 8) |
                          (pDataOut_3->VendorReadData[0]);

    for (int i = 0; i < NumElements_3; i++) {
      UINT32 ObjectHandlesFolder = pDataOut_3->VendorReadData[4 * i + 7] << 24 |
                                   pDataOut_3->VendorReadData[4 * i + 6] << 16 |
                                   pDataOut_3->VendorReadData[4 * i + 5] << 8 |
                                   pDataOut_3->VendorReadData[4 * i + 4];

      PTP_GetObjectInfo info = {0};
      hr = ptp.GetObjectInfo(ObjectHandlesFolder, info);

      BYTE *buffer_t = new BYTE[info.ObjCompSz];
      hr = ptp.ExecuteGetObject(ObjectHandlesFolder, buffer_t, info.ObjCompSz);
      if (hr != S_OK) {
        MessageBox(NULL,
                   __T("Error: ExecuteGetObject(OBJECT_HANDLE, buffer, "
                       "info.ObjCompSz)"),
                   __T("Error"), MB_OK);
        delete[] buffer_t;
        return;
      }
      hr = dMgr.saveImage(buffer_t, info.ObjCompSz, info.FileName, 0);
      if (hr != S_OK) {
        MessageBox(
            NULL,
            __T("Error: saveImage(buffer_t, info.ObjCompSz, info.FileName, 0)"),
            __T("Error"), MB_OK);
        delete[] buffer_t;
        return;
      }
      DisableProcessWindowsGhosting();
      delete[] buffer_t;
    }
    CoTaskMemFree(pDataOut_3);
  }
  CoTaskMemFree(pDataOut_2);
}

void CaptureDlg::mtpreadButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT32 MTP = 0x04;
  UINT32 StorageID = 0;
  HWND hWnd_mtp_list = GetDlgItem(hWnd, IDC_PHOTO_LIST);
  HWND hWnd_mtp_message = GetDlgItem(hWnd, IDC_MTP_MESSAGE);

  EnableWindow(GetDlgItem(hWnd, IDC_SD_READ), FALSE);
  EnableWindow(GetDlgItem(hWnd, IDC_TRANSFER), FALSE);

  SendMessage(hWnd_mtp_list, LB_RESETCONTENT, 0, 0);
  UpdateWindow(hWnd_mtp_list);
  SetWindowText(GetDlgItem(hWnd, IDC_MTP_MESSAGE),
                _TEXT("Reading files from SD card. Please wait."));
  UpdateWindow(hWnd_mtp_message);
  Sleep(200);

  hr = ptp.GetStorageID();
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: Escape GetStorageID"), __T("Error"), MB_OK);
    return;
  }

  PTP_VENDOR_DATA_OUT *pDataOut = dMgr.getStorageID();
  Sleep(200);
  INT32 num = pDataOut->VendorReadData[3] << 24 |
              pDataOut->VendorReadData[2] << 16 |
              pDataOut->VendorReadData[1] << 8 | pDataOut->VendorReadData[0];

  for (int i = 0; i < num; i++) {
    StorageID = pDataOut->VendorReadData[4 * i + 7] << 24 |
                pDataOut->VendorReadData[4 * i + 6] << 16 |
                pDataOut->VendorReadData[4 * i + 5] << 8 |
                pDataOut->VendorReadData[4 * i + 4];
    if (StorageID == 0x00010001) {
      hr = ptp.GetNumObjects();
      if (hr != S_OK) {
        MessageBox(NULL, __T("Error: Escape GetNumObjects"), __T("Error"),
                   MB_OK);
        return;
      }
      Sleep(200);
      PTP_VENDOR_DATA_OUT *pDataOut_1 = dMgr.getNumObjects();
      wchar_t buffer[1024] = {};
      num = pDataOut_1->Params[0];

      jpegfilelist(hWnd);

      rawfilelist(hWnd);

      mp4filelist(hWnd);

      CoTaskMemFree(pDataOut_1);
    }
  }

  SetWindowText(GetDlgItem(hWnd, IDC_MTP_MESSAGE),
                _TEXT("Reading files completed!"));
  UpdateWindow(hWnd_mtp_message);

  EnableWindow(GetDlgItem(hWnd, IDC_SD_READ), TRUE);
  EnableWindow(GetDlgItem(hWnd, IDC_TRANSFER), TRUE);

  CoTaskMemFree(pDataOut);
}

void CaptureDlg::transferButtonClicked(HWND hWnd) {
  HRESULT hr = 0;
  UINT32 MTP = 0x04;
  UINT32 StorageID = 0;
  HWND hWnd_mtp_message = GetDlgItem(hWnd, IDC_MTP_MESSAGE);

  EnableWindow(GetDlgItem(hWnd, IDC_SD_READ), FALSE);

  SetWindowText(GetDlgItem(hWnd, IDC_MTP_MESSAGE),
                _TEXT("Start transfereing files. Pleas wait."));
  UpdateWindow(hWnd_mtp_message);

  Sleep(200);
  hr = ptp.GetStorageID();
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: Escape GetStorageID"), __T("Error"), MB_OK);
    return;
  }

  PTP_VENDOR_DATA_OUT *pDataOut = dMgr.getStorageID();
  Sleep(200);
  INT32 num = pDataOut->VendorReadData[3] << 24 |
              pDataOut->VendorReadData[2] << 16 |
              pDataOut->VendorReadData[1] << 8 | pDataOut->VendorReadData[0];
  wchar_t buffer[1024] = {};

  for (int i = 0; i < num; i++) {
    StorageID = pDataOut->VendorReadData[4 * i + 7] << 24 |
                pDataOut->VendorReadData[4 * i + 6] << 16 |
                pDataOut->VendorReadData[4 * i + 5] << 8 |
                pDataOut->VendorReadData[4 * i + 4];
    if (StorageID == 0x00010001) {
      jpegfiletransfer(hWnd);

      rawfiletransfer(hWnd);

      mp4filetransfer(hWnd);
    }
  }
  CoTaskMemFree(pDataOut);
  SetWindowText(GetDlgItem(hWnd, IDC_MTP_MESSAGE),
                _TEXT("Compelted tranferring all files."));
  UpdateWindow(hWnd_mtp_message);
  EnableWindow(GetDlgItem(hWnd, IDC_SD_READ), TRUE);
}

void CaptureDlg::formatButtonClicked(HWND hWnd) {
  HRESULT hr = 0;

  int msgboxID = MessageBox(
      NULL, static_cast<LPCWSTR>(L"Do you want to format the SD card?"),
      static_cast<LPCWSTR>(L"Confirmation"), MB_OKCANCEL);

  switch (msgboxID) {
    case IDCANCEL:
      break;
    case IDOK:
      if (!isMediaFormatStarted) {
        hr = ptp.SDIOControlDevice(DPC_MEDIA_FORMAT, DOWN);
        if (hr != S_OK) {
          MessageBox(NULL,
                     __T("Please switch the media save destination to either "
                         "Memory or Memory & Host"),
                     __T("Information"), MB_OK);
          return;
        }
        CheckDlgButton(hWnd, IDC_MEDIA_FORMAT, 0);

        isMediaFormatStarted = TRUE;
      } else {
        hr = ptp.SDIOControlDevice(DPC_MEDIA_FORMAT, UP);
        if (hr != S_OK) {
          MessageBox(NULL,
                     __T("Error: SDIOControlDevice(DPC_MEDIA_FORMAT, UP)"),
                     __T("Error"), MB_OK);
          return;
        }
        CheckDlgButton(hWnd, IDC_MEDIA_FORMAT, 0);
        isMediaFormatStarted = FALSE;
      }

      break;
  }
}

void CaptureDlg::getDeviceInfoButtonClicked() {
  HRESULT hr = ptp.GetDeviceInfo();
  if (hr != S_OK) {
    MessageBox(NULL, __T("Error: Escape GetDeviceInfo"), __T("Error"), MB_OK);
    return;
  }

  PTP_VENDOR_DATA_OUT *pDataOut = dMgr.getDeviceInfoData();

  UINT offset = 0;

  offset += sizeof(UINT16);
  offset += sizeof(UINT);
  offset += sizeof(UINT16);
  UINT VendorExtensionLen = pDataOut->VendorReadData[offset];
  offset += sizeof(UINT8);
  offset += sizeof(UINT16) * VendorExtensionLen;
  offset += sizeof(UINT16);
  UINT OperationSupportedLen = pDataOut->VendorReadData[offset];
  offset += sizeof(UINT);
  offset += sizeof(UINT16) * OperationSupportedLen;
  UINT EventSupportedLen = pDataOut->VendorReadData[offset];
  offset += sizeof(UINT);
  offset += sizeof(UINT16) * EventSupportedLen;
  offset += sizeof(UINT);
  offset += sizeof(UINT);
  UINT ImageFormatsLen = pDataOut->VendorReadData[offset];
  offset += sizeof(UINT);
  offset += sizeof(UINT16) * ImageFormatsLen;
  UINT ManufactureLen = pDataOut->VendorReadData[offset];
  offset += sizeof(UINT8);
  offset += sizeof(UINT16) * ManufactureLen;

  UINT ModelLen = pDataOut->VendorReadData[offset];
  offset += sizeof(UINT8);
  UINT8 modelLength = (ModelLen * sizeof(UINT16));
  UINT8 *modelLenString = new UINT8[modelLength];
  memcpy(modelLenString,
         static_cast<UINT8 *>(&pDataOut->VendorReadData[offset]), modelLength);
  cCreateLPWSTRImpl(IDC_MODEL_EDIT, modelLenString, ModelLen);
  delete[] modelLenString;
  offset += sizeof(UINT16) * ModelLen;

  UINT DeviceVersionLen = pDataOut->VendorReadData[offset];
  offset += sizeof(UINT8);
  UINT8 deviceVersionlength = (DeviceVersionLen * sizeof(UINT16));
  UINT8 *deviceVersionLenString = new UINT8[deviceVersionlength];
  memcpy(deviceVersionLenString,
         static_cast<UINT8 *>(&pDataOut->VendorReadData[offset]),
         deviceVersionlength);
  cCreateLPWSTRImpl(IDC_DEVICE_VERSION_EDIT, deviceVersionLenString,
                    DeviceVersionLen);
  delete[] deviceVersionLenString;
  offset += sizeof(UINT16) * DeviceVersionLen;

  UINT SerialNumberLen = pDataOut->VendorReadData[offset];
  offset += sizeof(UINT8);

  UINT8 SerialNumberLength = (SerialNumberLen * sizeof(UINT16));

  UINT8 *SerialNumberLenString = new UINT8[SerialNumberLength];

  memcpy(SerialNumberLenString,
         static_cast<UINT8 *>(&pDataOut->VendorReadData[offset + 50]),
         SerialNumberLength);
  cCreateLPWSTRImpl(IDC_SERIAL_NUMBER_EDIT, SerialNumberLenString,
                    SerialNumberLen);
  delete[] SerialNumberLenString;
  offset += sizeof(UINT16) * SerialNumberLen;

  CoTaskMemFree(pDataOut);
}

void CaptureDlg::setSaveFolderPath() {
  BROWSEINFO bi;
  ITEMIDLIST *idl = NULL;
  LPMALLOC g_pMalloc = NULL;
  TCHAR szTmp[MAX_PATH];

  if (SHGetMalloc(&g_pMalloc) != S_OK) {
    return;
  }

  bi.hwndOwner = NULL;
  bi.pidlRoot = NULL;
  bi.pszDisplayName = szTmp;
  bi.lpszTitle = __TEXT("Select Folder");
  bi.ulFlags = BIF_RETURNONLYFSDIRS;
  bi.lpfn = NULL;
  bi.lParam = 0;
  bi.iImage = 0;
  idl = SHBrowseForFolder(&bi);
  if (idl != NULL) {
    if (SHGetPathFromIDList(idl, szTmp) != FALSE) {
      SetWindowText(GetDlgItem(cpDlg, IDC_SAVEPATH_EDIT), szTmp);
    }
    g_pMalloc->Free(idl);
  }
}

void CaptureDlg::getDisplayImageFromLiveview() {
  HRESULT hr = 0;
  UINT32 StorageID = 0;

  if (IsDlgButtonChecked(cpDlg, IDC_DISPLAY_LIVEVIEW_CHECK) == BST_CHECKED) {
    if (dMgr.getIsLiveviewValidFlag()) {
      PTP_GetObjectInfo info = {0};
      hr = ptp.GetObjectInfo(0xFFFFC002, info);
      StorageID = info.StorageID;
      if (hr != S_OK) {
        return;
      }

      if (info.ObjCompSz > 0) {
        BYTE *buffer = new BYTE[info.ObjCompSz];
        hr = ptp.ExecuteGetObject(0xFFFFC002, buffer, info.ObjCompSz);
        if (hr != S_OK) {
          delete[] buffer;
          return;
        }
        setDisplayImageFromByteArray(buffer);
        delete[] buffer;
      }
    }
  }
}

void CaptureDlg::DisplayImageFromLiveview(HWND hWnd) {

  if (!LiveViewclicked) {
    CheckDlgButton(cpDlg, IDC_DISPLAY_LIVEVIEW_CHECK, BST_CHECKED);
    LiveViewclicked = TRUE;
  } else {
    CheckDlgButton(cpDlg, IDC_DISPLAY_LIVEVIEW_CHECK, BST_UNCHECKED);
    Sleep(200);
    InvalidateRect(hWnd, NULL, TRUE);
    UpdateWindow(hWnd);
    LiveViewclicked = FALSE;
  }
}

void CaptureDlg::setDisplayImageFromByteArray(BYTE *buffer) {
  HBITMAP hBitmap = NULL;
  RECT rt;
  HDC hdc = NULL, hCompatDC = NULL;
  HWND hLvFrame = NULL;
  BITMAP bmp_info;
  UINT nImageWidth = 0;
  UINT nImageHeight = 0;
  UINT nWindowWidth = 0;
  UINT nWindowHeight = 0;
  UINT nWidth = 0;
  UINT nHeight = 0;
  int h_adjust = 0;
  int w_adjust = 0;
  BOOL ret = FALSE;

  ret = ptp.LoadDisplayImageFromLiveview(buffer, &bmp_info, hBitmap);

  if (ret == FALSE) {
    return;
  }

  hLvFrame = GetDlgItem(cpDlg, IDC_LIVEVIEW_FRAME);

  if (ExposureProgram != PreExposureProgram) {
    InvalidateRect(hLvFrame, NULL, TRUE);
    UpdateWindow(hLvFrame);
    PreExposureProgram = ExposureProgram;
  }

  GetWindowRect(hLvFrame, &rt);
  hdc = GetDC(hLvFrame);

  nImageWidth = bmp_info.bmWidth;
  nImageHeight = bmp_info.bmHeight;

  nWindowWidth = rt.right - rt.left;
  nWindowHeight = rt.bottom - rt.top;

  if (static_cast<double>(nImageWidth) / nImageHeight <
      static_cast<double>(nWindowWidth) / nWindowHeight) {
    nHeight = nWindowHeight;
    nWidth = nWindowHeight * nImageWidth / nImageHeight;
  } else {
    nWidth = nWindowWidth;
    nHeight = nWindowWidth * nImageHeight / nImageWidth;
  }
  if (nImageWidth == 0) {
  } else if (nWindowHeight / nImageWidth == 4 / 3)
  {
    h_adjust = 0;
  } else if (nHeight == nWidth)
  {
    w_adjust = (nWindowWidth - nWidth) / 2;
    h_adjust = 0;
  } else {
    h_adjust = (nWindowHeight - nHeight) / 2;
  }

  hCompatDC = CreateCompatibleDC(hdc);

  SelectObject(hCompatDC, hBitmap);
  DeleteObject(hBitmap);

  SetStretchBltMode(hdc, COLORONCOLOR);

  StretchBlt(hdc, w_adjust, h_adjust, nWidth, nHeight, hCompatDC, 0, 0,
             nImageWidth, nImageHeight, SRCCOPY);

  ReleaseDC(hLvFrame, hdc);
  DeleteDC(hCompatDC);
  DeleteObject(hBitmap);
}

bool CaptureDlg::memoryAlloc(DWORD bufsize) {
  if (m_buffer != NULL) {
    memoryDispose();
  }

  m_buffer = new UINT8[bufsize];

  return ((m_buffer == NULL) ? false : true);
}

void CaptureDlg::memoryDispose(void) {
  if (m_buffer != NULL) {
    delete[] m_buffer;
    m_buffer = NULL;
  }
}

void CaptureDlg::cCreateLPWSTRImpl(int idc, BYTE *value, UINT8 length) {
  TCHAR *str = new TCHAR[length];
  swprintf_s(&str[0], length, __TEXT("%s"), reinterpret_cast<wchar_t *>(value));
  SetWindowText(GetDlgItem(cpDlg, idc), str);
  delete[] str;
}

void CaptureDlg::updatemsg(UINT8 status) {
  TCHAR tcText[UCHAR_MAX];

  switch (status) {
    case UPDATE_STATUS_SYSTEMSTANDBY:
      wsprintf(update_msg, TEXT("%s"),
               __T("Update Status : SystemStandby Mode"));
      break;
    case UPDATE_STATUS_FILE_SENDING:
      wsprintf(update_msg, TEXT("%s"),
               __T("Update Status : Update File Sending"));
      break;
    case UPDATE_STATUS_UPDATING:
      wsprintf(update_msg, TEXT("%s"),
               __T("Update Status : Updating (Waiting for Reconnect)"));
      break;
    case UPDATE_STATUS_UPDATE_FIN:
      GetDlgItemText(cpDlg, IDC_UPDATE_BUTTON_STATUS, tcText,
                     sizeof(tcText) / sizeof(TCHAR));
      if (_tcscmp(tcText,
                  __TEXT("Update Status : Updating (Waiting for Reconnect)")) ==
          0) {
        wsprintf(update_msg, TEXT("%s"),
                 __T("Update Status : Update Complete"));
      } else {
        wsprintf(update_msg, TEXT("%s"), __T("Update Status : -"));
      }
      break;
  }
  SetDlgItemText(cpDlg, IDC_UPDATE_BUTTON_STATUS,
                 static_cast<LPCWSTR>(update_msg));
}
