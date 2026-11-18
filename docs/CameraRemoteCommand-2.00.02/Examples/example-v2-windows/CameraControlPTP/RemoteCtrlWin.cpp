#include "RemoteCtrlWin.h"

#include "CaptureDlg.h"
#include "PTPControl.h"
#include "UserMessage.h"

DWORD WINAPI getAllExtDevicePropInfoThread(LPVOID);
BOOL CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
void disConnectCall(HWND hWnd);
void expireUpdateCall(HWND hWnd);

struct CONNECTIONDATA {
  HWND hWnd;
  BOOL bEnd;
};

static CaptureDlg *captureDlg = NULL;
static PTPControl &ptp = PTPControl::getInstance();
static DataManager &dMgr = DataManager::getInstance();
const int waitTime = 1000;
HANDLE hMSP;

int WINAPI _tWinMain(_In_ HINSTANCE hCurInst, _In_opt_ HINSTANCE hPrevInst,
                     _In_ LPWSTR lpsCmdLine, _In_ int nCmdShow) {
  hMSP = CreateMutex(NULL, TRUE, __T("MultiplexStartingPrevention"));
  if (GetLastError() == ERROR_ALREADY_EXISTS) {
    MessageBox(NULL, __T("Error: Process already running"), __T("Error"),
               MB_OK);
  } else {
    DialogBox(hCurInst, __T("RCC_DIALOG"), NULL, (DLGPROC)WndProc);
  }

  return 0;
}

BOOL CALLBACK WndProc(HWND hDlg, UINT msg, WPARAM wp, LPARAM lp) {
  static CONNECTIONDATA connectData;
  static HANDLE hTh;
  static DWORD dwID;
  DWORD dwParam = 0;
  HRESULT hr = 0;

  switch (msg) {
    case WM_INITDIALOG:

      InitCommonControls();
      hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
      if (hr != S_OK) {
        MessageBox(NULL, __T("Error: CoInitialize"), __T("Error"), MB_OK);
        SendMessage(hDlg, WM_CLOSE, 0, 0);
      }
      captureDlg = new CaptureDlg();
      captureDlg->createOnDialog(hDlg);
      captureDlg->displayOnDialog();

      break;
    case WM_CAMERA_CONNECTION: {
      GetExitCodeThread(hTh, &dwParam);
      if (dwParam == STILL_ACTIVE) {
        break;
      }

      connectData.hWnd = hDlg;
      connectData.bEnd = FALSE;
      hTh = CreateThread(
          NULL, 0,
          static_cast<LPTHREAD_START_ROUTINE>(getAllExtDevicePropInfoThread),
          static_cast<LPVOID>(&connectData), 0, &dwID);

      break;
    }
    case WM_CAMERA_DISCONNECTION: {
      connectData.bEnd = TRUE;
      WaitForSingleObject(hTh, waitTime);
      CloseHandle(hTh);
      delete captureDlg;
      captureDlg = NULL;
      break;
    }
    case WM_CAMERA_EXPIRE_UPDATA: {
      dMgr.readCameraData();
      break;
    }
    case WM_CLOSE:
      connectData.bEnd = TRUE;

      if (hTh != NULL) {
        WaitForSingleObject(hTh, waitTime);
        CloseHandle(hTh);
      }

      delete captureDlg;
      ptp.closeMutex();
      if (hMSP) ReleaseMutex(hMSP);

      DestroyWindow(hDlg);
      return TRUE;
    case WM_DESTROY:
      PostQuitMessage(0);
      return TRUE;
  }
  return FALSE;
}

DWORD WINAPI getAllExtDevicePropInfoThread(LPVOID lpdata) {
  CONNECTIONDATA *lpmydata = NULL;
  lpmydata = static_cast<CONNECTIONDATA *>(lpdata);
  HRESULT hr = 0;

  while (!lpmydata->bEnd) {
    hr = ptp.SDIOGetAllExtDevicePropInfo(
        lpmydata->hWnd);

    if (hr != S_OK) {
      disConnectCall(lpmydata->hWnd);
      break;
    } else {
      expireUpdateCall(lpmydata->hWnd);
    }
    captureDlg->getDisplayImageFromLiveview();
  }
  return 0;
}

void disConnectCall(HWND hWnd) {
  SendMessage(hWnd, WM_CAMERA_DISCONNECTION, 0, 0);
}
void expireUpdateCall(HWND hWnd) {
  SendMessage(hWnd, WM_CAMERA_EXPIRE_UPDATA, 0, 0);
}
