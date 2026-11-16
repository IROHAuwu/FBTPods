/* SPDX-License-Identifier: Apache-2.0
 * FBTPods.exe — portable launcher (no console window).
 *   FBTPods.exe                -> runtime\pythonw.exe -m airpodsvr gui
 *   FBTPods.exe --autostart    -> runtime\pythonw.exe -m airpodsvr gui --autostart   (SteamVR)
 *   FBTPods.exe diag|setup|... -> runtime\python.exe  -m airpodsvr diag ...          (own console)
 * Layout: FBTPods.exe, app\ (program files, data\), runtime\ (Python).
 * Build: x86_64-w64-mingw32-gcc -municode -mwindows -O2 launcher.c icon.o -o FBTPods.exe
 */
#include <windows.h>
#include <shlwapi.h>
#include <stdio.h>
#include <wchar.h>

/* exit code of a Python stopped with Ctrl+C (STATUS_CONTROL_C_EXIT): the user meant to stop it */
#define CTRL_C_EXIT_CODE 0xC000013AUL

/* Settings the user made for another Python (PYTHONHOME left behind by an old install,
 * PYTHONWARNINGS, ...) must not reach the bundled runtime: PYTHONHOME alone makes it stop before it
 * can show or log anything. Every PYTHON* variable is dropped; main() sets the ones the app needs. */
static void drop_python_env(void) {
    wchar_t *block = GetEnvironmentStringsW();
    wchar_t *e;
    wchar_t name[256];
    if (!block)
        return;
    /* the block is a copy, so the real environment can be changed while walking it */
    for (e = block; *e; e += wcslen(e) + 1) {
        const wchar_t *eq = wcschr(e + 1, L'=');    /* +1: hidden per-drive entries start with '=' */
        size_t n = eq ? (size_t)(eq - e) : 0;
        if (n > 6 && n < 256 && _wcsnicmp(e, L"PYTHON", 6) == 0) {
            wcsncpy(name, e, n);
            name[n] = 0;
            SetEnvironmentVariableW(name, NULL);
        }
    }
    FreeEnvironmentStringsW(block);
}

/* Ctrl+C in the console is for python.exe (it stops tracking cleanly); the launcher must not quit
 * on it, or the console would close with it before an error can be read. */
static BOOL WINAPI ignore_ctrl_c(DWORD ev) {
    return ev == CTRL_C_EVENT || ev == CTRL_BREAK_EVENT;
}

/* A terminal command ended with an error: keep its console open until Enter, so the message can be
 * read (the window used to close the moment python.exe exited). */
static void wait_for_enter(DWORD code) {
    wchar_t msg[512], buf[16];
    DWORD n;
    HANDLE out = CreateFileW(L"CONOUT$", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                             NULL, OPEN_EXISTING, 0, NULL);
    HANDLE in = CreateFileW(L"CONIN$", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                            NULL, OPEN_EXISTING, 0, NULL);
    swprintf(msg, 512, L"\nFBT Pods stopped with an error (code %lu). Press Enter to close this window.\n"
                       L"FBT Pods 發生錯誤而結束（代碼 %lu）。按 Enter 關閉視窗。\n", code, code);
    if (out != INVALID_HANDLE_VALUE) {
        WriteConsoleW(out, msg, (DWORD)wcslen(msg), &n, NULL);
        CloseHandle(out);
    }
    if (in != INVALID_HANDLE_VALUE) {
        FlushConsoleInputBuffer(in);                 /* keys typed while it ran do not count */
        ReadConsoleW(in, buf, 16, &n, NULL);
        CloseHandle(in);
    }
}

int WINAPI wWinMain(HINSTANCE h, HINSTANCE p, PWSTR args, int show) {
    wchar_t dir[MAX_PATH], exe[MAX_PATH], app[MAX_PATH], cmd[4096], msg[2048];
    DWORD code = 0;
    GetModuleFileNameW(NULL, dir, MAX_PATH);
    wchar_t *slash = wcsrchr(dir, L'\\');
    if (slash) *slash = 0;

    while (*args == L' ') args++;
    int gui = (*args == 0) || (args[0] == L'-' && args[1] == L'-') || wcsncmp(args, L"gui", 3) == 0;
    swprintf(exe, MAX_PATH, L"%ls\\runtime\\%ls", dir, gui ? L"pythonw.exe" : L"python.exe");
    swprintf(app, MAX_PATH, L"%ls\\app", dir);
    if (GetFileAttributesW(exe) == INVALID_FILE_ATTRIBUTES || GetFileAttributesW(app) == INVALID_FILE_ATTRIBUTES) {
        MessageBoxW(NULL, L"The runtime or app folder is missing. Extract the whole FBT Pods folder from the zip "
                          L"file first, then run FBTPods.exe from the extracted folder.\n\n"
                          L"找不到 runtime 或 app 資料夾。請把整個 FBT Pods 資料夾完整解壓縮後再執行。",
                    L"FBT Pods", MB_ICONERROR);
        return 1;
    }
    if (*args == 0)
        swprintf(cmd, 4096, L"\"%ls\" -m airpodsvr gui", exe);
    else if (args[0] == L'-' && args[1] == L'-')
        swprintf(cmd, 4096, L"\"%ls\" -m airpodsvr gui %ls", exe, args);
    else
        swprintf(cmd, 4096, L"\"%ls\" -m airpodsvr %ls", exe, args);

    STARTUPINFOW si = {sizeof(si)};
    PROCESS_INFORMATION pi;
    drop_python_env();
    SetEnvironmentVariableW(L"PYTHONNOUSERSITE", L"1");
    SetEnvironmentVariableW(L"PYTHONUTF8", L"1");
    SetEnvironmentVariableW(L"PYTHONPATH", app);          /* the airpodsvr package lives in app\ */
    /* Terminal commands run in the caller's folder ("replay rec.txt" opens rec.txt there), so that
     * folder must not be searched for modules: a stray json.py in it would be imported instead of
     * Python's own. */
    SetEnvironmentVariableW(L"PYTHONSAFEPATH", L"1");
    if (!gui) {
        /* the console is the launcher's, shared with python.exe: it stays open while we wait below */
        AllocConsole();
        SetConsoleCtrlHandler(ignore_ctrl_c, TRUE);
    }
    /* terminal commands start in the caller's folder (NULL), so a file named on the command line is
     * found where it was typed; the window keeps app\ (it moves to its data folder at once anyway) */
    if (!CreateProcessW(NULL, cmd, NULL, NULL, FALSE, 0, NULL, gui ? app : NULL, &si, &pi)) {
        MessageBoxW(NULL, L"Cannot start FBT Pods.\n\n無法啟動 FBT Pods。", L"FBT Pods", MB_ICONERROR);
        return 1;
    }
    CloseHandle(pi.hThread);
    if (!gui) {
        WaitForSingleObject(pi.hProcess, INFINITE);
        GetExitCodeProcess(pi.hProcess, &code);
        if (code != 0 && code != CTRL_C_EXIT_CODE)
            wait_for_enter(code);
    } else if (WaitForSingleObject(pi.hProcess, 5000) == WAIT_OBJECT_0
               && GetExitCodeProcess(pi.hProcess, &code) && code != 0) {
        /* pythonw has no console: an error in its first seconds (most often a folder Windows does not
         * let it write to, such as Program Files) would otherwise look like nothing happened at all */
        swprintf(msg, 2048,
                 L"FBT Pods stopped right after starting (error code %lu).\n\n"
                 L"If the FBT Pods folder is in a place Windows does not let programs write to (such as "
                 L"Program Files or a read-only drive), move the whole folder to Documents or the Desktop "
                 L"and start it again. Details, if any could be written: %ls\\data\\airpodsvr.log or "
                 L"%%LOCALAPPDATA%%\\FBTPods\\airpodsvr.log\n\n"
                 L"FBT Pods 啟動後立刻結束（錯誤代碼 %lu）。\n"
                 L"如果 FBT Pods 資料夾放在 Windows 不允許程式寫入的位置（例如 Program Files 或唯讀磁碟），"
                 L"請把整個資料夾移到「文件」或桌面後再開啟。詳細記錄（如果寫得進去）：%ls\\data\\airpodsvr.log 或 "
                 L"%%LOCALAPPDATA%%\\FBTPods\\airpodsvr.log",
                 code, app, code, app);
        MessageBoxW(NULL, msg, L"FBT Pods", MB_ICONERROR);
    }
    CloseHandle(pi.hProcess);
    return (int)code;
}
