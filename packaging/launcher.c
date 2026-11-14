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

int WINAPI wWinMain(HINSTANCE h, HINSTANCE p, PWSTR args, int show) {
    wchar_t dir[MAX_PATH], exe[MAX_PATH], app[MAX_PATH], cmd[4096];
    GetModuleFileNameW(NULL, dir, MAX_PATH);
    wchar_t *slash = wcsrchr(dir, L'\\');
    if (slash) *slash = 0;

    while (*args == L' ') args++;
    int gui = (*args == 0) || (args[0] == L'-' && args[1] == L'-') || wcsncmp(args, L"gui", 3) == 0;
    swprintf(exe, MAX_PATH, L"%ls\\runtime\\%ls", dir, gui ? L"pythonw.exe" : L"python.exe");
    swprintf(app, MAX_PATH, L"%ls\\app", dir);
    if (GetFileAttributesW(exe) == INVALID_FILE_ATTRIBUTES || GetFileAttributesW(app) == INVALID_FILE_ATTRIBUTES) {
        MessageBoxW(NULL, L"找不到 runtime 或 app 資料夾。請把整個 FBT Pods 資料夾完整解壓縮後再執行。",
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
    DWORD flags = gui ? 0 : CREATE_NEW_CONSOLE;
    SetEnvironmentVariableW(L"PYTHONNOUSERSITE", L"1");
    SetEnvironmentVariableW(L"PYTHONUTF8", L"1");
    SetEnvironmentVariableW(L"PYTHONPATH", app);          /* the airpodsvr package lives in app\ */
    if (!CreateProcessW(NULL, cmd, NULL, NULL, FALSE, flags, NULL, app, &si, &pi)) {
        MessageBoxW(NULL, L"無法啟動 FBT Pods。", L"FBT Pods", MB_ICONERROR);
        return 1;
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return 0;
}
