; SPDX-License-Identifier: Apache-2.0
; Inno Setup script: per-user install (no admin needed to install; the dongle driver step asks for
; admin later, once). Build: iscc packaging\installer.iss  (after PyInstaller)

#define AppName "FBT Pods"
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{6E1A6F5C-3B7D-4C55-9A0E-4F2B7C1D9A31}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=FBT Pods (personal project)
DefaultDirName={localappdata}\Programs\FBTPods
DefaultGroupName=FBT Pods
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=FBTPods-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
LicenseFile=..\NOTICE
UninstallDisplayIcon={app}\FBTPods.exe

[Languages]
#ifexist "ChineseTraditional.isl"
Name: "chinesetrad"; MessagesFile: "ChineseTraditional.isl"
#endif
Name: "english"; MessagesFile: "compiler:Default.isl"

; Text of our own, in the language picked when installing (it used to be Chinese for everyone)
[CustomMessages]
english.DesktopIcon=Create a desktop shortcut
english.HardwareCheck=FBT Pods hardware check
english.UninstallApp=Uninstall FBT Pods
english.RunNow=Open FBT Pods now
#ifexist "ChineseTraditional.isl"
chinesetrad.DesktopIcon=建立桌面捷徑
chinesetrad.HardwareCheck=FBT Pods 硬體驗證
chinesetrad.UninstallApp=解除安裝 FBT Pods
chinesetrad.RunNow=立即開啟 FBT Pods
#endif

[Tasks]
Name: "desktopicon"; Description: "{cm:DesktopIcon}"; Flags: checkedonce

[Files]
Source: "..\dist\FBTPods\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[InstallDelete]
; 1.0 named these in Chinese whatever the setup language; the new ones follow the setup language
Type: files; Name: "{group}\FBT Pods 硬體驗證.lnk"
Type: files; Name: "{group}\解除安裝 FBT Pods.lnk"

[Icons]
Name: "{group}\FBT Pods"; Filename: "{app}\FBTPods.exe"
Name: "{group}\{cm:HardwareCheck}"; Filename: "{app}\FBTPods-console.exe"; Parameters: "diag"
Name: "{group}\{cm:UninstallApp}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\FBT Pods"; Filename: "{app}\FBTPods.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\FBTPods.exe"; Description: "{cm:RunNow}"; Flags: nowait postinstall skipifsilent
