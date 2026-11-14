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

[Tasks]
Name: "desktopicon"; Description: "建立桌面捷徑"; Flags: checkedonce

[Files]
Source: "..\dist\FBTPods\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\FBT Pods"; Filename: "{app}\FBTPods.exe"
Name: "{group}\FBT Pods 硬體驗證"; Filename: "{app}\FBTPods-console.exe"; Parameters: "diag"
Name: "{group}\解除安裝 FBT Pods"; Filename: "{uninstallexe}"
Name: "{userdesktop}\FBT Pods"; Filename: "{app}\FBTPods.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\FBTPods.exe"; Description: "立即開啟 FBT Pods"; Flags: nowait postinstall skipifsilent
