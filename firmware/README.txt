Bluetooth controller firmware, unmodified binaries from linux-firmware
(Ubuntu package version 20240318.git3b128b60-0ubuntu3.1, https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git).
Only the Bluetooth files the loaders use are included.

realtek/   rtl_bt/*  — Copyright Realtek Semiconductor Corporation. Redistributable, see realtek/LICENCE.txt
           (section "Redistributable-rtlwifi_firmware").
intel/     ibt-*     — Copyright Intel Corporation. Redistributable, see intel/LICENCE.txt
           (section "Redistributable-ibt_firmware").
mediatek/  BT_RAM_CODE_*, mt76x8pr2h — Copyright MediaTek Inc. Redistributable for use with MediaTek
           chipsets, see mediatek/LICENCE.txt (section "Redistributable-mediatek").
qca/       rampatch_usb_*, nvm_usb_* — Copyright Qualcomm Atheros, Inc. Redistributable, see qca/LICENCE.txt
           (section "Redistributable-QualcommAtheros_ath10k").

FBT Pods loads them into a USB Bluetooth adapter that has been switched to WinUSB, the same way
Linux does at boot. They are not modified, decompiled or reverse engineered.
