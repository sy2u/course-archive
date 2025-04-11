#ifndef KEYBOARD_H
#define KEYBOARD_H

#include "i8259.h"
#include "x86_desc.h"
#include "types.h"
#include "lib.h"
#include "asm_linkage.h"
#include "rtc.h"
#include "speaker.h"
#include "cmos.h"
#include "history.h"
#include "tab.h"

#define SHELL_MODE      0
#define SPEAKER_MODE    1

#define BKS         (0x08)
#define TAB         (0x09)
#define CAPS        (0x3A)
#define KB_IRQ      (0x01)
#define PS2_DATA    (0x60)
#define NUM_SCAN    (58)

#define L_CTRL      (0x1D)
#define L_CTRL_R    (0x9D)

#define L_ALT       (0x38)
#define L_ALT_R     (0xB8)

#define L_SHIFT     (0x2A)
#define L_SHIFT_R   (0xAA)

#define R_SHIFT     (0x36)
#define R_SHIFT_R   (0xB6)

#define F1          (0X3B)
#define F2          (0X3C)
#define F3          (0X3D)

#define EXTENDED    (0xE0)
#define UP          (0X48)
#define DOWN        (0X50)

#define MAX_INPUT   127

void keyboard_init(void);
void keyboard_handler(void);

void switch_speaker_mode();
void clear_screen();

#endif
