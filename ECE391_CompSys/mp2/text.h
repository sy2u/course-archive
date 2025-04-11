/*
 * tab:4
 *
 * text.h - font data and text to mode X conversion utility header file
 *
 * "Copyright (c) 2004-2009 by Steven S. Lumetta."
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose, without fee, and without written agreement is
 * hereby granted, provided that the above copyright notice and the following
 * two paragraphs appear in all copies of this software.
 * 
 * IN NO EVENT SHALL THE AUTHOR OR THE UNIVERSITY OF ILLINOIS BE LIABLE TO 
 * ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL 
 * DAMAGES ARISING OUT  OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, 
 * EVEN IF THE AUTHOR AND/OR THE UNIVERSITY OF ILLINOIS HAS BEEN ADVISED 
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * THE AUTHOR AND THE UNIVERSITY OF ILLINOIS SPECIFICALLY DISCLAIM ANY 
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE 
 * PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND NEITHER THE AUTHOR NOR
 * THE UNIVERSITY OF ILLINOIS HAS ANY OBLIGATION TO PROVIDE MAINTENANCE, 
 * SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS."
 *
 * Author:        Steve Lumetta
 * Version:       2
 * Creation Date: Thu Sep  9 22:08:16 2004
 * Filename:      text.h
 * History:
 *    SL    1    Thu Sep  9 22:08:16 2004
 *        First written.
 *    SL    2    Sat Sep 12 13:40:11 2009
 *        Integrated original release back into main code base.
 */

#ifndef TEXT_H
#define TEXT_H

/* The default VGA text mode font is 8x16 pixels. */
#define FONT_WIDTH   8
#define FONT_HEIGHT  16

/* Standard VGA text font. */
extern unsigned char font_data[256][16];

// checkpoint 1
extern void text_to_buf(unsigned char* bar_buffer, char* str, unsigned char back_color);
#define WHITE           0x1F
#define BAR_X_DIM       320
#define BAR_Y_DIM       18
#define BAR_X_WIDTH     BAR_X_DIM / 4
#define BAR_SIZE        BAR_X_DIM * BAR_Y_DIM
#define BAR_PLANE_SIZE  BAR_SIZE / 4

// checkpoint 2
#define FRUIT_TEXT_COLOR    0xFE   // place holder, no actual color
#define FRUIT_BACK_COLOR    0xFF   // place holder, no actual color
#define FRUIT_X_DIM         104    // maxl lengt: "a strawberry!", text won't exceed 13 fonts
#define FRUIT_Y_DIM         16
#define FRUIT_SIZE          FRUIT_X_DIM * FRUIT_Y_DIM
extern void fruit_to_buf(unsigned char* buffer, char* str);

#endif /* TEXT_H */
