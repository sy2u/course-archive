## Checkpoint 1

### Status Bar Register Setup (2024/2/15)
**Description:** Status bar remains black after I setup copy_to_status_bar function and some of other registers.  
**Reason:** I forgot to set the 6th bit of ctrl_09h register to 0. So the line compare value is much larger than my desired one. (Should be 0x016B, but mistakenly set to 0x036B).  
**Solution:** Set mode_X_CRTC[9] = 0x0109.  

### Timing Format (2024/2/16)
**Description:** The format for time in the status bar is different from the one in demo. When minutes or seconds are less than ten, it only displays one digit instead of two digits.  
**Reason:** I directly pass the time value to sprintf function, and it's not standardized as the display form of tens and ones digits.  
**Solution:** I explicitly computer the first digit as time/60, and the second digit as time%60. And then put these two digits into sprintf function.  

### Fruit Corner Case (2024/2/24)
**Description:** The fruit doesn't disappear if it's in the corner.  
**Reason:** I put my masking routine in the wrong place. When the fruit is in the corner, the player can't go over it, and the dir would be DIR_STOP. The original draw_full_block function was put inside a if(dir!=DIR_STOP) condition, and I directly replace it with my masking function. So the block won't be rendered again if the fruit is in the corner.  
**Solution:** I add another condition check for dir==DIR_STOP. If dir==DIR_STOP, then check if there's any fruit by getting the return value from check_for_fruit(). If there's a fruit, then set need_redraw = 1. I also put all masking routine outside the if block, if need_redraw = 1, then the build buffer would be updated and whole screen would be rendered again.  


## Checkpoint 2

### Fruit Text Segmentation Fault (2024/2/23)
**Description:** The whole mazegame stucks when the player reaches a fruit. It can't receive any keyboard signal and can't be exited.  
**Reason:** There's a segmentation fault occured in the draw_fruit_text function. I declared an array for the buffer, but the size is smaller than the actual size. My write to memory exceeded the declared valid range. Actually, when computer the block_x_dim, I forgot to multiply the number of fonts by font_width.  
**Solution:** I updated the block_x_dim and the buffer size.

### Fruit Text Messy Background (2024/2/25)
**Description:** When the fruit text is on, and the player goes near the screen corner, the background would become messy.  
**Reason:** In draw_fruit_text function, I forgot to keep the buffer (for original background) updated with the blk. So the left and upper borders of the buffer are not aligned with the area where text are actually drawn. Wrong pixels outside the displayed screen are mixed in when doing unmask.  
**Solution:** I kept updating buffer pointer with blk in draw_fruit_text function, so to make the mask and unmask area match.  

### Fruit Text Transparent Color (2024/2/25)
**Description:** Fruit text over wall is always light blue, no matter what color is the wall.    
**Reason:** Transparent color is only initialized in fill_palette at the beginning of the program. The wall color keeps changing with the level, but its corresponding transparent (wall color + 0x40) color is fixed. So they no longer matches. Light blue is the transparent color for the first wall color.   
**Solution:** Every time the wall color is changed, I also updated its corresponding transparent color.  

### Tux Controller not Working at all (2024/3/2)
**Description:** After implementing button and led function, tux controller doesn't work and keep displaying OOPS.    
**Reason:** tuxctl-ioctl.c needs to run in kernel and should be compiled indenpendently and loaded into kernel. I forgot to compile it.  
**Solution:** Compile the driver in module folder and load it to the kernel.  

### Tux Controller Clear Segments Randomly (2024/3/3)
**Description:** By calling ioctl(fd, TUX_SET_LED, 0), sometimes the driver can clear the LED and sometimes it can not, which is very unstable.  
**Reason:** It's caused by slow ackonwledgement from tux, the previous command have't finished when calling clear LED.  
**Solution:**  Put clearing instruction in a while loop to make sure it is executed.  