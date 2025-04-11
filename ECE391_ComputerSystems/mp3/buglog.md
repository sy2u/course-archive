# BUGLOG
## Checkpoint 5
- Can't receive input \
After launching test machine, keyboard input can't be written into terminal. It looks like the terminal freezes. It's caused by forgetting send_eoi and sti in the pit handler, so the program can't receive interrupt after the first PIT interrupt.  
- Invalid tss and infinite halt \
The shells keeps printing out "can NOT halt base shell" when there's no command typed. By gdb, we found that's caused by invalid_tss exception. The reason is in the schduler, there's a variable decared after saving ebp and esp, so the esp sent into IRET context is not correct. Fixed by declaring all local variables at the beginning of the function. Make sure the stack is never changed after saving ebp and esp.  
- Random terminal input \
When typing into the terminal, the keyboard echo chars are seperated in all three terminals. It's writing into the current active terminal, instead of the current shown terminal. Fixed by updating terminal structure and set up the special case for keyboard input in lib.c, which can make sure for putc called by keyboard, every update is made in the real VRAM and the current terminal.  
- Program execute in the wrong terminal \
Sometimes when command is typed in terminal 1, the program would be executed in terminal 2 or 3. This is caused by error index in terminal_read(). Since terminal read is always managed by the current active process and wouldn't make change to the VRAM, the corresponding terminal buffer should always be the cur_sched, instead of cur_term. Fixed by changeing buffer index.
- Cursor update latency \
When switching terminal, cursor is not always immediately updated. Sometimes it would wait until the keyboard interrupt happened to update the cursor position. Reason: Cursor is intially only updated in lib.c and in scheduler, but not in terminal_switch. Though scheduler should be fast enough to update the cursor without being notices, it's not working ideally. Solved by manully update cursor each time terminal_switch is called.
- Random page fault \
After opening test machine for some time, page faults would be raised after some random time. Caused by not CLI in syscall, interrupt and exception handlers. PIT interrupt sometimes would happened in the middle of pushing or popping, causing register values to be messed up. Fixed by handle all CLI and STI in assembly linkage.
## Checkpoint 4
- Page Fault for vidmap  
When fish is running, page fault is raised. This is caused by when adding a new page table, write priority is not given. Solved by set writing priority bit in both PDE and PTE.  
- cat "executable" can only print out first few chars  
We initially only write min(nbytes,strlen(buf)) to terminal. But for executables, buf is not a string, there're a lot of NUL before the file actually ends. Fixed by write nbytes to the terminal.  
- fish: all slashes  \
We fish is called for the first time, the whole screen was occupied by slashes. This is caused by not supporting file_position function in file descriptor tables. So the program keep reading the first char in frame0/1.txt, but can't move forward. Fixed by adding file_position support in read_data and fread.  
- cat verylargetextwithverylongname.txt  \
Originally we did not have a filter for argument length so cat verylargetextwithverylongname.txt would read the file. This is not expected behavior so we added a string length filter in our get args function. Now cat verylargetextwithverylongname.txt prints "could not read arguments". cat verylargetextwithverylongname.tx now works as the command to print the very large text file.
## Checkpoint 3
- Page Fault in execute  
Page fault was thrown when it comes to assign file operation to f_ops->open directly. It's caused by we only allocate the address of file descriptor, but not file operation pointer. The f_op ptr is 0x00 which can't be accessed. Solved by instantiate file operation pointers as global variable in syscall.c. Please make sure those table in .c file but not .h file. Placing it in header file could cause strange compile errors.  
- File type indescrapancy  
in the file descriptor I labeled the file type flag as opposite of the dentry file type flag on accident. This caused our program to fail reading from rtc and files since the file type indescrapancy would fail argument checks. fixed this by making dentry file type and file descriptor file type flags match exactly
- Open function only allowed regular files  
our open function was copied directly from file open function, because of this it had a file type checker. this file type checker would return -1 if the file opened was not a regular file. this caused our LS function to not work for a while
- Pingpong freezes at the first line  
Open in syscall initially only allocate the file in file_desc_table, but never call the corresponding open function. So for pingpong, rtc_open is never called and rtc is never enabled. Pingpong stucked in waiting for rtc_read() return. Fixed by calling corresponding open function after file_desc allocation.
## Checkpoint 2
- RTC test first frame too slow  
When ceating tests for frequencies 1, 2, 4, 8, ..., 1024Hz, the first frame of each frequency is always slower than desired. That's caused by not updating current rtc counter when rtc_write, so the first frame would remain in old frequency. Fixed by set rtc_cnt in rtc_set_freq().
- `terminal_read, terminal_write` generate some random characters besides the buffer that was read, number of bytes also incorrect: xyin16\  
Clear the keyboard buffer every time and reset the length field to 0  
- File system:  
I had a bug where i had a for loop indexed by i but also another for loop inside that used counter as the index. i used i for a 3rd nested for loop and it messed up my indexing for the output of which dentry to pick.   
Adding on to those bugs, I wrote very bad loops to create my file name strings and had to rewrite it once because i forgot what it did.  

## Checkpoint 1
- IDT error handlers: romanl2  
our error screen "green screen" used to start printing at the bottom of the screen. I changed this by printing extra newlines at top of the exception handler.
- Access invalid page: siyingy3  
After confirming all PDE and PTE are set correctly, paging still can't be enabled, and GDB outputs "cannot access memory at 0x400052". Solved by moving page_directory and page_table and set them as global variable. They were intially declared as local variable.
- String printing Page Fault: siyingy3  
When 0x0D General Protection Exception is triggered, the screen instead would be printed as 0x0E Page Fault. This is caused by putc not update screen_y when input char is '\n', which makes screen_y>NUM_ROWS. Fixed by adding `screen_y = (screen_y + (screen_x / NUM_COLS)) % NUM_ROWS;` to this condition.
- PIC initialization leading to constant 0x0D Fault: xyin16  
During initilization, we should store PIC's original masks from data port, but initially we wrote as PIC port, leading to constant 0x0D Fault.
- IDT exception tests: romanl2  
while writing tests I messed up syntax for `asm volatile("int $0x0C");` by forgetting the $.
This caused a scare in the group chat because I pushed this version that did not compile
to master on github. 
- Paging: vxyu2  
When stepping through the program, There would be a memory error as soon as paging was enabled via the cr0 regiser. Fixed by declaring page directory and page table as global, rather than local variables.
