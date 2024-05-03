# Development Log
Congrats! All demo finished. It's a nice semester with 391! This log is maintained by Siying Yu.
`demo`: Final version used for extra credit demo & competition.  
## Extra Credit
All features below are implemented and demoed.
- [x] **Terminal Related**
    - [x] input history
        - Press `up` or `down` to restore input history 
        - Support all terminals and nested process calls
        - Should only be tested in shell mode (speaker mode has no terminal read)
    - [x] automatic command completion
        - Roman and Victor had some good try
        - for all cmds and args
- [x] **Mouse:**
    - [x] driver
        - Left click for scroll to last history command
        - right click for scroll to next history command
        - mid button for clear screen
- [x] **Speaker:** 
    - [x] driver
    - [x] demo program
        - Press `F1` to enter or exit speaker testing mode.
        - Add `-soundhw pcspk` to the property of test machine.
- [x] **Text Editor:**
    - [x] writable file system
        - [x] create new file (dwrite with nbytes > 0)
        - [x] remove file (dwrite with nbytes < 0)
        - [x] write date into file (fwrite) 
    - [x] demo user program (linux style)
        - [x] touch: `touch filename`
        - [x] rm: `rm filenmae`
        - [x] echo: 
            - append: `echo "content" >> new.txt`
            - overwrite: `echo "content" > new.txt`
            - show usage guidence: `echo -h`
    - Load user program into filesystem: [piazza guidence](https://piazza.com/class/lr8pz9xmnozbh/post/500)
        - filesystem_img is already updated on `vim` branch, so no need to compile system calls again.  
        - In syscall: `make mytest.exe`, `make mytest`
        - In fsdir: `dos2unix frame0.txt`, `dos2unix frame1.txt`
        - In mp3_group_43: `cp ./syscalls/to_fsdir/mytest fsdir/`, `./createfs -i fsdir -o student-distrib/filesys_img`
        - In student-distrib: `make clean && make dep && sudo make`
- [x] **Reboot Shutdown Power Control:**
    - [x] kernel support
        - added syscalls
    - [x] demo program
        - added support function and user program (reboot & halt)
- [x] **CMOS:** 
    - [x] driver  
    - [x] demo: Status Bar
        - Showing real time and current mode (Shell/Speaker)  
## Checkpoint 5:
- [x] **terminal switch** 
- [x] **scheduling**
## Checkpoint 3 & 4
**Disc Note:** iret pop out 32-bit data -> pushl is needed.
- [x] **execute:** 
    - paging.c: add function to set up user page & TLB flush
    - filesystem.c: add get_file_size() to copy user program to 128MB virtual memory space.  
    - syscall.c: context switch, details in comments
- [x] **halt:**
    - paging.c: switch user program page
    - syscall.c: context switch, details in comments
- [x] **getargs:** 
    - syscall.c: get stored arguments in pcb
- [x] **vidmap:** 
    - syscall.c: map user pointer to the fixed pre-set user vedio memory address
- [x] **updating filesystem and terminal:**
    - filesystem: fread, dread, read_data: add file_postion function, change return value
    - terminal_write: write nbytes, not strlen
- [x] **open, close, read, write:** 
## Checkpoint 2
[+Full points in demo! Congrats!+]  
- [x] **RTC Driver:** 
- rtc.c: rtc_open(), rtc_close(), rtc_write(), rtc_read()
    - Virtualization Enabled
    - Only support single process
    - Only power of 2 frequency allowed in cp2, comment out one sanity check line in rtc_write() can enable arbitrary frequency RTC
- lib.c: Add set_text_up(), used to print char from the top left of the screen. Wouldn't move cursor.
- test.c: Press key 'r' to proceed RTC test.
    - test_rtc_driver(): Main test function for RTC test. Cover all 4 functions. Use helper function test_rtc_helper().
    - wait(): used to wait for control signal, pause the test program to make prompts readable.
- [x] **Keyboard Handler:** 
Make sure term.len not exceeding 128  
Write to keyboard buffer when terminal_read is called. Should input length now not limited. Discarding all inputs after 127.  
- [X] **Terminal Driver:** 
Terminal test pushed. Support keyboard buffer up to 128 bytes(including '\n') \
Replace magic number with macros in lib.c  
- [x] **File System:**  
- filesystem.c: init_filesys_addr(), read_dentry_by_name(), read_dentry_by_index(), read_data()
- fread(), fwrite(), fopen(), fclose(), dread(), dwrite, dopen(), dclose()
    - enables read data from files and directories
    - allows up to 8 files open at once
    - modifications were made to kernel.c to pass address of file system.
    - fopen and dopen return correct file descriptor indexes.
## Checkpoint 1
- [x] **Init GDT:**  
- boot.S: lgdt gdt_desc_ptr  
- x86_desc.c: creat struct gdt_desc_ptr (Limit|BasePtr)
- [x] **Init IDT:**  
- idt.c: IDT initialization, exception handlers and system call handler  
    - Descriptor: use Interrupt gate for all entries for simplicity  
    - Exception Handler: didn't use cli(), cause INT gate would disable interrupt  
- asm_linkage.S: assembly linakges for all handlers 
    - Macro used for simplicity, interrupt and system call share the same linkage
- lib.c: Enable Exception Blue Screen, set High 4 bit (0x1 for blue) in attribute byte for VGA Text Mode Vedio (refer to mp1 document)  
- [x] **Init PIC:**  
- i8259.c: Follow [osdev_8259PIC](https://wiki.osdev.org/8259_PIC)
    - Add master port setting for irq_num>8
    - Add senity check, make sure 0<=irq_num<=15
    - Add slave PIC CAS port check, for first slave PIC port enabled and last slave PIC port disabled
    - Edit EOI format (0x60|irq_port_num)
- [x] **Init RTC:** 
- rtc.c: Initialization and handler, test_interrupt used for checkpoint 1
- [x] **Init Keyboard:** 
- keyboard.c: Initialization and handler, handler currently can handle all characters for checkpoint 1
- [x] **Init Paging:** 
- paging.c: Edit PDE and PTE for first 4kB and first 4MB
- setup_paging.S: Setting cr0, cr3, cr4 to enable paging, 4MB page and TLB flash.
