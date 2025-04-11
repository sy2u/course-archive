#include "syscall.h"

uint8_t pre_hist_list[MAX_PIDS-MAX_SHELL] = {0,0,0}; // 0 for free, 1 for busy
uint8_t pid_list[MAX_PIDS] = {0,0,0,0,0,0}; // 0 for free, 1 for busy
extern int32_t cur_sched;
extern int32_t sched_arr[SCHED_NUM];
extern terminal term_buf[NUM_TERM];
history_t pre_hist[MAX_PIDS-MAX_SHELL]; // at most 3 previous history would be needed

// instantiate operations
// LEAVE THESE TABLE HERE, IT CAUSE STRANGE ERROR IN HEADERR FILE
file_operations_t stdin_op = {
    .open = terminal_open,
    .close = terminal_close,
    .read = terminal_read,
    .write = illegal_write
};

file_operations_t stdout_op = {
    .open = terminal_open,
    .close = terminal_close,
    .read = illegal_read,
    .write = terminal_write
};

// Define the file_operations struct for a regular file
file_operations_t reg_file_ops = {
    .open = fopen,
    .read = fread,
    .write = fwrite,
    .close = fclose
};

file_operations_t dir_file_ops = {
    .open = dopen,
    .read = dread,
    .write = dwrite,
    .close = dclose
};

file_operations_t rtc_file_ops = {
    .open = rtc_open,
    .read = rtc_read,
    .write = rtc_write,
    .close = rtc_close
};

/* get_cur_pcb
 * Description: Get PCB pointer for current process
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Return Value: pcb pointer
 */
pcb_t* get_pcb(uint32_t pid) {
    return (pcb_t*) (EIGHT_MB - (pid+1) * EIGHT_KB);
}

/* get_cur_pid
 * Description: Get PID for current process
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Return Value: PID
 */

uint32_t get_cur_pid() {
    register uint32_t saved_esp asm("esp"); 
    uint32_t cur_pid = (EIGHT_MB - saved_esp) / EIGHT_KB;
    return cur_pid;
}

/* get_pcb
 * Description: Get PCB for given PID
 * Inputs: pid -- Process ID for the PCB
 * Outputs: None
 * Side Effects: None
 * Return Value: PCB pointer
 */
pcb_t* get_cur_pcb() {
    uint32_t cur_pid = get_cur_pid();
    return get_pcb(cur_pid);
}


/* execute
 * Description: Attempting to load and execute a new program, handing off the processor to the new progra until it terminates.
                This function interface is mainly copied from mp3 Appendix B.
 * Inputs: Command -- a space-separated sequence of words
        First word -- e file name of the program to be executed
        Other cmds -- stripped of leading spaces, should be provided to the new program on request via the getargs system call
 * Outputs: Depending on the functionality of called program
 * Side Effects: may allocate new page, new pcb in the memory
 * Return Value: -1 if the command cannot be executed, 0 for success
 */
int32_t execute(const uint8_t* command){
    uint8_t fname[MAX_CMD];
    uint8_t fbuf[4]; // 4 bytes reserved to read magic bytes for ELF and EIP
    uint8_t argbuf[MAX_CMD];
    int32_t i, j, ret, pid, entry;
    dentry_t dentry;
    pcb_t *new_pcb, *cur_pcb;

    cli();

    // check file validity
    if( command == 0 ){ return -1; } // check NULL ptr
    // get file name
    for( i = 0; command[i] != ' ' && command[i] != '\0' && i < MAX_CMD; i++ ){ fname[i] = command[i]; } 
    fname[i] = '\0'; // make sure file name is NULL terminated
    ret = read_dentry_by_name(fname, &dentry);
    if( ret == -1 ){ return -1; } // file doesn't exist
     // check ELF, first 4 bytes
    if( read_data(dentry.inode_num, 0, fbuf, 4) == -1 ){ return -1; }
    if( !(fbuf[0]== 0x7f && fbuf[1]=='E' && fbuf[2]=='L' && fbuf[3]=='F') ){ return -1; }
    // read EIP, byte 24-27
    if( read_data(dentry.inode_num, 24, fbuf, 4) == -1 ){ return -1; } 
    entry = (fbuf[3]<<24) | (fbuf[2]<<16) | (fbuf[1]<<8) | fbuf[0];

    // parse args
    for( i++, j = 0; command[i] != '\0'; i++, j++){ argbuf[j] = command[i]; }
    argbuf[j] = '\0'; 
    
    // get new pid
    for( i = 0; i < MAX_PIDS; i++ ){
        if( pid_list[i] == 0 ){
            pid = i;
            pid_list[i] = 1;
            break;
        }
    }
    // reach maximum numbers of processes
    if (i == MAX_PIDS) { 
        printf("Maximum number of processes reached!\n");
        return -1; 
    }
    
    // set up paging
    set_user_page(EIGHT_MB + pid * FOUR_MB);
    // load file into memory
    if( read_data(dentry.inode_num, 0, (uint8_t*)PROGRAM_IMAGE, get_file_size(dentry.inode_num)) == -1 ){ return -1; }

    // save current process esp, ebp
    if( pid > 2 ){ // no need when launching base shells
        cur_pcb = get_cur_pcb();
        register uint32_t saved_ebp asm("ebp");
        register uint32_t saved_esp asm("esp");
        cur_pcb->ebp = saved_ebp;
        cur_pcb->esp = saved_esp;
    }

    // update sched_arr
    sched_arr[cur_sched] = pid;

    // create pcb
    new_pcb = get_pcb(pid);
    new_pcb->cur_pid = pid;
    new_pcb->parent_pid = (pid == 0) ? 0 : get_cur_pid(); // get_cur_pid() may not work for first shell
    
    // set exception flag
    new_pcb->exception = 0;
    // open fds
    // stdin
    new_pcb->fds[0].f_ops = &stdin_op;
    new_pcb->fds[0].usage_flag = 0;
    // stdout
    new_pcb->fds[1].f_ops = &stdout_op;
    new_pcb->fds[1].usage_flag = 0;

    // init terminals
    if( sched_arr[cur_sched]>= 2 ){ // not base shell, store parent history
        // assign pre_hist id
        for( i = 0; i < MAX_PIDS-MAX_SHELL; i++ ){
            if( pre_hist_list[i] == 0 ){
                new_pcb->pre_hist_id = i;
                pre_hist_list[i] = 1;
                break;
            }
        } // no sanity check, cause this is secured by os itself, won't fail
        for( i = 0; i < HIST_DEPTH; i++ ){
            pre_hist[new_pcb->pre_hist_id] = term_buf[cur_sched].hist;
        }
    }
    terminal_open(0);

    //set fds
    for (i = 2; i < 8; i ++){
        new_pcb->fds[i].usage_flag = -1;   //set the 6 files to unalloc
        new_pcb->fds[i].file_position = 0;
    }

    // add arguments
    memcpy(new_pcb->args, argbuf, MAX_CMD);

    // prepare context switch
    // setting ss0 to kernel’s stack segment
    tss.ss0 = KERNEL_DS;
    // kernel stack for a process grows form lower side of the 8-kb block
    // -4 to make sure it doesn't go to another pcb
    tss.esp0 = (uint32_t)get_pcb(pid - 1) - 4;

    // syscall uses INT gate, enable interrupt
    sti();

    // setting up IRET context
    // IRET needs 5 elements on stack
    //  User_DS
    //  ESP(user stack)
    //  EFLAG
    //  CS
    //  EIP
    asm volatile("pushl %0 \n\
                  pushl %1 \n\
                  pushfl   \n\
                  pushl %2 \n\
                  pushl %3 \n\
                  iret     \n"
                : /* no output */
                : "r" (USER_DS), \
                  "r" (PROGRAM_VIRT_ADDR + FOUR_MB - 4), \
                  "r" (USER_CS), \
                  "r" (entry)
                : "memory");
    // return
    return 0;
}

/* halt
 * Description: terminates a process, returning the specified value to its parent process.
                This function interface is mainly copied from mp3 Appendix B.
 * Inputs: Status -- return value from child process to be passed to parent process
 * Outputs: Error message if error happens
 * Side Effects: None
 * Return Value: return value from child process, or 0 for first shell
 */
int32_t halt(uint8_t status){
    pcb_t* cur_pcb = get_cur_pcb();
    pcb_t* par_pcb = get_pcb(cur_pcb->parent_pid);
    uint32_t cur_pid = cur_pcb->cur_pid;
    uint32_t ret = (uint32_t)status;
    int i;

    cli();
    
    // cannot halt base terminal
    if ( (cur_pid==0) || (cur_pid==1) || (cur_pid==2) ){
        printf("can NOT halt base shell!\n");
        pid_list[cur_pid] = 0; // reset base shell pid for next run
        execute((uint8_t*)"shell");
        return 0;
    }

    // Validate the current PID is within the expected range and the parent PID is active
    if (cur_pid < 0 || cur_pid >= MAX_PIDS || !pid_list[cur_pcb->parent_pid]) {
        return -1;
    }

    // restore terminals
    if( sched_arr[cur_sched]>= 2 ){
        for( i = 0; i < HIST_DEPTH; i++ ){ term_buf[cur_sched].hist = pre_hist[cur_pcb->pre_hist_id]; }
    }
    term_buf[cur_sched].read_idx = -1; // reset
    pre_hist_list[cur_pcb->pre_hist_id] = 0;

    // update sched_arr
    sched_arr[cur_sched] = cur_pcb->parent_pid;

    // clear opened file descriptors
    for (i = 0; i < MAX_DESCS; i++)
        if (cur_pcb->fds[i].usage_flag >= 0) {
            cur_pcb->fds[i].f_ops->close(i);
            cur_pcb->fds[i].usage_flag = -1;    //set to -1 as unused
            cur_pcb->fds[i].file_position = 0;
        }
    pid_list[cur_pid] = 0;

    // restore parent paging
    uint32_t parent_pid = cur_pcb->parent_pid;
    set_user_page(EIGHT_MB + parent_pid * FOUR_MB);
    
    // write parent process info into tss
    // kernel stack for a process grows form lower side of the 8-kb block
    // -4 to make sure it doesn't go to another pcb
    tss.ss0 = KERNEL_DS;
    tss.esp0 = EIGHT_MB - parent_pid * EIGHT_KB - 4;

    // handle exception
    if(cur_pcb->exception == 1){
        printf("%s\n", exceptions[status]);
        ret = 256; // error code set by shell
        cur_pcb->exception = 0;
    }
    
    // syscall uses INT gate, enable interrupt
    sti();

    // Jump to execute return 
    asm volatile("movl %0, %%eax \n\
                  movl %1, %%ebp \n\
                  movl %2, %%esp \n\
                  leave          \n\
                  ret            \n"
                : /* no output */
                : "r" (ret), \
                  "r" (par_pcb->ebp), \
                  "r" (par_pcb->esp)
                : "eax", "ebp", "esp");

    return 0;
}


/* read
 * Description: reads data from the keyboard, a file, device (RTC), or directory.
                This function interface is mainly copied from mp3 Appendix B.
 * Inputs: fd -- file index in pcb fd array
           buf -- pointer to the buffer
           nbytes -- number of bytes need to be read
 * Outputs: None
 * Side Effects: None
 * Return Value: 0 if success, -1 if fail
 */
int32_t read (int32_t fd, void* buf, int32_t nbytes){
    int32_t ret;
    //validate inputs
    if (fd < 0 || fd > 7) return -1;
    if (buf == 0) return -1;
    if (nbytes < 0) return -1;

    //use file descriptor table to call read
    pcb_t* curr_pcb = get_cur_pcb();
    // make sure it's already opened and not in use
    if( curr_pcb->fds[fd].usage_flag != 0){ return -1; }
    // read
    curr_pcb->fds[fd].usage_flag = 1;
    ret = curr_pcb->fds[fd].f_ops->read(fd, buf, nbytes);
    curr_pcb->fds[fd].usage_flag = 0;

    return ret;
}

/* write
 * Description: writes data to the terminal or to a device (RTC).
                This function interface is mainly copied from mp3 Appendix B.
 * Inputs: fd -- file index in pcb fd array
           buf -- pointer to the buffer
           nbytes -- number of bytes need to be read
 * Outputs: None
 * Side Effects: None
 * Return Value: 0 if success, -1 if fail
 */
int32_t write (int32_t fd, const void* buf, int32_t nbytes){
    int32_t ret;
    //use file descriptor table to call write
    pcb_t* curr_pcb = get_cur_pcb();

    //validate inputs
    if (fd < 0 || fd > 7) {
        printf("inval fd");
        return -1;
    }
    if (buf == 0) {
        printf("inval buf ptr");
        return -1;
    }
    if ( (curr_pcb->fds[fd].file_type_flag == 2) && (nbytes == 0)) {// allow negative nbyte for file write
        printf("inval nbytes for files");
        return -1; 
    }
    
    // make sure it's already opened and not in use
    if( curr_pcb->fds[fd].usage_flag != 0){ return -1; } 
    // write
    curr_pcb->fds[fd].usage_flag = 1;
    ret = curr_pcb->fds[fd].f_ops->write(fd, buf, nbytes);
    curr_pcb->fds[fd].usage_flag = 0;
    if( ret == -1 ) return -1;
    // handle new file
    if( curr_pcb->fds[fd].file_type_flag == 1 ){
        curr_pcb->fds[fd].file_type_flag = 2;
        curr_pcb->fds[fd].f_ops = &reg_file_ops;
    }
    return ret;
}

/* open 
 * Description: provides access to the file system
                This function interface is mainly copied from mp3 Appendix B.
 * Inputs: filename -- name of the file to be opened
 * Outputs: None
 * Side Effects: None
 * Return Value: file descriptor index on success, -1 if fail
 */
int32_t open (const uint8_t* filename){
    //check invalid ptr filename
    if(filename == 0) return -1;
    //get pcb
    pcb_t* curr_pcb = get_cur_pcb();
    //attempt to allocate file descriptor
    int32_t i = 0;
    int32_t file_desc_idx = -1;
    
    // Attempt to find the directory entry by name.
    dentry_t dentry;
	if (read_dentry_by_name(filename, &dentry) == -1) {
        return -1;
	}

    for (i = 2; i < MAX_DESCS; i++){
        //check if flag is set to unalloc
        if(curr_pcb->fds[i].usage_flag == -1){
            //allocate this file descriptor
            file_desc_idx = i;
            curr_pcb->fds[i].usage_flag = 0; //set this fd flag to not in use
            curr_pcb->fds[i].file_position = 0;   //reset position
            break;  // break out of for loop
        }
    }
    // check if not allocated
    if(file_desc_idx == -1) return -1;

    //change file descriptor inode and file type
    curr_pcb->fds[file_desc_idx].inode = dentry.inode_num;

    //based off of file type fill in the f_ops
    curr_pcb->fds[file_desc_idx].file_type_flag = dentry.file_type;
    switch (dentry.file_type){
        case 0: //rtc
            curr_pcb->fds[file_desc_idx].f_ops = &rtc_file_ops;
            break;
        case 1: //dir
            curr_pcb->fds[file_desc_idx].f_ops = &dir_file_ops;
            break;
        case 2: //reg
            curr_pcb->fds[file_desc_idx].f_ops = &reg_file_ops;
            break;
    }
    
    // open
    curr_pcb->fds[file_desc_idx].f_ops->open(filename);

    return file_desc_idx;   //return file descriptor index on success *ta said so for checkpoint 3
}

/* close 
 * Description: closes the specified file descriptor and makes it available for return from later calls to open.
                This function interface is mainly copied from mp3 Appendix B.
 * Inputs: fd -- file index in pcb fd array
 * Outputs: None
 * Side Effects: None
 * Return Value: 0 if success, -1 if fail
 */
int32_t close (int32_t fd){
    //make sure fd is correct value
    if (fd < 2 || fd > 7) return -1;
    //get pcb
    pcb_t* curr_pcb = get_cur_pcb();
    //check if already closed
    if (curr_pcb->fds[fd].usage_flag == -1) return -1;
    //set usage to -1 for unlink
    curr_pcb->fds[fd].usage_flag = -1;
    // close
    return curr_pcb->fds[fd].f_ops->close(fd); 
}

/* getargs 
 * Description: reads the program’s command line arguments into a user-level buffer.
                This function interface is mainly copied from mp3 Appendix B.
 * Inputs: buf -- buffer to hold arguments
           nbytes -- number of bytes of argument to be read
 * Outputs: None
 * Side Effects: None
 * Return Value: 0 if success, -1 if fail
 */
int32_t getargs (uint8_t* buf, int32_t nbytes){
    // sanity check
    if((uint32_t)buf < PROGRAM_VIRT_ADDR || (uint32_t)buf >= PROGRAM_VIRT_ADDR+FOUR_MB ){ return -1; }
    pcb_t* curr_pcb = get_cur_pcb();
    //if argument length is longer than 32, fail entry
    int arglen = strlen((int8_t*)curr_pcb->args);
    if( arglen > MAX_ARGLEN) {return -1;}  
    // if( nbytes > MAX_CMD ){ nbytes = MAX_CMD; }
    memcpy(buf, curr_pcb->args, nbytes);
    return 0;
}

/* vidmap 
 * Description: maps the text-mode video memory into user space at a pre-set virtual address.
                This function interface is mainly copied from mp3 Appendix B.
 * Inputs: screen_start -- start address of the video memory virtual address
 * Outputs: None
 * Side Effects: None
 * Return Value: 0 if success, -1 if fail
 */
int32_t vidmap (uint8_t** screen_start){
    // sanity check
    if((uint32_t)screen_start < PROGRAM_VIRT_ADDR || (uint32_t)screen_start >= PROGRAM_VIRT_ADDR+FOUR_MB ){ return -1; }
    // set vidmap
    *screen_start = (uint8_t*)VIDMAP_ADDR;
    return 0;
}

// below function for for extra credit (signals)
int32_t set_handler (int32_t signum, void* handler_address){
    // place holder
    return -1;
}

int32_t sigreturn (void){
    // place holder
    return -1;
}

//storing the reasoning strings
static char *exceptions[] = {
        "Exception 0x00 Divide Error is raised.",
        "Exception 0x01 Intel Reserved is raised.",
        "Exception 0x02 NMI Interrupt is raised.",
        "Exception 0x03 Breakpoint is raised.",
        "Exception 0x04 Overflow is raised.",
        "Exception 0x05 BOUND Range Exceeded is raised.",
        "Exception 0x06 Invalid Opcode is raised.",
        "Exception 0x07 Device Not Available is raised.",
        "Exception 0x08 Double Fault is raised.",
        "Exception 0x09 Coprocessor Segment Overrun is raised.",
        "Exception 0x0A Invalid TSS is raised.",
        "Exception 0x0B Segment Not Present is raised.",
        "Exception 0x0C Stack-Segment Fault is raised.",
        "Exception 0x0D General Protection is raised.",
        "Exception 0x0E Page Fault is raised.",
        "Exception 0x0F Intel Reserved is raised.",
        "Exception 0x10 x87 FPU Floating-Point Error is raised.",
        "Exception 0x11 Alignment Check is raised.",
        "Exception 0x12 Machine Check is raised.",
        "Exception 0x13 SIMD Floating-Point is raised."
};
