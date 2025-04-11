#ifndef _SYSCALL_H
#define _SYSCALL_H

#include "filesystem.h"
#include "terminal.h"
#include "paging.h"
#include "schedule.h"

#define MIN_PROCESS_PID     3
#define MAX_DESCS           8
#define MAX_PIDS            6
#define MAX_SHELL           3
#define MAX_CMD             128
#define MAX_ARGLEN          32
#define EIGHT_MB            0x800000
#define EIGHT_KB            0x2000
#define PROGRAM_IMAGE       0x08048000

// File operations jump table
typedef struct file_operations {
    int32_t (*open)(const uint8_t*);
    int32_t (*read)(int32_t, void*, int32_t);
    int32_t (*write)(int32_t, const void*, int32_t);
    int32_t (*close)(int32_t);
} file_operations_t;

// Represents an open file as associated with a process
typedef struct {
    file_operations_t* f_ops; // Pointer to the file operations jump table
    uint32_t inode; // Inode number for this file (0 for directories and RTC)
    uint32_t file_position; // Current read position in the file
    int8_t usage_flag; // -1 unalloc, 0 not in use, 1 in use
    int8_t file_type_flag; // 0 rtc, 1 dir, 2 reg file
    int8_t reserved_flags[2];  //reserved, change as needed
} file_descriptor_t;

typedef struct pcb_t { 
    file_descriptor_t fds[MAX_DESCS];
    uint32_t cur_pid;
    uint32_t parent_pid;
    uint32_t esp;
    uint32_t ebp;
    uint8_t args[BUF_SIZE];
    uint8_t exception;
    uint8_t pre_hist_id;
} pcb_t;

extern pcb_t* get_pcb(uint32_t pid);
extern pcb_t* get_cur_pcb();
extern uint32_t get_cur_pid();

extern uint8_t pid_list[MAX_PIDS];

// system call function interface from Appendix B
extern int32_t execute (const uint8_t* command);
extern int32_t halt (uint8_t status);
extern int32_t read (int32_t fd, void* buf, int32_t nbytes);
extern int32_t write (int32_t fd, const void* buf, int32_t nbytes);
extern int32_t open (const uint8_t* filename);
extern int32_t close (int32_t fd);
extern int32_t getargs (uint8_t* buf, int32_t nbytes);
extern int32_t vidmap (uint8_t** screen_start);
extern int32_t set_handler (int32_t signum, void* handler_address);
extern int32_t sigreturn (void);

static char *exceptions[];

#endif // _SYSCALL_H
