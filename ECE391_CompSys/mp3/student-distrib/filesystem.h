#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include "types.h"
#include "lib.h"
#include "syscall.h"

#define MAX_FILE_BLOCKS     1023
#define LEN_BLOCK           4096
#define MAX_DENTRIES        63
#define FILE_NAME_LENGTH    32

#define MAX_INODE           64
#define MAX_DATA_BLOCK      59
#define ORIG_DENTRY_NUM     19      //must be updated every time a user program is inserted

// Represents a specific file
typedef struct {
    uint32_t file_size; // File size in bytes
    uint32_t data_blocks[MAX_FILE_BLOCKS]; // Array of data block numbers (4kB per block, adjust size as necessary)
} inode_t;

typedef struct {
    uint8_t data[LEN_BLOCK];
} data_t;

// Represents a directory entry, a single component of a path
typedef struct {
    uint8_t file_name[FILE_NAME_LENGTH]; // File name (up to 32 characters, zero-padded)
    uint32_t file_type; // File type (0: RTC, 1: Directory, 2: Regular file)    
    uint32_t inode_num; // Inode number (meaningful for regular files)
    uint8_t reserved[24]; //reserved bits
} dentry_t;


typedef struct {
    uint32_t num_entries;
    uint32_t num_inodes;
    uint32_t num_data_block;
    uint8_t reserved[52];
    dentry_t dentries[MAX_DENTRIES];
} bootBlock_t;

void init_filesys(uint32_t start_addr,uint32_t addr_end);

int32_t read_dentry_by_name (const uint8_t* fname, dentry_t* dentry);
int32_t read_dentry_by_index (uint32_t index, dentry_t* dentry);
int32_t read_data (uint32_t inode, uint32_t offset, uint8_t* buf, uint32_t length);

void clear_data_block(uint32_t idx);
int32_t free_data_block(uint32_t idx);
int32_t alloc_data_block(void);

// reads nbytes bytes of data from file into buf, uses read_data
int32_t fread (int32_t fd, void* buf, int32_t nbytes);
// do nothing, return -1
int32_t fwrite (int32_t fd, const void* buf, int32_t nbytes);
// initialize temporary structures, uses read_dentry_by_name, return 0
int32_t fopen (const uint8_t* filename);
// undo fopen, return 0
int32_t fclose (int32_t fd);

// read files filename by filename, including ".", uses read_dentry_by_index
int32_t dread (int32_t fd, void* buf, int32_t nbytes);
// do nothing, return -1
int32_t dwrite (int32_t fd, const void* buf, int32_t nbytes);
// opens a directory file, return 0
int32_t dopen (const uint8_t* filename);
// probably does nothing, return 0
int32_t dclose (int32_t fd);

// helper function for dwrite
int32_t create_new_file(int32_t fd, const void* buf);
int32_t remove_file(const void* filename);
int32_t get_dentry_index(const uint8_t* fname);
int32_t write_data(uint32_t inode, uint32_t offset, uint8_t* buf, uint32_t length);

// helper function for system call
int32_t get_file_size(int32_t inode);

#endif // FILESYSTEM_H
