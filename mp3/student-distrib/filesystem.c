// #include <stdio.h>      //this line is causing a compilation error

#include "filesystem.h" 

//create a filescope variables
// get address start
uint32_t addr_start, addr_end;
uint8_t inode_cnt = 0, db_cnt = 0;
// flag for usage: -1 can't change, 0 free, 1 in use
int8_t dir_list[MAX_DENTRIES], inode_list[MAX_INODE], db_list[MAX_DATA_BLOCK]; 

bootBlock_t* boot_ptr;

/* init_filesys_addr
 * Description: inits the file system
 * Inputs: start_addr: the address of the start of the file system link
 *          end_addr: the address of the end of the file system link
 * Outputs: None
 * Side Effects: None
 * Return Value: None
 */     
void init_filesys(uint32_t start_addr, uint32_t end_addr){
    int i, j;
    inode_t* inode_ptr;
    dentry_t dentry;

    addr_start = start_addr;
    addr_end = end_addr;        //currently the difference is 124 blocks
    boot_ptr = (bootBlock_t*)addr_start;

    // ec info
    memset(dir_list, 0, MAX_DENTRIES);
    memset(inode_list, 0, MAX_INODE);
    memset(db_list, 0, MAX_DATA_BLOCK);
    // printf("dir:%d inode:%d db:%d ", dir_cnt, inode_cnt, db_cnt);

    // init bit maps
    for( i = 0; i < boot_ptr->num_entries; i++ ){
        dentry = boot_ptr->dentries[i];
        inode_ptr = (inode_t*)(addr_start + dentry.inode_num*LEN_BLOCK + LEN_BLOCK);
        dir_list[i] = -1;
        inode_list[dentry.inode_num] = -1;
        for (j = 0; j < MAX_FILE_BLOCKS; j++){
            if (inode_ptr->data_blocks[j] == 0){
                break;
            }
            db_list[inode_ptr->data_blocks[j]] = -1;
            db_cnt++;
        }
        inode_cnt++;
    }

    printf("filesystem init done.\n");
}

/* clear_data_block
 * Description: helper function to clear data blocks
 * Inputs: idx - file block index
 * Outputs: None
 * Side Effects: fills a data block with \0
 * Return Value: None
 */  
void clear_data_block(uint32_t idx){
    if (idx >= MAX_DATA_BLOCK || idx == 0) {    //idx 0 for dir and rtc
        return;
    }
    uint32_t datablock_addr;
    data_t* block_data;
    int32_t j;
    
    datablock_addr = addr_start + boot_ptr->num_inodes * LEN_BLOCK + LEN_BLOCK + idx * LEN_BLOCK ;
    //get data_t struct
    block_data = (data_t*) datablock_addr;
    for (j = 0; j < LEN_BLOCK; j++){
        block_data->data[j] = '\0';
    }
}

/* free_data_block
 * Description: helper function to free data blocks
 * Inputs: idx - file block index
 * Outputs: None
 * Side Effects: fills a data block with \0 and sets as not used in from db_list
 * Return Value: -1 - failed to free
 *                0  - freed a block
 */  
int32_t free_data_block(uint32_t idx){
// flag for usage: -1 can't change, 0 free, 1 in use
    if (idx >= MAX_DATA_BLOCK || idx == 0) {    //idx 0 for dir and rtc
        return -1;
    }
    switch (db_list[idx])
    {
    case -1:    // cant change, return error
        return -1;
        break;
    case 0:     // already free, return 0
        // printf("already free: %d", idx);
        return 0;
        break;
    case 1:
        // printf("trying to free: %d", idx);
        clear_data_block(idx);
        db_list[idx] = 0;
        db_cnt--;
        break;
    }
    return 0;
}

/* alloc_data_block
 * Description: helper function to alloc data blocks
 * Inputs: None
 * Outputs: None
 * Side Effects: sets a datablock as used through db_list
 * Return Value: -1 - failed to allocate
 *                int32_t index of the block that was allocated
 */  
int32_t alloc_data_block(void){
    int32_t i;
    int32_t datablock = -1;
    for( i = 1; i < MAX_DATA_BLOCK ; i++ ){     //do not alloc 0 idx 0 for dir and rtc
        if( db_list[i] == 0 ){
            db_list[i] = 1;
            datablock = i;
            db_cnt++;
            break;
        }
    }    
    return datablock;
}

/* read_dentry_by_name
 * Description: given a file name, fills out the dentry pointer input with matching data
 * Inputs: fname: the file name to search for
 *          dentry: the pass by pointer of a dentry to fill out
 * Outputs: None
 * Side Effects: None
 * Return Value: Returns 0 when everything is good, returns -1 if failed
 */     
int32_t read_dentry_by_name(const uint8_t* fname, dentry_t* dentry) {
    uint8_t file_name[FILE_NAME_LENGTH], dentry_name[FILE_NAME_LENGTH];
    int i = 0, counter = 0, fname_len = 0, dentry_found_idx = -1;
    uint8_t* dname_ptr;

    //check null entries
    if (fname == 0 || dentry == 0 || *fname == '\0'){
        return -1;      //invalid ptrs
    }
    //strncmp exists in lib.c so this was unnecesarry
    // will not be changing cause its not a problem
    //clean out fname to only include word

    for(counter = 0; counter < FILE_NAME_LENGTH; counter ++){
        if (*fname == '\0') { //if first time reaching a null
            fname_len = counter;
            break;
        }
        file_name[counter] = *fname;
        fname++;
        
    }
    file_name[fname_len] = '\0';  //if end of string fill in file  name with '\0'
        
    //loop through dentry list to find one with correct name, unable to do direct comparison
    // roman is an uncommitted change smhhh
    // char dentry_name[32];
    for(i = 0; i < MAX_DENTRIES; i ++){
        //clean out dentry name and compare
        
        dname_ptr = boot_ptr->dentries[i].file_name;
        int dname_len = 0;
        for(counter = 0; counter < FILE_NAME_LENGTH; counter ++){
            if (*dname_ptr == '\0') {      //if first time reaching a null
                dname_len = counter;
                break;
            }
            dentry_name[counter] = *dname_ptr;
            dname_ptr++;
        }
        dentry_name[dname_len] = '\0';    //if end of string fill in file  name with '\0'

        //once done making strings check string similarity        
        if (fname_len == dname_len){  //check lengths match
            int truth = 0;
            for (counter = 0; counter < fname_len; counter++){    //check if chars match
                if(file_name[counter] != dentry_name[counter]){
                    truth = 1;              //if truth is 1 chars do not match
                    break;
                }
            }
            if(truth == 0){
                dentry_found_idx = i;
                // printf("found at: %d\n input name: %s output name: %s\n", dentry_found_idx, fname, boot_ptr->dentries[dentry_found_idx].file_name);
                // i += MAX_DENTRIES;
                break;
            }
            
        }
        if(dentry_found_idx != -1) break;
    }

    //set output dentry values
    //debug prints
    // printf("DEBUG: file name was: %s\n", boot_ptr->dentries[dentry_found_idx].file_name);
	// printf("DEBUG: file type was: %d\n", boot_ptr->dentries[dentry_found_idx].file_type);
	// printf("DEBUG: inode was: %d\n", boot_ptr->dentries[dentry_found_idx].inode_num);
    if (dentry_found_idx == -1) return -1;  //if not found return -1
    *dentry = boot_ptr->dentries[dentry_found_idx]; //modify dentry input
    return 0;
}

/* read_dentry_by_index
 * Description: given an index for dentry in boot block,
 *              fills out the dentry pointer input with matching data
 * Inputs: index: boot block index
 *          dentry: the pass by pointer of a dentry to fill out
 * Outputs: None
 * Side Effects: None
 * Return Value: Returns 0 when everything is good, returns -1 if failed
 */ 
int32_t read_dentry_by_index(uint32_t index, dentry_t* dentry) {
    //check null entry
    if (dentry == 0){
        return -1;
    }

    if (index < MAX_DENTRIES) {
        *dentry = boot_ptr->dentries[index];    //confirmed index by boot block
        return 0;
    }
    return -1; // Index out of bounds
}

/* read_data
 * Description: given an inode, offset, and length, output the file contents in buf
 * Inputs: inode: inode index
 *          offset: number of bytes to move ahead before starting read
 *          buf: the buffer to output file contents in
 *          length: the number of bytes to read
 * Outputs: None
 * Side Effects: None
 * Return Value: Returns number of bytes read, returns -1 if failed
 */   
int32_t read_data(uint32_t inode, uint32_t offset, uint8_t* buf, uint32_t length) {
    //null check
    uint32_t inode_addr, datablock_idx, datablock_addr;
    inode_t* inode_ptr;
    data_t* block_data;

    if(inode > boot_ptr->num_inodes) return -1;
    if(buf == 0) return -1;

    inode_addr = addr_start + inode*LEN_BLOCK + LEN_BLOCK; //start address + inode offset + bootblock offset
    inode_ptr = (inode_t*) inode_addr; //create inode

    // position check
    if(offset == inode_ptr->file_size + 1) return 0;
    if(offset > inode_ptr->file_size + 1) return -1;

    //calculate how many bytes to check
    //every 4096 bytes, move to the next block
    //end when length exhausted

    //init starting values
    int index_block = offset / LEN_BLOCK;
    int index_byte = offset - index_block*LEN_BLOCK;
    int i = 0;
    int true_length = length; 
    if (true_length > inode_ptr->file_size - offset ) {
        true_length = inode_ptr->file_size - offset + 1; //include the termination at the end
        // printf("original: %d new: %d, max length: %d", length, offset+true_length, inode_ptr->file_size);
    }

    for (i = 0; i < true_length; i++){
        if(index_byte < LEN_BLOCK){     //if index_byte is less than 4096, just read
            datablock_idx = inode_ptr->data_blocks[index_block];
            //datablock addr indexed by: start + inode * # bytes + boot block bytes + datablock_idx * # of bytes
            //currently should be the start of the block
            datablock_addr = addr_start + boot_ptr->num_inodes * LEN_BLOCK + LEN_BLOCK + datablock_idx * LEN_BLOCK ;
            //get data_t struct
            block_data = (data_t*) datablock_addr;

            *buf = block_data->data[index_byte];
            buf++;  // move buffer after write
            index_byte++;
        } else {
            index_byte = 0;
            index_block++;
            datablock_idx = inode_ptr->data_blocks[index_block];
            //datablock addr indexed by: start + inode * # bytes + boot block bytes + datablock_idx * # of bytes
            //currently should be the start of the block
            datablock_addr = addr_start + boot_ptr->num_inodes * LEN_BLOCK + LEN_BLOCK + datablock_idx * LEN_BLOCK ;
            //get data_t struct
            block_data = (data_t*) datablock_addr;

            *buf = block_data->data[index_byte];
            buf++;  // move buffer after write
            index_byte++;
        }
    }

    return true_length;
}

/* fread
 * Description: given the inode index in fd and nbytes, output the file contents in buf
 * Inputs: fd: file descriptor index
 *          buf: the buffer with file contents output
 *          nbytes: number of bytes to read
 * Outputs: None
 * Side Effects: changes file_desc_table contents
 * Return Value: Returns number of bytes read, returns -1 if failed
 */   
int32_t fread(int32_t fd, void* buf, int32_t nbytes) {
	// printf("entered fread\n");
    int32_t read_len;
    uint32_t inode_idx;
    uint32_t inode_addr;
    inode_t* inode_ptr;
    pcb_t* curr_pcb = get_cur_pcb();

    if (fd < 0 || fd > 7 || nbytes < 0) return -1; // Invalid inputs
    if (curr_pcb->fds[fd].usage_flag == -1) {
        printf("Invalid file descriptor\n");
        return -1;
    }
    if (curr_pcb->fds[fd].file_type_flag != 2) {
        printf("Invalid file type\n");
        return -1;
    }
    inode_idx = curr_pcb->fds[fd].inode;
    
    read_len = read_data(inode_idx, curr_pcb->fds[fd].file_position, (uint8_t*)buf, (uint32_t)nbytes);
    inode_addr = addr_start + inode_idx*LEN_BLOCK + LEN_BLOCK; //start address + inode offset + bootblock offset
    inode_ptr = (inode_t*) inode_addr; //create inode
    //printf("%d", &result);
    if (read_len == -1) {
        // Reading data failed.
        return -1;
    }

    curr_pcb->fds[fd].file_position += read_len;
    if(curr_pcb->fds[fd].file_position == inode_ptr->file_size){ return 0; }
    
    return read_len;
}


/* fwrite
 * Description: writes to the file system
 * Inputs: fd: inode index given through fopen
 *          buf: the buffer with file contents output
 *          nbytes: number of bytes to write
 *                  positive - append, negative - clear and write
 * Outputs: None
 * Side Effects: None
 * Return Value: returns -1
 */   
int32_t fwrite(int32_t fd, const void* buf, int32_t nbytes) {
    uint32_t inode_idx, inode_addr;
    int32_t i, write_len;
    inode_t* inode_ptr;
    pcb_t* curr_pcb = get_cur_pcb();

    // sanity check
    if ((fd < 0) || (fd > 7) || (buf == 0)) return -1; // Invalid inputs
    if (curr_pcb->fds[fd].usage_flag == -1) {
        printf("Invalid file descriptor\n");
        return -1;
    }
    if (curr_pcb->fds[fd].file_type_flag != 2) {
        printf("Invalid file type\n");
        return -1;
    }
    if( nbytes == 0 ){ return 0; } // no need to write
    
    // priviledge check
    inode_idx = curr_pcb->fds[fd].inode;  
    if( inode_list[inode_idx] == -1 ){
        printf("No file modification permissions\n");
        return -1;
    } else if ( inode_list[inode_idx] == 0 ){
        printf("File not created yet\n");
        return -1;
    }

    //get inode
    inode_addr = addr_start + curr_pcb->fds[fd].inode*LEN_BLOCK + LEN_BLOCK; //start address + inode offset + bootblock offset
    inode_ptr = (inode_t*) inode_addr; //create inode

    // parse arg and do write
    if( nbytes > 0 ){
        // append to original file
        write_len = write_data(inode_idx, inode_ptr->file_size, (uint8_t*)buf, nbytes);
    } else {
        // clear data blocks
        curr_pcb->fds[fd].file_position = 0;
        for(i = 0; i < MAX_FILE_BLOCKS; i ++){
            free_data_block(inode_ptr->data_blocks[i]);
        }
        inode_ptr->data_blocks[0] = alloc_data_block();
        inode_ptr->file_size = 0;
        write_len = write_data(inode_idx, 0, (uint8_t*)buf, -nbytes);  //changed this to no negative nbytes
    }

    if (write_len == -1) {
        // writing data failed.
        return -1;
    }

    curr_pcb->fds[fd].file_position += write_len;
    
    return write_len;
}

/* fopen
 * Description: based on the file name input, output the inode index of that file
 * Inputs: filename: the string file name to open
 * Outputs: None
 * Side Effects: None
 * Return Value: Returns file descriptor index on success, returns -1 if failed
 */   
int32_t fopen(const uint8_t* filename) {
	// functionality is already handled in syscall.c
    return 0;
}

/* fclose
 * Description: clears out item in file descriptor table
 * Inputs: fd: file descriptor index
 * Outputs: None
 * Side Effects: None
 * Return Value: returns 0 on success, returns -1 if fail
 */   
int32_t fclose(int32_t fd) {
    // functionality is already handled in syscall.c
    return 0;
}

/* dread
 * Description: prints out all the file names in the directory
 * Inputs: fd: inode index given through dopen
 *         buf: buffer to hold file names
 *         nbytes: number of bytes to read
 * Outputs: prints all the files in the directory
 * Side Effects: prints the file names out into the kernel
 * Return Value: returns -1 if a file failed to read, otherwise return 0
 */   
int32_t dread (int32_t fd, void* buf, int32_t nbytes){
    int i;
    dentry_t dentry;
    pcb_t* curr_pcb = get_cur_pcb();

    //check fd input
    if(fd < 0 || fd > 7) return -1; //fd out of range
    
    if (curr_pcb->fds[fd].usage_flag == -1) {
        printf("Invalid file descriptor\n");
        return -1;
    }
    if (curr_pcb->fds[fd].file_type_flag != 1) {
        printf("Invalid dir type\n");
        return -1;
    }

    // read and position check
    if (curr_pcb->fds[fd].file_position >= boot_ptr->num_entries) return 0; 
    if (read_dentry_by_index(curr_pcb->fds[fd].file_position, &dentry) == -1) return -1;
    curr_pcb->fds[fd].file_position++;

    for(i = 0; i < FILE_NAME_LENGTH && dentry.file_name[i]!='\0'; i++){
        ((uint8_t*)buf)[i] = dentry.file_name[i];
    }
    ((uint8_t*)buf)[i] = '\0';

    // printf("filename:%s, inode:%d\n", dentry.file_name, dentry.inode_num);

    return i; // return length of file name
}

/* dwrite
 * Description: create a new dentry or remove a dentry
 * Note: To create a new file: open arbitrary dentry to get a fd -> call dwrite
 * Inputs: fd: inode index given through fopen
 *          buf: the buffer with new file name / file name to be deleted
 *          nbytes: >= 0 to create new file, <0 to delete target file
 * Outputs: None
 * Side Effects: None
 * Return Value: 0 if success, -1 if fail
 */   
int32_t dwrite (int32_t fd, const void* buf, int32_t nbytes){
    int32_t ret;
    if ( nbytes >= 0 ){
        ret = create_new_file(fd, buf);
    } else {
        ret = remove_file(buf);
    }
    return ret;
}

/* dopen
 * Description: finds the directory file name and returns the inode index of it.
 * Inputs: filename: the name of the directory to open
 * Outputs: None
 * Side Effects: None
 * Return Value: returns -1 if failed to read, returns file descriptor index on success
 */   
int32_t dopen (const uint8_t* filename){
    // functionality is already handled in syscall.c
    return 0;
}

/* dclose
 * Description: frees file descriptor related to opened dir
 * Inputs: fd: inode index given through fopen
 * Outputs: None
 * Side Effects: changes file_desc_table
 * Return Value: returns 0
 */    
int32_t dclose (int32_t fd){
    return 0;   //success
}

/* get_file_size
 * Description: get file size of a specific inode
 * Inputs: inode_num -- unique number for a inode
 * Outputs: None
 * Side Effects: None
 * Return Value: file size in bytes
 */   
int32_t get_file_size(int32_t inode){
    uint32_t inode_addr = addr_start + inode*LEN_BLOCK + LEN_BLOCK; //start address + inode offset + bootblock offset
    inode_t* inode_ptr =  (inode_t*)inode_addr; //create inode
    return inode_ptr->file_size;
}

/* create_new_file
 * Description: create file with certain file name
 * Inputs: fname: the file name to be removed
 * Outputs: None
 * Side Effects: None
 * Return Value: -1 if failed, 0 if success
 */  
int32_t create_new_file(int32_t fd, const void* buf){
    int i;
    uint32_t inode_num, datablock;
    inode_t* inode_ptr;
    pcb_t* cur_pcb = get_cur_pcb();
    // printf("entries: %d, inode: %d, db: %d\n", boot_ptr->num_entries, inode_cnt, db_cnt); 

    // sanity check
    if( boot_ptr->num_entries >= MAX_DENTRIES || inode_cnt >= MAX_INODE || db_cnt >= MAX_DATA_BLOCK ){ 
        printf("fail\n"); 
        return -1; 
    }
    if( strlen((int8_t*)buf) > 32 ){ return -1; }
    // allocate available dentry & inode & first data block
    datablock = alloc_data_block();
    if (datablock == -1) {return -1;}   //fail to assign datablock (no space left)
    for( i = 0; i < MAX_INODE; i++ ){
        if( inode_list[i] == 0 ){
            inode_list[i] = 1;
            inode_num = i;
            inode_ptr = (inode_t*)(addr_start + inode_num*LEN_BLOCK + LEN_BLOCK);
            inode_ptr->data_blocks[0] = datablock;
            inode_ptr->file_size = 0;
            // update file descriptor
            cur_pcb->fds[fd].inode = inode_num;
            break;
        }
    }
    for( i = ORIG_DENTRY_NUM; i < MAX_DENTRIES; i++ ){
        if( dir_list[i] == 0 ){ 
            dir_list[i] = 1;
            memcpy(boot_ptr->dentries[i].file_name, buf, strlen((int8_t*)buf)); // set name
            boot_ptr->dentries[i].file_type = 2; // only support regular file for now
            boot_ptr->dentries[i].inode_num = inode_num;
            break;
        }
    }

    // update counts
    boot_ptr->num_entries++; inode_cnt++; 
    // db_cnt++; // this is alloced through alloc_data_block()
    return 0;
}

/* remove_file
 * Description: remove file with certain file name
 * Inputs: fname: the file name to be removed
 * Outputs: None
 * Side Effects: None
 * Return Value: -1 if failed, 0 if success
 */     
int32_t remove_file(const void* filename){
    int i;
    int32_t inode_num, dentry_idx;
    inode_t* inode_ptr;
    dentry_t dentry;

    // sanity check
    if( strlen((int8_t*)filename) > 32 ){ return -1; }
    dentry_idx = get_dentry_index((uint8_t*)filename);
    // printf("dent idx: %d\n", dentry_idx);
    if(dentry_idx <= ORIG_DENTRY_NUM){ return -1; }
    dentry = boot_ptr->dentries[dentry_idx];

    // free data block
    inode_num = dentry.inode_num;
    inode_ptr = (inode_t*)(addr_start + inode_num*LEN_BLOCK + LEN_BLOCK);

    for (i = 0; i < MAX_FILE_BLOCKS; i ++){
        free_data_block(inode_ptr->data_blocks[i]);
        inode_ptr->data_blocks[i] = 0;
    }
    
    // free inode
    if( inode_list[inode_num] != 1 ){ return -1; }
    inode_list[inode_num] = 0;
    inode_ptr->file_size = 0;
    inode_cnt--;

    // free dentry
    if( dir_list[dentry_idx] != 1 ){ return -1; }
    boot_ptr->num_entries--;

    // keep all dentries consecutive
    if( dentry_idx < boot_ptr->num_entries ){ 
        for( i = dentry_idx; i < boot_ptr->num_entries; i++ ){
            boot_ptr->dentries[i] = boot_ptr->dentries[i+1];
        }   
    }
    dir_list[boot_ptr->num_entries] = 0;
    memset(boot_ptr->dentries[boot_ptr->num_entries].file_name, '\0', 32); // clear file name

    return 0;
}

/* get_dentry_index
 * Description: given a file name, return its index in dentry array
 * Inputs: fname: the file name to search for
 * Outputs: None
 * Side Effects: None
 * Return Value: -1 if failed, otherwise return dentry index
 */     
int32_t get_dentry_index(const uint8_t* fname) {
    uint8_t file_name[FILE_NAME_LENGTH], dentry_name[FILE_NAME_LENGTH];
    int i = 0, counter = 0, fname_len = 0, dentry_found_idx = -1;
    uint8_t* dname_ptr;

    //check null entries
    if (fname == 0 || *fname == '\0'){ return -1; }
    // strncmp exists in lib.c so this was unnecesarry
    // will not be changing cause its not a problem
    // clean out fname to only include word

    for(counter = 0; counter < FILE_NAME_LENGTH; counter ++){
        if (*fname == '\0') { //if first time reaching a null
            fname_len = counter;
            break;
        }
        file_name[counter] = *fname;
        fname++;
        
    }
    file_name[fname_len] = '\0';  //if end of string fill in file  name with '\0'
        
    //loop through dentry list to find one with correct name, unable to do direct comparison
    // roman is an uncommitted change smhhh
    // char dentry_name[32];
    for(i = 0; i < MAX_DENTRIES; i ++){
        //clean out dentry name and compare
        
        dname_ptr = boot_ptr->dentries[i].file_name;
        int dname_len = 0;
        for(counter = 0; counter < FILE_NAME_LENGTH; counter ++){
            if (*dname_ptr == '\0') {      //if first time reaching a null
                dname_len = counter;
                break;
            }
            dentry_name[counter] = *dname_ptr;
            dname_ptr++;
        }
        dentry_name[dname_len] = '\0';    //if end of string fill in file  name with '\0'

        //once done making strings check string similarity        
        if (fname_len == dname_len){  //check lengths match
            int truth = 0;
            for (counter = 0; counter < fname_len; counter++){    //check if chars match
                if(file_name[counter] != dentry_name[counter]){
                    truth = 1;              //if truth is 1 chars do not match
                    break;
                }
            }
            if(truth == 0){
                dentry_found_idx = i;
                // printf("found at: %d\n input name: %s output name: %s\n", dentry_found_idx, fname, boot_ptr->dentries[dentry_found_idx].file_name);
                // i += MAX_DENTRIES;
                break;
            }
            
        }
        if(dentry_found_idx != -1) break;
    }
    
    return dentry_found_idx;
}

/* write_data
 * Description: given an inode, offset, and length, write buf content into file
 * Inputs: inode: inode index
 *          offset: number of bytes to move ahead before starting write
 *          buf: the buffer to input file contents in
 *          length: the number of bytes to written
 * Outputs: None
 * Side Effects: None
 * Return Value: Returns number of bytes wrote, returns -1 if failed
 */   
int32_t write_data(uint32_t inode, uint32_t offset, uint8_t* buf, uint32_t length){
    int i;
    uint32_t inode_addr, datablock_addr;
    int32_t datablock_idx;
    inode_t* inode_ptr;
    data_t* block_data;

    //null check
    if(inode > boot_ptr->num_inodes) return -1;
    if(buf == 0) return -1;

    inode_addr = addr_start + inode*LEN_BLOCK + LEN_BLOCK; //start address + inode offset + bootblock offset
    inode_ptr = (inode_t*) inode_addr; //create inode

    inode_ptr->file_size = offset + length; // off by 1 error somewhere 

    //calculate how many bytes to check
    //every 4096 bytes, move to the next block
    //end when length exhausted

    //init starting values
    int index_block = offset / LEN_BLOCK;
    int index_byte = offset - index_block*LEN_BLOCK;
    uint8_t blank = (uint8_t)'\0';
    uint8_t length_end = 0;

    for (i = 0; i <= length; i++){
        if(i == length) {
            length_end = 1;
        }
        if(index_byte < LEN_BLOCK){     //if index_byte is less than 4096, just write
            datablock_idx = inode_ptr->data_blocks[index_block];
            // check data block availablity
            if( db_list[datablock_idx] != 1 ){ return -1; }
            //datablock addr indexed by: start + inode * # bytes + boot block bytes + datablock_idx * # of bytes
            //currently should be the start of the block
            datablock_addr = addr_start + boot_ptr->num_inodes * LEN_BLOCK + LEN_BLOCK + datablock_idx * LEN_BLOCK ;
            //get data_t struct
            block_data = (data_t*) datablock_addr;

            // copy data into filesystem
            block_data->data[index_byte] = *buf;
            if (length_end){
                block_data->data[index_byte] = blank;
            }
            // printf("char in less: %c\n", block_data->data[index_byte]);
            buf++;  // move buffer after write
            index_byte++;
            // if (i == length - 1 && index_byte != 4095) {block_data->data[index_byte] = (uint8_t)"\n";} // if index byte is at last byte, do not do this
        } else {
            index_byte = 0; datablock_idx = -1;
            index_block++;
            datablock_idx = alloc_data_block();
            if( datablock_idx == -1 ){
                printf("no more available data block\n");
                return -1;
            }
            // start writing
            inode_ptr->data_blocks[index_block] = datablock_idx;
            //datablock addr indexed by: start + inode * # bytes + boot block bytes + datablock_idx * # of bytes
            //currently should be the start of the block
            datablock_addr = addr_start + boot_ptr->num_inodes * LEN_BLOCK + LEN_BLOCK + datablock_idx * LEN_BLOCK ;
            //get data_t struct
            block_data = (data_t*) datablock_addr;

            block_data->data[index_byte] = *buf;
            if (length_end){
                block_data->data[index_byte] = blank;
            }
            // printf("char in else: %c\n", block_data->data[index_byte]);
            buf++;  // move buffer after write
            index_byte++;
            // if (i+1 == length) {block_data->data[index_byte] = (uint8_t)"\n";} //safe since at new block
        }
    }


    return length;
}


