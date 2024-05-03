#include "tests.h"

#define INT_MAX 2147483647
#define INT_MIN	-2147483648

#define PASS 1
#define FAIL 0

/* format these macros as you see fit */
#define TEST_HEADER 	\
	printf("[TEST %s] Running %s at %s:%d\n", __FUNCTION__, __FUNCTION__, __FILE__, __LINE__)
#define TEST_OUTPUT(name, result)	\
	printf("[TEST %s] Result = %s\n", name, (result) ? "PASS" : "FAIL");

static inline void assertion_failure(){
	/* Use exception #15 for assertions, otherwise
	   reserved by Intel */
	asm volatile("int $15");
}

/* Global Variables*/
int next;

/* Checkpoint 1 tests */

/* IDT Test - Example
 * 
 * Asserts that first 10 IDT entries are not NULL
 * Inputs: None
 * Outputs: PASS/FAIL
 * Side Effects: None
 * Coverage: Load IDT, IDT definition
 * Files: x86_desc.h/S
 */
int idt_test(){
	TEST_HEADER;

	int i;
	int result = PASS;
	for (i = 0; i < 10; ++i){
		if ((idt[i].offset_15_00 == NULL) && 
			(idt[i].offset_31_16 == NULL)){
			assertion_failure();
			result = FAIL;
		}
	}

	return result;
}

// add more tests here

/* test div by 0
 * Inputs: None
 * Outputs: None
 * Side Effects: Program stucks in while(1) loop
 * Coverage: Exception 0x00
 * Files: idt.h/c
 */
void test_div_err() {	//works
	int a = 0, b = 1;
	b= b/a;

	printf("Divide by 0 test passed (this message shows only if no fault detected). \n");
}

/* test_bound_range_err
 * Inputs: None
 * Outputs: None
 * Side Effects: Program stucks in while(1) loop
 * Coverage: Exception 0x05
 * Files: idt.h/c
 */
void test_bound_range_err(){
	asm volatile("int $0x05"); // just test going to exception

	printf("bound range test passed (this message shows only if no fault detected). \n");
}

/* test_page_fault_err
 * Inputs: None
 * Outputs: None
 * Side Effects: Program stucks in while(1) loop
 * Coverage: Exception 0x0E Page Fault
 * Files: idt.h/c
 */
void test_page_fault_err(){	//works
	int testing[10];
	int i;
	int test;
	while(1){		// this loop forces this table to go past what is present in physical memory
		test = testing[i];
		i++;
	}

	printf("page fault test passed (this message shows only if no fault detected). \n");
}

/* general_protection_err
 * Inputs: None
 * Outputs: None
 * Side Effects: Program stucks in while(1) loop
 * Coverage: Exception 0x0D
 * Files: idt.h/c
 */
void general_protection_err(){	//testing page fault
	asm volatile("int $0x0D"); // just test going to exception

	printf("general protection test passed (this message shows only if no fault detected). \n");
}

/* test_inval_opcode_err
 * Inputs: None
 * Outputs: None
 * Side Effects: Program stucks in while(1) loop
 * Coverage: Exception 0x06
 * Files: idt.h/c
 */
void test_inval_opcode_err() {	//works 
	asm volatile("mov %cr7, %eax");		//recommended way to test invalid opcode https://wiki.osdev.org/Double_Fault#Exceptions 
	printf("test invalid opcode passed (this message shows only if no fault detected). \n");
}

/* test_stack_seg_err
 * Inputs: None
 * Outputs: None
 * Side Effects: Program stucks in while(1) loop
 * Coverage: Exception 0x0C
 * Files: idt.h/c
 */
void test_stack_seg_err() {
	asm volatile("int $0x0C"); // just test going to exception

	printf("test stack segment passed (this message shows only if no fault detected). \n");
}

/* test_syscall
 * Inputs: None
 * Outputs: None
 * Side Effects: Program stucks in while(1) loop
 * Coverage: Exception 0x80
 * Files: idt.h/c
 */
void test_syscall(){ asm volatile("int $0x80"); }

/* test_accessible_memory
 * Inputs: None
 * Outputs: None
 * Side Effects: Program stucks in while(1) loop
 * Coverage: Exception 0x0E, shouldn't raise exception
 * Files: idt.h/c
 */
void test_accessible_memory() {
    uint32_t* video_memory = (uint32_t*)0xB8000; // Common video memory address
    *video_memory = 0xFFFFFFFF; // Attempt to write to video memory

    uint32_t* mapped_memory = (uint32_t*)0x401000; // Just beyond 4MB
    *mapped_memory = 0x12345678; // Attempt to write

    printf("Accessible memory test passed (this message shows only if no fault occurred).\n");
}

/* test_inaccessible_memory
 * Inputs: None
 * Outputs: None
 * Side Effects: Program stucks in while(1) loop
 * Coverage: Exception 0x0E Page Fault
 * Files: idt.h/c
 */
void test_inaccessible_memory() {
    uint32_t* invalid_memory = (uint32_t*)0x80000000; // Address outside mapped regions

	// page fault expected
    uint32_t value = *invalid_memory;
	printf("%d", value);

    printf("Inaccessible memory test failed (this message should not be reachable).\n");
}


/* Checkpoint 2 tests */

/* read_data_test
 * Inputs: None
 * Outputs: None
 * Side Effects: Program can get stuck in while(1) loop
 * Coverage: read_dentry_by_name(), read_data() on multiple files
 * Files: filesystem.h/c
 */
// void read_data_test(){
// 	int32_t freq = 1;
// 	rtc_open("");
// 	rtc_write(0,&freq,4);

// 	uint8_t	idx = 0;
// 	char* name;
// 	while(1){
// 		switch (idx){
// 		case 0: name="frame0.txt"; break;
// 		case 1: name="frame1.txt"; break;
// 		case 2: name ="ls"; break;
// 		case 3: name ="verylargetextwithverylongname.txt"; break;
// 		case 4: name ="grep"; break;
// 		default: break;
// 		}
// 		dentry_t temp;
// 		clear(); set_text_top();
// 		if (read_dentry_by_name((uint8_t*)name, &temp) == -1) {
// 			printf("\nfailed to find %s\n", name);
// 			while(1);
// 		}
// 		uint32_t inode_ = temp.inode_num;
// 		uint32_t offset_ = 0;
// 		uint8_t buf_[0x2000];
// 		uint32_t length_ = sizeof(buf_)/sizeof(uint8_t);
// 		int i;
// 		for( i = 0; i < length_; i++ ){ buf_[i] = 0x00; }	// init buf

// 		if (read_data(inode_, offset_, buf_, length_) == -1) {
// 			printf("\nread data fail\n");
// 			while(1);
// 		}

// 		printf("file: %s\n",name);
// 		printf("\nbuffer reads:\n\n");
// 		for( i = 0; i <length_; i++ ){ if( buf_[i] != 0x00 ){  putc(buf_[i]); }}
		
// 		rtc_read(buf_); rtc_read(""); // display each file for 2 secs
// 		idx++;
// 		if( idx==3 ){ idx=0; }
// 	}
// }

/* read_dentry_name_test
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Coverage: read_dentry_by_name() on single file
 * Files: filesystem.h/c
 */
void read_dentry_name_test(){
	char* name = "verylargetextwithverylongname.txt";
	dentry_t temp;
	clear(); set_text_top();
	if (read_dentry_by_name((uint8_t*)name, &temp) == -1) printf("failed to find\n");
	printf("file name was: %s\n", temp.file_name);
	printf("file type was: %d\n", temp.file_type);
	printf("inode was: %d\n", temp.inode_num);
}

/* read_dentry_idx_test
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Coverage: read_dentry_by_index() on single file
 * Files: filesystem.h/c
 */
void read_dentry_idx_test(){
	dentry_t temp;	
	clear();
	if (read_dentry_by_index(10, &temp) == -1) printf("failed to find\n");
	printf("file name was: %s\n", temp.file_name);
	printf("file type was: %d\n", temp.file_type);
	printf("inode was: %d\n", temp.inode_num);
}

/* wait
 * Inputs: None
 * Outputs: None
 * Side Effects: Stuck the kernel until "next" signal is received from keyboard
 */
void wait(){
	while(!next);
	printf("\n");
	next = 0;
}

// /* test_rtc_helper
//  * Inputs: buf - buffer that should contain 4-byte integer
// 		   n_bytes - number of bytes to be passed to rtc_write
//  * Outputs: Write to screen
//  * Side Effects: Modify VRAM
//  * Descripton: Helper function for test_rtc_driver
//  */
// void test_rtc_helper(int32_t* buf, int32_t n_bytes){
// 	int i;
// 	if( rtc_write(1, buf, n_bytes) ){
// 		printf("Invalid Arguments!\n");
// 		return;
// 	}
// 	for ( i = 0; i <= (*buf)*2+4; i++ ){ // printed number is randomly chosen for better performance
// 		rtc_read(buf);	
// 		printf("1");
// 	}
// }

// /* test_rtc_driver
//  * Inputs: None
//  * Outputs: Prompts and chars shown on the screen
//  * Side Effects: Modify VRAM
//  * Coverage: RTC Driver, all four functions
//  * Files: rtc.h/c
//  */
// void test_rtc_driver(){
// 	int i;
// 	clear(); set_text_top(); printf("RTC Test.\n");
// 	// printf("Press 'r' to run.\n");
// 	rtc_open("");
// 	// init test
// 	// wait(); printf("Test rtc_open with 2Hz default frequency\n");
// 	// for( i = 0; i < 5; i++ ){ rtc_read(); test_interrupts();}
// 	// clear();
// 	// frequency test
// 	printf("Test rtc_write with frequencies from 1 to 1024 Hz\n\n");
// 	for( i = 1; i <= 1024; i = i*2){ 
// 		printf("Test Frequency: %dHz\n",i);
// 		test_rtc_helper(&i,4);
// 		putc('\n');
// 		clear(); set_text_top();
// 	}
// 	// argument test
// 	// int32_t freq = 2;
// 	// printf("Test Invalid Arguments\n");
// 	// printf("\nInvalid buf pointer = NULL "); wait(); test_rtc_helper(NULL,4);
// 	// printf("\nTest Invalid n_bytes, Passing n_bytes = 8 "); wait(); test_rtc_helper(&freq,8);
// 	// freq = 3; printf("\nTest Invalid frequency = %dHz ", freq); wait(); test_rtc_helper(&freq,4);
// 	// freq = 511; printf("\nTest Invalid frequency = %dHz ", freq); wait(); test_rtc_helper(&freq,4);
// 	rtc_close(0);
// 	clear(); set_text_top(); printf("RTC Driver tests finished\n");
// }

/* terminal_test
 * Inputs: None
 * Outputs: read from keyboard buffer and its length, then write to screen
 * Side Effects: none
 * Coverage: terminal_read, terminal_write
 * Files: terminal.c/h, keyboard.c/h
 */
void terminal_test() {
	char buf[128]; // max length 128 char
	clear(); set_text_top(); printf("Limit nbytes test\n");
	memset(buf, 0, sizeof (buf));
	printf("read bytes %d\n",terminal_read(0, buf, 10));
	printf("read from terminal:\n");
	printf("write bytes %d\n",terminal_write(0, buf, 128));
	printf("write bytes %d\n",terminal_write(0, buf, 5));
	printf("\nEcho Test\n");
	while (1) {
		memset(buf, 0, sizeof (buf));
		printf("read bytes %d\n",terminal_read(0, buf, 128));
		printf("read from terminal:\n");
		printf("write bytes %d\n",terminal_write(0, buf, 128));
		printf("\n\n");
	}
}

/* fread_test
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Coverage: fopen, fread on single file
 * Files: filesystem.h/c
 */
void fread_test(){
	uint8_t	idx = 1;
	char* name;
	switch (idx){
		case 0: name="frame0.txt"; break;
		case 1: name="frame1.txt"; break;
		case 2: name ="verylargetextwithverylongname.txt"; break;
		case 3: name ="ls"; break;
		case 4: name ="grep"; break;
		default: break;
	}
    int32_t fd = fopen((uint8_t*)name);
    if (fd == -1) {
        printf("\nfailed to open %s\n", name);
        while(1);
    }

	char buf[25000*10];
	memset(buf, 0, sizeof(buf));
    fd = fread(fd, buf, sizeof(buf));
	if (fd == -1) {
        printf("\nfailed to read %s\n", name);
        while(1);
    }

	// printf("\n\n%s", buf);
	printf("\n\nReading File: %s\n\n", name);
	int i;
	for( i = 0; i <sizeof(buf); i++ ){ if( buf[i] != 0x00 ){  putc(buf[i]); }}
	fclose(2);
	

}

/* fread_test
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Coverage: fopen, fread on single directory
 * Files: filesystem.h/c
 */
void dread_test(){
	uint8_t* name = (uint8_t*)".";
    int32_t fd = dopen(name);
    if (fd == -1) {
        printf("\nfailed to open %s\n", name);
        while(1);
    }

	char buf[25*10];
    dread(fd, buf, sizeof(buf));

	//printf(buf);
}

void mult_open_test(){
	uint8_t* name = (uint8_t*)".";
    int32_t dir_fd = dopen(name);		//1
	uint8_t* fname = (uint8_t*)"frame0.txt";
	int32_t frame0_fd = fopen(fname);	//2
	fname = (uint8_t*)"verylargetextwithverylongname.txt";
	int32_t verylarge_fd = fopen(fname);//3
	fname = (uint8_t*)"ls";
	int32_t ls_fd = fopen(fname);		//4
	fname = (uint8_t*)"grep";
	int32_t grep_fd0 = fopen(fname);	//5
	int32_t grep_fd1 = fopen(fname);	//6
	int32_t grep_fd2 = fopen(fname);	//7th one, this should ret -1

    if (grep_fd2 == -1) {
        printf("\nfailed to open (good thing) %s\n", fname);
		printf("dumping fds: dir: %d, frame: %d, very: %d, ls: %d", dir_fd, frame0_fd, verylarge_fd, ls_fd);
		printf("dumping fds: grep 0: %d, grep 1: %d, grep 2: %d", grep_fd0, grep_fd1, grep_fd2);
    }

	char buf1[25*10];
	printf("Reading Dir\n");
    dread(dir_fd, buf1, sizeof(buf1));
	dclose(dir_fd);

	char buf2[25000*10];
	memset(buf2, 0, sizeof(buf2));
	int fd = fread(ls_fd, buf2, sizeof(buf2));
	if (fd == -1) {
        printf("\nfailed to read ls");
        while(1);
    }

	// printf("\n\n%s", buf);
	printf("\n\nReading File: ls");
	int i;
	for( i = 0; i <sizeof(buf2); i++ ){ if( buf2[i] != 0x00 ){  putc(buf2[i]); }}
	fclose(ls_fd);

	//printf(buf);
}


/* Checkpoint 3 tests */
void test_base_shell_halt(){
	asm volatile ("                    \
			PUSHL	%EBX              ;\
			MOVL	$1,%EAX ;\
			MOVL	8(%ESP),%EBX      ;\
			MOVL	12(%ESP),%ECX     ;\
			MOVL	16(%ESP),%EDX     ;\
			INT	$0x80             ;\
			CMP	$0xFFFFC000,%EAX  ;\
			JBE	1f                ;\
			MOVL	$-1,%EAX	  ;\
		1:	POPL	%EBX              ;\
			RET                        \
		");
}
/* Checkpoint 4 tests */
/* Checkpoint 5 tests */
void filesys_memtest(){
	int i;
	//alloc and free stress test
    for (i = 0; i < LEN_BLOCK; i ++){
        int32_t temp = alloc_data_block();
        int32_t temp2 = alloc_data_block();
        printf("block 1: %d block 2: %d\n", temp, temp2);
        if (temp == -1 || temp2 == -1){
            printf("failed, block 1: %d block 2: %d\n", temp, temp2);
            while(1);
        }
        int32_t freeret = free_data_block(temp);
        if(freeret == -1){
            printf("ran out of space or something failed at %d\n", temp);
            while(1);
        } else {
            printf("all good");
        }
    }
}


/* Test suite entry point */
void launch_tests(){
	/* Checkpoint 3 */
    /* Run Shell */
    // clear();
    // execute((uint8_t*)"shell");

	/* Checkpoint 2 */
	// test_rtc_driver();
	// terminal_test();
	// fread_test();
	// dread_test();
	// mult_open_test();
	// read_data_test();
	// read_dentry_name_test();
	// read_dentry_idx_test();

	/* Checkpoint 1 */
	// test_div_err();
	// test_bound_range_err();
	// test_page_fault_err();
	// general_protection_err();
	// test_inval_opcode_err();
	// test_stack_seg_err();
	// test_interrupts();
	// test_syscall();
	// test_accessible_memory();
	// test_inaccessible_memory();
}
