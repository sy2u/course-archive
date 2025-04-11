/* store error codes */
#define DIVIDE_ERROR_CODE       0
#define INTEL_RESERVE_CODE      1
#define NMI_INTERUPT_CODE       2
#define BREAKPOINT_CODE         3
#define OVERFLOW_CODE           4
#define BOUND_RANGE_CODE        5
#define INVALID_OPCODE_CODE     6
#define DEVICE_NA_CODE          7
#define DOUBLE_FAULT_CODE       8
#define COPROCESSOR_CODE        9
#define INVAILD_TSS_CODE        10
#define SEGMENT_NA_CODE         11
#define SEGMENT_FAULT_CODE      12
#define GENERAL_PROTECT_CODE    13
#define PAGE_FAULT_CODE         14
#define INTEL_RESERVE2_CODE     15
#define FPU_FP_ERROR_CODE       16
#define ALIGNMENT_ERROR_CODE    17
#define MACHINE_ERROR_CODE      18
#define SIMD_FP_ERROR_CODE      19

/* write a macro to handle all the exceptions*/
/*   copy the template from ece391syscall.S */

#define exception_link(function_name, vector) \
    .GLOBL function_name                    ;\
    function_name:                          ;\
        PUSHAL                              ;\
        PUSHFL                              ;\
        PUSHL $vector                       ;\
        CALL exception_handler              ;\
        ADDL $4, %esp                       ;\
        POPFL                               ;\
        POPAL                               ;\
        IRET                                


#define exception_link_err(function_name, vector)\
    .GLOBL function_name                        ;\
    function_name:                              ;\
        PUSHAL                                  ;\
        PUSHFL                                  ;\
        PUSHL $vector                           ;\
        CALL exception_handler                  ;\
        ADDL $8, %esp                           ;\
        POPFL                                   ;\
        POPAL                                   ;\
        IRET                                

;/* exception call wrappers */
exception_link(divide_error, DIVIDE_ERROR_CODE);
exception_link(intel_reserved, INTEL_RESERVE_CODE);
exception_link(nmi_int, NMI_INTERUPT_CODE);
exception_link(breakpoint, BREAKPOINT_CODE);
exception_link(overflow, OVERFLOW_CODE);
exception_link(bound_range, BOUND_RANGE_CODE);
exception_link(inval_opcode, INVALID_OPCODE_CODE);
exception_link(device_na, DEVICE_NA_CODE);
exception_link_err(double_fault, DOUBLE_FAULT_CODE);
exception_link(coprocessor, COPROCESSOR_CODE);
exception_link_err(inval_tss, INVAILD_TSS_CODE);
exception_link_err(segment_na, SEGMENT_NA_CODE);
exception_link_err(segment_fault, SEGMENT_FAULT_CODE);
exception_link_err(general_protect, GENERAL_PROTECT_CODE);
exception_link_err(page_fault, PAGE_FAULT_CODE);
exception_link(intel_reserved_2, INTEL_RESERVE2_CODE);
exception_link(FPU_FP, FPU_FP_ERROR_CODE);
exception_link_err(alignment, ALIGNMENT_ERROR_CODE);
exception_link(machine, MACHINE_ERROR_CODE);
exception_link(SIMD_FP, SIMD_FP_ERROR_CODE);



/* Interrupt Handler */
#define int_link(function_name, handler, IRQ)    \
    .GLOBL function_name                            ;\
    function_name:                                  ;\
        PUSHAL                                      ;\
        PUSHFL                                      ;\
        PUSHL $IRQ                                  ;\
        CALL handler                                ;\
        ADDL $4, %esp                               ;\
        POPFL                                       ;\
        POPAL                                       ;\
        STI                                         ;\
        IRET                                

/* interrupt call wrappers */
int_link(pit_lnk, pit_handler, 0x20);
int_link(rtc_lnk, rtc_hdl, 0x28);
int_link(kb_lnk, keyboard_handler, 0x21);
int_link(ms_lnk, mouse_handler, 0x2C);


/* System Call Handler */
.GLOBL sys_lnk                            
sys_lnk:            
    # eax shouldn't be saved, so don't use pushal
    CLI
    PUSHL       %edi
    PUSHL       %esi
    PUSHL       %ecx
    PUSHL       %edx
    PUSHL       %ebx
    PUSHFL
    # check eax parameter                                      
    CMPL        $1, %eax
    JL          invalid
    CMPL        $12, %eax
    JG          invalid
    # parameters
    PUSHL       %edx
    PUSHL       %ecx
    PUSHL       %ebx
    STI
    # jump to corresponding system call
    CALL         *jumptable(, %eax, 4)
    # end of exec, pop all parameters
    CLI
    ADDL        $12, %esp
    POPFL      
    POPL        %ebx                                 
    POPL        %edx
    POPL        %ecx
    POPL        %esi
    POPL        %edi
    STI
    IRET
invalid:
    POPFL                                       
    POPL    %ebx                                 
    POPL    %edx
    POPL    %ecx
    POPL    %esi
    POPL    %edi
    MOVL    $-1, %eax
    STI
    IRET 
jumptable:
    .long   0           # place holder
    .long   halt
    .long   execute
    .long   read
    .long   write
    .long   open
    .long   close
    .long   getargs
    .long   vidmap
    .long   set_handler
    .long   sigreturn
    .long   reboot
    .long   shutdown
