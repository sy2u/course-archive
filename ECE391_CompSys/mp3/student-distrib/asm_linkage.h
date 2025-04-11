#ifndef _ASM_LINKAGE_H
#define _ASM_LINKAGE_H

/* all error handlers*/
extern void divide_error(void);
extern void intel_reserved(void);
extern void nmi_int(void);
extern void breakpoint(void);
extern void overflow(void);
extern void bound_range(void);
extern void inval_opcode(void);
extern void device_na(void);
extern void double_fault(void);
extern void coprocessor(void);
extern void inval_tss(void);
extern void segment_na(void);
extern void segment_fault(void);
extern void general_protect(void);
extern void page_fault(void);
extern void intel_reserved_2(void);
extern void FPU_FP(void);
extern void alignment(void);
extern void machine(void);
extern void SIMD_FP(void);

/* interrupt handlers */
extern void pit_lnk(void);
extern void rtc_lnk(void);
extern void kb_lnk(void);  
extern void ms_lnk(void); 

/* system call handlers */
extern void sys_lnk(void); // args in reg, no arg on stack


#endif /*_ASM_LINKAGE_H*/
