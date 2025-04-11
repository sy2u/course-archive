
.globl enable_paging
.globl flush_tlb

.align 4
enable_paging:
    # load page directory
    movl 4(%esp), %eax
    movl %eax, %cr3
    # enable 4MB page
    movl %cr4, %eax
    orl  $0x00000010, %eax   # set PSE
    movl %eax, %cr4
    # set paging flag
    movl %cr0, %eax          # Copy CR0 into EAX
    orl  $0x80000000, %eax   # Set the paging bit
    movl %eax, %cr0          # Write back to CR0
    # flash TLB
    movl %cr3, %eax 
	movl %eax, %cr3
    ret

flush_tlb:
    movl %cr3, %eax 
	movl %eax, %cr3
    ret
