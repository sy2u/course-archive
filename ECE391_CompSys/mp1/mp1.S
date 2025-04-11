//////////////////////
.data					# section declaration

        # Useful offset constants for accessing members of a 
        # struct mp1_blink_struct structure
        LOCATION   = 0    
        ON_CHAR    = 2
        OFF_CHAR   = 3 
        ON_LENGTH  = 4
        OFF_LENGTH = 6
        COUNTDOWN  = 8
        STATUS     = 10
        NEXT       = 12

        STRUCT_SIZE = 16

# Pointer to head of list (initialized to NULL)
mp1_list_head:
        .long   0

.text					# section declaration

# Export the function symbol names

.global mp1_rtc_tasklet
.global mp1_ioctl

# void mp1_poke(void);
#
# Interface: Register-based arguments (not C-style)
#    Inputs: %cl  - The byte you wish to write
#            %eax - Offset from the start of video memory that you wish
#                   to write to
#   Outputs: Text-mode video screen is written to at location %eax with
#            the byte in %cl
# Registers: Clobbers EDX
mp1_poke:
        
	movl    vmem_base_addr(,1),%edx
	movb    %cl,(%edx,%eax,1)
	ret


# void mp1_rtc_tasklet(unsigned long);
#
# Interface: C-style
#    Inputs: NO parameter used
#    Outputs: Display the image in the linked-list on terminal
#             NO return value
# Registers: callee-saved
mp1_rtc_tasklet:
        # NO error handling in this function?
        pushl   %ebp
        movl    %esp, %ebp
        pushl   %ebx                    # ebx -- countdown / status
        pushl   %ecx                    # ecx -- hold next char
        pushl   %edx                    # edx -- linked-list ptr
        pushl   %esi                    # esi -- hold next length
        movl    mp1_list_head,%edx
list_track:
        cmpl    $0,%edx                 # check list tail
        je      list_end                
        movw    COUNTDOWN(%edx),%bx
        subw    $1,%bx
        cmpw    $0,%bx                  # check countdown
        je      call_poke
        movw    %bx,COUNTDOWN(%edx)
        movl    NEXT(%edx),%edx
        jmp     list_track
call_poke:
        movw    STATUS(%edx),%bx
        movb    ON_CHAR(%edx),%cl
        movw    ON_LENGTH(%edx),%si
        cmpw    $0,%bx                  # check status
        je      char_on
        movb    OFF_CHAR(%edx),%cl
        movw    OFF_LENGTH(%edx),%si
char_on:
        xorw    $1,%bx
        movw    %bx,STATUS(%edx)        # flip status
        movw    %si,COUNTDOWN(%edx)     # update countdown
        xorl    %eax,%eax
        movw    LOCATION(%edx),%ax
        shlw    $1,%ax
        pushl   %edx
        call    mp1_poke
        popl    %edx
        movl    NEXT(%edx),%edx
        jmp     list_track
list_end:
        popl    %esi
        popl    %edx
        popl    %ecx
        popl    %ebx
        leave
        ret

# void mp1_ioctl(unsigned long arg, unsigned long cmd);
#
# Interface: C-style
#    Inputs: arg -- NOT used, reserved for following core function
#            cmd -- number used to decide which core function to jump to
#    Outputs: %eax -- 0 if success, -1 if failed
# Registers: callee-saved, only EAX modified as return value
mp1_ioctl:
        movl    8(%esp),%eax            # eax <- cmd
        cmpl    $0,%eax
        jl      ioctl_invalid
        cmpl    $3,%eax                 # 0 <= cmd <= 3
        jg      ioctl_invalid
        jmp     *jumptable(,%eax,4)
	ret
ioctl_invalid:
        xorl    %eax,%eax
        subl    $1,%eax
        ret
jumptable:
        .long   mp1_ioctl_add, mp1_ioctl_remove, mp1_ioctl_find, mp1_ioctl_sync


# void mp1_ioctl_add(unsigned long arg);
#
# Interface: C-style
#    Inputs: arg -- user-level pointer to the mp1_blink_struct structure
#    Outputs: %eax -- 0 if success, -1 if failed
# Registers: callee-saved, only EAX modified as return value
mp1_ioctl_add:
        pushl   %ebp
        movl    %esp, %ebp
        pushl   %ebx                    # ebx -- temporary use
        pushl   %ecx                    
        pushl   %edx                    
        movl    8(%ebp),%ecx            # ecx -- arg
        cmpl    $0,%ecx
        je      add_invalid             # check NULL arg && invalid location
        # call malloc
        pushl   $STRUCT_SIZE            
        call    mp1_malloc              
        addl    $4,%esp
        movl    %eax,%edx               # edx -- new allocated struct
        cmpl    $0,%edx                 
        je      add_invalid             # malloc NULL, return -1
        # call copy_from_user
        pushl   $STRUCT_SIZE             
        pushl   8(%ebp)                 # user memory struct
        pushl   %edx                    
        call    mp1_copy_from_user      # eax -- number of FAILED bits
        popl    %edx
        addl    $8,%esp
        cmpl    $0,%eax                 # more than 0 copy failed, return -1
        jne     malloc_invalid
        # check location
        movw    LOCATION(%edx),%bx
        cmpw    $0,%bx
        jl      malloc_invalid
        cmpw    $2000,%bx               # 0 <= location <= 1999
        jge     malloc_invalid
        # update infos
        movw    ON_LENGTH(%edx),%bx
        movw    %bx,COUNTDOWN(%edx)     # countdown <- on_length
        xorw    %bx,%bx
        addw    $1,%bx
        movw    %bx,STATUS(%edx)        # status <- 1
        movl    mp1_list_head,%ebx
        movl    %ebx,NEXT(%edx)         # set next pointer
        movl    %edx,mp1_list_head      # set list head
        movb    ON_CHAR(%edx),%cl       
        # call poke
        xorl    %eax,%eax
        movw    LOCATION(%edx),%ax
        shlw    $1,%ax                  # No color byte, eax <- location*2
        call    mp1_poke
        xorl    %eax,%eax               # return 0
        popl    %edx
        popl    %ecx
        popl    %ebx
        leave
	ret
malloc_invalid:
        pushl   %edx
        call    mp1_free
        addl    $4,%esp
add_invalid:
        xorl    %eax,%eax               # return -1
        subl    $1,%eax
        popl    %edx
        popl    %ecx
        popl    %ebx
        leave
        ret

# void mp1_ioctl_remove(unsigned long arg);
#
# Interface: C-style
#    Inputs: arg -- the location to be removed
#    Outputs: %eax -- 0 if success, -1 if failed
# Registers: callee-saved, only EAX modified as return value
mp1_ioctl_remove:
        pushl   %ebp
        movl    %esp, %ebp
        pushl   %ebx                    
        pushl   %ecx                    
        pushl   %edx
        movl    8(%ebp),%ebx
        call    linked_list_search      # eax -- struct to be removed
        cmpl    $0,%eax
        je      remove_fail
        movl    mp1_list_head,%ecx      
        cmpl    %ecx,%eax               # check list head
        jne     remove_find_previous
        movl    NEXT(%eax),%ebx         # ebx -- next node of removed struct
        movl    %ebx,mp1_list_head
        jmp     remove_free
remove_find_previous:                   # since node is found, NULL check is not neccessary
        cmpl    NEXT(%ecx),%eax         # ecx -- previous node of removed struct
        je      remove_link
        movl    NEXT(%ecx),%ecx
        cmpl    $0,%ecx
        je      remove_fail
        jmp     remove_find_previous
remove_link:
        movl    NEXT(%eax),%ebx
        movl    %ebx,NEXT(%ecx)
remove_free:
        pushl   %eax
        call    mp1_free
        addl    $4,%esp
        xorl    %eax,%eax               # return 0
        popl    %edx
        popl    %ecx
        popl    %ebx
        leave        
	ret
remove_fail:
        xorl    %eax,%eax               # return -1
        subl    $1,%eax
        popl    %edx
        popl    %ecx
        popl    %ebx
        leave        
	ret        


# void mp1_ioctl_find(unsigned long arg);
#
# Interface: C-style
#    Inputs: arg -- location to be found
#    Outputs: %eax -- 0 if success, -1 if failed
# Registers: callee-saved, only EAX modified as return value
mp1_ioctl_find:
        pushl   %ebp
        movl    %esp, %ebp
        pushl   %ebx
        pushl   %ecx
        pushl   %edx
	movl    8(%ebp),%edx            # edx -- user level struct
        # check NULL ptr
        cmpl    $0,%edx                
        je      find_fail
        # malloc
        pushl   %edx
        pushl   $STRUCT_SIZE
        call    mp1_malloc
        addl    $4,%esp
        popl    %edx
        cmpl    $0,%eax
        je      find_fail
        # copy from user
        pushl   $STRUCT_SIZE
        pushl   %edx
        pushl   %eax
        call    mp1_copy_from_user
        popl    %ecx                    # ecx -- kernel level struct
        popl    %edx
        addl    $4,%esp
        cmpl    $0,%eax
        jne     find_malloc_fail
        # extract location
        movw    LOCATION(%ecx),%bx      # bx -- target location   
        pushw   %bx
        pushl   %edx
        # free kernel copied struct
        pushl   %ecx
        call    mp1_free                
        addl    $4,%esp
        popl    %edx
        popw    %bx
        # find struct
        call    linked_list_search
        cmpl    $0,%eax                 # eax -- kernel-level target ptr
        je      find_fail
        # copy to user level
        pushl   $STRUCT_SIZE
        pushl   %eax
        pushl   %edx
        call    mp1_copy_to_user
        popl    %edx
        addl    $8,%esp
        cmpl    $0,%eax
        jne     find_fail
        # end
        xorl    %eax,%eax               # return 0
        popl    %edx
        popl    %ecx
        popl    %ebx
        leave
        ret
find_malloc_fail:
        pushl   %ecx
        call    mp1_free
        addl    $4,%esp
find_fail:
        xorl    %eax,%eax               # return -1
        subl    $1,%eax
        popl    %edx
        popl    %ecx
        popl    %ebx
        leave
        ret


# void mp1_ioctl_sync(unsigned long arg);
#
# Interface: C-style
#    Inputs: arg with upper 16bits repersenting location-1, lower 16bits repersenting location-2
#    Outputs: %eax -- 0 if success, -1 if failed
# Registers: callee-saved, only EAX modified as return value
mp1_ioctl_sync:
        pushl   %ebp
        movl    %esp, %ebp
        pushl   %ebx                    # bx -- first location / temporary use
        pushl   %ecx                    # cx -- second location
        pushl   %edx
        movl    8(%ebp),%ebx            # load arg
        movl    %ebx,%ecx
        shr     $16,%ebx                # store upper 16 bits (location 1)
        call    linked_list_search
        cmpl    $0,%eax
        je      sync_fail
        movl    %eax,%edx               # edx -- first struct pointer
        movw    %cx,%bx
        call    linked_list_search      # eax -- second struct pointer
        cmpl    $0,%eax
        je      sync_fail          
        movw    ON_LENGTH(%edx),%bx
        movw    %bx,ON_LENGTH(%eax)
        movw    OFF_LENGTH(%edx),%bx
        movw    %bx,OFF_LENGTH(%eax)
        movw    COUNTDOWN(%edx),%bx
        movw    %bx,COUNTDOWN(%eax)
        movw    STATUS(%edx),%bx
        movw    %bx,STATUS(%eax)
        movb    ON_CHAR(%eax),%cl
        cmpw    $0,%bx                  # status = 0
        jne     sync_poke
        movb    OFF_CHAR(%eax),%cl
sync_poke:
        movl    %eax,%ebx
        xorl    %eax,%eax
        movw    LOCATION(%ebx),%ax
        shlw    $1,%ax
        call    mp1_poke
        xorl    %eax,%eax               # return 0
        popl    %edx
        popl    %ecx
        popl    %ebx
        leave
	ret
sync_fail:
        xorl    %eax,%eax
        subl    $1,%eax
        popl    %edx
        popl    %ecx
        popl    %ebx
        leave
	ret        


# void linked_list_search(unsigned short location);
#
# Interface: Register-based arguments (not C-style)
#    Inputs: %bx  - The location to be found
#    Outputs: %eax - pointer to the structure, NULL if not found
# Registers: Clobbers EAX
linked_list_search:
        pushl   %ebp
        movl    %esp, %ebp
        pushl   %ecx
        cmpw    $0,%bx                  # check invalid location
        jl      search_fail
        cmpw    $2000,%bx
        jge     search_fail
        movl    mp1_list_head,%eax      # eax -- structure ptr
search_loop:
        cmpl    $0,%eax                 # check NULL ptr
        je      search_fail
        movw    LOCATION(%eax),%cx      # cx -- current struct location
        cmpw    %cx,%bx
        je      search_found
        movl    NEXT(%eax),%eax
        jmp     search_loop
search_fail:
        xorl    %eax,%eax               # return NULL
search_found:
        popl    %ecx
        leave
        ret


.end
