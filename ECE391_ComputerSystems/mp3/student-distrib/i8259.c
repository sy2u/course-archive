/* i8259.c - Functions to interact with the 8259 interrupt controller
 * vim:ts=4 noexpandtab
 */

#include "i8259.h"
#include "lib.h"

/* Interrupt masks to determine which interrupts are enabled and disabled */
uint8_t master_mask; /* IRQs 0-7  */
uint8_t slave_mask;  /* IRQs 8-15 */
uint8_t a1, a2;

/* i8259_init
 * Description: Initialize the 8259 PIC
 * Inputs: None
 * Outputs: None
 * Side Effects: Modify PIC registers
 * Return Value: None
 */
void i8259_init(void) {

    // mask out all the interrupts
    master_mask = ALL_MASK;
    slave_mask = ALL_MASK;

    // store original mask of both PIC
    outb(slave_mask, SLAVE_8259_DATA);
    outb(master_mask, MASTER_8259_DATA);

    // setting master PIC
    outb(ICW1,          MASTER_8259_PORT);  // select
    outb(ICW2_MASTER,   MASTER_8259_DATA);  // mapping
    outb(ICW3_MASTER,   MASTER_8259_DATA);  // set Master/Slave
    outb(ICW4,          MASTER_8259_DATA);  // set 8086 mode

    // setting slave PIC
    outb(ICW1,          SLAVE_8259_PORT);
    outb(ICW2_SLAVE,    SLAVE_8259_DATA);
    outb(ICW3_SLAVE,    SLAVE_8259_DATA);
    outb(ICW4,          SLAVE_8259_DATA);
    
}

/* enable_irq
 * Description: Enable (unmask) the specified IRQ
 * Inputs: irq_num
 * Outputs: None
 * Side Effects: Modify PIC masking registers
 * Return Value: None
 */
void enable_irq(uint32_t irq_num) {
    if((irq_num < 0)||(irq_num > MAX_PORT_IDX)){ return; }
    if (irq_num >= MASTER_PORT_NUM){
        if(slave_mask==ALL_MASK){   // for the first irq on slave pic, enable irq2
            master_mask &= ~(1<<SLAVE_CAS_PORT);
            outb(master_mask, MASTER_8259_DATA);
        }
        slave_mask &= ~(1<<(irq_num - MASTER_PORT_NUM));
        outb(slave_mask, SLAVE_8259_DATA);
    } else {
        master_mask &= ~(1<<irq_num);
        outb(master_mask, MASTER_8259_DATA);
    }
}

/* disable_irq
 * Description: Disable (mask) the specified IRQ
 * Inputs: irq_num
 * Outputs: None
 * Side Effects: Modify PIC masking registers
 * Return Value: None
 */
void disable_irq(uint32_t irq_num) {
    if((irq_num < 0)||(irq_num > MAX_PORT_IDX)){ return; }
    if (irq_num >= MASTER_PORT_NUM){
        slave_mask |= (1<<(irq_num - MASTER_PORT_NUM));
        outb(slave_mask, SLAVE_8259_DATA);
        if(slave_mask==ALL_MASK){   // if slave pic irq are disabled, disable irq2
            master_mask |= (1<<SLAVE_CAS_PORT);
            outb(master_mask, MASTER_8259_DATA);
        }
    } else {
        master_mask |= (1<<irq_num);
        outb(master_mask, MASTER_8259_DATA);
    }
}

/* send_eoi
 * Description: Send end-of-interrupt signal for the specified IRQ
 * Inputs: irq_num
 * Outputs: None
 * Side Effects: None
 * Return Value: None
 */
void send_eoi(uint32_t irq_num) {
    if((irq_num < 0)||(irq_num > MAX_PORT_IDX)){ return; }
    if (irq_num >= MASTER_PORT_NUM){
        outb((EOI|SLAVE_CAS_PORT), MASTER_8259_PORT);
        outb((EOI|(irq_num-MASTER_PORT_NUM)), SLAVE_8259_PORT);
    } else {
        outb((EOI|irq_num), MASTER_8259_PORT);
    }
}
