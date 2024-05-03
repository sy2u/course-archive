#include <stdint.h>

#include "ece391support.h"
#include "ece391syscall.h"

#define BUFSIZE 32
#define OPSIZE  5
#define PARSE_START     0
#define PARSE_CONTENT   1
#define PARSE_WAIT_1    2
#define PARSE_OP        3
#define PARSE_WAIT_2    4
#define PARSE_FILE      5

int main(){
    int32_t fd, ret, i, j, state, op, parse_flag;
    uint8_t buf[BUFSIZE], content[BUFSIZE], filename[BUFSIZE];

    for( i = 0; i < BUFSIZE; i++ ){
        content[i] = '\0';
        filename[i] = '\0';
    }

    if (0 != ece391_getargs (buf, BUFSIZE)) {
        ece391_fdputs (1, (uint8_t*)"could not read argument\n");
        return 3;
    }

    if ( 0 == ece391_strncmp((uint8_t*)"-h", buf, 2) ){
        ece391_fdputs (1, (uint8_t*)"Usage Example:\n");
        ece391_fdputs (1, (uint8_t*)"append: echo \"content\" >> new.txt\n");
        ece391_fdputs (1, (uint8_t*)"overwrite: echo \"content\" > new.txt\n");
        return 0;
    }

    // parse args
    parse_flag = 0;
    state = PARSE_START;
    for( i = 0; i < BUFSIZE; i++ ){
        if( !parse_flag ){
            switch (state) {
                case PARSE_START:
                    if( buf[i] == '\"' ){ 
                        state = PARSE_CONTENT;
                        j = 0;
                    }
                    break;
                case PARSE_CONTENT:
                    if( buf[i] == '\"' ){
                        state = PARSE_OP; 
                        op = -1;
                    } else {
                        content[j++] = buf[i];
                    }
                    break;
                case PARSE_OP:
                    if( buf[i] != ' ' ){
                        if( op < 2 ){
                            if( buf[i] == '>' ){ 
                                op++; 
                            } else {
                                if( op == -1 ){
                                    ece391_fdputs (1, (uint8_t*)"wrong sytax, use [-h] to check usage\n");
                                    return 3;
                                } else {
                                    state = PARSE_FILE;
                                    j = 0;
                                    filename[j++] = buf[i];
                                }
                            }
                        } else {
                            ece391_fdputs (1, (uint8_t*)"wrong sytax, use [-h] to check usage\n");
                            return 3;
                        }
                    }
                    break;
                case PARSE_FILE:
                    if( buf[i] == '\0' ){
                        parse_flag = 1;
                    } else {
                        filename[j++] = buf[i];
                    }
                    break;
                default: break;
            }
        } else {
            break;
        }

    }

    // check if this file exists
    if (-1 == (fd = ece391_open (filename))) {
        ece391_fdputs (1, (uint8_t*)"file not found\n");
	return 2;
    }

    // do write
    switch (op){
    case 0: ret = ece391_write(fd, content, -ece391_strlen(content)); break; // overwrite
    case 1: ret = ece391_write(fd, content, ece391_strlen(content)); break; // append
    default: break;
    }

    if( -1 == ece391_close(fd) ){
        ece391_fdputs (1, (uint8_t*)"file descriptor closing failed\n");
        return 3;
    }

    if( ret == -1 ){ 
        ece391_fdputs (1, (uint8_t*)"Data writing failed\n");
        return 3;
    } else {
        ece391_fdputs (1, (uint8_t*)"Data writing succeed\n");
        return 0;
    }
}
