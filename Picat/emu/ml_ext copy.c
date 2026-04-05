/********************************************************************
 * ml_ext.c
 * ML Socket Extension for Picat - Graceful Network Failure
 * Author: Beza (2025) 空历史可用

 ********************************************************************/

#include "bprolog.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <arpa/inet.h>

// Global state
static int ml_socket_fd = -1;
static char ml_socket_path[256] = "/tmp/ml_picat.sock";

// Connect to ML daemon
static int ml_ensure_connection() {
    struct sockaddr_un addr;
    
    if (ml_socket_fd >= 0) return 0;
    
    ml_socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (ml_socket_fd < 0) return -1;
    
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, ml_socket_path, sizeof(addr.sun_path) - 1);
    
    if (connect(ml_socket_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(ml_socket_fd);
        ml_socket_fd = -1;
        return -1;
    }
    
    return 0;
}

// Send request and get response
static int ml_send_request_full(const char* json_request, char* response_buffer, size_t buffer_size) {
    size_t msg_len;
    uint32_t net_len;
    uint32_t resp_net_len;
    uint32_t resp_len;
    
    if (ml_ensure_connection() < 0) return -1;
    
    msg_len = strlen(json_request);
    net_len = htonl((uint32_t)msg_len);
    
    if (send(ml_socket_fd, &net_len, 4, 0) != 4) goto error;
    if (send(ml_socket_fd, json_request, msg_len, 0) != (ssize_t)msg_len) goto error;
    
    if (recv(ml_socket_fd, &resp_net_len, 4, MSG_WAITALL) != 4) goto error;
    
    resp_len = ntohl(resp_net_len);
    if (resp_len > buffer_size - 1) goto error;
    
    if (recv(ml_socket_fd, response_buffer, resp_len, MSG_WAITALL) != (ssize_t)resp_len) goto error;
    response_buffer[resp_len] = '\0';
    
    return 0;

error:
    close(ml_socket_fd);
    ml_socket_fd = -1;
    return -1;
}

// Convert history equations to JSON
static void ml_history_to_json(BPLONG history_list, char* buffer, int size) {
    BPLONG_PTR top;
    int pos;
    int first;
    BPLONG head;
    BPLONG_PTR struct_ptr;
    SYM_REC_PTR sym_ptr;
    int arity;
    BPLONG mult_val, shift_val, op_val, left_val, right_val;
    
    pos = 0;
    buffer[pos++] = '[';
    
    first = 1;
    while (ISLIST(history_list)) {
        head = bp_get_car(history_list);
        DEREF(head);
        
        if (!first && pos < size - 1) buffer[pos++] = ',';
        first = 0;
        
        if (ISSTRUCT(head)) {
            struct_ptr = (BPLONG_PTR)UNTAGGED_ADDR(head);
            sym_ptr = (SYM_REC_PTR)FOLLOW(struct_ptr);
            arity = GET_ARITY(sym_ptr);
            
            if (arity == 5 && pos < size - 100) {
                mult_val = FOLLOW(struct_ptr + 1);
                shift_val = FOLLOW(struct_ptr + 2);
                op_val = FOLLOW(struct_ptr + 3);
                left_val = FOLLOW(struct_ptr + 4);
                right_val = FOLLOW(struct_ptr + 5);
                
                DEREF(mult_val);
                DEREF(shift_val);
                DEREF(op_val);
                DEREF(left_val);
                DEREF(right_val);
                
                pos += snprintf(buffer + pos, size - pos,
                    "{\"mult\":%ld,\"shift\":%ld,\"op\":%ld,\"left\":%ld,\"right\":%ld}",
                    ISINT(mult_val) ? INTVAL(mult_val) : 0,
                    ISINT(shift_val) ? INTVAL(shift_val) : 0,
                    ISINT(op_val) ? INTVAL(op_val) : 0,
                    ISINT(left_val) ? INTVAL(left_val) : -1,
                    ISINT(right_val) ? INTVAL(right_val) : -1
                );
            }
        }
        
        history_list = bp_get_cdr(history_list);
    }
    
    if (pos < size - 1) buffer[pos++] = ']';
    buffer[pos] = '\0';
}

// ========== Picat Callable C Functions ==========

// c_ML_PREDICT(+Target, +CurrentMult, +History, -Result)
int c_ML_PREDICT() {
    BPLONG target_arg, current_mult_arg, history_arg, result_var;
    BPLONG_PTR top;
    long target, current_mult;
    char history_json[8192];
    char request_json[10240];
    char response[16384];
    int op_predicted;
    char* result_pos;
    
    target_arg = bp_get_call_arg(1, 4);
    current_mult_arg = bp_get_call_arg(2, 4);
    history_arg = bp_get_call_arg(3, 4);
    result_var = bp_get_call_arg(4, 4);
    
    DEREF(target_arg);
    DEREF(current_mult_arg);
    DEREF(history_arg);
    
    if (!ISINT(target_arg) || !ISINT(current_mult_arg)) {
        return BP_FALSE;
    }
    
    target = INTVAL(target_arg);
    current_mult = INTVAL(current_mult_arg);
    
    /* Convert history to JSON */
    if (ISLIST(history_arg) || ISNIL(history_arg)) {
        ml_history_to_json(history_arg, history_json, sizeof(history_json));
    } else {
        strcpy(history_json, "[]");
    }
    
    snprintf(request_json, sizeof(request_json),
        "{\"target\":%ld,\"current_mult\":%ld,\"history\":%s}",
        target, current_mult, history_json);
    
    /* Send request - if fails, return default */
    if (ml_send_request_full(request_json, response, sizeof(response)) < 0) {
        return bp_unify(result_var, MAKEINT(3));
    }
    
    /* Parse response */
    op_predicted = 3;
    result_pos = strstr(response, "\"op_predicted\":");
    if (result_pos) {
        result_pos += 15;
        while (*result_pos == ' ' || *result_pos == ':') result_pos++;
        op_predicted = atoi(result_pos);
    }
    
    return bp_unify(result_var, MAKEINT(op_predicted));
}

// c_ML_PREDICT_DETAILED(+Target, +CurrentMult, +History, -Op, -Probs)
int c_ML_PREDICT_DETAILED() {
    BPLONG target_arg, current_mult_arg, history_arg, op_var, probs_var;
    BPLONG_PTR top;
    long target, current_mult;
    char history_json[8192];
    char request_json[10240];
    char response[16384];
    int op_predicted;
    char* op_pos;
    double probs[4];
    char* probs_pos;
    int i;
    char search_key[32];
    char* prob_pos;
    BPLONG list, elem;
    
    target_arg = bp_get_call_arg(1, 5);
    current_mult_arg = bp_get_call_arg(2, 5);
    history_arg = bp_get_call_arg(3, 5);
    op_var = bp_get_call_arg(4, 5);
    probs_var = bp_get_call_arg(5, 5);
    
    DEREF(target_arg);
    DEREF(current_mult_arg);
    DEREF(history_arg);
    
    if (!ISINT(target_arg) || !ISINT(current_mult_arg)) {
        return BP_FALSE;
    }
    
    target = INTVAL(target_arg);
    current_mult = INTVAL(current_mult_arg);
    
    /* Convert history to JSON */
    if (ISLIST(history_arg) || ISNIL(history_arg)) {
        ml_history_to_json(history_arg, history_json, sizeof(history_json));
    } else {
        strcpy(history_json, "[]");
    }
    
    snprintf(request_json, sizeof(request_json),
        "{\"target\":%ld,\"current_mult\":%ld,\"history\":%s}",
        target, current_mult, history_json);
    
    /* Send request - if fails, return defaults */
    if (ml_send_request_full(request_json, response, sizeof(response)) < 0) {
        /* Build default list [0.0, 0.0, 0.0, 1.0] */
        list = bp_build_nil();
        for (i = 3; i >= 0; i--) {
            elem = bp_build_list();
            bp_unify(bp_get_car(elem), bp_build_float(i == 3 ? 1.0 : 0.0));
            bp_unify(bp_get_cdr(elem), list);
            list = elem;
        }
        bp_unify(op_var, MAKEINT(3));
        bp_unify(probs_var, list);
        return BP_TRUE;
    }
    
    /* Parse op_predicted */
    op_predicted = 3;
    op_pos = strstr(response, "\"op_predicted\":");
    if (op_pos) {
        op_pos += 15;
        while (*op_pos == ' ' || *op_pos == ':') op_pos++;
        op_predicted = atoi(op_pos);
    }
    
    /* Parse probabilities */
    probs[0] = 0.0;
    probs[1] = 0.0;
    probs[2] = 0.0;
    probs[3] = 1.0;
    
    probs_pos = strstr(response, "\"op_probabilities\":");
    if (probs_pos) {
        for (i = 0; i < 4; i++) {
            snprintf(search_key, sizeof(search_key), "\"%d\":", i);
            prob_pos = strstr(probs_pos, search_key);
            if (prob_pos) {
                prob_pos += strlen(search_key);
                while (*prob_pos == ' ') prob_pos++;
                probs[i] = atof(prob_pos);
            }
        }
    }
    
    /* Build probability list */
    list = bp_build_nil();
    for (i = 3; i >= 0; i--) {
        elem = bp_build_list();
        bp_unify(bp_get_car(elem), bp_build_float(probs[i]));
        bp_unify(bp_get_cdr(elem), list);
        list = elem;
    }
    
    if (!bp_unify(op_var, MAKEINT(op_predicted))) return BP_FALSE;
    if (!bp_unify(probs_var, list)) return BP_FALSE;
    
    return BP_TRUE;
}

// c_ML_CONNECT(+Path)
int c_ML_CONNECT() {
    BPLONG path_arg = bp_get_call_arg(1, 1);
    BPLONG_PTR top;
    
    DEREF(path_arg);
    if (ISATOM(path_arg)) {
        char* path_str = bp_get_name(path_arg);
        strncpy(ml_socket_path, path_str, sizeof(ml_socket_path) - 1);
        ml_socket_path[sizeof(ml_socket_path) - 1] = '\0';
    } else {
        return BP_FALSE;
    }
    
    if (ml_socket_fd >= 0) {
        close(ml_socket_fd);
        ml_socket_fd = -1;
    }
    
    return BP_TRUE;
}

// c_ML_DISCONNECT()
int c_ML_DISCONNECT() {
    if (ml_socket_fd >= 0) {
        close(ml_socket_fd);
        ml_socket_fd = -1;
    }
    return BP_TRUE;
}