// fast_ml_client.c
// 高性能 ML 客户端

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <arpa/inet.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("0.0");
        return 0;
    }
    
    // 连接 Unix Socket
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        printf("0.0");
        return 0;
    }
    
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, "/tmp/ml_picat.sock");
    
    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock);
        printf("0.0");
        return 0;
    }
    
    // 发送请求
    const char* request = argv[1];
    size_t req_len = strlen(request);
    uint32_t net_len = htonl((uint32_t)req_len);
    
    if (send(sock, &net_len, 4, 0) != 4) {
        close(sock);
        printf("0.0");
        return 0;
    }
    
    if (send(sock, request, req_len, 0) != (ssize_t)req_len) {
        close(sock);
        printf("0.0");
        return 0;
    }
    
    // 接收响应
    uint32_t resp_net_len;
    if (recv(sock, &resp_net_len, 4, MSG_WAITALL) != 4) {
        close(sock);
        printf("0.0");
        return 0;
    }
    
    uint32_t resp_len = ntohl(resp_net_len);
    if (resp_len > 65536) {
        close(sock);
        printf("0.0");
        return 0;
    }
    
    char* response = malloc(resp_len + 1);
    if (!response) {
        close(sock);
        printf("0.0");
        return 0;
    }
    
    if (recv(sock, response, resp_len, MSG_WAITALL) != (ssize_t)resp_len) {
        free(response);
        close(sock);
        printf("0.0");
        return 0;
    }
    response[resp_len] = '\0';
    
    // 提取 "result" 的值
    char* result_pos = strstr(response, "\"result\":");
    if (result_pos) {
        result_pos += 9;
        while (*result_pos == ' ' || *result_pos == ':') {
            result_pos++;
        }
        // 输出数字（不要换行）
        while (*result_pos && 
               (*result_pos == '-' || *result_pos == '.' || 
                (*result_pos >= '0' && *result_pos <= '9') ||
                *result_pos == 'e' || *result_pos == 'E')) {
            putchar(*result_pos++);
        }
    } else {
        printf("0.0");
    }
    
    free(response);
    close(sock);
    return 0;
}
