#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#include "picat.h"   // Picat 提供的 C API

// 创建 TCP socket
int picat_socket() {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    return sockfd;  // 返回 socket 描述符
}

// 连接服务器
int picat_connect(int sockfd, const char* host, int port) {
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
    if(inet_pton(AF_INET, host, &serv_addr.sin_addr) <= 0) return -1;

    return connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
}

// 发送数据
int picat_send(int sockfd, const char* msg) {
    return send(sockfd, msg, strlen(msg), 0);
}

// 接收数据
int picat_recv(int sockfd, char* buffer, int size) {
    int n = recv(sockfd, buffer, size - 1, 0);
    if (n >= 0) buffer[n] = '\0';
    return n;
}

// 关闭 socket
void picat_close(int sockfd) {
    close(sockfd);
}
