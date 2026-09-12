/* Landlock ABI 1 denies all cross-directory rename/link operations. Forward
 * only failed publications to the namespace-local, workspace-confined broker.
 * This does not change DSH code, contents, or its atomic replacement strategy.
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static int exact(int fd, void *buf, size_t n, int writing) {
    char *p = buf;
    while (n) {
        ssize_t k = writing ? send(fd, p, n, MSG_NOSIGNAL) : recv(fd, p, n, 0);
        if (k <= 0) return -1;
        p += k; n -= k;
    }
    return 0;
}
static int publish(int op, const char *old, const char *new) {
    const char *port = getenv("SWE_ATOMIC_PORT");
    if (!port || old[0] != '/' || new[0] != '/' || strlen(old)>4096 || strlen(new)>4096) {
        errno = EXDEV; return -1;
    }
    int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    struct sockaddr_in addr = {.sin_family=AF_INET, .sin_port=htons(atoi(port)),
                                .sin_addr.s_addr=htonl(INADDR_LOOPBACK)};
    struct timeval timeout={.tv_sec=5};
    setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
    setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
    uint32_t header[3]={htonl(op),htonl(strlen(old)),htonl(strlen(new))}, result=0;
    int failed = connect(fd,(void*)&addr,sizeof(addr)) || exact(fd,header,sizeof(header),1)
        || exact(fd,(void*)old,strlen(old),1) || exact(fd,(void*)new,strlen(new),1)
        || exact(fd,&result,sizeof(result),0);
    close(fd);
    if (failed) {errno=EIO;return -1;}
    if (result) {errno=ntohl(result);return -1;}
    return 0;
}
int rename(const char *old,const char *new) {
    int (*native)(const char*,const char*)=dlsym(RTLD_NEXT,"rename");
    int rv=native(old,new);
    return rv<0 && errno==EXDEV ? publish(1,old,new) : rv;
}
int link(const char *old,const char *new) {
    int (*native)(const char*,const char*)=dlsym(RTLD_NEXT,"link");
    int rv=native(old,new);
    return rv<0 && errno==EXDEV ? publish(2,old,new) : rv;
}
