// Loaded into the engine worker (see engine.crash_tracer): on SIGSEGV, SIGBUS or SIGABRT it writes
// the crashing thread's backtrace to $SIM_CRASH_TRACE_DIR/crash-<pid>.txt, then re-raises the
// signal so the worker still dies of it. Only the first crashing thread writes.
#define _GNU_SOURCE
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char path[600];

static void on_crash(int sig, siginfo_t* info, void* context) {
    (void)context;
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd >= 0) {
        char head[128];
        int n = snprintf(head, sizeof head, "signal %d address %p\n", sig, info->si_addr);
        if (write(fd, head, n) == n) {
            void* frames[128];
            backtrace_symbols_fd(frames, backtrace(frames, 128), fd);
        }
        close(fd);
    }
    signal(sig, SIG_DFL);
    raise(sig);
}

__attribute__((constructor)) static void install(void) {
    const char* dir = getenv("SIM_CRASH_TRACE_DIR");
    if (dir == NULL) {
        return;
    }
    snprintf(path, sizeof path, "%s/crash-%d.txt", dir, getpid());
    void* warm[1];
    backtrace(warm, 1);  // loads libgcc's unwinder now rather than inside the handler
    struct sigaction action;
    memset(&action, 0, sizeof action);
    action.sa_sigaction = on_crash;
    action.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &action, NULL);
    sigaction(SIGBUS, &action, NULL);
    sigaction(SIGABRT, &action, NULL);
}
