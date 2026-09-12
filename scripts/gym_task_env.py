"""Restricted task process: Landlock filesystem + user/PID/network namespaces.

No Docker daemon, mount changes, or harness plugins. A byte relay is the only
network route. The native DSH and all its shell/editor descendants share policy.
"""
import argparse
import ctypes as C
import errno
import fcntl
import json
import hashlib
import os
from pathlib import Path
import resource
import socket
import socketserver
import struct
import stat
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / '.venv-dsh/bin/python'
NATIVE = ROOT / '.venv-dsh/lib/python3.12/site-packages/deepseek_harness_runtime/runtime/deepseek-harness-sdk-runtime-linux-x64'
TASK_PY = ROOT / '.venv-swe/bin/python'
def task_uid(sandbox):
    return 1000000 + int(hashlib.sha256(str(sandbox.resolve()).encode()).hexdigest()[:8], 16) % 1000000000


def restrict(sandbox):
    """Fail closed on kernel policy errors; used before exec, with no threads."""
    libc = C.CDLL(None, use_errno=True)
    def check(rv):
        if rv < 0:
            raise OSError(C.get_errno(), os.strerror(C.get_errno()))
        return rv
    class Rules(C.Structure):
        _fields_ = [('fs', C.c_uint64)]
    class Beneath(C.Structure):
        _pack_ = 1
        _fields_ = [('access', C.c_uint64), ('fd', C.c_int32)]
    abi = check(libc.syscall(444, 0, 0, 1))
    handled = (1 << 13) - 1  # ABI 1; pathname truncate is blocked below.
    rules = Rules(handled)
    fd = check(libc.syscall(444, C.byref(rules), C.sizeof(rules), 0))
    ro = ['/usr/bin', '/usr/lib', '/usr/lib64', '/lib', '/lib64', '/bin',
          '/usr/share/zoneinfo', '/etc/ld.so.cache', '/etc/localtime', '/etc/passwd',
          '/etc/group', '/etc/nsswitch.conf', '/etc/hostname', '/etc/hosts', '/etc/ssl/openssl.cnf',
          '/proc/cpuinfo', '/proc/meminfo', str(PY.parent.parent),
          str(PY.resolve().parent.parent), str(TASK_PY.parent.parent), str(TASK_PY.resolve().parent.parent),
          str(ROOT / 'runtime/gym-baseline/bin'),
          str(ROOT / 'runtime/gym-baseline/graphviz/root'), str(ROOT / 'runtime/gym-baseline/graphviz/fonts.conf'),
          str(sandbox.parent / 'qwen.patch.yml'), str(ROOT / 'runtime/tools/swe-atomic-compat.so')]
    rw = [str(sandbox), '/dev/null', '/dev/zero', '/dev/urandom', '/dev/random',
          '/dev/ptmx', '/dev/pts']
    for path, mask in [(p, 13) for p in ro] + [(p, handled) for p in rw]:
        if not os.path.exists(path):
            continue
        if not os.path.isdir(path):
            mask &= 7
        pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        rule = Beneath(mask, pfd)
        check(libc.syscall(445, fd, 1, C.byref(rule), 0))
        os.close(pfd)
    check(libc.prctl(38, 1, 0, 0, 0))  # no_new_privs
    check(libc.syscall(446, fd, 0))
    os.close(fd)
    # Drop namespace capabilities; block pathname truncation and ownership
    # changes absent in ABI 1. chmod remains usable for atomic editor writes;
    # the unique host UID owns only this task's files, so DAC protects others.
    class CapHeader(C.Structure):
        _fields_ = [('version', C.c_uint32), ('pid', C.c_int)]
    class CapData(C.Structure):
        _fields_ = [('effective', C.c_uint32), ('permitted', C.c_uint32), ('inheritable', C.c_uint32)]
    check(libc.syscall(126, C.byref(CapHeader(0x20080522, 0)), C.byref((CapData * 2)())))
    class Filter(C.Structure):
        _fields_ = [('code', C.c_ushort), ('jt', C.c_ubyte), ('jf', C.c_ubyte), ('k', C.c_uint32)]
    class Program(C.Structure):
        _fields_ = [('len', C.c_ushort), ('filter', C.POINTER(Filter))]
    deny = 0x50000 | errno.EPERM
    ops = [(0x20, 0, 0, 4), (0x15, 1, 0, 0xC000003E), (6, 0, 0, 0x80000000),
           (0x20, 0, 0, 0)]
    # x86_64 only; block x32 ABI as well as namespace, ptrace, BPF,
    # pathname truncation/metadata changes, and async I/O syscall bypasses.
    ops += [(0x35, 0, 1, 0x40000000), (6, 0, 0, deny)]
    for nr in [76, 92, 93, 94, 101, 132, 155, 161, 165, 166, 167,
               175, 176, 246, 248, 249, 250, 251, 252, 260, 272, 280,
               298, 304, 308, 310, 311, 313, 321, 323, 425, 426, 427, 434, 438]:
        ops += [(0x15, 0, 1, nr), (6, 0, 0, deny)]
    # AF_UNIX socket creation would permit connection to host pathname
    # sockets even in a network namespace. Allow only INET sockets here.
    ops += [(0x15, 0, 5, 41), (0x20, 0, 0, 16),
            (0x15, 3, 0, socket.AF_INET), (0x15, 2, 0, socket.AF_INET6),
            (6, 0, 0, deny), (6, 0, 0, deny), (6, 0, 0, 0x7FFF0000)]
    arr = (Filter * len(ops))(*(Filter(*x) for x in ops))
    check(libc.prctl(22, 2, C.byref(Program(len(ops), arr)), 0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**3, 1024**3))


class Bridge(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


class BridgeHandler(socketserver.BaseRequestHandler):
    def handle(self):
        import select
        with socket.socket(socket.AF_UNIX) as upstream:
            upstream.connect(self.server.upstream)
            peers = [self.request, upstream]
            while True:
                readable, _, _ = select.select(peers, [], [], 65)
                if not readable:
                    return
                for src in readable:
                    data = src.recv(65536)
                    if not data:
                        return
                    peers[1 if src is peers[0] else 0].sendall(data)


class AtomicHandler(socketserver.BaseRequestHandler):
    def handle(self):
        def receive(n):
            result = b''
            while len(result) < n:
                chunk = self.request.recv(n - len(result))
                if not chunk:
                    raise OSError(errno.EIO, 'short request')
                result += chunk
            return result
        error = 0
        try:
            op, n, m = struct.unpack('!III', receive(12))
            if op not in (1, 2) or not 0 < n <= 4096 or not 0 < m <= 4096:
                raise OSError(errno.EINVAL, 'invalid publication')
            source, target = Path(os.fsdecode(receive(n))), Path(os.fsdecode(receive(m)))
            # Open directories first, then verify their resolved inode paths.
            # dir_fd operations stay confined even if the agent renames a
            # parent directory concurrently. Source must be a regular file.
            fds = []
            try:
                for path in (source.parent, target.parent):
                    fd = os.open(path, os.O_PATH | os.O_DIRECTORY)
                    fds.append(fd)
                    resolved = Path(os.readlink(f'/proc/self/fd/{fd}'))
                    if not resolved.is_relative_to(self.server.workspace):
                        raise OSError(errno.EACCES, 'outside task workspace')
                if source.name in ('', '.', '..') or target.name in ('', '.', '..'):
                    raise OSError(errno.EINVAL, 'invalid basename')
                info = os.stat(source.name, dir_fd=fds[0], follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    raise OSError(errno.EACCES, 'source must be regular')
                if op == 1:
                    os.rename(source.name, target.name, src_dir_fd=fds[0], dst_dir_fd=fds[1])
                else:
                    os.link(source.name, target.name, src_dir_fd=fds[0], dst_dir_fd=fds[1], follow_symlinks=False)
            finally:
                for fd in fds:
                    os.close(fd)
        except OSError as exc:
            error = exc.errno or errno.EIO
        self.request.sendall(struct.pack('!I', error))


def main():
    global TASK_PY
    parser = argparse.ArgumentParser()
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--sandbox', type=Path, required=True)
    parser.add_argument('--socket')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.command
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]
    sandbox = args.sandbox.resolve()
    settings_path = sandbox.parent / 'runtime-config.json'
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    TASK_PY = Path(settings.get('task_python', str(TASK_PY)))
    if not args.inside:
        # Only this task UID is used; it has no ownership in other projects.
        os.setgroups([])
        os.setgid(task_uid(sandbox))
        os.setuid(task_uid(sandbox))
        argv = ['/usr/bin/unshare', '-Urnpf', '--kill-child=KILL', str(PY),
                str(Path(__file__).resolve()), '--inside', '--sandbox', str(sandbox)]
        if args.socket:
            argv += ['--socket', args.socket]
        os.execv(argv[0], argv + ['--'] + cmd)
    # Bring up only the new namespace's loopback interface.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        fcntl.ioctl(sock, 0x8914, struct.pack('16sh', b'lo', 0x49))
    env = {'PATH': f'{ROOT}/runtime/gym-baseline/bin:{TASK_PY.parent}:/usr/bin:/bin', 'HOME': str(sandbox / 'home'),
           'TMPDIR': str(sandbox / 'tmp'), 'DSH_HOME': str(sandbox / 'dsh-home'),
           'XDG_CACHE_HOME': str(sandbox / 'cache'), 'XDG_CONFIG_HOME': str(sandbox / 'home'),
           'XDG_DATA_HOME': str(sandbox / 'home'), 'XDG_STATE_HOME': str(sandbox / 'home'),
           'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
           'CUDA_VISIBLE_DEVICES': '', 'QWEN_LOCAL_API_KEY': 'local-qwen',
           'NODE_OPTIONS': '--max-old-space-size=1024', 'TERM': 'xterm-256color',
           'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null'}
    env.update(settings.get('task_env', {}))
    bridge = None
    if args.socket:
        bridge = Bridge(('127.0.0.1', 0), BridgeHandler)
        bridge.upstream = args.socket
        env['QWEN_BASE_URL'] = f'http://127.0.0.1:{bridge.server_address[1]}/v1'
    atomic = Bridge(('127.0.0.1', 0), AtomicHandler)
    atomic.workspace = (sandbox / 'workspace').resolve()
    env['SWE_ATOMIC_PORT'] = str(atomic.server_address[1])
    env['LD_PRELOAD'] = str(ROOT / 'runtime/tools/swe-atomic-compat.so')
    proc = subprocess.Popen(cmd, cwd=sandbox / 'workspace', env=env,
                            preexec_fn=lambda: restrict(sandbox))
    threading.Thread(target=atomic.serve_forever, daemon=True).start()
    if bridge:
        threading.Thread(target=bridge.serve_forever, daemon=True).start()
    code = proc.wait()
    atomic.shutdown()
    atomic.server_close()
    if bridge:
        bridge.shutdown()
        bridge.server_close()
    sys.exit(code)


if __name__ == '__main__':
    main()
