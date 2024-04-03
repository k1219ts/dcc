import sys
sys.path.insert(0, '/dexter/Cache_DATA/RND/taehyung/external_module')
import pexpect

server = '220.73.45.250'
srcFile = sys.argv[1]
dstFile = sys.argv[2]



cmd = 'scp root@%s:%s %s' %(server, srcFile, dstFile)
# child = pexpect.spawn(cmd)

child = pexpect.spawn(cmd, timeout=None)
r = child.expect("root@%s's password:" % server)

if r == 0:
    child.sendline('$dexter!')
    child.logfile = sys.stdout
    child.expect(pexpect.EOF)
    print '*** outlog', child.read()

child.close()
