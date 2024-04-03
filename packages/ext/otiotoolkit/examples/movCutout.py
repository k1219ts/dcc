# import opentimelineio as otio
import os
import subprocess


currentDir = os.path.dirname(__file__)
srcMov = os.path.join(currentDir, 'data', 'Mogadishu_D_2_200624_5th_F_Apple_ProRes_422_Proxy.mov')
outDir = os.path.join(currentDir, 'data', '_shot_mov_')
movName = 'ABC_0010'
movRationalTime = (432484, 148) # start Time, duration

# first mov to jpg using rvio
startFrame = movRationalTime[0]
duration = movRationalTime[1]

command = ['DCC.server', 'rez-env', 'rv-7.3.4', '--', 'rvio']
command += ['-v', srcMov]
command += ['-t', '%s-%s' % (startFrame, startFrame + duration)]
command += ['-in709', '-outsrgb']

outputJpgName = os.path.join(outDir, movName, '%s.#.jpg' % movName)
if not os.path.exists(os.path.dirname(outputJpgName)):
    os.makedirs(os.path.dirname(outputJpgName))
command += ['-o', outputJpgName]

command = ' '.join(command)
print command
proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

while proc.poll() == None:
    output = proc.stdout.readline()
    if output:
        print output.strip()

# Second burn in using nuke
command = ['DCC.server', 'rez-env', 'nuke-10.0v4', '--', 'nukeX']
command += ['-i', '-t', '-X', 'Write1']
# burnIn.py --jpgdir ${CUROUT_JPG_DIR) --shotname
command += [os.path.join(currentDir, 'burnIn.py'), ] # Python Script
command += ['--jpgdir', os.path.dirname(outputJpgName)]
command += ['--color', '1', '0', '1', '1']
command += ['--textpos', 'RT']

command = ' '.join(command)
print command
proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

while proc.poll() == None:
    output = proc.stdout.readline()
    if output:
        print output.strip()


# third, burn in to mov
burnInDir = os.path.join(os.path.dirname(outputJpgName), 'burnin')
command = ['DCC.server', 'rez-env', 'rv-7.3.4', '--']
command += ['rvio', '-v', '-fps', '23.976']
command += [os.path.join(burnInDir, '%s.#.jpg' % movName), '-o', os.path.join(outDir, '%s.mov' % movName)]

command = ' '.join(command)
print command
proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

while proc.poll() == None:
    output = proc.stdout.readline()
    if output:
        print output.strip()

os.system('rm -rf %s' % os.path.dirname(outputJpgName))