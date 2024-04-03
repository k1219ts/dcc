
import subprocess

def convert(inputFileList, outputFileList):
    for i in range(len(inputFileList)):
        cmd = 'ffmpeg  -r 24 -i {filename} -an -vcodec '
        cmd += 'libx264 -preset slow -profile:v baseline -b 6000k '
        cmd += '-tune zerolatency -y {outFile}'
        cmd = cmd.format(filename=inputFileList[i], outFile=outputFileList[i])

        p = subprocess.Popen(cmd, shell=True)
        p.wait()

