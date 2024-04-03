import subprocess
import os


ocioConfig = os.path.join(os.getenv('REZ_OCIO_CONFIGS_ROOT'), 'config.ocio')

# ACES - ACEScg -> aces REC.709
cmdRule = 'oiiotool {INPUT_RULE} --colorconfig {OCIO_CONFIG} --colorconvert "ACES - ACEScg" "Output - Rec.709" -o {OUTPUT_RULE}'

def ExrToJpg(input):
    output = input.replace(".exr", ".jpg")
    cmd = cmdRule.format(INPUT_RULE=input, OCIO_CONFIG=ocioConfig, OUTPUT_RULE=output)
    subprocess.Popen(cmd, shell=True).wait()
