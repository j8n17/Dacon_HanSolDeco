import subprocess
import sys

def run_command(command, shell=False, background=False):
    print("Executing command:", command)
    if background:
        subprocess.Popen(command, shell=shell)
    else:
        subprocess.check_call(command, shell=shell)

def main():
    # 1. sudo apt-get install -y pciutils
    run_command(["sudo", "apt-get", "install", "-y", "pciutils"])
    
    # 2. curl https://ollama.ai/install.sh | sh
    run_command("curl https://ollama.ai/install.sh | sh", shell=True)
    # 3. nohup ollama serve & (백그라운드 실행)
    run_command("nohup ollama serve &", shell=True, background=True)
    run_command('ollama pull gemma3:4b', shell=True)
    run_command('ollama cp gemma3:4b gemma3:4bcp', shell=True)

    # 4. requirements.txt 설치
    run_command([sys.executable, "-m", "pip", "-q", "install", "-r", "requirements.txt"])

if __name__ == "__main__":
    main()