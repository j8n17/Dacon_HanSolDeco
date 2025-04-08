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
    
    # 4. pip -q install langchain-community
    run_command([sys.executable, "-m", "pip", "-q", "install", "langchain-community"])
    
    # 6. pip -q install langchain-ollama
    run_command([sys.executable, "-m", "pip", "-q", "install", "langchain-ollama"])
    
    # 7. pip -q install faiss-cpu
    run_command([sys.executable, "-m", "pip", "-q", "install", "faiss-cpu"])

    # 8. ollama create exaone-3.5 -f "/content/drive/MyDrive/Colab Notebooks/dacon/inference/EXAONE-3.5/Modelfile"
    run_command('ollama pull gemma3:4b', shell=True)

    run_command('ollama cp gemma3:4b gemma3:4bcp', shell=True)
    
    run_command([sys.executable, "-m", "pip", "-q", "install", "langchain-huggingface"])

if __name__ == "__main__":
    main()