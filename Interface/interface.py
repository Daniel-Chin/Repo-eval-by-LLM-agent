from openai import OpenAI

client = OpenAI()
import os
from openai import OpenAI
import subprocess
import signal
import time

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

os.environ['OPENAI_API_KEY'] = "API_KEY_HERE"

def get_completion(prompt, model = "gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model = model,
    messages = messages,
    temperature = 0)
    return response.choices[0].message.content

sys_msg = """
You are a fully autonomous agent (instead of an assistant) whose job is to evaluate the usability of a given Github repo. \
You will be interacting with a persistent terminal through a rule-based interface that you chat with. \
No human will be filling in blanks for you. You are in charge of the entire system via the shell. \
The RULE-BASED interface will deliver inputs and outputs between you and the terminal (shell). \
> If there is an output from the terminal, the interface will provide the output with you. If there is no output, the interface will say 'no return'.
Your goal is to evaluate how easy it is to use this repository with the help of the README/doc. Therefore, you must access the README/doc to know what you should do. \
All your actions will be realized through the terminal. You need to provide commands to test the repository by yourself. \
You have to provide ONLY ONE line of command to the user after each run. \
You need to debug by yourself when meeting errors. \
You also need to prepare any materials that is not provided in the repository but is needed for the testing.

Here are your tasks:
Clone the repository onto our ubuntu system.
Navigate to the repository directory.
Read the README to know the job of this repository.
Use bullet points to propose the tasks mentioned in the README.
Propose the outcomes that are expected from the repository.
Execute the main program file(s).
Test out features by following the instructions provided in the repository's README.
Report any errors encountered and provide suggestions for improvement.
Make system calls to fix any possible errors during running.

Please note:
Always place the command in the very first line of your response without any additional characters.
Sometimes the command will not lead to an output but can still finish executing the command. In that case, the output provided to you will also be empty.
If you think the testing is finished or you want to stop the process, please put 'stop' for the command.
"""

prompt = """Hi ChatGPT, you are going to run and test codes from GitHub repositories given by me. \
I will be here to help you interact with the terminal. You should tell me what I should put for the input \
and I will provide you with the output from the terminal after each runtime. \
You should provide me with the command. Do not ask me to write command. \
I will run the command given by you and tell you the outputs. \
Note that I can only tell you the exact output or results. You need to process and analyze the output by yourself. \
Please give me step-by-step instructions on interacting with the terminal. Please give commands ONLY ONE at each time.
"""

# Function to run a command using subprocess
def run_command(command):
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate()
    return {
        'stdout': stdout.strip(),
        'stderr': stderr.strip(),
        'returncode': process.returncode
    }

# Function to interact with GPT and process commands
def interact_with_gpt(chat_history):
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=chat_history,
        temperature=0
    )

    gpt_response = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": gpt_response})

    return gpt_response.strip()

def handle_ctrl_c(signum, frame):
    print('Ctrl+C detected. GPT has chosen to stop the process.')

chat_history = []

chat_history.append({"role": "system", "content": sys_msg})

chat_history.append({"role": "user", "content": prompt})

client = OpenAI()

repo_url = "https://github.com/Lightning-AI/pytorch-lightning"
chat_history.append({"role": "user", "content": f"Here is a GitHub repo to test: {repo_url}"})

gpt_command = interact_with_gpt(chat_history)

signal.signal(signal.SIGINT, handle_ctrl_c)

while True:
    chat_history.append({"role": "assistant", "content": gpt_command})

    print(f"GPT command:\n{gpt_command}")
    terminal_command = gpt_command.split('\n')[0]
    print(f"terminal command:{terminal_command}")
    command_result = run_command(gpt_command)

    if command_result['returncode'] == 0:
        output = command_result['stdout']
        print(f"Command succeeded:\n{output}")
    else:
        output = command_result['stderr']
        print(f"Command failed:\n{output}")

    chat_history.append({"role": "user", "content": output})

    chat_history.append({"role": "user", "content": "Do you want to stop the process with Ctrl+C? If yes, include 'ctrl+c' in your response. If no, say 'no'."})
    ctrl_c_decision = interact_with_gpt(chat_history).lower()

    if "ctrl+c" in ctrl_c_decision:
        print("Process finished by GPT.")
        break

    gpt_command = interact_with_gpt(chat_history)

