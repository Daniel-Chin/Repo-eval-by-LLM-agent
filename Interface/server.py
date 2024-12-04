import socket
import subprocess
import threading
import time
import signal
import openai
import os

from dotenv import load_dotenv, find_dotenv

assert load_dotenv(find_dotenv('openai_api.env'))
openai_api_key = os.getenv('OPENAI_API_KEY')

from env import SERVER_IP, SERVER_PORT, max_output_length


def run_command(command, current_directory):
    print(f"Executing command in {current_directory}: {command.split()}")

    start_time = time.time()
    process = subprocess.Popen(
        command.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=current_directory
    )
    
    def read_output(pipe, output_list):
        lines = 0
        for line in iter(pipe.readline, ''):
            if lines < max_output_length:
                print(line.strip())
                lines += 1
            output_list.append(line.strip())
        pipe.close()

    stdout_lines = []
    stderr_lines = []

    stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_lines))
    stderr_thread = threading.Thread(target=read_output, args=(process.stderr, stderr_lines))

    stdout_thread.start()
    stderr_thread.start()

    process.wait()
    stdout_thread.join()
    stderr_thread.join()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Command executed in {execution_time:.2f} seconds.")

    stdout = "\n".join(stdout_lines)
    stderr = "\n".join(stderr_lines)

    return stdout, stderr, execution_time


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
    print("Ctrl+C received, stopping execution.")
    raise KeyboardInterrupt

def handle_cd_command(directory):
    if not os.path.isabs(directory):
        directory = os.path.join("/home/yw5343", directory)

    try:
        os.chdir(directory)
        print(f"Changed directory to {directory}")
        return directory
    except FileNotFoundError:
        print(f"Directory '{directory}' does not exist.")
        return os.getcwd()


def handle_client_commands():
    current_directory = "/home/yw5343"

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_IP, SERVER_PORT))
    server_socket.listen(1)
    print(f"Server listening on {SERVER_IP}:{SERVER_PORT}...")

    client_socket, client_address = server_socket.accept()
    print(f"Connection from {client_address}")

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
    1. You will provide a chain of thought reasoning followed by a single command to interact with the system. Exactly one at each time.
    Strictly follow the output format:
    <Reasoning>
    <Command>

    Here is an example output:
    To begin testing, I will clone the repository to our system.

    ```bash
    git clone https://github.com/example/repo.git
    ```
    2. The current category is not saved. Each time when you want to change the direction, please include the full direction.
    3. Sometimes the command will not lead to an output but can still finish executing the command. In that case, the output provided to you will also be empty.
    4. If you think the testing is finished or you want to stop the process at any time, please put 'stop' for the command.
    """ 
    

    chat_history = [{"role": "system", "content": sys_msg}]
    data = client_socket.recv(1024).decode('utf-8')

    if data == "STOP_PROCESS":
        print("Stop signal received from client.")
        client_socket.send("Stopping process.".encode('utf-8'))
        client_socket.close()
        return

    chat_history.append({"role": "user", "content": f"Here is a GitHub repo to test: {data}"})

    while True:
        responses = interact_with_gpt(chat_history).replace("```bash", "").replace("```", "").strip()
        response_parts = responses.split('\n')
        chain_of_thought = "\n".join(response_parts[:-1]).strip()
        gpt_command = response_parts[-1].strip()

        print(f"Chain of Thought:\n{chain_of_thought}")
        print(f"GPT command:\n{gpt_command}")

        chat_history.append({"role": "assistant", "content": f"{chain_of_thought}\n\n{gpt_command}"})

        if "stop" in gpt_command.lower():
            print("Process finished by GPT.")
            break

        if gpt_command.startswith("cd"):
            target_directory = gpt_command.split()[1]
            if not os.path.isabs(target_directory):
                target_directory = os.path.join(current_directory, target_directory)
            try:
                os.chdir(target_directory)
                current_directory = target_directory
                print(f"Changed directory to {current_directory}")
            except FileNotFoundError:
                print(f"Directory '{target_directory}' does not exist.")
            continue

        stdout, stderr, execution_time = run_command(gpt_command, current_directory)

        if stdout:
            chat_history.append({"role": "user", "content": f"Standard Output:\n{stdout}"})
        if stderr:
            chat_history.append({"role": "user", "content": f"Standard Error:\n{stderr}"})
        chat_history.append({"role": "user", "content": f"Command executed in {execution_time:.2f} seconds."})

        combined_output = (
            f"Execution Time: {execution_time:.2f} seconds\n" +
            stdout + ("\n" + stderr if stderr else "") or "no return"
        )

        try:
            client_socket.send(combined_output.encode('utf-8'))
        except BrokenPipeError:
            print("Client disconnected. Closing connection.")
            break

    client_socket.close()
    server_socket.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_ctrl_c)
    handle_client_commands()
