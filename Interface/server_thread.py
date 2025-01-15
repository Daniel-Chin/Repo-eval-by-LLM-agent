import socket
import subprocess
import threading
import time
import signal
import openai
import os
from queue import Queue

from dotenv import load_dotenv, find_dotenv

assert load_dotenv(find_dotenv('openai_api.env'))
openai_api_key = os.getenv('OPENAI_API_KEY')

from env import SERVER_IP, SERVER_PORT, max_output_length

# Function to log conversation
'''
def log_conversation(entry):
    with open("/mnt/shared/conversation_log.txt", "a") as log_file:
        log_file.write(f"{entry}\n\n")
'''

# Function to run the command and provide updates
def run_command_with_feedback(command, current_directory, status_queue):
    print(f"Executing command in {current_directory}: {command.split()}")

    start_time = time.time()
    process = subprocess.Popen(
        command.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=current_directory
    )
    
    def read_output(pipe, output_list, pipe_name):
        lines = 0
        for line in iter(pipe.readline, ''):
            if lines < max_output_length:
                print(f"{pipe_name}: {line.strip()}")
                lines += 1
            output_list.append(line.strip())
        pipe.close()

    stdout_lines = []
    stderr_lines = []

    stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_lines, "STDOUT"))
    stderr_thread = threading.Thread(target=read_output, args=(process.stderr, stderr_lines, "STDERR"))

    stdout_thread.start()
    stderr_thread.start()

    while process.poll() is None:
        elapsed_time = time.time() - start_time
        status_queue.put({
            "status": "running",
            "elapsed_time": elapsed_time,
            "stdout_preview": "\n".join(stdout_lines[-5:]),
            "stderr_preview": "\n".join(stderr_lines[-5:])
        })
        time.sleep(5)  # Provide updates every 5 seconds

    stdout_thread.join()
    stderr_thread.join()

    end_time = time.time()
    execution_time = end_time - start_time

    stdout = "\n".join(stdout_lines)
    stderr = "\n".join(stderr_lines)

    status_queue.put({
        "status": "finished",
        "elapsed_time": execution_time,
        "stdout": stdout,
        "stderr": stderr
    })


# GPT interaction function
def interact_with_gpt(chat_history):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=chat_history,
        temperature=0
    )
    gpt_response = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": gpt_response})

    # Log GPT response
    # log_conversation(f"GPT Response:\n{gpt_response}")

    return gpt_response.strip()


# Handle Ctrl+C signal
def handle_ctrl_c(signum, frame):
    print("Ctrl+C received, stopping execution.")
    raise KeyboardInterrupt


# Handle cd command
def handle_cd_command(directory, current_directory):
    if not os.path.isabs(directory):
        directory = os.path.join(current_directory, directory)

    try:
        os.chdir(directory)
        print(f"Changed directory to {directory}")
        return directory
    except FileNotFoundError:
        print(f"Directory '{directory}' does not exist.")
        return current_directory


# Main function to handle incoming commands and GPT integration
def handle_client_commands():
    current_directory = "/home/yw5343"
    status_queue = Queue()

    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_IP, SERVER_PORT))
    server_socket.listen(1)
    print(f"Server listening on {SERVER_IP}:{SERVER_PORT}...")

    client_socket, client_address = server_socket.accept()
    print(f"Connection from {client_address}")

    # > If there is an output from the terminal, the interface will provide the output with you. If there is no output, the interface will say 'no return'. 
    sys_msg = """
### Your job
You are a fully autonomous agent (instead of an assistant) whose job is to evaluate the usability of a given Github repo. Play the role of a user getting to use the repo for the first time. Is the README easy to follow? Is installation troublesome? Are the features working as expected? And so on. After you test it, summarize your findings and provide a rating.

### You are a lone operator of an OS
You will be interacting with a persistent terminal through a rule-based interface that you chat with. The RULE-BASED interface will deliver inputs and outputs between you and the terminal (shell). 

You have full agency. No human will be filling in blanks for you. You are in charge of the entire system via the shell. The system is all yours, resettable, and not shared with anyone else, so don't hesitate to tune the system to fit what you are doing.

### Tips for action planning
Your goal is to evaluate how easy it is to use a repository. Here's an example high-level plan. 
- Clone the repository onto your ubuntu system.
- Navigate to the repository directory.
- Read the README to know the job of this repository.
- Use a numbered list to propose your action plan. Install dependencies? Run main programs? Test some features? Etc.
- Formulate expected outcomes of each step.
- Follow your own plan and reflect on what the repo is doing well/poorly as you go.
- At the end, summarize. Report any errors encountered and miscommunication/confusion in readme/doc. Provide suggestions for improvement. Rate the repo w.r.t. the following aspects (1-5 scale, or N/A):
    - How smooth is the installation?
    - After installing, Can smooth can you run the main program? (in terms of errors, parameter supplying, warnings...)
    - Semantically, does the test outcome deliver what the README promises?
    - How well written is the readme/doc?
    - How much expertise is required to install and run everything? 1 being computer-hobbiest, 5 being postdoc-level developer.

### How you interact with the terminal
All your actions will be realized through the terminal. Provide at most ONE line of command in each response. When encountering error, debug by yourself. Prepare any materials that is not provided in the repository but is needed for testing. You have access to the internet. In each of your response:
- Think out loud. Make sense of past system outputs and make a short-term plan.
- Choose either to wait, to interrupt the system, or give a new line of command.
    - Call the wait function: the system will continue to run the current command.
    - Call the interrupt function: emulates a keyboard interrupt.
    - Call the run_command function: provide exactly ONE line of bash command to the system via the function parameter.
After each of your response, the interface will return you with the current situation. Maybe the command finished with some stdout and stderr. Maybe it is still running. You will be informed of the current time.

### More tips
- If a `cd` fails with relative path, try using the full path.
""".strip('\n')
    chat_history = [{"role": "system", "content": sys_msg}]
    # log_conversation(f"System Message:\n{sys_msg}")
    data = client_socket.recv(1024).decode('utf-8')

    if data == "STOP_PROCESS":
        print("Stop signal received from client.")
        client_socket.send("Stopping process.".encode('utf-8'))
        client_socket.close()
        return

    chat_history.append({"role": "user", "content": f"Here is a GitHub repo to test: {data}"})
    # log_conversation(f"User Input:\nHere is a GitHub repo to test: {data}")

    while True:
        responses = interact_with_gpt(chat_history).replace("```bash", "").replace("```", "").strip()
        response_parts = responses.split('\n')
        chain_of_thought = "\n".join(response_parts[:-1]).strip()
        gpt_command = response_parts[-1].strip()

        print(f"Chain of Thought:\n{chain_of_thought}")
        print(f"GPT command:\n{gpt_command}")

        # log_conversation(f"Chain of Thought:\n{chain_of_thought}\nGPT Command:\n{gpt_command}")

        chat_history.append({"role": "assistant", "content": f"{chain_of_thought}\n\n{gpt_command}"})

        if "stop" in gpt_command.lower():
            print("Process finished by GPT.")
            break

        if gpt_command.startswith("cd"):
            target_directory = gpt_command.split()[1]
            current_directory = handle_cd_command(target_directory, current_directory)
            continue

        # Run command with feedback in a separate thread
        command_thread = threading.Thread(
            target=run_command_with_feedback,
            args=(gpt_command, current_directory, status_queue)
        )
        command_thread.start()

        while command_thread.is_alive():
            try:
                status_update = status_queue.get(timeout=5)
                if status_update["status"] == "running":
                    print(f"Elapsed time: {status_update['elapsed_time']:.2f} seconds")
                    print(f"STDOUT Preview:\n{status_update['stdout_preview']}")
                    print(f"STDERR Preview:\n{status_update['stderr_preview']}")

                    status_message = (
                        f"Elapsed time: {status_update['elapsed_time']:.2f} seconds\n"
                        f"STDOUT Preview:\n{status_update['stdout_preview']}\n"
                        f"STDERR Preview:\n{status_update['stderr_preview']}"
                        f"Do you want to terminate this process? (yes/no): "
                    )

                    chat_history.append({"role": "user", "content": status_message})

                    if "yes" in gpt_response:
                        print("Terminating process as instructed by GPT...")
                        process.terminate()
                        break
                    # decision = input("Do you want to terminate this process? (yes/no): ").strip().lower()
                    # if decision == "yes":
                    #     print("Terminating process...")
                    #     break
                elif status_update["status"] == "finished":
                    print(f"Command completed in {status_update['elapsed_time']:.2f} seconds.")
                    break
            except Exception:
                continue

        command_thread.join()

    client_socket.close()
    server_socket.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_ctrl_c)
    handle_client_commands()
