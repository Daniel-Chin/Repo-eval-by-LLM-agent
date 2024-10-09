import socket
import subprocess
import multiprocessing
import time
import signal
import openai
import os
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv

assert load_dotenv(find_dotenv('openai_api.env'))

# os.environ['OPENAI_API_KEY'] = "API_KEY_HERE"

# Function to run the command and return stdout and stderr
def run_command(command, output_queue):
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    while True:
        retcode = process.poll()
        stdout_line = process.stdout.readline().strip()
        stderr_line = process.stderr.readline().strip()

        if stdout_line:
            output_queue.put(("stdout", stdout_line))
        if stderr_line:
            output_queue.put(("stderr", stderr_line))

        if retcode is not None:
            break

        time.sleep(0.1)

    process.stdout.close()
    process.stderr.close()
    output_queue.put(("done", process.returncode))

# Function to periodically check the output
def check_output_periodically(output_queue):
    while True:
        if not output_queue.empty():
            output_type, content = output_queue.get()
            if output_type == "done":
                print(f"Subprocess finished with return code: {content}")
                break
            elif output_type == "stdout":
                print(f"Standard Output: {content}")
            elif output_type == "stderr":
                print(f"Standard Error: {content}")

        time.sleep(10)

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
    return gpt_response.strip()

# Handle Ctrl+C signal
def handle_ctrl_c(signum, frame):
    print("Ctrl+C received, stopping execution.")
    raise KeyboardInterrupt

# Main function to handle incoming commands and GPT integration
def handle_client_commands():
    # Define the server IP address and port
    SERVER_IP = '10.0.0.22'  # VM IP
    SERVER_PORT = 5001

    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_IP, SERVER_PORT))
    server_socket.listen(1)
    print(f"Server listening on {SERVER_IP}:{SERVER_PORT}...")

    client_socket, client_address = server_socket.accept()
    print(f"Connection from {client_address}")

    # Initialize chat history with your system message
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
    If you think the testing is finished or you want to stop the process at any time, please put 'stop' for the command.
    """ 
    
    chat_history = [{"role": "system", "content": sys_msg}]

    # Receive the repo URL or command from client
    data = client_socket.recv(1024).decode('utf-8')

    if data == "STOP_PROCESS":
        print("Stop signal received from client.")
        client_socket.send("Stopping process.".encode('utf-8'))
        client_socket.close()
        return  # Exit the function to stop further execution

    # If it's not STOP_PROCESS, proceed with normal GPT interaction
    chat_history.append({"role": "user", "content": f"Here is a GitHub repo to test: {data}"})

    # Chain of Thought (CoT) reasoning
    chat_history.append({"role": "user", "content": "Explain your reasoning and steps before giving the command."})
    chain_of_thought = interact_with_gpt(chat_history)
    print(f"Chain of thought:\n{chain_of_thought}")

    # GPT generates the first terminal command
    gpt_command = interact_with_gpt(chat_history)
    print(f"GPT command:\n{gpt_command}")

    while True:
        if "stop" in gpt_command.lower():
            print("Process finished by GPT.")
            break
        
        print('while True entered')
        # Shared output queue for subprocess communication
        output_queue = multiprocessing.Queue()

        # Create subprocess and output checking process
        command_process = multiprocessing.Process(target=run_command, args=(gpt_command, output_queue))
        output_process = multiprocessing.Process(target=check_output_periodically, args=(output_queue,))
        print('multiprocess set')
        # Start both processes
        command_process.start()
        output_process.start()
        print('multiprocess started')
        # Wait for both processes to finish
        command_process.join()
        output_process.join()
        print('multiprocess joined')
        # Send acknowledgment or result back to client
        client_socket.send("Command executed".encode('utf-8'))

        # Update chat history with output
        gpt_command = interact_with_gpt(chat_history)
        chat_history.append({"role": "user", "content": gpt_command})

    client_socket.close()
    server_socket.close()

if __name__ == "__main__":
    # Set up Ctrl+C handling
    signal.signal(signal.SIGINT, handle_ctrl_c)
    
    handle_client_commands()
