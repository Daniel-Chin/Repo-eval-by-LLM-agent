import socket

from env import SERVER_IP, SERVER_PORT

def send_message(message):
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Connect to the server
        client_socket.connect((SERVER_IP, SERVER_PORT))

        # Send a message to the server
        client_socket.send(message.encode('utf-8'))

        # Receive a response from the server
        response = client_socket.recv(4096).decode('utf-8')
        print(f"Received from server: {response}")

    finally:
        # Close the client socket
        client_socket.close()

# Send initial message (Repo URL or any other command)
repo_url = "https://github.com/Lightning-AI/pytorch-lightning"
send_message(f"Here is a GitHub repo to test: {repo_url}")

# Wait for some time, then send stop signal (or based on user input)
input("Press Enter to stop the process...")
send_message("STOP_PROCESS")
