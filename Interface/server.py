import subprocess
import threading
import time
import signal
from openai import OpenAI, AsyncOpenAI
import os
import json
from queue import Queue, Empty
from dotenv import load_dotenv, find_dotenv
import xml.etree.ElementTree as ET

# --- Type Hinting Imports ---
import typing as tp
import asyncio
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
from functools import cached_property
import itertools
import traceback

# Import OpenAI Pydantic models for Function Calling
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.shared.function_definition import FunctionDefinition
from openai.types.chat.completion_create_params import Function as FunctionDefinitionParam


# --- Initialize OpenAI Client ---
env_path = find_dotenv(filename='openai_api.env', raise_error_if_not_found=False)
if env_path:
    assert load_dotenv(env_path)
else:
    print("INFO: 'openai_api.env' not found. Relying on environment variables for OPENAI_API_KEY.")

client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
if not os.getenv('OPENAI_API_KEY'):
    print("CRITICAL ERROR: OPENAI_API_KEY is not set. Ensure it's in your environment or 'openai_api.env'.")
    exit(1)

# --- Constants ---
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "agent_test_log.txt")
INITIAL_WORKING_DIRECTORY = os.path.expanduser("~")
DEFAULT_REPO_URL_TO_TEST: str = "https://github.com/cookiecutter/cookiecutter"
# DEFAULT_REPO_URL_TO_TEST: str = "https://github.com/pwaller/pyfiglet"


# ===============================================================
# 0. Process Management (NEW SECTION)
# ===============================================================

@dataclass
class ManagedProcess:
    pid: int
    command: str
    subprocess: subprocess.Popen
    status: str = "running"
    start_time: float = field(default_factory=time.time)
    stdout_queue: Queue = field(default_factory=Queue)
    stderr_queue: Queue = field(default_factory=Queue)
    return_code: tp.Optional[int] = None
    
    # Fields to store the complete output permanently
    full_stdout: str = ""
    full_stderr: str = ""

    def get_age(self) -> float:
        return time.time() - self.start_time

    def read_stdout(self) -> str:
        lines = []
        while not self.stdout_queue.empty():
            try: lines.append(self.stdout_queue.get_nowait())
            except Empty: break
        return "".join(lines)

    def read_stderr(self) -> str:
        lines = []
        while not self.stderr_queue.empty():
            try: lines.append(self.stderr_queue.get_nowait())
            except Empty: break
        return "".join(lines)
    

# ===============================================================
# 1. LAIAsUI Framework Base
# ===============================================================

UIEventHandler = tp.Callable[..., float]

@dataclass(frozen=True)
class LAIReply:
    id_: int
    instructions: str
    screen_as_text: str
    screen_summary: tp.Optional[str] = None

    @cached_property
    def for_agent(self) -> str:
        content = f"{self.instructions}\n"
        if self.screen_summary is not None and self.screen_summary.strip() and not self.screen_summary.lower().startswith("summary not generated"):
            content += f"<screen_summary id={self.id_}>\n{self.screen_summary}\n</screen_summary>\n"
        content += f"<screen id={self.id_}>\n{self.screen_as_text}\n</screen>"
        return content.strip()

SUMMARIZE_SCREEN_FUNCTION_DEF = FunctionDefinition(
    name='summarizeScreen',
    description=(
        "Carefully review the current screen. Create a concise yet comprehensive summary of all new and important information, events, outputs, and errors. "
        "This summary is crucial for your future reference as the full screen text may not be shown again in the history. "
        "If nothing significant changed, state that clearly."
    ),
    parameters=dict(
        type='object',
        properties={
            'screen_id': {'type': 'integer', 'description': 'ID of the screen you are summarizing.'},
            'summary': {'type': 'string', 'description': 'Your comprehensive summary.'}
        },
        required=['screen_id', 'summary'],
    ),
)


class LAIAsUI(metaclass=ABCMeta):
    @abstractmethod
    def systemPrinciples(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def render(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    async def cleanup(self):
        pass

# ===============================================================
# 2. Job State Machine
# ===============================================================
class JobStateMachine:
    def __init__(self):
        self._state = "idle" # idle, running_command, command_finished, error
        self._lock = threading.Lock()

    @property
    def state(self):
        with self._lock:
            return self._state

    def set_state(self, new_state):
        with self._lock:
            self._state = new_state

    def is_state(self, state_name):
        with self._lock:
            return self._state == state_name

# ===============================================================
# 3. GitHubRepoTesterUI Implementation
# ===============================================================
class GithubRepoTesterUI(LAIAsUI):
    def __init__(self, openai_client: AsyncOpenAI, initial_repo_url: tp.Optional[str] = None):
        self.client = openai_client
        self.current_directory: str = INITIAL_WORKING_DIRECTORY
        self.github_url_to_test: str = initial_repo_url or DEFAULT_REPO_URL_TO_TEST
        self.program_should_terminate: bool = False

        self.processes: tp.Dict[int, ManagedProcess] = {}
        self.summary_history: tp.List[str] = []
        self._lock = threading.Lock()

        self.current_stage: str = "Setup" # Start in the Setup stage
        self.stage_prompts = {
            "Setup": "Goal: Make the repository runnable. This may involve cloning, exploring files, or installing dependencies.",
            "Exploration": "Goal: Understand the repository's features. This may involve reading the documentation or using help commands.",
            "Testing": "Goal: Verify the core features work correctly by running one or more examples from the documentation. After verifying, move on to the next stage to report.",
            "Reporting": "Goal: Summarize your findings and provide a final score by calling the report function."
        }
        
        self.SYSTEM_PROMPT = f"""
### Your job
You are a fully autonomous agent whose job is to test the usability of a given GitHub repository: {self.github_url_to_test}. Play the role of a user invoking the repo for the first time. After completing your evaluation, summarize your findings in a concise report.  
Design and execute comprehensive test plans that exercise as many repository features as possible, acknowledging that some capabilities may lie outside your immediate testing scope.  
Assess documentation clarity, verifying that examples are understandable and appropriately concise, and ensure that the repository\'s tone aligns with its intended functionality.  
Maintain a professional, exploratory tone, explicitly stating your reasoning and observations.

### Operator Authority
You have full control over a dedicated OS environment via a rule-based terminal interface. No human assistance will fill in blanks for you. You are responsible for configuring the system, debugging errors, and retuning the environment as needed.

### Testing Workflow & Stages
You MUST follow a structured testing process. In every "Think" phase, you must identify which of the following stages you are in.

1.  **Setup:** Your initial goal. Clone the repo, explore the file system, and install all necessary dependencies (`pip install`, etc.). The goal is to prepare a runnable environment.
2.  **Exploration:** After setup is complete. Read the README file and run the primary command with `--help` to understand the core functionality.
3.  **Testing:** After you understand the tool. Execute one or two of the main examples or features described in the documentation to verify they work.
4.  **Reporting:** Once you have gathered enough information, you MUST call the `report_test_findings_and_terminate` function. Do not use `echo` to create a report; you must call the function directly.

### Interaction Cycle
You operate in a strict three-phase cycle: **Think -> Act -> Summarize.**
- **Think:** Based on the screen and your current stage, call `propose_next_action` with your `current_stage`, `reasoning`, and `proposed_command`.
- **Act:** Execute the proposed command.
- **Summarize:** Summarize the results of your action for your long-term memory.

### Important Tips
- **ONE COMMAND PER TURN**: Propose only one single command per run. Do not propose something like `<command 1> && <command 2>`.
- If `cd` fails with a relative path, retry with the absolute path.
- Never use interactive browsers or TUIs (e.g., nano, vim); interact only via text-based commands.
- Allow sufficient time for lengthy operations if they continuously produce new output.
- You are fully responsible for resolving any errors you encounter. Set up the environment and prepare all necessary materials yourself. Never give up until you have tried your best to debug.
- Use python3 for python commands.

### Evaluation and Scoring
Whether succeeded or not, after your testing process, assign star-ratings (0.0-10.0) for each of these criteria:
- **Environment Setup Ease**: how simple it is to configure and install dependencies.
- **Execution Difficulty**: how straightforward it is to run the core functionality.
- **Documentation Quality**: clarity, completeness, and usefulness of the README and examples.
Then calculate and report an **Overall Score** (0.0-10.0) along with a brief justification for each rating.
""".strip()

    def systemPrinciples(self) -> str:
        return self.SYSTEM_PROMPT

    def _read_process_output_queue(self) -> str:
        output_lines = []
        while not self.process_output_queue.empty():
            try:
                line_info = self.process_output_queue.get_nowait()
                output_lines.append(f"[{line_info['source'].upper()}] {line_info['line']}")
            except Empty: break
        return "\n".join(output_lines)

    # --- Tool/Function Definitions ---
    NEW_SHELL_COMMAND_DEF = FunctionDefinition(name="new_shell_command",description="Starts a new shell command in the background and returns its PID.",parameters={"type": "object","properties": {"command_str": {"type": "string", "description": "The shell command to execute."}},"required": ["command_str"],},)
    WAIT_DEF = FunctionDefinition(name="wait",description="Pauses execution for a specified duration to allow processes to run.",parameters={"type": "object","properties": {"time_s": {"type": "number", "description": "Seconds to wait."}},"required": ["time_s"],},)
    KILL_DEF = FunctionDefinition(name="kill",description="Terminates a specific running process by its PID.",parameters={"type": "object","properties": {"pid": {"type": "integer", "description": "The PID of the process to terminate."}},"required": ["pid"],},)
    SUMMARIZE_SCREEN_FUNCTION_DEF = FunctionDefinition(name='summarizeScreen',description=("Carefully review the current XML screen. Create a concise yet comprehensive summary of all new and important information, events, outputs, and errors from the <processes> block."),parameters={'type': 'object','properties': {'summary': {'type': 'string', 'description': 'Your comprehensive summary of the latest events.'}},'required': ['summary'],},)
    PROPOSE_ACTION_DEF = FunctionDefinition(name="propose_next_action",description="Propose the next shell command to execute based on your reasoning.",
        parameters={
            "type": "object",
            "properties": {
                "current_stage": {
                    "type": "string",
                    "description": "The current stage of your testing process.",
                    "enum": ["Setup", "Exploration", "Testing", "Reporting"]
                },
                "reasoning": {"type": "string", "description": "Your detailed chain-of-thought and rationale for the proposed command, based on the past summaries and current screen."},
                "proposed_command": {"type": "string", "description": "The exact shell command you propose to run next (e.g., 'ls -l', 'cd repo', 'python3 setup.py')."}
            },
            "required": ["reasoning", "proposed_command"]
        }
    )
    REPORT_FINDINGS_DEF = FunctionDefinition(
        name="report_test_findings_and_terminate",
        description="Submits your final analysis and star-ratings, then terminates the agent.",
        parameters={
            "type": "object",
            "properties": {
                "setup_ease_score": {"type": "number", "description": "Rating (0.0-10.0) for setup ease."},
                "execution_difficulty_score": {"type": "number", "description": "Rating (0.0-10.0) for execution difficulty."},
                "doc_quality_score": {"type": "number", "description": "Rating (0.0-10.0) for documentation quality."},
                "overall_score": {"type": "number", "description": "Final overall usability rating (0.0-10.0)."},
                "justification": {"type": "string", "description": "Concise justification for your scores and summary."}
            },
            "required": ["setup_ease_score", "execution_difficulty_score", "doc_quality_score", "overall_score", "justification"],
        }
    )

    def get_available_functions(self) -> tp.List[tp.Tuple[FunctionDefinition, UIEventHandler]]:
        """Returns a list of all available tools and their handlers."""
        
        # NOTE: In this new architecture, all tools are considered available at all times.
        # The LLM is expected to use its reasoning to decide if a command like 'kill' is
        # appropriate based on the process states it sees on the screen.
        
        return [
            (self.NEW_SHELL_COMMAND_DEF, self.handle_new_shell_command),
            (self.WAIT_DEF, self.handle_wait),
            (self.KILL_DEF, self.handle_kill),
            (self.REPORT_FINDINGS_DEF, self.handle_report_test_findings),
        ]

    def render(self) -> str:
        """Generates an XML-based screen for the LLM. Does not delete processes."""
        with self._lock:
            root = ET.Element("screen")
            state_elem = ET.SubElement(root, "current_state")
            ET.SubElement(state_elem, "working_directory").text = self.current_directory
            prompt_text = self.stage_prompts.get(self.current_stage, "Review processes and decide the next action.")
            ET.SubElement(state_elem, "prompt", attrib={"stage": self.current_stage}).text = prompt_text
            processes_elem = ET.SubElement(root, "processes")

            for pid, proc in self.processes.items():
                proc_elem = ET.SubElement(processes_elem, "process", attrib={"pid": str(pid)})
                ET.SubElement(proc_elem, "command").text = proc.command
                ET.SubElement(proc_elem, "status").text = proc.status
                ET.SubElement(proc_elem, "age_seconds").text = f"{proc.get_age():.1f}"
                ET.SubElement(proc_elem, "return_code").text = str(proc.return_code) if proc.return_code is not None else "N/A"

                stdout_text = proc.read_stdout() if proc.status == "running" else proc.full_stdout
                stderr_text = proc.read_stderr() if proc.status == "running" else proc.full_stderr

                ET.SubElement(proc_elem, "stdout").text = stdout_text
                ET.SubElement(proc_elem, "stderr").text = stderr_text

            ET.indent(root, space="\t")
            return ET.tostring(root, encoding='unicode')
    
    def render_as_graph(self, xml_string: str) -> str:
        """Parses the screen XML and renders it as a human-readable text graph with full output."""
        try:
            root = ET.fromstring(xml_string)
            lines = []

            # --- Current Prompt Section ---
            prompt = root.find(".//prompt")
            stage = prompt.get("stage")
            lines.append("-------------------------- (screen) -----------------------")
            lines.append(f"| Stage: {stage}")
            lines.append(f"| Prompt: {prompt.text}")
            lines.append("|--------------------------------------------------------")

            # --- Current Processes Section ---
            lines.append("|------------------- current processes --------------------")
            
            for proc_elem in root.findall(".//process"):
                pid = proc_elem.get("pid")
                command = proc_elem.find("command").text or ""
                status = proc_elem.find("status").text or ""
                age = float(proc_elem.find("age_seconds").text or 0)
                return_code = proc_elem.find("return_code").text or "N/A"
                
                cmd_display = (command[:60] + '...') if len(command) > 63 else command
                
                # Print the main process info line
                lines.append(f"| PID: {pid:<5} | Status: {status:<10} | Age: {age:<7.1f}s | Code: {return_code:<3} |")
                lines.append(f"| Command: {cmd_display}")

                # Get the full stdout and stderr
                stdout = proc_elem.find("stdout").text or ""
                stderr = proc_elem.find("stderr").text or ""

                # If there is stdout, format and add it.
                if stdout.strip():
                    lines.append("|--[stdout]--------------------------------------------")
                    for line in stdout.strip().split('\n'):
                        lines.append(f"|  > {line}")

                # If there is stderr, format and add it.
                if stderr.strip():
                    lines.append("|--[stderr]--------------------------------------------")
                    for line in stderr.strip().split('\n'):
                        lines.append(f"|  > {line}")
                lines.append("|--------------------------------------------------------")
            
            return "\n".join(lines)
        except ET.ParseError as e:
            return f"Error parsing XML for graph rendering: {e}\n\n{xml_string}"
    
    # --- Tool/Function Definitions & Handlers ---

    def _command_execution_thread(self, command: str):
        # Initialize lists to collect the full output
        full_stdout_lines, full_stderr_lines = [], []
        
        try:
            process = subprocess.Popen(
                ['/bin/bash', '-c', command],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=self.current_directory, universal_newlines=True, bufsize=1
            )
            
            with self._lock:
                managed_proc = ManagedProcess(pid=process.pid, command=command, subprocess=process)
                self.processes[process.pid] = managed_proc

            # The stream_reader now correctly takes 'line_list' to store full output
            def stream_reader(pipe: tp.IO[str], queue: Queue, line_list: tp.List[str]):
                for line in iter(pipe.readline, ''):
                    queue.put(line)      # For live reading
                    line_list.append(line) # For permanent storage
                pipe.close()

            # The threading.Thread calls now correctly pass 'line_list' as the third argument
            stdout_thread = threading.Thread(target=stream_reader, args=(process.stdout, managed_proc.stdout_queue, full_stdout_lines))
            stderr_thread = threading.Thread(target=stream_reader, args=(process.stderr, managed_proc.stderr_queue, full_stderr_lines))
            stdout_thread.start()
            stderr_thread.start()
            
            process.wait()
            stdout_thread.join()
            stderr_thread.join()

            with self._lock:
                # Save the collected full output to the process object
                managed_proc.full_stdout = "".join(full_stdout_lines)
                managed_proc.full_stderr = "".join(full_stderr_lines)
                managed_proc.return_code = process.returncode
                managed_proc.status = "finished" if process.returncode == 0 else "error"
        except Exception as e:
            print(f"Failed to start command '{command}': {e}")

    NEW_SHELL_COMMAND_DEF = FunctionDefinition(name="new_shell_command", description="Starts a new shell command in the background.", parameters={"type": "object", "properties": {"command_str": {"type": "string", "description": "The shell command to execute."}}, "required": ["command_str"]})
    def handle_new_shell_command(self, command_str: str) -> str:
        # Handle simple cd commands only
        if command_str.strip().startswith("cd "):
            try:
                target_dir = command_str.strip().split(" ", 1)[1]
                effective_target = os.path.join(self.current_directory, target_dir) if not target_dir.startswith(('/', '~')) else os.path.expanduser(target_dir)
                os.chdir(os.path.normpath(effective_target))
                self.current_directory = os.getcwd()
                return f"Changed directory to {self.current_directory}."
            except Exception as e:
                return f"Error changing directory: {e}"

        # Handle all other commands as background processes
        thread = threading.Thread(target=self._command_execution_thread, args=(command_str,))
        thread.daemon = True
        thread.start()
        
        # Wait for the process to actually start and be registered
        max_attempts = 20  # 2 seconds total
        for attempt in range(max_attempts):
            time.sleep(0.1)
            with self._lock:
                # Look for a process that matches our command and is running
                for proc in self.processes.values():
                    if proc.command == command_str and proc.status == "running":
                        return f"Command initiated with PID: {proc.pid}"
        
        # If we get here, the process didn't start as expected
        return f"Command '{command_str}' was initiated but process status unclear."

    WAIT_DEF = FunctionDefinition(name="wait", description="Pauses execution for a specified duration.", parameters={"type": "object", "properties": {"time_s": {"type": "number", "description": "Seconds to wait."}}, "required": ["time_s"]})
    def handle_wait(self, time_s: float = 3.0) -> str:  # Add default parameter
        actual_wait = max(0, min(time_s, 300))
        time.sleep(actual_wait)
        return f"Waited for {actual_wait:.1f} seconds."

    KILL_DEF = FunctionDefinition(name="kill", description="Terminates a specific running process by its PID.", parameters={"type": "object", "properties": {"pid": {"type": "integer", "description": "The PID of the process to terminate."}}, "required": ["pid"]})
    def handle_kill(self, pid: int) -> str:
        with self._lock:
            if pid in self.processes and self.processes[pid].status == "running":
                try:
                    self.processes[pid].subprocess.terminate()
                    return f"Sent termination signal to process with PID {pid}."
                except Exception as e:
                    return f"Error terminating process {pid}: {e}"
            else:
                return f"No running process with PID {pid} found."

    def handle_report_test_findings(self, setup_ease_score: float, execution_difficulty_score: float, doc_quality_score: float, overall_score: float, justification: str) -> str:
        final_report = f"""
================================
=== GitHub Repo Test Report ===
================================
Repository: {self.github_url_to_test}

--- SCORING ---
- Environment Setup Ease:     {setup_ease_score:.1f} / 10.0
- Execution Difficulty:       {execution_difficulty_score:.1f} / 10.0
- Documentation Quality:      {doc_quality_score:.1f} / 10.0
---------------------------------
- Overall Score:              {overall_score:.1f} / 10.0

--- JUSTIFICATION & SUMMARY ---
{justification}
================================
        """
        print(final_report)
        self.program_should_terminate = True
        return "Test findings reported. Program will terminate."
    
    async def cleanup(self):
        print("[CLEANUP] Cleanup called.")
        with self._lock:
            # MODIFIED: Iterate through the new processes dictionary
            for pid, proc in self.processes.items():
                if proc.status == "running":
                    print(f"[CLEANUP] Terminating active process {pid}.")
                    try:
                        # Use the stored subprocess object to kill the process
                        proc.subprocess.kill()
                    except Exception as e:
                        print(f"[CLEANUP] Error killing process {pid}: {e}")
    
    def cull_finished_processes(self):
        """Removes any processes that have finished from the list."""
        with self._lock:
            pids_to_delete = [pid for pid, proc in self.processes.items() if proc.status in ["finished", "error"]]
            for pid in pids_to_delete:
                if pid in self.processes:
                    del self.processes[pid]

# ===============================================================
# 4. Logging
# ===============================================================
interaction_log_buffer: tp.List[tp.Dict[str, str]] = []
def log_interaction_agent(message: str, role: str = "system_internal"):
    print(message) 
    interaction_log_buffer.append({"role": role, "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'), "content": message})

def save_interaction_log_file(screen_id: int, api_call_payloads: list):
    try:
        with open(LOG_FILE_PATH, "a", encoding='utf-8') as f:
            f.write(f"\n==================== FULL LOG FOR TURN {screen_id} ({time.strftime('%Y-%m-%d %H:%M:%S')}) ====================\n")
            f.write("\n--- Other System & Action Logs for this Turn ---\n")
            for entry in interaction_log_buffer: f.write(f"[{entry['timestamp']}] {entry['role'].upper()}: {entry['content']}\n")
            for i, payload in enumerate(api_call_payloads):
                f.write(f"\n--- API Call #{i+1} ---\n--- PAYLOAD SENT TO GPT ---\n")
                f.write(json.dumps(payload['messages_sent'], indent=2))
                f.write("\n\n--- REPLY RECEIVED FROM GPT ---\n")
                f.write(json.dumps(payload['gpt_reply'], indent=2))
                f.write("\n---------------------------\n")
            f.write(f"==================== END OF TURN {screen_id} ====================\n\n")
        interaction_log_buffer.clear() 
    except Exception as e: print(f"Error saving log file: {e}")


# ===============================================================
# 5. Main LAI Loop
# ===============================================================

async def main_lai_loop(laiAsUI: GithubRepoTesterUI, client_openai: AsyncOpenAI, model_name: str = "gpt-4.1-mini"):
    # This chat_history is the persistent, long-term memory built from summaries.
    chat_history: tp.List[ChatCompletionMessageParam] = []

    while not laiAsUI.program_should_terminate:
        
        # --- Phase 1: RENDER and THINK ---
        print("\n--- AGENT STATE ---")
        # Step 1: Render the current state, including any finished processes from the last turn.
        screen_xml = laiAsUI.render()
        print(laiAsUI.render_as_graph(screen_xml))

        # Step 2: The agent "thinks" based on the screen it just saw.
        past_summaries = "\n".join(laiAsUI.summary_history) or "No summaries yet."
        think_prompt_content = f"<past_summaries>\n{past_summaries}\n</past_summaries>\n\n{screen_xml}"
        
        think_messages: tp.List[ChatCompletionMessageParam] = [
            {'role': 'system', 'content': laiAsUI.systemPrinciples()},
            *chat_history,
            {'role': 'user', 'content': think_prompt_content}
        ]

        api_call_logs_for_turn = []
        
        try:
            # The agent can now choose between proposing a command or reporting findings.
            think_tools = [
                {'type': 'function', 'function': tp.cast(FunctionDefinitionParam, laiAsUI.PROPOSE_ACTION_DEF.model_dump(exclude_none=True))},
                {'type': 'function', 'function': tp.cast(FunctionDefinitionParam, laiAsUI.REPORT_FINDINGS_DEF.model_dump(exclude_none=True))}
            ]
            think_completion = await client_openai.chat.completions.create(
                model=model_name, 
                messages=think_messages, 
                tools=think_tools,
                tool_choice='auto',
                temperature=0.8
            )
            think_response = think_completion.choices[0].message
            api_call_logs_for_turn.append({'messages_sent': think_messages, 'gpt_reply': think_response.model_dump()})

            # Step 3: Now that the agent has seen the finished process and thought, we can clear it.
            # This is the key to the "show once" logic.
            laiAsUI.cull_finished_processes()

            proposed_command = "" 
            if think_response.tool_calls:
                tool_call = think_response.tool_calls[0]
                function_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if function_name == laiAsUI.PROPOSE_ACTION_DEF.name:
                    current_stage = args.get("current_stage", "Unknown")
                    reasoning = args.get("reasoning", "")
                    proposed_command = args.get("proposed_command", "")
                    
                    laiAsUI.current_stage = current_stage
                    print(f"LLM STAGE: {current_stage}")
                    print(f"LLM THOUGHT: {reasoning}")

                elif function_name == laiAsUI.REPORT_FINDINGS_DEF.name:
                    print("LLM is reporting final findings and will terminate.")
                    laiAsUI.handle_report_test_findings(**args)
                    break # This correctly exits the loop.

            # --- Phase 2: EXECUTE ---
            if proposed_command:
                print(f"EXECUTING: {proposed_command}")
                result_message = laiAsUI.handle_new_shell_command(proposed_command)
                print(f"COMMAND RESULT: {result_message}")
                
                # Give more time for processes to produce output, especially for npm install
                if "npm install" in proposed_command:
                    await asyncio.sleep(5.0)  # npm install needs more time to show progress
                else:
                    await asyncio.sleep(1.5)  # General commands

            # --- Phase 3: SUMMARIZE ---
            final_screen_xml = laiAsUI.render() # Render again to get final state for summary
            summary_prompt_content = f"The turn is complete. Here is the final screen state. Summarize the key events and outcomes of your last action.\n{final_screen_xml}"
            summary_messages: tp.List[ChatCompletionMessageParam] = [{'role': 'system', 'content': laiAsUI.systemPrinciples()}, *chat_history, {'role': 'user', 'content': summary_prompt_content}]
            
            summarize_tools = [{'type': 'function', 'function': tp.cast(FunctionDefinitionParam, laiAsUI.SUMMARIZE_SCREEN_FUNCTION_DEF.model_dump(exclude_none=True))}]
            summary_completion = await client_openai.chat.completions.create(
                model=model_name, messages=summary_messages, tools=summarize_tools,
                tool_choice={'type': 'function', 'function': {'name': laiAsUI.SUMMARIZE_SCREEN_FUNCTION_DEF.name}}
            )
            summary_response = summary_completion.choices[0].message
            api_call_logs_for_turn.append({'messages_sent': summary_messages, 'gpt_reply': summary_response.model_dump()})

            if summary_response.tool_calls:
                # ... (rest of summary logic is the same)
                tool_call = summary_response.tool_calls[0]
                args = json.loads(tool_call.function.arguments)
                summary = args.get("summary", "No summary provided.")
                turn_summary_text = f"Summary of Turn {len(laiAsUI.summary_history)}: {summary}"
                laiAsUI.summary_history.append(turn_summary_text)
                print(f"SCREEN SUMMARY: {summary}")
                
                # Update chat history for context in the next turn
                if think_response.tool_calls:
                    chat_history.append({'role': 'assistant', 'content': think_response.tool_calls[0].function.arguments})
                chat_history.append({'role': 'user', 'content': summary})


        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            traceback.print_exc()
            await asyncio.sleep(10)
        
        save_interaction_log_file(len(laiAsUI.summary_history), api_call_logs_for_turn)
        


# ===============================================================
# 6. Entry Point
# ===============================================================
if __name__ == "__main__":
    # ... (__main__ block remains the same as the last version) ...
    ui_instance = GithubRepoTesterUI(client, initial_repo_url=DEFAULT_REPO_URL_TO_TEST)
    loop = asyncio.get_event_loop()
    main_task = None
    def signal_handler_sigint(signum, frame): 
        print("\nCtrl+C: Shutdown...")
        log_interaction_agent("Ctrl+C shutdown.", "signal_event")
        if main_task and not main_task.done(): main_task.cancel()
    signal.signal(signal.SIGINT, signal_handler_sigint)
    try:
        main_task = loop.create_task(main_lai_loop(ui_instance, client))
        loop.run_until_complete(main_task)
    except asyncio.CancelledError: log_interaction_agent("Main task cancelled.", "system_event")
    except Exception as e:
        log_interaction_agent(f"UNHANDLED EXCEPTION: {e}", "critical_error"); log_interaction_agent(traceback.format_exc(), "critical_error_trace")
    finally:
        print("Final log save."); save_interaction_log_file(-1, []); print("Exited.")