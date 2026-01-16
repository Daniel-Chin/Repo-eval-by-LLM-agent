import subprocess
import threading
import time
import signal
from openai import AsyncOpenAI
import os
import json
from queue import Queue, Empty
from dotenv import load_dotenv, find_dotenv
import xml.etree.ElementTree as ET
import typing as tp
import asyncio
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import traceback
from enum import Enum, auto
import re

# Import OpenAI Pydantic models for Function Calling
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
    print("CRITICAL ERROR: OPENAI_API_KEY is not set.")
    exit(1)

# --- Constants ---
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "agent_test_log.txt")
INITIAL_WORKING_DIRECTORY = os.path.expanduser("~")
# DEFAULT_REPO_URL_TO_TEST: str = "https://github.com/ZZWaang/audio2midi"
DEFAULT_REPO_URL_TO_TEST: str = "https://github.com/mkaz/termgraph"
TAKEAWAYS_FILE_PATH = os.path.join(os.path.dirname(__file__), "takeaways.txt")
MAX_SCREEN_OUTPUT_CHARS = 2000  # Truncate output if it exceeds this length

# ===============================================================
# 0. Event System & Process Management
# ===============================================================

class AgentEventType(Enum):
    STARTUP = auto()           # Trigger for the initial wake-up
    PROCESS_OUTPUT = auto()    # Stdout/Stderr from a running process
    PROCESS_FINISHED = auto()  # A process has terminated
    ALARM_EXPIRATION = auto()  # The agent's requested alarm has fired
    USER_INTERVENTION = auto() # Human triggered 'interrogate'
    TOOL_COMPLETION = auto()   # Triggered after a tool finishes so LLM can see result

@dataclass
class AgentEvent:
    type: AgentEventType
    source_pid: tp.Optional[int] = None
    data: tp.Any = None

@dataclass
class ManagedProcess:
    pid: int
    command: str
    subprocess: tp.Optional[subprocess.Popen]
    status: str = "running"
    start_time: float = field(default_factory=time.time)
    
    # "Full" history for cat/read operations
    full_stdout: str = ""
    full_stderr: str = ""
    
    # "Unread" buffer for the ephemeral screen (cleared after render)
    unread_stdout: str = ""
    unread_stderr: str = ""
    
    return_code: tp.Optional[int] = None
    finished_time: tp.Optional[float] = None
    
    # Lock for thread-safe buffer updates
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get_age(self) -> float:
        return time.time() - self.start_time

    def append_stdout(self, text: str):
        with self._lock:
            self.full_stdout += text
            self.unread_stdout += text

    def append_stderr(self, text: str):
        with self._lock:
            self.full_stderr += text
            self.unread_stderr += text
            
    def read_and_clear_buffer(self) -> tp.Tuple[str, str]:
        """Returns unread output and clears the unread buffer."""
        with self._lock:
            out = self.unread_stdout
            err = self.unread_stderr
            self.unread_stdout = ""
            self.unread_stderr = ""
            return out, err

# ===============================================================
# 1. LAIAsUI Framework Base
# ===============================================================

UIEventHandler = tp.Callable[..., tp.Coroutine[tp.Any, tp.Any, str]]

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
# 2. GitHubRepoTesterUI Implementation
# ===============================================================
class GithubRepoTesterUI(LAIAsUI):
    def __init__(self, openai_client: AsyncOpenAI, event_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, initial_repo_url: tp.Optional[str] = None):
        self.client = openai_client
        self.event_queue = event_queue
        self.loop = loop # Reference to main asyncio loop for threadsafe calls
        
        self.current_directory: str = INITIAL_WORKING_DIRECTORY
        self.github_url_to_test: str = initial_repo_url or DEFAULT_REPO_URL_TO_TEST
        self.program_should_terminate: bool = False
        self.processes: tp.Dict[int, ManagedProcess] = {}
        self.summary_history: tp.List[str] = []
        self._lock = threading.Lock()
        
        self.current_stage: str = "Setup"
        self.stage_prompts = {
            "Setup": "Goal: Make the repository runnable. Clone, explore, install dependencies.",
            "Exploration": "Goal: Understand the repository's features (README, help commands).",
            "Testing": "Goal: Verify core features by running examples.",
            "Reporting": "Goal: Summarize findings and call the report function."
        }

        # Load takeaways
        takeaways_section = ""
        try:
            if os.path.exists(TAKEAWAYS_FILE_PATH):
                with open(TAKEAWAYS_FILE_PATH, "r", encoding="utf-8") as f:
                    takeaways = [line.strip() for line in f if line.strip()]
                if takeaways:
                    takeaways_section = "\n".join(f"- {t}" for t in takeaways)
        except Exception: pass
        
        self.SYSTEM_PROMPT = f"""
### Your job
You are a fully autonomous agent whose job is to test the usability of a given GitHub repository: {self.github_url_to_test}. Play the role of a user invoking the repo for the first time. After completing your evaluation, summarize your findings in a concise report.  
Design and execute comprehensive test plans that exercise as many repository features as possible, focusing only on features that are realistically testable through text-based interaction. Many repositories may include non-text assets (videos, audio, images, large binaries, datasets, etc.) which are out of scope; in such cases, explicitly mark them as out of scope and continue testing text-based functionality to the largest extent possible. If you cannot test any, at least read its code and see if the functions seem to work with each other, or if there is any obvious mistake.
Assess documentation clarity, verifying that examples are understandable and appropriately concise, and ensure that the repository\'s tone aligns with its intended functionality.  
Maintain a professional, exploratory tone, explicitly stating your reasoning and observations.

### Operator Authority
You have full control over a dedicated OS environment via a rule-based terminal interface. No human assistance will fill in blanks for you. You are responsible for configuring the system, debugging errors, and retuning the environment as needed.

### Strategic Lessons (Takeaways)
{takeaways_section}

### Event-Driven Operation
You do not run in a continuous loop. You operate on an **Alarm/Event** basis:
1. **Screen is a Buffer:** The information shown to you in <screen> is NEW information since you last looked. Old output is cleared to save tokens.
2. **Missing Output:** If you see "Output Truncated", use the `read_process_output` tool to read the specific log.
3. **Setting Alarms:** You MUST use `set_alarm(seconds)` when you start a long-running process (like installation) or when you want to "think" again in the future.
4. **Interruption:** If a process produces output *before* your alarm goes off, your alarm is CANCELLED, and you are woken up immediately to handle the new info.
You operate in a strict **Two-Phase Loop** for every event:
1.  **OBSERVATION PHASE:** You will see the new screen output. You MUST use the `summarizeScreen` tool to record what just happened into your long-term memory.
2.  **ACTION PHASE:** You will see your past summaries and the screen again. You MUST then decide the next move (Command, Alarm, Report, etc.).

### Workflow Stages
1. **Setup:** Clone, Install dependencies. (Allow time for installs!).
2. **Exploration:** Read README, run --help.
3. **Testing:** Run actual examples.
4. **Reporting:** Call `report_test_findings_and_terminate`.

### Important Tips
- **ONE COMMAND PER TURN**: Propose only one single command per run. Do not propose something like `<command 1> && <command 2>`.
- If `cd` fails with a relative path, retry with the absolute path.
- Never use interactive browsers or TUIs (e.g., nano, vim); interact only via text-based commands.
- If a command is running, `set_alarm` to wait for it. Do not just loop `wait`.
- **Constructive Feedback:** When reporting, provide actionable advice for the repo owner.
- You are fully responsible for resolving any errors you encounter. Set up the environment and prepare all necessary materials yourself. Never give up until you have tried your best to debug.
- Use python3 for python commands.
- If you already notified something that is related to a function which is out of your testing scope while setting up, you can choose to skip that specific part.
- All running processes will be printed on the screen. If you find a running process disappears, see that process as finished.

### Evaluation and Scoring
At the end of your testing process, assign star-ratings (0.0-10.0) for each of the following criteria.
**Important**: Each score must be evaluated independently. A problem in one area should not automatically reduce ratings in other areas. Always distinguish whether difficulties arise from the repository itself or from your own system environment to avoid unfair grading.
- **Environment Setup Ease**: How easy it is to configure the environment and install dependencies. Consider whether errors are caused by the repoâ€™s instructions or by external/system issues. If it is caused by external/system issues, you should not lower the score of this field.
- **Execution Difficulty**: How smoothly the programs or scripts run once setup is complete. Evaluate whether the code achieves its intended functions. If certain features require out-of-scope resources (e.g., videos, audio, large binaries, GUIs), ignore those and grade only the text-based parts you could test. This should only focus on whether the functions provided by the repository are running properly.
- **Documentation Quality**: Clarity, completeness, and usefulness of the README and examples. Focus on whether it provides clear setup steps and runnable test/execution instructions.
Finally, calculate and report an Overall Score (0.0-10.0) with a concise justification for each criterion.
"""

    def systemPrinciples(self) -> str:
        return self.SYSTEM_PROMPT

    # --- Tool Definitions ---
    
    NEW_SHELL_COMMAND_DEF = FunctionDefinition(
        name="new_shell_command",
        description="Starts a new shell command in the background.",
        parameters={"type": "object","properties": {"command_str": {"type": "string", "description": "The shell command."}},"required": ["command_str"]},
    )
    
    SET_ALARM_DEF = FunctionDefinition(
        name="set_alarm",
        description="Sets an alarm to wake you up after T seconds. If a process produces output before T, you wake up early.",
        parameters={"type": "object","properties": {"time_s": {"type": "number", "description": "Seconds to wait."}},"required": ["time_s"]},
    )
    
    READ_OUTPUT_DEF = FunctionDefinition(
        name="read_process_output",
        description="Reads the full history of a process's output, useful if the screen buffer was truncated.",
        parameters={"type": "object","properties": {
            "pid": {"type": "integer", "description": "PID of the process."},
            "offset": {"type": "integer", "description": "Character offset to start reading from (default 0)."}
        },"required": ["pid"]},
    )

    KILL_DEF = FunctionDefinition(
        name="kill",
        description="Terminates a process.",
        parameters={"type": "object","properties": {"pid": {"type": "integer"}},"required": ["pid"]},
    )
    
    # PROPOSE_ACTION_DEF = FunctionDefinition(
    #     name="propose_next_action",
    #     description="Propose the next shell command or tool action.",
    #     parameters={
    #         "type": "object",
    #         "properties": {
    #             "current_stage": {"type": "string", "enum": ["Setup", "Exploration", "Testing", "Reporting"]},
    #             "reasoning": {"type": "string", "description": "Chain-of-thought."},
    #             "proposed_command": {"type": "string", "description": "Shell command (optional)."}
    #         },
    #         "required": ["reasoning"]
    #     }
    # )
    
    REPORT_FINDINGS_DEF = FunctionDefinition(
        name="report_test_findings_and_terminate",
        description="Submit final report, ratings, and feedback, then terminate the agent.",
        parameters={
            "type": "object",
            "properties": {
                "setup_ease_score": {"type": "number", "description": "Rating (0.0-10.0) for setup ease."},
                "execution_difficulty_score": {"type": "number", "description": "Rating (0.0-10.0) for execution difficulty."},
                "doc_quality_score": {"type": "number", "description": "Rating (0.0-10.0) for documentation quality."},
                "overall_score": {"type": "number", "description": "Final overall usability rating (0.0-10.0)."},
                "justification": {"type": "string", "description": "Concise justification for your scores."},
                "constructive_feedback": {"type": "string", "description": "Actionable advice for the repo owner to improve the project."},
                "key_takeaway": {"type": "string", "description": "A single, concise strategic lesson for future runs."}
            },
            "required": ["setup_ease_score", "execution_difficulty_score", "doc_quality_score", "overall_score", "justification", "constructive_feedback", "key_takeaway"],
        }
    )

    SUMMARIZE_SCREEN_DEF = FunctionDefinition(
        name='summarizeScreen',
        description="Summarize the new events on screen.",
        parameters={'type': 'object','properties': {'summary': {'type': 'string'}},'required': ['summary']},
    )

    
    # Helper to get tools for Phase 1 (Observation)
    def get_summary_tools(self):
        return [
            (self.SUMMARIZE_SCREEN_DEF, None)
        ]

    # Helper to get tools for Phase 2 (Action)
    def get_action_tools(self):
        return [
            (self.NEW_SHELL_COMMAND_DEF, self.handle_new_shell_command),
            (self.SET_ALARM_DEF, self.handle_set_alarm),
            (self.READ_OUTPUT_DEF, self.handle_read_output),
            (self.KILL_DEF, self.handle_kill),
            (self.REPORT_FINDINGS_DEF, self.handle_report_test_findings),
            (self.PROPOSE_ACTION_DEF, None), 
        ]

    # --- XML Safe Helper ---
    def make_xml_safe(self, text: str) -> str:
        """
        General cleaning function:
        Removes or escapes all low-level control characters according to the XML 1.0 standard.
        Preserves: \\n, \\r, \\t
        Escapes: All other 0x00-0x1F characters (e.g. \\x1b becomes the literal string "\\x1b")
        """
        if not text:
            return ""
            
        illegal_xml_chars_pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

        def _escape_control_char(match):
            char_code = ord(match.group())
            return f"\\x{char_code:02x}"

        return illegal_xml_chars_pattern.sub(_escape_control_char, text)

    def render(self) -> str:
        """
        Generates the Screen XML. 
        CRITICAL: This method consumes (clears) the 'unread' buffer of processes.
        """
        with self._lock:
            root = ET.Element("screen")
            state_elem = ET.SubElement(root, "current_state")
            ET.SubElement(state_elem, "working_directory").text = self.current_directory
            ET.SubElement(state_elem, "prompt", attrib={"stage": self.current_stage}).text = \
                self.stage_prompts.get(self.current_stage, "Action required.")
            
            processes_elem = ET.SubElement(root, "processes")

            has_new_content = False

            for pid, proc in self.processes.items():
                # Get unread content and clear it from the process object
                new_out, new_err = proc.read_and_clear_buffer()
                
                # Truncation Logic
                display_out = new_out
                if len(display_out) > MAX_SCREEN_OUTPUT_CHARS:
                    display_out = display_out[:MAX_SCREEN_OUTPUT_CHARS] + \
                        f"\n... [OUTPUT TRUNCATED at {MAX_SCREEN_OUTPUT_CHARS} chars. Use 'read_process_output' to see full log] ..."
                
                display_err = new_err
                if len(display_err) > MAX_SCREEN_OUTPUT_CHARS:
                    display_err = display_err[:MAX_SCREEN_OUTPUT_CHARS] + \
                        f"\n... [STDERR TRUNCATED] ..."

                # Only show process block if it's running OR has new output to show
                if proc.status == "running" or display_out or display_err:
                    has_new_content = True
                    proc_elem = ET.SubElement(processes_elem, "process", attrib={"pid": str(pid)})
                    ET.SubElement(proc_elem, "command").text = proc.command
                    ET.SubElement(proc_elem, "status").text = proc.status
                    ET.SubElement(proc_elem, "age_seconds").text = f"{proc.get_age():.1f}"
                    ET.SubElement(proc_elem, "new_stdout").text = self.make_xml_safe(display_out)
                    ET.SubElement(proc_elem, "new_stderr").text = self.make_xml_safe(display_err)
            
            # If nothing happened, indicate silence
            if not has_new_content:
                ET.SubElement(root, "info").text = "No new process output since last screen."

            ET.indent(root, space="\t")
            return ET.tostring(root, encoding='unicode')

    def render_as_graph(self, xml_string: str) -> str:
        """
        [ADAPTED FROM server.py]
        Parses the screen XML and renders it as a human-readable text graph.
        Handles both 'stdout' (server.py style) and 'new_stdout' (server_UI.py style).
        """
        try:
            root = ET.fromstring(xml_string)
            lines = []

            # --- Current Prompt Section ---
            prompt = root.find(".//prompt")
            stage = prompt.get("stage") if prompt is not None else "Unknown"
            prompt_text = prompt.text if prompt is not None else "N/A"
            
            lines.append("-------------------------- (screen) -----------------------")
            lines.append(f"| Stage: {stage}")
            lines.append(f"| Prompt: {prompt_text}")
            lines.append("|--------------------------------------------------------")

            # --- Current Processes Section ---
            lines.append("|------------------- current processes --------------------")
            
            processes = root.findall(".//process")
            if not processes:
                lines.append("| (No active processes or new output)")
            
            for proc_elem in processes:
                pid = proc_elem.get("pid")
                
                cmd_elem = proc_elem.find("command")
                command = cmd_elem.text if cmd_elem is not None else ""
                
                status_elem = proc_elem.find("status")
                status = status_elem.text if status_elem is not None else ""
                
                age_elem = proc_elem.find("age_seconds")
                age = float(age_elem.text) if age_elem is not None and age_elem.text else 0.0
                
                cmd_display = (command[:60] + '...') if len(command) > 63 else command
                
                # Print the main process info line
                lines.append(f"| PID: {pid:<5} | Status: {status:<10} | Age: {age:<7.1f}s |")
                lines.append(f"| Command: {cmd_display}")

                # Get stdout/stderr (handling naming differences)
                stdout = proc_elem.find("new_stdout")
                if stdout is None: stdout = proc_elem.find("stdout")
                stdout_text = stdout.text if stdout is not None else ""
                
                stderr = proc_elem.find("new_stderr")
                if stderr is None: stderr = proc_elem.find("stderr")
                stderr_text = stderr.text if stderr is not None else ""

                # If there is stdout, format and add it.
                if stdout_text and stdout_text.strip():
                    lines.append("|--[stdout]--------------------------------------------")
                    for line in stdout_text.strip().split('\n'):
                        lines.append(f"|  > {line}")

                # If there is stderr, format and add it.
                if stderr_text and stderr_text.strip():
                    lines.append("|--[stderr]--------------------------------------------")
                    for line in stderr_text.strip().split('\n'):
                        lines.append(f"|  > {line}")
                lines.append("|--------------------------------------------------------")
            
            return "\n".join(lines)
        except ET.ParseError as e:
            return f"Error parsing XML for graph rendering: {e}\n\n{xml_string}"

    async def handle_interrogation(self, last_screen: str, chat_history: tp.List[ChatCompletionMessageParam]):
        """
        [ADAPTED FROM server.py] 
        Pauses the agent for a multi-turn conversation and gets responses from the agent.
        """
        print("\n--- AGENT PAUSED: CONVERSATION MODE ---")
        print("Type your questions below. Type 'end', 'resume', or 'exit' to continue the agent's task.")
        
        interrogation_history: tp.List[ChatCompletionMessageParam] = []

        initial_context = (
            f"A human operator has paused you to ask clarifying questions about your last action.\n\n"
            f"--- LAST SCREEN YOU SAW ---\n{last_screen}\n\n"
        )
        interrogation_history.append({'role': 'user', 'content': initial_context})

        try:
            while True:
                question = await asyncio.to_thread(input, "\nYour question ('end' to resume): ")

                if question.strip().lower() in ["end", "resume", "exit"]:
                    break
                if not question.strip():
                    continue

                interrogation_history.append({'role': 'user', 'content': question})

                # Combine histories
                messages: tp.List[ChatCompletionMessageParam] = [
                    {'role': 'system', 'content': self.systemPrinciples()},
                    *chat_history,
                    *interrogation_history
                ]

                print("\n🤔 Agent is thinking...")
                response = await self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                )
                answer = response.choices[0].message.content
                print(f"\n AGENT RESPONSE:\n{answer}")

                if answer:
                    interrogation_history.append({'role': 'assistant', 'content': answer})

        except Exception as e:
            print(f"An error occurred during interrogation: {e}")
        finally:
            print("\n--- AGENT RESUMING ---")

    # --- Execution Handlers ---

    def _command_execution_thread(self, command: str, managed_proc: ManagedProcess):
        """
        Thread that runs the subprocess and pushes output events to the async queue.
        """
        def stream_reader(pipe, is_stderr: bool):
            for line in iter(pipe.readline, ''):
                # Update the process object (history + buffer)
                if is_stderr:
                    managed_proc.append_stderr(line)
                else:
                    managed_proc.append_stdout(line)
                
                # Push Event to Main Async Loop (Thread-Safe)
                # We do not debounce here; we let the main loop aggregate events.
                self.loop.call_soon_threadsafe(
                    self.event_queue.put_nowait,
                    AgentEvent(type=AgentEventType.PROCESS_OUTPUT, source_pid=managed_proc.pid)
                )
            pipe.close()

        try:
            process = subprocess.Popen(
                ['/bin/bash', '-c', command],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=self.current_directory, universal_newlines=True, bufsize=1
            )
            managed_proc.subprocess = process
            
            t_out = threading.Thread(target=stream_reader, args=(process.stdout, False))
            t_err = threading.Thread(target=stream_reader, args=(process.stderr, True))
            t_out.start()
            t_err.start()
            
            process.wait()
            t_out.join()
            t_err.join()

            managed_proc.return_code = process.returncode
            managed_proc.status = "finished" if process.returncode == 0 else "error"
            managed_proc.finished_time = time.time()
            
            # Push Finished Event
            self.loop.call_soon_threadsafe(
                self.event_queue.put_nowait,
                AgentEvent(type=AgentEventType.PROCESS_FINISHED, source_pid=managed_proc.pid)
            )

        except Exception as e:
            print(f"Error in execution thread: {e}")
            managed_proc.status = "failed_start"

    async def handle_new_shell_command(self, command_str: str) -> str:
        if command_str.strip().startswith("cd "):
            try:
                target_dir = command_str.strip().split(" ", 1)[1]
                # Simple expand handling
                if target_dir.startswith("~"): target_dir = os.path.expanduser(target_dir)
                else: target_dir = os.path.join(self.current_directory, target_dir)
                
                os.chdir(target_dir)
                self.current_directory = os.getcwd()
                return f"Changed directory to {self.current_directory}."
            except Exception as e:
                return f"Error changing directory: {e}"

        # Setup Process Object
        proc_container = ManagedProcess(pid=0, command=command_str, subprocess=None) # type: ignore
        
        thread = threading.Thread(target=self._command_execution_thread, args=(command_str, proc_container))
        thread.daemon = True
        thread.start()
        
        # Wait for PID assignment
        for _ in range(20):
            if proc_container.subprocess is not None:
                with self._lock:
                    self.processes[proc_container.subprocess.pid] = proc_container
                    proc_container.pid = proc_container.subprocess.pid
                return f"Started command '{command_str}' (PID: {proc_container.pid})"
            await asyncio.sleep(0.05)
            
        return "Command started, but PID fetch timed out."

    async def handle_set_alarm(self, time_s: float) -> str:
        return f"ALARM_SET:{time_s}"

    async def handle_read_output(self, pid: int, offset: int = 0) -> str:
        with self._lock:
            proc = self.processes.get(pid)
            if not proc: return f"PID {pid} not found."
            
            full_log = proc.full_stdout + "\n--- STDERR ---\n" + proc.full_stderr
            if offset >= len(full_log):
                return "Offset beyond content length."
            
            content = full_log[offset:]
            if len(content) > MAX_SCREEN_OUTPUT_CHARS:
                content = content[:MAX_SCREEN_OUTPUT_CHARS] + "\n... [Remaining content truncated] ..."
            
            return f"--- Output for PID {pid} (Offset {offset}) ---\n{content}"

    async def handle_kill(self, pid: int) -> str:
        with self._lock:
            proc = self.processes.get(pid)
            if proc and proc.status == "running" and proc.subprocess:
                proc.subprocess.terminate()
                return f"Terminated PID {pid}."
        return "Process not found or not running."

    async def handle_report_test_findings(
        self, 
        setup_ease_score: float, 
        execution_difficulty_score: float,
        doc_quality_score: float,
        overall_score: float, 
        justification: str, 
        constructive_feedback: str, 
        key_takeaway: str, 
        **kwargs
    ) -> str:
        report = f"""
================================
=== GitHub Repo Test Report ===
================================
Repo: {self.github_url_to_test}

SCORES:
- Setup Ease:          {setup_ease_score:.1f}/10
- Execution Difficulty:{execution_difficulty_score:.1f}/10
- Doc Quality:         {doc_quality_score:.1f}/10
---------------------------------
- Overall Score:       {overall_score:.1f}/10

JUSTIFICATION:
{justification}

CONSTRUCTIVE FEEDBACK:
{constructive_feedback}

KEY TAKEAWAY:
{key_takeaway}
================================
"""
        print(report)
        try:
            with open(TAKEAWAYS_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(key_takeaway + "\n")
        except: pass
        self.program_should_terminate = True
        return "Report submitted."

    async def cleanup(self):
        with self._lock:
            for pid, proc in self.processes.items():
                if proc.status == "running" and proc.subprocess:
                    try:
                        proc.subprocess.kill()
                    except: pass

# ===============================================================
# 3. Input Listener
# ===============================================================
def input_listener_thread(event_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    while True:
        try:
            cmd = input()
            if cmd.strip().lower() in ["i", "interrogate"]:
                loop.call_soon_threadsafe(
                    event_queue.put_nowait, 
                    AgentEvent(type=AgentEventType.USER_INTERVENTION)
                )
        except: break

# ===============================================================
# 4. Main Event Loop (Replaces main_lai_loop)
# ===============================================================

async def run_agent_event_loop(ui: GithubRepoTesterUI):
    chat_history: tp.List[ChatCompletionMessageParam] = []
    
    # Current active alarm task
    active_alarm_task: tp.Optional[asyncio.Task] = None
    
    print("--- AGENT STARTED (Event-Driven Mode) ---")
    
    # Inject a startup event to wake the agent immediately
    await ui.event_queue.put(AgentEvent(type=AgentEventType.STARTUP))
    
    while not ui.program_should_terminate:
        
        # 1. Wait for an event
        try:
            event = await ui.event_queue.get()
        except asyncio.CancelledError:
            break

        # 2. Debounce / Aggregation Logic
        if event.type in [AgentEventType.PROCESS_OUTPUT, AgentEventType.PROCESS_FINISHED]:
            await asyncio.sleep(2.0) 
            while not ui.event_queue.empty():
                try:
                    next_evt = ui.event_queue.get_nowait()
                    if next_evt.type == AgentEventType.USER_INTERVENTION:
                        event = next_evt
                        break
                    elif next_evt.type == AgentEventType.ALARM_EXPIRATION:
                        pass 
                except Empty:
                    break
        
        # 3. Check Logic: Process Event vs Alarm
        if event.type in [AgentEventType.PROCESS_OUTPUT, AgentEventType.PROCESS_FINISHED, AgentEventType.USER_INTERVENTION]:
            if active_alarm_task and not active_alarm_task.done():
                # print(f"[EventLoop] Clearing Alarm due to {event.type.name}")
                active_alarm_task.cancel()
                active_alarm_task = None
        
        # 4. Render Screen
        screen_xml = ui.render()
        print(f"\n--- SCREEN ({event.type.name}) ---")
        print(ui.render_as_graph(screen_xml))
        
        # 5. Interrogation Handling
        if event.type == AgentEventType.USER_INTERVENTION:
            await ui.handle_interrogation(screen_xml, chat_history)
            continue

        # ================================================================
        # PHASE 1: OBSERVATION (Mandatory Summary)
        # ================================================================
        # We force the LLM to summarize the screen before letting it do anything else.
        # ================================================================
        
        summary_prompt = (
            f"Current Event: {event.type.name}\n\n"
            f"{screen_xml}\n\n"
            f"**PHASE 1: OBSERVATION**\n"
            f"Analyze the screen above. You MUST use the `summarizeScreen` tool to record what just happened into your long-term memory.\n"
            f"If the event is trivial (e.g. just started a command), the summary can be very brief.\n"
            f"Do NOT propose actions yet."
        )
        
        phase1_messages = [
            {'role': 'system', 'content': ui.systemPrinciples()},
            *chat_history,
            {'role': 'user', 'content': summary_prompt}
        ]
        
        phase1_tools = [{'type': 'function', 'function': t[0].model_dump(exclude_none=True)} for t in ui.get_summary_tools()]

        try:
            # print("[Phase 1] Force Summarize...")
            p1_completion = await ui.client.chat.completions.create(model="gpt-4o", messages=phase1_messages, tools=phase1_tools)
            p1_msg = p1_completion.choices[0].message
            chat_history.append(p1_msg) # Add Assistant Message
            
            # Execute Summary Tool
            if p1_msg.tool_calls:
                for tc in p1_msg.tool_calls:
                    if tc.function.name == "summarizeScreen":
                        args = json.loads(tc.function.arguments)
                        summary = args.get("summary", "")
                        ui.summary_history.append(summary)
                        print(f"SCREEN SUMMARY: {summary}")
                        chat_history.append({'role': 'tool', 'tool_call_id': tc.id, 'content': "Summary recorded."})
            else:
                # If LLM failed to call tool (rare with strict prompting), mock it
                print("[WARN] LLM skipped summary. Injecting empty summary.")
        except Exception as e:
            print(f"Error in Summary Phase: {e}")
        
        # =========================================================================
        # PHASE 2: ACTION (Decide & Act)
        # =========================================================================
        # Now we provide the long-term memory and allow actions.
        # =========================================================================
        
        past_summaries = "\n".join(ui.summary_history) or "No summaries yet."
        action_prompt = (
            f"<past_summaries>\n{past_summaries}\n</past_summaries>\n\n"
            f"**PHASE 2: ACTION**\n"
            f"You have summarized the latest event. Now, based on your long-term memory (above) and the current screen, decide the next step.\n"
            f"Use `propose_next_action` to run commands, or `report_test_findings_and_terminate` if done."
        )
        
        phase2_messages = [
            {'role': 'system', 'content': ui.systemPrinciples()},
            *chat_history, # Contains Phase 1 interaction
            {'role': 'user', 'content': action_prompt}
        ]
        
        phase2_tools = [{'type': 'function', 'function': t[0].model_dump(exclude_none=True)} for t in ui.get_action_tools()]

        try:
            # print("[Phase 2] Deciding Action...")
            async def execute_action_tools(tool_calls):
                nonlocal active_alarm_task
                for tc in tool_calls:
                    func_name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    content_result = ""
                    
                    # Logic for all Action Tools
                    if func_name == "new_shell_command":
                        cmd = args.get("command_str")
                        res = await ui.handle_new_shell_command(cmd)
                        print(f"EXECUTING: {cmd}")
                        print(f"COMMAND RESULT: {res}")
                        content_result = res
                    elif func_name == "set_alarm":
                        time_s = float(args.get("time_s", 0))
                        async def alarm_coro():
                            await asyncio.sleep(time_s)
                            await ui.event_queue.put(AgentEvent(type=AgentEventType.ALARM_EXPIRATION))
                        if active_alarm_task: active_alarm_task.cancel()
                        active_alarm_task = asyncio.create_task(alarm_coro())
                        print(f"ACTION: Set Alarm for {time_s}s")
                        content_result = f"Alarm set for {time_s}s."
                    elif func_name == "propose_next_action":
                        ui.current_stage = args.get("current_stage", ui.current_stage)
                        print(f"LLM THOUGHT: {args.get('reasoning')}")
                        cmd = args.get("proposed_command")
                        if cmd:
                            res = await ui.handle_new_shell_command(cmd)
                            print(f"EXECUTING: {cmd}")
                            print(f"COMMAND RESULT: {res}")
                            content_result = res
                        else: content_result = "Stage updated."
                    elif func_name == "report_test_findings_and_terminate":
                        await ui.handle_report_test_findings(**args)
                        content_result = "Report submitted."
                    elif func_name == "read_process_output":
                        res = await ui.handle_read_output(**args)
                        content_result = res
                    elif func_name == "kill":
                        res = await ui.handle_kill(**args)
                        content_result = res
                    
                    chat_history.append({'role': 'tool', 'tool_call_id': tc.id, 'content': str(content_result)})
                
                # Close the loop
                await ui.event_queue.put(AgentEvent(type=AgentEventType.TOOL_COMPLETION))

            p2_completion = await ui.client.chat.completions.create(model="gpt-5", messages=phase2_messages, tools=phase2_tools)
            p2_msg = p2_completion.choices[0].message
            chat_history.append(p2_msg)

            if p2_msg.tool_calls:
                await execute_action_tools(p2_msg.tool_calls)
            

        except Exception as e:
            print(f"Error in Action Phase: {e}")
            traceback.print_exc()

# ===============================================================
# 5. Entry Point
# ===============================================================
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    event_queue = asyncio.Queue()
    
    ui_instance = GithubRepoTesterUI(client, event_queue, loop, initial_repo_url=DEFAULT_REPO_URL_TO_TEST)
    
    # Start Input Listener
    t_input = threading.Thread(target=input_listener_thread, args=(event_queue, loop))
    t_input.daemon = True
    t_input.start()
    
    print("Agent is running. Press 'i' or 'interrogate' then Enter to pause and ask a question.")

    try:
        loop.run_until_complete(run_agent_event_loop(ui_instance))
    except KeyboardInterrupt:
        print("\nManual Shutdown.")
    finally:
        loop.run_until_complete(ui_instance.cleanup())
        loop.close()