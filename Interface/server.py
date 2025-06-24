import subprocess
import threading
import time
import signal
from openai import OpenAI, AsyncOpenAI
import os
import json
from queue import Queue, Empty
from dotenv import load_dotenv, find_dotenv

# --- Type Hinting Imports ---
import typing as tp
import asyncio
from dataclasses import dataclass
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
DEFAULT_REPO_URL_TO_TEST: str = "https://github.com/mkaz/termgraph"

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
        "If nothing significant changed that you don't already know from previous summaries, state that clearly (e.g., 'Screen shows command X still running, no new output.')."
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
    def render(self) -> tp.Tuple[str, str, tp.List[tp.Tuple[FunctionDefinition, UIEventHandler]]]:
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
        self.job_state: JobStateMachine = JobStateMachine()
        self.process_output_queue: Queue = Queue()
        
        self.current_subprocess: tp.Optional[subprocess.Popen] = None
        self.subprocess_thread: tp.Optional[threading.Thread] = None
        self.kill_event: threading.Event = threading.Event()

        self.github_url_to_test: str = initial_repo_url or DEFAULT_REPO_URL_TO_TEST
        self.cloned_repo_name: str = self.github_url_to_test.split('/')[-1].replace('.git', '') if self.github_url_to_test else "repository"
        
        # MODIFIED: Reinstated last_command_output to hold full results for render()
        self.last_command_output: tp.Dict[str, tp.Any] = {"stdout": "", "stderr": "", "return_code": None, "elapsed_time": 0}
        self.last_action_outcome_for_tool_message: tp.Optional[str] = None
        self.program_should_terminate: bool = False
        
        self.SYSTEM_PROMPT = f"""
### Your job
You are a fully autonomous agent whose job is to test the usability of a given GitHub repository. Play the role of a user invoking the repo for the first time. After completing your evaluation, summarize your findings in a concise report.  
Design and execute comprehensive test plans that exercise as many repository features as possible, acknowledging that some capabilities may lie outside your immediate testing scope.  
Assess documentation clarity, verifying that examples are understandable and appropriately concise, and ensure that the repository\'s tone aligns with its intended functionality.  
Maintain a professional, exploratory tone, explicitly stating your Reasoning and observations.

### Operator Authority
You have full control over a dedicated OS environment via a rule-based terminal interface. No human assistance will fill in blanks for you. You are responsible for configuring the system, debugging errors, and retuning the environment as needed. Try your best before giving up.

### Interaction
- You will be provided with several functions to execute commands and manage processes within the terminal environment.  
- The \'screen\' displays the current working directory, process status, and live output.
- When calling a function, do not forgot to include your chain-of-thought in your response.
- Execute **ONE** shell command per `run_shell_command` call.
- Prioritize core CLI commands and features; deprioritize or skip features that require extensive GUI interaction or are clearly out of the intended testing scope.

### Guidelines
- If `cd` fails with a relative path, retry with the absolute path.
- Never use interactive browsers or TUIs (e.g., nano, vim); interact only via text-based commands.
- Allow sufficient time for lengthy operations if they continuously produce new output.

### Evaluation and Scoring
Upon completing your tests, assign star-ratings (1.0-5.0) for each of these criteria:
- **Environment Setup Ease**: how simple it is to configure and install dependencies.
- **Execution Difficulty**: how straightforward it is to run the core functionality.
- **Documentation Quality**: clarity, completeness, and usefulness of the README and examples.
Then calculate and report an **Overall Score** (1.0-5.0) along with a brief justification for each rating.
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
    RUN_SHELL_COMMAND_DEF = FunctionDefinition(name="run_shell_command", description="Executes a single shell command (e.g., 'git clone ...', 'ls', 'cd ...', 'cat ...', 'pip install ...').", parameters={"type": "object", "properties": {"command": {"type": "string", "description": "The shell command to execute."}}, "required": ["command"]})
    WAIT_DEF = FunctionDefinition(name="wait", description="Pauses execution. Use if a command is running to see more output, or to briefly pause and plan. Can optionally specify duration.", parameters={"type": "object", "properties": {"duration": {"type": "number", "description": "Optional. Seconds to wait."}}})
    KILL_COMMAND_DEF = FunctionDefinition(name="kill_current_command", description="Terminates the currently running shell command, if any.", parameters={"type": "object", "properties": {}})
    REPORT_FINDINGS_DEF = FunctionDefinition(
        name="report_test_findings_and_terminate",
        description="Submits your final analysis, including star-ratings for specific criteria and an overall score, then terminates the agent.",
        parameters={
            "type": "object",
            "properties": {
                "setup_ease_score": {"type": "number", "description": "Star-rating (1.0-5.0) for environment setup and dependency installation ease."},
                "execution_difficulty_score": {"type": "number", "description": "Star-rating (1.0-5.0) for how straightforward it was to run core functionality."},
                "doc_quality_score": {"type": "number", "description": "Star-rating (1.0-5.0) for the quality and clarity of the documentation."},
                "overall_score": {"type": "number", "description": "Your final overall star-rating (1.0-5.0) for the repository's usability."},
                "justification": {"type": "string", "description": "A concise justification for your scores and a summary of your experience."}
            },
            "required": ["setup_ease_score", "execution_difficulty_score", "doc_quality_score", "overall_score", "justification"],
        }
    )
    def render(self) -> tp.Tuple[str, str, tp.List[tp.Tuple[FunctionDefinition, UIEventHandler]]]:
        screen_as_text = f"Time: {time.strftime('%H:%M:%S')} | Repo: {self.github_url_to_test} | CWD: {self.current_directory} | Cmd Status: {self.job_state.state}\n"
        
        # MODIFIED: Always show the result of the last completed command on the screen.
        if self.job_state.is_state("command_finished") or self.job_state.is_state("error"):
            screen_as_text += f"\n--- Last Command Result ---\n"
            MAX_OUTPUT_LEN_SCREEN = 2000 
            stdout_display = self.last_command_output.get('stdout', '')
            stderr_display = self.last_command_output.get('stderr', '')
            if len(stdout_display) > MAX_OUTPUT_LEN_SCREEN:
                stdout_display = stdout_display[:MAX_OUTPUT_LEN_SCREEN] + "\n... (stdout truncated in screen display) ..."
            if len(stderr_display) > MAX_OUTPUT_LEN_SCREEN:
                stderr_display = stderr_display[:MAX_OUTPUT_LEN_SCREEN] + "\n... (stderr truncated in screen display) ..."
            screen_as_text += f"STDOUT:\n{stdout_display}\n"
            screen_as_text += f"STDERR:\n{stderr_display}\n"
            screen_as_text += f"Return Code: {self.last_command_output.get('return_code')}\n"
            screen_as_text += "---------------------------\n"

        if self.job_state.is_state("running_command"):
            live_output = self._read_process_output_queue()
            screen_as_text += f"--- Live Output ---\n{live_output or 'Awaiting output...'}\n-------------------\n"
        
        instructions = "Review screen. Decide next action towards testing the repository."
        if self.program_should_terminate:
            instructions = "Termination initiated."
            return instructions, screen_as_text, []

        available_functions: tp.List[tp.Tuple[FunctionDefinition, UIEventHandler]] = [
            (self.RUN_SHELL_COMMAND_DEF, self.handle_run_shell_command),
            (self.WAIT_DEF, self.handle_wait),
            (self.REPORT_FINDINGS_DEF, self.handle_report_test_findings)
        ]
        if self.job_state.is_state("running_command"):
            available_functions.append((self.KILL_COMMAND_DEF, self.handle_kill_current_command))
        
        return instructions, screen_as_text, available_functions

    # --- Handler Method Implementations ---
    def _command_execution_thread(self, command: str):
        self.kill_event.clear()
        start_time = time.time()
        
        # This thread populates the final result into self.last_command_output
        self.last_command_output = {"stdout": "", "stderr": "", "return_code": None, "elapsed_time": 0}
        full_stdout_lines, full_stderr_lines = [], []

        while not self.process_output_queue.empty():
            try: self.process_output_queue.get_nowait()
            except Empty: break

        try:
            self.current_subprocess = subprocess.Popen(
                ['/bin/bash', '-c', command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=self.current_directory, universal_newlines=True, bufsize=1
            )
            def stream_reader(pipe, source_name, line_list):
                if pipe:
                    for line in iter(pipe.readline, ''):
                        if self.kill_event.is_set(): break
                        self.process_output_queue.put({"source": source_name, "line": line.strip()})
                        line_list.append(line.strip())
                    pipe.close()
            
            stdout_thread = threading.Thread(target=stream_reader, args=(self.current_subprocess.stdout, "stdout", full_stdout_lines))
            stderr_thread = threading.Thread(target=stream_reader, args=(self.current_subprocess.stderr, "stderr", full_stderr_lines))
            stdout_thread.start(); stderr_thread.start()
            
            stdout_thread.join(); stderr_thread.join()
            self.current_subprocess.wait()

            if self.kill_event.is_set():
                self.last_command_output["return_code"] = -1
                full_stderr_lines.append("Process terminated by kill signal.")
            else:
                self.last_command_output["return_code"] = self.current_subprocess.returncode
        except Exception as e:
            full_stderr_lines.append(f"System error executing command: {e}")
            self.last_command_output["return_code"] = -2
        finally:
            self.last_command_output["stdout"] = "\n".join(full_stdout_lines)
            self.last_command_output["stderr"] = "\n".join(full_stderr_lines)
            self.last_command_output["elapsed_time"] = time.time() - start_time
            self.job_state.set_state("command_finished" if self.last_command_output["return_code"] == 0 else "error")
            self.current_subprocess = None

    def handle_run_shell_command(self, command: str) -> float:
        log_interaction_agent(f"[ACTION] Run Shell Command: {command}")
        # Reset last output before new command
        self.last_command_output = {}

        if self.job_state.is_state("running_command"):
            self.last_action_outcome_for_tool_message = "Error: Another command is already running."
            return 1.0

        if command.strip().startswith("cd "):
            parts = command.strip().split(" ", 1)
            target_dir = parts[1] if len(parts) > 1 else "~"
            original_dir = self.current_directory
            try:
                effective_target = os.path.join(self.current_directory, target_dir) if not target_dir.startswith(('/', '~')) else os.path.expanduser(target_dir)
                os.chdir(os.path.normpath(effective_target))
                self.current_directory = os.getcwd()
                self.job_state.set_state("command_finished")
                self.last_action_outcome_for_tool_message = f"Command '{command}' succeeded. CWD is now {self.current_directory}."
                self.last_command_output = {"stdout": f"Changed directory to {self.current_directory}", "stderr": "", "return_code": 0}
            except Exception as e:
                self.current_directory = original_dir
                self.job_state.set_state("error")
                self.last_action_outcome_for_tool_message = f"Command '{command}' failed: {e}"
                self.last_command_output = {"stdout": "", "stderr": str(e), "return_code": 1}
            return 0.1
        
        self.job_state.set_state("running_command")
        self.subprocess_thread = threading.Thread(
            target=self._command_execution_thread, args=(command,)
        )
        self.subprocess_thread.daemon = True
        self.subprocess_thread.start()
        self.subprocess_thread.join()
        
        rc = self.last_command_output.get('return_code', 'N/A')
        out = self.last_command_output.get('stdout', '')[:500]
        err = self.last_command_output.get('stderr', '')[:500]
        self.last_action_outcome_for_tool_message = f"Command finished with RC={rc}. Stdout: '{out}...', Stderr: '{err}...'"
        
        return 1.0

    def handle_wait(self, duration: tp.Optional[float] = None) -> float:
        actual_duration = duration if duration is not None and duration > 0 else 3.0
        log_interaction_agent(f"[ACTION] Wait ({actual_duration:.1f}s)")
        self.last_action_outcome_for_tool_message = f"Waited for {actual_duration:.1f}s."
        return actual_duration

    def handle_kill_current_command(self) -> float:
        log_interaction_agent("[ACTION] Kill Current Command")
        if self.job_state.is_state("running_command") and self.current_subprocess:
            self.kill_event.set()
            self.last_action_outcome_for_tool_message = "Kill signal sent to running command."
        else:
            self.last_action_outcome_for_tool_message = "No command was running to kill."
        return 1.0

    def handle_report_test_findings(
        self, 
        setup_ease_score: float, 
        execution_difficulty_score: float, 
        doc_quality_score: float, 
        overall_score: float, 
        justification: str
    ) -> float:
        # Format the new scored report
        final_report = f"""
================================
=== GitHub Repo Test Report ===
================================
Repository: {self.github_url_to_test}

--- SCORING ---
- Environment Setup Ease:     {setup_ease_score:.1f} / 5.0
- Execution Difficulty:       {execution_difficulty_score:.1f} / 5.0
- Documentation Quality:      {doc_quality_score:.1f} / 5.0
---------------------------------
- Overall Score:              {overall_score:.1f} / 5.0

--- JUSTIFICATION & SUMMARY ---
{justification}
================================
        """
        print(final_report) # Print to console
        log_interaction_agent(f"[ACTION] Test Report Submitted:\n{final_report}") # Log it
        
        self.program_should_terminate = True
        self.job_state.set_state("terminated")
        self.last_action_outcome_for_tool_message = "Test findings and scores reported; program will terminate."
        return 0.1

    async def cleanup(self):
        print("[CLEANUP] Cleanup called.")
        if self.current_subprocess and self.current_subprocess.poll() is None:
            print("[CLEANUP] Active subprocess found. Signaling kill_event.")
            self.kill_event.set()
            if self.subprocess_thread and self.subprocess_thread.is_alive():
                self.subprocess_thread.join(timeout=2.0)
            if self.current_subprocess and self.current_subprocess.poll() is None:
                print("[CLEANUP] Forcing kill on subprocess.")
                self.current_subprocess.kill()

# ===============================================================
# 4. Logging
# ===============================================================
interaction_log_buffer: tp.List[tp.Dict[str, str]] = []
def log_interaction_agent(message: str, role: str = "system_internal"):
    print(message) 
    interaction_log_buffer.append({"role": role, "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'), "content": message})

# MODIFIED: Logging function updated to save full API call transcript for a turn
def save_interaction_log_file(screen_id: int, api_call_payloads: list, other_logs: list):
    try:
        with open(LOG_FILE_PATH, "a", encoding='utf-8') as f:
            f.write(f"\n==================== FULL LOG FOR TURN {screen_id} ({time.strftime('%Y-%m-%d %H:%M:%S')}) ====================\n")
            
            for i, payload in enumerate(api_call_payloads):
                f.write(f"\n--- API Call #{i+1} ---\n")
                f.write("--- PAYLOAD SENT TO GPT ---\n")
                f.write(json.dumps(payload['messages_sent'], indent=2))
                f.write("\n\n--- REPLY RECEIVED FROM GPT ---\n")
                f.write(json.dumps(payload['gpt_reply'], indent=2))
                f.write("\n---------------------------\n")

            f.write("\n--- Other System & Action Logs for this Turn ---\n")
            for entry in other_logs:
                f.write(f"[{entry['timestamp']}] {entry['role'].upper()}: {entry['content']}\n")
            f.write(f"==================== END OF TURN {screen_id} ====================\n\n")

        interaction_log_buffer.clear() 
    except Exception as e:
        print(f"Error saving log file: {e}")


# ===============================================================
# 5. Main LAI Loop
# ===============================================================
async def main_lai_loop(
    laiAsUI: GithubRepoTesterUI, client_openai: AsyncOpenAI,
    model_name: str = "gpt-4.1-mini", max_tokens: int = 2000, temperature: float = 0.05,
):
    chat_history_for_llm: tp.List[tp.Union[LAIReply, ChatCompletionMessage, ChatCompletionMessageParam]] = []
    id_iter = itertools.count(0)

    def Messages(last_lai_reply: LAIReply) -> tp.List[ChatCompletionMessageParam]:
        processed_messages: tp.List[ChatCompletionMessageParam] = [{'role': 'system', 'content': laiAsUI.systemPrinciples()}]
        for msg in chat_history_for_llm:
            if isinstance(msg, LAIReply):
                processed_messages.append({'role': 'user', 'content': msg.for_agent})
            elif isinstance(msg, ChatCompletionMessage): 
                assistant_msg: ChatCompletionMessageParam = {'role': msg.role, 'content': msg.content}
                if msg.tool_calls:
                    assistant_msg['tool_calls'] = [{'id': tc.id, 'type': tc.type, 'function': {'name': tc.function.name, 'arguments': tc.function.arguments}} for tc in msg.tool_calls]
                if assistant_msg['content'] is None and not assistant_msg.get('tool_calls'): assistant_msg['content'] = "" 
                elif assistant_msg.get('content') == "" and assistant_msg.get('tool_calls'): assistant_msg['content'] = None
                processed_messages.append(assistant_msg)
            elif isinstance(msg, dict) and msg.get('role') == 'tool':
                processed_messages.append(msg)
        processed_messages.append({'role': 'user', 'content': last_lai_reply.for_agent})
        return processed_messages

    try:
        while not laiAsUI.program_should_terminate:
            current_screen_id = next(id_iter)
            api_call_logs_for_turn = [] # To store payloads for logging
            raw_instructions, raw_screen_as_text, raw_available_funcs = laiAsUI.render()
            
            log_interaction_agent(f"\n--- Screen ID: {current_screen_id} ---\nInstructions: {raw_instructions}\nScreen Text:\n{raw_screen_as_text}\n--- End Screen ---", role="agent_ui_render")

            # Summarization Step
            summary = "Summary not generated or an error occurred."
            to_summarize_reply = LAIReply(id_=current_screen_id, instructions="Review the screen. Summarize all new and important information for your future reference.", screen_as_text=raw_screen_as_text)
            try:
                summary_messages_to_send = Messages(to_summarize_reply)
                summary_completion = await client_openai.chat.completions.create(
                    model=model_name, messages=summary_messages_to_send,
                    tools=[{'type': 'function', 'function': tp.cast(FunctionDefinitionParam, SUMMARIZE_SCREEN_FUNCTION_DEF.model_dump(exclude_none=True))}],
                    tool_choice={'type': 'function', 'function': {'name': SUMMARIZE_SCREEN_FUNCTION_DEF.name}},
                    max_tokens=500, temperature=0.0)
                assistant_summary_message = summary_completion.choices[0].message
                
                api_call_logs_for_turn.append({'messages_sent': summary_messages_to_send, 'gpt_reply': assistant_summary_message.model_dump()})

                chat_history_for_llm.append(to_summarize_reply) 
                chat_history_for_llm.append(assistant_summary_message)
                if assistant_summary_message.tool_calls:
                    tool_call = assistant_summary_message.tool_calls[0]
                    args = json.loads(tool_call.function.arguments)
                    summary = args.get('summary', 'Summary was empty.')
                    chat_history_for_llm.append({"tool_call_id": tool_call.id, "role": "tool", "content": summary})
                else: summary = assistant_summary_message.content or "Summary call: no tool use or content."
                log_interaction_agent(f"Screen Summary (ID: {current_screen_id}): {summary}", role="llm_summary")
            except Exception as e: log_interaction_agent(f"Summarization API call failed: {e}", role="error")

            # Action Step
            lai_reply_for_action = LAIReply(id_=current_screen_id, instructions=raw_instructions, screen_as_text=raw_screen_as_text, screen_summary=summary)
            tool_definitions_for_llm = [{'type': 'function', 'function': tp.cast(FunctionDefinitionParam, f[0].model_dump(exclude_none=True))} for f in raw_available_funcs]
            min_wait_time = 5.0 

            if not tool_definitions_for_llm and not laiAsUI.program_should_terminate:
                chat_history_for_llm.append(lai_reply_for_action); await asyncio.sleep(min_wait_time); continue

            try:
                action_messages_to_send = Messages(lai_reply_for_action)
                action_completion = await client_openai.chat.completions.create(
                    model=model_name, messages=action_messages_to_send, tools=tool_definitions_for_llm or None,
                    tool_choice="auto" if tool_definitions_for_llm else None, max_tokens=max_tokens, temperature=temperature)
                llm_action_message = action_completion.choices[0].message
                
                api_call_logs_for_turn.append({'messages_sent': action_messages_to_send, 'gpt_reply': llm_action_message.model_dump()})
                
                chat_history_for_llm.append(lai_reply_for_action) 
                chat_history_for_llm.append(llm_action_message)

                if llm_action_message.content:
                    log_interaction_agent(f"LLM Chain of Thought: {llm_action_message.content}", role="llm_thought")
                if llm_action_message.tool_calls:
                    wait_times_from_handlers = []
                    for tool_call in llm_action_message.tool_calls:
                        function_name = tool_call.function.name
                        handler = next((h for f, h in raw_available_funcs if f.name == function_name), None)
                        if handler:
                            try:
                                args = json.loads(tool_call.function.arguments or '{}')
                                wait_time = await asyncio.to_thread(handler, **args)
                                wait_times_from_handlers.append(wait_time)
                                chat_history_for_llm.append({"tool_call_id": tool_call.id, "role": "tool", "content": str(laiAsUI.last_action_outcome_for_tool_message)[:2000]})
                            except Exception as he: log_interaction_agent(f"Handler error for {function_name}: {he}", role="error"); wait_times_from_handlers.append(1.0)
                        else: log_interaction_agent(f"Unknown func: {function_name}", role="error"); wait_times_from_handlers.append(1.0)
                    min_wait_time = min(wait_times_from_handlers) if wait_times_from_handlers else 5.0
                else: log_interaction_agent(f"LLM no tool use: {llm_action_message.content}", role="llm_action"); min_wait_time = 10.0
            except Exception as e: log_interaction_agent(f"Action API/processing error: {e}", role="error"); min_wait_time = 10.0 
            
            # Save logs for the completed turn
            save_interaction_log_file(current_screen_id, api_call_logs_for_turn, interaction_log_buffer)

            log_interaction_agent(f"Loop end {current_screen_id}. Wait: {min_wait_time:.2f}s.", "system_internal")
            await asyncio.sleep(max(0.1, min_wait_time))
            max_hist = 15
            if len(chat_history_for_llm) > max_hist: chat_history_for_llm = chat_history_for_llm[-max_hist:]

    except asyncio.CancelledError: log_interaction_agent("LAI loop cancelled.", "system_internal")
    finally: await laiAsUI.cleanup(); log_interaction_agent("LAI loop finished.", "system_internal")

# ===============================================================
# 6. Entry Point
# ===============================================================
if __name__ == "__main__":
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
    except asyncio.CancelledError:
        log_interaction_agent("Main task cancelled.", "system_event")
    except Exception as e:
        log_interaction_agent(f"UNHANDLED EXCEPTION: {e}", "critical_error")
        log_interaction_agent(traceback.format_exc(), "critical_error_trace")
    finally:
        print("Final log save."); 
        save_interaction_log_file(-1, [], interaction_log_buffer) # Save any final entries
        print("Exited.")