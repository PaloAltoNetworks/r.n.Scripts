from google import genai

from vertexai.preview.generative_models import GenerativeModel, Part
import networkx as nx
import re
import docker  # Import the Docker library
import logging
from tree_sitter import Language, Parser
from utils import node_to_dict, write_to_file, LANGUAGE_MAP
from docker_orchestrator import run_deobfuscation_in_docker

import questionary
from config import (
    GENERATION_CONFIG, SAFETY_SETTING, USER_MODEL, VERTEX_AI_INITIALIZED
)
# We don't import PROMPT_TEMPLATES, SYSTEM_INSTRUCTION_RENAME here anymore, they are passed dynamically
# from config import PROMPT_TEMPLATES, SYSTEM_INSTRUCTION_RENAME

import google.generativeai as genai  # Need this alias for token counting
from colors import *

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    def __init__(self, model_name: str = list(USER_MODEL.keys())[0], logger: logging.Logger = logger):
        self.logger = logger
        self.model_name = model_name
        self._current_system_instruction = None  # To be set dynamically
        self._current_prompt_template = None  # To be set dynamically

        if not VERTEX_AI_INITIALIZED:
            self.logger.error("Vertex AI not initialized. LLMAnalyzer cannot be used.")
            print(f"{ERROR_COLOR}[-] Warning: Vertex AI not initialized. LLM features will likely fail.{COLOR_RESET}")

        self.base_model = None  # Initialize here, but build in set_analysis_context

    def set_analysis_context(self, system_instruction: str, prompt_template: str):
        """Sets the system instruction and prompt template for subsequent LLM calls."""
        self._current_system_instruction = system_instruction
        self._current_prompt_template = prompt_template
        if VERTEX_AI_INITIALIZED:
            try:
                self.base_model = GenerativeModel(self.model_name, system_instruction=self._current_system_instruction)
                self.logger.info(f"LLM model '{self.model_name}' context set with specific system instruction.")
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize LLM model '{self.model_name}' with new system instruction: {e}")
                print(
                    f"{ERROR_COLOR}[-] Error: Failed to initialize LLM model '{self.model_name}'. Check model name or config: {e}{COLOR_RESET}")

    @staticmethod
    def analyze_code(ct_type, code):
        try:
            language = LANGUAGE_MAP.get(ct_type)
            if language is None:
                logger.warning(f"Unsupported language type for analysis: {ct_type}")
                print(
                    f"{WARNING_COLOR}[!] Warning: Unsupported language type '{ct_type}' for analysis. Returning None.{COLOR_RESET}")
                return None, None

            LANGUAGE = Language(language)
            parser = Parser(LANGUAGE)
            tree = parser.parse(code)
            root_node = tree.root_node
            node_dict = node_to_dict(root_node, code)
            return tree, node_dict
        except Exception as e:
            logger.error(f"An error occurred during code analysis for type '{ct_type}': {e}")
            print(f"{ERROR_COLOR}[!] Error during code analysis for type '{ct_type}': {e}{COLOR_RESET}")
            return None, None

    def _generate_model_response(self, documents, stream=True):
        if not self.base_model:
            self.logger.error("LLM model not initialized or context not set. Cannot generate response.")
            print(f"{ERROR_COLOR}[!] LLM model not ready. Did you call set_analysis_context?{COLOR_RESET}")
            return None
        if not self._current_prompt_template:
            self.logger.error("No prompt template set. Cannot generate response.")
            print(f"{ERROR_COLOR}[!] No prompt template specified. Did you call set_analysis_context?{COLOR_RESET}")
            return None

        try:
            print(f"\n{STREAM_COLOR}---LLM Response Streaming from Vertex AI:---{COLOR_RESET}")

            responses = self.base_model.generate_content([
                self._current_prompt_template,  # Use the selected prompt template
                documents],
                generation_config=GENERATION_CONFIG,
                safety_settings=SAFETY_SETTING,
                stream=stream,
            )
            response_text = ""
            for response in responses:
                if response.text:
                    print(response.text, end="", flush=True)
                    response_text += response.text
                else:
                    self.logger.warning(
                        f"Received empty or problematic response chunk. Model might have issues or content was blocked.")
            print(f"\n{STREAM_COLOR}---LLM Response Streaming completed.---{COLOR_RESET}")
            return response_text
        except Exception as e:
            self.logger.error(f"Error during streaming response: {e}")
            print(f"{ERROR_COLOR}[!] Error during LLM streaming response: {e}{COLOR_RESET}")
            return None

    def generate_ast(self, ast_json):
        # The system instruction is already set via set_analysis_context
        document1 = Part.from_data(mime_type="text/plain", data=ast_json.encode('utf-8'))
        return self._generate_model_response(document1)

    def generate_graph(self, graph_json):
        # The system instruction is already set via set_analysis_context
        document1 = Part.from_data(mime_type="text/plain", data=graph_json.encode('utf-8'))
        return self._generate_model_response(document1)

    def generate_ast_graph(self, ast_json, graph_json):
        # The system instruction is already set via set_analysis_context
        AST_CONTENT = f"AST JSON:\n\n{ast_json}\n"
        GRAPH_CONTENT = f"Graph JSON:\n{graph_json}\n"
        combine = AST_CONTENT + GRAPH_CONTENT
        document1 = Part.from_data(mime_type="text/plain", data=combine.encode('utf-8'))
        return self._generate_model_response(document1)

    def generate_code_summary(self, code):
        # The system instruction is already set via set_analysis_context
        document1 = Part.from_data(mime_type="text/plain", data=code.encode('utf-8'))
        return self._generate_model_response(document1)

    # REVISED: This function now RETURNS the formatted string content
    # It no longer calls write_to_file directly
    def process_llm_response(self, response_text_from_llm: str, prompt_key_for_report: str, combine_llm_header: str,
                             docker_running: str, original_code: str, hash_value: str, model_selection: str) -> str:
        """
        Processes LLM response, potentially runs Docker, and returns the formatted
        string content (including Docker output if applicable) ready for file writing.
        This function does NOT write to file.
        """

        # Start with the basic LLM response section
        processed_section_content = f"Gemini Response:\n{'-' * 100}\n\n{response_text_from_llm}\n\n{'-' * 100}"

        # Handle Docker execution specifically for Deobfuscation prompt
        # Use prompt_key_for_report to check whether it's the specific Deobfuscation prompt
        if prompt_key_for_report == "Focus: Deobfuscation Code Generation (Option: Execute Code in Docker Container)":
            python_code_match = re.search(r"```python(.*?)```", response_text_from_llm, re.DOTALL | re.IGNORECASE)
            docker_file_match = re.search(r"```dockerfile(.*?)```", response_text_from_llm, re.DOTALL | re.IGNORECASE)

            dockerfile = docker_file_match.group(1).strip() if docker_file_match else ""

            if python_code_match and docker_file_match and docker_running == "Yes":
                print(f"{INFO_COLOR}\n[+] - Docker daemon is running.{COLOR_RESET}")
                run_container = questionary.select(
                    f"\nChoose to execute Python code in Docker container ?:",
                    choices=["Yes", "No"]
                ).ask()

                if run_container == "Yes":
                    python_code = python_code_match.group(1).strip()
                    self.logger.info(
                        f"Attempting to run Docker with Python code:\n{python_code}\nDockerfile:\n{dockerfile}")
                    docker_output, exit_code = run_deobfuscation_in_docker(python_code, dockerfile, original_code)
                    print(f"{INFO_COLOR}[+] Docker execution finished with exit code {exit_code}.{COLOR_RESET}")
                    print(f"{INFO_COLOR}Docker Output (raw) will be included below.{COLOR_RESET}")

                    # Append Docker output directly to the processed section content
                    processed_section_content += f"\n\nDocker Output:\n```\n{docker_output}\n```"
                    self.logger.info("Docker results appended to LLM response.")

                    # Ask for re-run if Docker execution was part of it
                    rerun_sample = questionary.select(
                        f"\nChoose to re-run sample:",
                        choices=["Yes", "No"]
                    ).ask()

                    if rerun_sample == "Yes":
                        troubleshoot_prompt = f"Troubleshoot the error, something went wrong with the python code. " \
                                              f"The original LLM response was:\n{response_text_from_llm}\n\n" \
                                              f"The following error occurred, fix it so the Docker container can run correctly. " \
                                              f"Docker Output:\n```\n{docker_output}\n```"

                        print(f"{INFO_COLOR}\n[+] - Re-running generation for troubleshooting...{COLOR_RESET}")
                        # For troubleshooting, reuse the current fixed system instruction and just pass the prompt
                        new_llm_response_text = self._generate_model_response(
                            Part.from_data(mime_type="text/plain", data=troubleshoot_prompt.encode('utf-8'))
                        )

                        python_code_match_rerun = re.search(r"```python(.*?)```", new_llm_response_text,
                                                            re.DOTALL | re.IGNORECASE)
                        docker_file_match_rerun = re.search(r"```dockerfile(.*?)```", new_llm_response_text,
                                                            re.DOTALL | re.IGNORECASE)

                        if python_code_match_rerun and docker_file_match_rerun:
                            python_code_rerun = python_code_match_rerun.group(1).strip()
                            dockerfile_rerun = docker_file_match_rerun.group(1).strip()
                            self.logger.info("Attempting re-run with fixed code from LLM.")
                            docker_output_rerun, exit_code_rerun = run_deobfuscation_in_docker(python_code_rerun,
                                                                                               dockerfile_rerun,
                                                                                               original_code)
                            print(
                                f"{INFO_COLOR}[+] Docker (re-run) execution finished with exit code {exit_code_rerun}.{COLOR_RESET}")

                            # Replace the entire processed section content with the re-run version
                            processed_section_content = f"Gemini Response (Troubleshooted):\n{'-' * 100}\n\n{new_llm_response_text}\n\n{'-' * 100}" + \
                                                        f"\n\nDocker Output (Rerun):\n```\n{docker_output_rerun}\n```"
                            self.logger.info("Re-run Docker results added to output.")
                        else:
                            self.logger.error(
                                "LLM did not provide valid Python or Dockerfile in troubleshooting response.")
                            print(
                                f"{ERROR_COLOR}[!] LLM did not provide valid Python or Dockerfile in troubleshooting response. No re-run output generated.{COLOR_RESET}")
                            # Indicate that the re-run failed and original output is kept
                            processed_section_content += f"\n\n[!] Troubleshooting re-run failed: LLM did not output valid Python/Dockerfile. Keeping initial results."

                elif run_container == "No":
                    self.logger.info("User chose not to execute Docker container.")
                    print(f"{INFO_COLOR}[+] Skipping Docker container execution as requested.{COLOR_RESET}")
            else:
                print(
                    f"{WARNING_COLOR}\nDocker daemon is NOT running or valid Python/Dockerfile patterns not found in response. Skipping Docker execution.{COLOR_RESET}")
                self.logger.warning(
                    "Docker daemon is NOT running or valid patterns not found. Skipping Docker execution.")

        return processed_section_content  # Return the compiled string content

    def count_tokens(self, text: str) -> int:
        """Counts tokens using the currently selected analysis model."""
        if not VERTEX_AI_INITIALIZED or not text:
            return 0
        try:
            model = genai.GenerativeModel(self.model_name)
            token_count = model.count_tokens(text).total_tokens
            return token_count
        except Exception as e:
            self.logger.error(f"Error counting tokens for model '{self.model_name}': {e}")
            print(f"{ERROR_COLOR}[-] Warning: Could not count tokens for model '{self.model_name}': {e}{COLOR_RESET}")
            return 0