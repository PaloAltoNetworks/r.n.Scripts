import sys
import logging
import os
import questionary
import networkx as nx
import json
from config import (INPUT_TOKEN_PRICING_PER_MILLION, USER_MODEL, PROMPT_TEMPLATES, ANALYSIS_METHOD,
                    SYSTEM_INSTRUCTION_SUMMARY_PERSONAS, SUMMARY_PROMPT_TEMPLATES)
from utils import generate_rn_script_ascii_art, magika_type, read_file, write_to_file, sha256_hash, \
    shannon_entropy_scipy, convert_ast_to_graph, generate_ast, get_all_shortest_path_edges_subgraph, trim_ast_json_dict
from docker_orchestrator import is_docker_running
# Ensure correct import of LLMAnalyzer and the module-level logger
from llm_analyzer import LLMAnalyzer, logger as module_logger  # Renamed to avoid confusion with local logger

import colorama
colorama.init()
from colors import *


# Configure logging - This should be done once when the script starts
# Using module_logger for configuration
log_initialized = False

def configure_main_logger(sha256_hash, output_dir="."):
    global log_initialized
    if not log_initialized:
        log_filename = f"{sha256_hash}_rnScript.log"
        log_filepath = os.path.join(output_dir, log_filename)

        module_logger.setLevel(logging.INFO)  # Set the level for the imported logger

        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        module_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        module_logger.addHandler(console_handler)
        log_initialized = True
    return module_logger  # Return the configured logger instance


# Use module_logger throughout main.py for consistent logging
logger = module_logger


def calculate_input_cost(model_name: str, input_tokens: int) -> float | None:
    """
    Calculates the estimated input cost based on the model name and token count.
    """
    pricing_info = INPUT_TOKEN_PRICING_PER_MILLION.get(model_name)

    if pricing_info is None:
        logger.warning(f"Pricing data not available for model '{model_name}'. Cannot estimate cost.")
        print(
            f"\n{WARNING_COLOR}[!] Warning: Pricing data not available for model '{model_name}'. Cannot estimate cost.{COLOR_RESET}")
        return None

    price_per_million = 0.0
    if isinstance(pricing_info, dict):
        threshold = pricing_info.get("threshold", 0)
        if input_tokens <= threshold:
            price_per_million = pricing_info.get("under_threshold", 0)
        else:
            price_per_million = pricing_info.get("over_threshold", pricing_info.get("under_threshold", 0))
    else:
        price_per_million = pricing_info

    cost = (input_tokens / 1_000_000.0) * price_per_million
    return cost


if __name__ == "__main__":
    ascii_art = generate_rn_script_ascii_art(font="standard")
    print(f"{PROMPT_COLOR}{ascii_art}{COLOR_RESET}")
    print("\nWelcome to r.n.Scripts: Script Analysis Tool\n")
    user_input = input(f"{PROMPT_COLOR}\nEnter the path to the file: {COLOR_RESET}").strip("\"")

    code = read_file(user_input)
    if code is None:
        print(f"{ERROR_COLOR}[!] Error reading the input file. Exiting.{COLOR_RESET}")
        logger.error(f"Error reading file from path: {user_input}")
        sys.exit(1)

    hash_value = sha256_hash(code.encode())
    entropy_value = shannon_entropy_scipy(code.encode())
    magika_out = magika_type(user_input)

    logger = configure_main_logger(hash_value)

    docker_running = "Yes" if is_docker_running() else "No"

    llm_analyzer = LLMAnalyzer(logger=logger)
    default_model_for_costing = list(USER_MODEL.keys())[0]

    input_tokens_raw_code = llm_analyzer.count_tokens(code)
    estimated_cost_raw_code = calculate_input_cost(default_model_for_costing, input_tokens_raw_code)

    print(f"\n{HEADER_COLOR}--- Analysis Report Header ---{COLOR_RESET}")
    print(f"{INFO_COLOR}[+] - User Input: {user_input}{COLOR_RESET}")
    print(f"{INFO_COLOR}[+] - SHA256:{hash_value}{COLOR_RESET}")
    print(f"{INFO_COLOR}[+] - Shannon Entropy: {entropy_value}{COLOR_RESET}")
    print(f"{INFO_COLOR}[+] - Magika File Type: {magika_out.label}{COLOR_RESET}")
    print(f"{INFO_COLOR}[+] - Magika MimeType: {magika_out.mime_type}{COLOR_RESET}")
    print(f"{INFO_COLOR}[+] - Magika Description: {magika_out.description}{COLOR_RESET}")
    print(f"{INFO_COLOR}[+] - Initial Raw Code Tokens: {HIGHLIGHT_COLOR}{input_tokens_raw_code}")
    if estimated_cost_raw_code is not None:
        print(
            f"{INFO_COLOR}[+] - Estimated Cost for Raw Code: {HIGHLIGHT_COLOR}${estimated_cost_raw_code:.6f} USD{COLOR_RESET}")

    model_selection = questionary.select(
        f"\n[!] - Choose a Gemini Model for analyzing functions: (Not responsible for the cost.)",
        choices=list(USER_MODEL.keys())
    ).ask()
    llm_analyzer.model_name = model_selection

    persona_key = questionary.select(
        f"\n[!] - Choose a System Instruction Persona:",
        choices=list(SYSTEM_INSTRUCTION_SUMMARY_PERSONAS.keys())
    ).ask()

    prompt_key = questionary.select(
        f"\n[!] - Choose a Detailed Analysis Prompt:",
        choices=list(PROMPT_TEMPLATES.keys())
    ).ask()



    selected_prompt_template_primary = PROMPT_TEMPLATES[prompt_key]
    selected_system_instruction_primary = SYSTEM_INSTRUCTION_SUMMARY_PERSONAS[persona_key]

    # Set the context in LLMAnalyzer for the primary analysis
    llm_analyzer.set_analysis_context(selected_system_instruction_primary, selected_prompt_template_primary)
    #  End Primary Analysis Prompt Selection Logic

    combine_llm_report_header = (
        f"{ascii_art}\n\n\n[+] - User Input: {user_input}\n[+] - SHA256:{hash_value}\n[+] - Shannon Entropy: {entropy_value}\n"
        f"[+] - Magika File Type: {magika_out.label}\n"
        f"[+] - Magika MimeType: {magika_out.mime_type}\n[+] - Magika Description: {magika_out.description}\n"
        f"[+] - Initial Raw Code Tokens: {input_tokens_raw_code}\n[+] - Estimated Cost for Raw Code: ${estimated_cost_raw_code if estimated_cost_raw_code is not None else 'N/A'} USD\n"
        f"[+] - Gemini Model: {model_selection}\n"
        f"[+] - Primary Analysis Focus: ({prompt_key})\n"
    )

    print(f"{INFO_COLOR}[+] - Gemini Model: {model_selection}{COLOR_RESET}")
    print(f"{INFO_COLOR}[+] - Primary Analysis Focus: ({prompt_key}){COLOR_RESET}")

    available_analysis_methods = list(ANALYSIS_METHOD.keys())
    # If a prompt allows Docker execution, `Graph Only` is not suitable with it
    if prompt_key == "Focus: Deobfuscation Code Generation (Option: Execute Code in Docker Container)":
        if "Graph Only" in available_analysis_methods:
            available_analysis_methods.remove("Graph Only")
            logger.info("Removed 'Graph Only' option as 'Deobfuscation' prompt was chosen.")
            print(
                f"{WARNING_COLOR}[!] 'Graph Only' is not directly suitable for Deobfuscation Code Generation. Removed from options.{COLOR_RESET}")

    tree, analysis_results_raw_dict = generate_ast(magika_out.label, code.encode())

    #  Initialize primary analysis variables here, so they're always defined
    primary_analysis_llm_response_content = ""
    analysis_method = "N/A"  # Default value, will be updated by questionary if options present

    # Set default description if no specific analysis method is chosen or callable below
    primary_analysis_description = "[No primary analysis performed due to user choice or technical limitations.]"
    #  End initialization


    if analysis_results_raw_dict and tree:
        analysis_method = questionary.select(
            f"\nChoose analysis method:",
            choices=list(available_analysis_methods)
        ).ask()

    print(f"{INFO_COLOR}[+] - Analysis Method: {analysis_method}{COLOR_RESET}\n")
    combine_llm_report_header += f"[+] - Analysis Method: {analysis_method}\n"

    # Restructure the analysis flow based on `analysis_method`
    if analysis_method == "Code Summary Only":
        primary_analysis_description = "Code Summary"
        print(f"{HEADER_COLOR}[+] - Performing code summary only...{COLOR_RESET}")
        code_tokens = llm_analyzer.count_tokens(code)
        estimated_cost_summary = calculate_input_cost(llm_analyzer.model_name, code_tokens)
        print(f"{INFO_COLOR}[+] - Code Token Count: {HIGHLIGHT_COLOR}{code_tokens}{COLOR_RESET}")
        if estimated_cost_summary is not None:
            print(
                f"{INFO_COLOR}[+] - Gemini Estimated Cost For Code Summary: {HIGHLIGHT_COLOR}${estimated_cost_summary:.6f} USD{COLOR_RESET}")

        if code_tokens < 2000000:
            response = llm_analyzer.generate_code_summary(code)
            if response:
                primary_analysis_llm_response_content = llm_analyzer.process_llm_response(response,
                                                                                          prompt_key,
                                                                                          combine_llm_report_header,
                                                                                          docker_running,
                                                                                          code, hash_value,
                                                                                          model_selection)
        else:
            print(
                f"{WARNING_COLOR}[!] Token limit exceeded for Code Summary. Skipping LLM generation.{COLOR_RESET}")
            logger.warning(
                f"Code Summary token limit exceeded ({code_tokens} tokens). Skipping LLM response generation.")

    elif analysis_results_raw_dict and tree:  # This block handles AST/Graph for Detailed Analysis only
        apply_ast_trimming = False
        if analysis_method in ["AST Only", "AST and Graph Analysis"]:
            apply_ast_trimming = questionary.confirm(
                f"Apply AST noise reduction (flattening) to the AST?"
            ).ask()

        ast_dict_for_analysis = analysis_results_raw_dict
        if apply_ast_trimming:
            logger.info("Applying AST noise reduction.")
            ast_dict_for_analysis = trim_ast_json_dict(analysis_results_raw_dict)
            if ast_dict_for_analysis is None or (isinstance(ast_dict_for_analysis, list) and not ast_dict_for_analysis):
                print(
                    f"{WARNING_COLOR}[!] Warning: AST became empty after trimming. Skipping AST analysis.{COLOR_RESET}")
                logger.warning("Trimmed AST is empty. Skipping analysis.")
                ast_json = "{}"
            elif isinstance(ast_dict_for_analysis, list):
                logger.info("Root node was flattened, re-wrapping under 'program' style node.")
                ast_dict_for_analysis = {
                    "type": f"trimmed_program_root_{magika_out.ct_label}",
                    "text": "--- TRIMMED ROOT ---",
                    "children": ast_dict_for_analysis
                }
        ast_json = json.dumps(ast_dict_for_analysis, indent=2)

        if analysis_method == "AST Only":
            primary_analysis_description = "AST Only"
            write_to_file(f"{hash_value}_AST_out.txt", ast_json)

            ast_tokens = llm_analyzer.count_tokens(ast_json)
            estimated_cost_ast = calculate_input_cost(llm_analyzer.model_name, ast_tokens)
            print(
                f"{INFO_COLOR}[+] - AST Token Count: {HIGHLIGHT_COLOR}{ast_tokens}{COLOR_RESET} ({(f'Trimmed' if apply_ast_trimming else 'Raw')})")
            if estimated_cost_ast is not None:
                print(
                    f"{INFO_COLOR}[+] - Gemini Estimated Cost For AST Analysis: {HIGHLIGHT_COLOR}${estimated_cost_ast:.6f} USD{COLOR_RESET}")

            if ast_tokens < 2000000:
                response = llm_analyzer.generate_ast(ast_json)
                if response:
                    primary_analysis_llm_response_content = llm_analyzer.process_llm_response(response,
                                                                                              prompt_key,
                                                                                              combine_llm_report_header,
                                                                                              docker_running,
                                                                                              code, hash_value,
                                                                                              model_selection)
            else:
                print(
                    f"{WARNING_COLOR}[!] Token limit exceeded for AST analysis. Skipping LLM generation.{COLOR_RESET}")
                logger.warning(f"AST token limit exceeded ({ast_tokens} tokens). Skipping LLM response generation.")


        elif analysis_method == "Graph Only":
            primary_analysis_description = "Graph Only"
            graph = convert_ast_to_graph(tree.root_node)

            apply_graph_trimming = questionary.confirm(
                f"Apply shortest path trimming to the graph?"
            ).ask()
            if apply_graph_trimming:
                logger.info("Applying shortest path trimming to the graph.")
                graph_to_analyze = get_all_shortest_path_edges_subgraph(graph, weight_attribute=None)
            else:
                graph_to_analyze = graph
                logger.info("Skipping shortest path trimming.")

            graph_json = json.dumps(nx.node_link_data(graph_to_analyze), indent=2)
            write_to_file(f"{hash_value}_GRAPH_out.txt", graph_json)

            graph_tokens = llm_analyzer.count_tokens(graph_json)
            estimated_cost_graph = calculate_input_cost(llm_analyzer.model_name, graph_tokens)
            print(
                f"{INFO_COLOR}[+] - NetworkX Graph Token Count: {HIGHLIGHT_COLOR}{graph_tokens}{COLOR_RESET} ({(f'Trimmed from {graph.number_of_edges()} edges' if apply_graph_trimming else 'Raw')})"
            )
            if estimated_cost_graph is not None:
                print(
                    f"{INFO_COLOR}[+] - Gemini Estimated Cost For Graph Analysis: {HIGHLIGHT_COLOR}${estimated_cost_graph:.6f} USD{COLOR_RESET}")

            if graph_tokens < 2000000:
                response = llm_analyzer.generate_graph(graph_json)
                if response:
                    primary_analysis_llm_response_content = llm_analyzer.process_llm_response(response,
                                                                                              prompt_key,
                                                                                              combine_llm_report_header,
                                                                                              docker_running,
                                                                                              code, hash_value,
                                                                                              model_selection)
            else:
                print(
                    f"{WARNING_COLOR}[!] Token limit exceeded for Graph analysis. Skipping LLM generation.{COLOR_RESET}")
                logger.warning(f"Graph token limit exceeded ({graph_tokens} tokens). Skipping LLM response generation.")


        elif analysis_method == "AST and Graph Analysis":
            primary_analysis_description = "AST and Graph Analysis"
            write_to_file(f"{hash_value}_AST_out.txt", ast_json)

            graph = convert_ast_to_graph(tree.root_node)
            apply_graph_trimming = questionary.confirm(
                f"Apply shortest path trimming to the graph?"
            ).ask()
            if apply_graph_trimming:
                logger.info("Applying shortest path trimming to the graph.")
                graph_to_analyze = get_all_shortest_path_edges_subgraph(graph, weight_attribute=None)
            else:
                graph_to_analyze = graph
                logger.info("Skipping shortest path trimming.")

            graph_json = json.dumps(nx.node_link_data(graph_to_analyze), indent=2)
            write_to_file(f"{hash_value}_GRAPH_out.txt", graph_json)

            ast_tokens = llm_analyzer.count_tokens(ast_json)
            graph_tokens = llm_analyzer.count_tokens(graph_json)
            print(
                f"{INFO_COLOR}[+] - AST Token Count: {HIGHLIGHT_COLOR}{ast_tokens}{COLOR_RESET} ({(f'Trimmed' if apply_ast_trimming else 'Raw')})")
            print(
                f"{INFO_COLOR}[+] - NetworkX Graph Token Count: {HIGHLIGHT_COLOR}{graph_tokens}{COLOR_RESET} ({(f'Trimmed from {graph.number_of_edges()} edges' if apply_graph_trimming else 'Raw')})"
            )
            total_tokens = ast_tokens + graph_tokens
            estimated_cost_combined = calculate_input_cost(llm_analyzer.model_name, total_tokens)
            if estimated_cost_combined is not None:
                print(
                    f"{INFO_COLOR}[+] - Gemini Estimated Cost For AST & Graph Analysis: {HIGHLIGHT_COLOR}${estimated_cost_combined:.6f} USD{COLOR_RESET}")

            if total_tokens < 2000000:
                response = llm_analyzer.generate_ast_graph(ast_json, graph_json)
                if response:
                    primary_analysis_llm_response_content = llm_analyzer.process_llm_response(response,
                                                                                              prompt_key,
                                                                                              combine_llm_report_header,
                                                                                              docker_running,
                                                                                              code, hash_value,
                                                                                              model_selection)
            else:
                print(
                    f"{WARNING_COLOR}[!] Token limit exceeded for AST/Graph analysis. Skipping LLM generation.{COLOR_RESET}")
                logger.warning(
                    f"Combined AST/Graph token limit exceeded ({total_tokens} tokens). Skipping LLM response generation.")

    else:  # Tree-sitter error or no tree, fallback to general summarization for primary analysis
        primary_analysis_description = f"Primary Analysis: Fallback Code Summary (Prompt: {prompt_key} due to Tree-Sitter error)"
        print(
            f"{WARNING_COLOR}[!] - Tree Sitter error or grammar not supported for Magika type. Falling back to general summarization for this primary analysis.{COLOR_RESET}")
        code_tokens = llm_analyzer.count_tokens(code)
        estimated_cost_fallback = calculate_input_cost(llm_analyzer.model_name, code_tokens)
        print(f"{INFO_COLOR}[+] - Code Token Count: {HIGHLIGHT_COLOR}{code_tokens}{COLOR_RESET}")
        if estimated_cost_fallback is not None:
            print(
                f"{INFO_COLOR}[+] - Gemini Estimated Cost For Fallback Summary: {HIGHLIGHT_COLOR}${estimated_cost_fallback:.6f} USD{COLOR_RESET}")

        if code_tokens < 2000000:
            # For fallback, use the 'General Summary' persona prompt as a default summary path
            fallback_system_instruction = SYSTEM_INSTRUCTION_SUMMARY_PERSONAS["General Summary"]
            fallback_prompt_template = SUMMARY_PROMPT_TEMPLATES["General Summary"]
            llm_analyzer.set_analysis_context(fallback_system_instruction, fallback_prompt_template)

            print(
                f"{INFO_COLOR}[+] - Using Fallback System Instruction for Primary Analysis: 'General Summary' persona.{COLOR_RESET}")

            response_from_llm = llm_analyzer.generate_code_summary(code)
            if response_from_llm:
                primary_analysis_llm_response_content = llm_analyzer.process_llm_response(response_from_llm,
                                                                                          prompt_key,
                                                                                          "", docker_running, code,
                                                                                          hash_value, model_selection)
        else:
            print(
                f"{WARNING_COLOR}[!] Token limit exceeded for fallback Code Summary. Skipping LLM generation.{COLOR_RESET}")
            logger.warning(
                f"Fallback Code Summary token limit exceeded ({code_tokens} tokens). Skipping LLM response generation.")

    #  Overall Code Summary Section (User-selectable Persona)
    print(f"\n{HEADER_COLOR}--- Overall Code Summary ---{COLOR_RESET}")

    # Allow user to select persona for the FINAL summary
    final_summary_persona_choice = questionary.select(
        f"\n[!] - Choose a prompt for the Overall Code Summary Report:",
        choices=list(SUMMARY_PROMPT_TEMPLATES.keys())
    ).ask()

    final_summary_prompt_template = SUMMARY_PROMPT_TEMPLATES[final_summary_persona_choice]
    llm_analyzer.set_analysis_context(selected_system_instruction_primary, final_summary_prompt_template)
    print(
        f"{INFO_COLOR}[+] - Using System Instruction for Final Summary: '{final_summary_persona_choice}' persona.{COLOR_RESET}")

    final_summary_code_tokens = llm_analyzer.count_tokens(code)
    estimated_cost_final_summary = calculate_input_cost(llm_analyzer.model_name, final_summary_code_tokens)

    print(f"{INFO_COLOR}[+] - Final Summary Code Tokens: {HIGHLIGHT_COLOR}{final_summary_code_tokens}{COLOR_RESET}")
    if estimated_cost_final_summary is not None:
        print(
            f"{INFO_COLOR}[+] - Estimated Cost for Final Summary: {HIGHLIGHT_COLOR}${estimated_cost_final_summary:.6f} USD{COLOR_RESET}")

    final_summary_response_text = ""
    if final_summary_code_tokens < 2000000:
        final_summary_response_text = llm_analyzer.generate_code_summary(code)
        if not final_summary_response_text:
            print(f"{WARNING_COLOR}[!] No final summary response generated or it was empty.{COLOR_RESET}")
            logger.warning("No final summary response generated for final summary.")
    else:
        print(f"{WARNING_COLOR}[!] Token limit exceeded for Final Summary. Skipping LLM generation.{COLOR_RESET}")
        logger.warning(
            f"Final Summary token limit exceeded ({final_summary_code_tokens} tokens). Skipping LLM response generation.")

    #  UNIFIED OUTPUT FILE GENERATION
    final_output_filename = f"{hash_value}_{model_selection}_Report.txt"

    # Use combine_llm_report_header as the base
    full_report_content = f"{combine_llm_report_header}"

    # Primary Analysis Section
    full_report_content += f"\n\n--- Primary Analysis Report (Prompt: {prompt_key} | Method: {analysis_method}) ---\n"
    full_report_content += f"This section provides the LLM's primary analysis based on the selected prompt and chosen analysis method.\n"
    full_report_content += f"{primary_analysis_llm_response_content if primary_analysis_llm_response_content else '[No primary LLM analysis content generated or applicable, either due to an error, token limits, or user choice.]'}\n"

    # Overall Code Summary Section
    full_report_content += f"\n\n--- Overall Code Summary Report (Persona: {final_summary_persona_choice}) ---\n"
    full_report_content += f"This section provides a high-level summary of the entire code, leveraging the selected persona.\n"
    if final_summary_response_text:
        full_report_content += f"\nLLM Summary:\n{'-' * 100}\n\n{final_summary_response_text}\n\n{'-' * 100}\n"
    else:
        full_report_content += "[No final code summary generated due to token limits or LLM issues.]\n"

    write_to_file(final_output_filename, full_report_content)
    print(f"{INFO_COLOR}[+] All analysis results written to: {final_output_filename}{COLOR_RESET}")
    logger.info(f"Final report written to {final_output_filename}")